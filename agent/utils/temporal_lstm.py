"""
CNN + bidirectional-LSTM + additive temporal attention extractor.

Moved out of ``game_env.py`` unchanged in behaviour: the network, the internal
feature buffer and the gradient handling are byte-for-byte equivalent to the
version that produced the published benchmark numbers, so results stay
comparable. The only change is that the temporal window comes from
``TRAINING_CONFIG['window_size']`` (default 10, the previous hard-coded value)
instead of being unreachable.

Known property, deliberately not "fixed" here: the module is stateful across
``forward()`` calls, which makes its output depend on call order. That is
documented in docs/LIMITATIONS.md (L12) and changing it would invalidate the
CNN+LSTM row of the benchmark.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalAttentionLSTM(BaseFeaturesExtractor):
    """Remembers the last ``window_size`` frames instead of reacting to one."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        window_size: int = 10,
        debug: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.n_frames = observation_space.shape[0]
        self.frame_height = observation_space.shape[1]
        self.frame_width = observation_space.shape[2]
        self.debug = debug

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_frames, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size * 2, 1),
        )

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )

        self.hidden_state = None
        self.hidden_reset = True
        self.window_size = int(window_size)
        self.feature_buffer: deque = deque(maxlen=self.window_size)

    def repackage_hidden(self, h):
        """Detach the computation history to keep the graph bounded."""
        if isinstance(h, th.Tensor):
            return h.detach()
        return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        device = observations.device

        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5

        cnn_features = self.cnn(x)

        if batch_size == 1:
            self.feature_buffer.append(cnn_features.detach())
        else:
            self.feature_buffer.append(cnn_features)
            for i in range(len(self.feature_buffer) - 1):
                self.feature_buffer[i] = self.feature_buffer[i].detach()

        if len(self.feature_buffer) == 1:
            while len(self.feature_buffer) < self.window_size:
                self.feature_buffer.append(self.feature_buffer[-1])

        sequence = th.stack(list(self.feature_buffer), dim=1)
        sequence.requires_grad_(True)

        should_reset = (
            self.hidden_reset
            or self.hidden_state is None
            or self.hidden_state[0].shape[1] != batch_size
        )

        if should_reset:
            num_directions = 2
            hidden_size_total = self.lstm_num_layers * num_directions
            h0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            c0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            current_hidden = (h0, c0)
            self.hidden_reset = False
        else:
            current_hidden = self.repackage_hidden(self.hidden_state)

        lstm_out, last_hidden = self.lstm(sequence, current_hidden)
        self.hidden_state = last_hidden

        attention_weights = th.softmax(self.attention(lstm_out), dim=1)
        context = th.sum(attention_weights * lstm_out, dim=1)

        return self.linear(context)

    def reset_hidden(self, dones: th.Tensor | None = None) -> None:
        """Clear the recurrent state. Call this at episode boundaries."""
        self.hidden_state = None
        self.hidden_reset = True
        self.feature_buffer.clear()
