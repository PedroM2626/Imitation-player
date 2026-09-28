"""
Impoola-CNN feature extractor for Imitation Learning.

An architecture based on Impala-CNN, but using Global Average Pooling (GAP)
instead of the flatten layer (Flatten) before the linear projection.
This makes the model highly invariant to spatial translations, extremely light
(fewer parameters) and with excellent generalization ability.
"""

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """Standard residual block for the Impala/Impoola CNN."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return x + out


class ImpoolaBlock(nn.Module):
    """Main Impoola block: Conv2d -> MaxPool2d -> 2x ResidualBlocks."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpoolaCNNExtractor(BaseFeaturesExtractor):
    """
    Impoola-CNN extractor for Stable-Baselines3.
    Replaces the classic NatureCNN with a residual extractor using Global Average Pooling (GAP).
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 channels_list = [32, 64, 128]):
        super().__init__(observation_space, features_dim)
        
        # The observation space has shape (n_frames, H, W)
        self.n_frames = observation_space.shape[0]
        
        # Build the stack of Impoola blocks
        blocks = []
        in_ch = self.n_frames
        for out_ch in channels_list:
            blocks.append(ImpoolaBlock(in_ch, out_ch))
            in_ch = out_ch
            
        self.features_net = nn.Sequential(*blocks)
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Linear projection to features_dim
        self.head = nn.Sequential(
            nn.Linear(channels_list[-1], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalize pixels to [0, 1] if needed
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        x = self.features_net(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
