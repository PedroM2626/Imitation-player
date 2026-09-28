"""
Architecture registry.

One place that knows which feature extractor corresponds to which name, its
constructor defaults, and the checkpoint prefix its training script writes. The
trainer, the deployer and the benchmark all read this, so adding an encoder is
one entry instead of three copies of a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import gymnasium as gym


@dataclass
class Architecture:
    name: str
    checkpoint_prefix: str
    factory: Callable[..., Any]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    #: None means "let stable_baselines3 use its default NatureCNN encoder".
    policy_class: Optional[str] = "ActorCriticCnnPolicy"
    note: str = ""


def _registry():
    from agent.utils.impoola_cnn import ImpoolaCNNExtractor
    from agent.utils.new_architectures import ImpalaCNNExtractor, ResNet18Extractor
    from agent.utils.temporal_lstm import TemporalAttentionLSTM
    from agent.utils.vision_transformer import VisionTransformerExtractor

    return {
        "naturecnn": Architecture(
            name="NatureCNN", checkpoint_prefix="NatureCNN_policy", factory=None, kwargs={},
            policy_class=None,
            note="stable-baselines3 default three-layer CNN; the study's reference point",
        ),
        "lstm": Architecture(
            name="CNN_LSTM", checkpoint_prefix="bc_policy_lstm", factory=TemporalAttentionLSTM,
            kwargs=dict(features_dim=512, lstm_hidden_size=256, lstm_num_layers=2),
            note="5-conv CNN + bidirectional LSTM + additive temporal attention (stateful)",
        ),
        "transformer": Architecture(
            name="ViT_Transformer", checkpoint_prefix="bc_policy_transformer",
            factory=VisionTransformerExtractor,
            kwargs=dict(features_dim=512, embed_dim=256, patch_size=16, num_heads=4, num_layers=4),
            note="ViT over 256 patches of a 4-frame stack, with temporal embeddings",
        ),
        "impoola": Architecture(
            name="Impoola_CNN", checkpoint_prefix="ImpoolaCNN_policy", factory=ImpoolaCNNExtractor,
            kwargs=dict(features_dim=512, channels_list=[32, 64, 128]),
            note="residual CNN with Global Average Pooling instead of Flatten",
        ),
        "impala": Architecture(
            name="Impala_CNN", checkpoint_prefix="Impala_CNN_policy", factory=ImpalaCNNExtractor,
            kwargs=dict(features_dim=512, channels_list=[32, 64, 128]),
            note="IMPALA CNN terminating in a 32,768-wide Flatten head",
        ),
        "resnet18": Architecture(
            name="ResNet18", checkpoint_prefix="ResNet18_policy", factory=ResNet18Extractor,
            kwargs=dict(features_dim=512), note="torchvision ResNet-18, trained from scratch",
        ),
    }


ARCHITECTURES: Dict[str, Architecture] = _registry()


def get(key: str) -> Architecture:
    try:
        return ARCHITECTURES[key]
    except KeyError:
        raise KeyError(f"unknown architecture {key!r}; choose from "
                       f"{', '.join(sorted(ARCHITECTURES))}") from None


def policy_kwargs_for(arch: Architecture, observation_space: gym.spaces.Box,
                      training_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """``policy_kwargs`` for stable-baselines3, or ``None`` for the default CNN."""
    if arch.factory is None:
        return None
    kwargs = dict(arch.kwargs)
    if arch.name == "CNN_LSTM" and training_config:
        # The extractor's window used to be unreachable; honour the config now.
        kwargs["window_size"] = int(training_config.get("window_size", kwargs.get("window_size", 10)))
    return {"features_extractor_class": arch.factory, "features_extractor_kwargs": kwargs}
