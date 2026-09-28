"""Inference helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th


class PolicyRunner:
    """Thin wrapper that keeps recurrent extractors honest between episodes."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.extractor = getattr(policy, "features_extractor", None)
        self.has_recurrence = self.extractor is not None and hasattr(self.extractor, "reset_hidden")

    def reset(self) -> None:
        if self.has_recurrence:
            self.extractor.reset_hidden()

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, Any]:
        return self.policy.predict(observation, deterministic=deterministic)

    def sharpened(self, observation: np.ndarray, multiplier: float, indices) -> np.ndarray:
        """Bernoulli-sample per-bit probabilities, boosting ``indices``.

        Used by the ``deploy.aggressiveness`` knob. Bypasses the policy's own
        action post-processing, so it is only valid for MultiBinary spaces.
        """
        from stable_baselines3.common.utils import obs_as_tensor

        with th.no_grad():
            tensor = obs_as_tensor(np.asarray(observation), self.policy.device)
            features = self.policy.extract_features(tensor)
            latent_pi, _ = self.policy.mlp_extractor(features)
            probs = th.sigmoid(self.policy.action_net(latent_pi)).cpu().numpy().reshape(-1)

        probs = np.clip(probs, 0.0, 1.0)
        for i in indices:
            if i < probs.shape[0]:
                probs[i] = min(probs[i] * multiplier, 1.0)
        return (np.random.rand(probs.shape[0]) < probs).astype(np.float32)


def load_policy(path, device: str = "cpu"):
    """Load an SB3 checkpoint, preferring the CNN actor-critic used for training."""
    from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

    for klass in (ActorCriticCnnPolicy, ActorCriticPolicy):
        try:
            return klass.load(path, device=device)
        except Exception:  # noqa: BLE001 - fall through to the next candidate
            continue
    raise RuntimeError(f"could not load a policy from {path}")
