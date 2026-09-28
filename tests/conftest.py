"""Shared fixtures. Every test runs against a throwaway ``runs/`` tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th
from imitation.data.types import Trajectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def runs_root(tmp_path, monkeypatch):
    """Point the runs/ root at a temporary directory for the duration of a test."""
    monkeypatch.setenv("IMITATION_RUNS", str(tmp_path))
    return tmp_path


def make_demos(directory: Path, *, num_actions: int = 18, trajectories: int = 2,
               frames: int = 40, seed: int = 0, p_fire: float = 0.3,
               width_override: int | None = None) -> list:
    """Write synthetic ``demo*.pt`` files and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(trajectories):
        width = width_override or num_actions
        obs = rng.integers(0, 256, size=(frames + 1, 4, 128, 128), dtype=np.uint8)
        acts = (rng.random((frames, width)) > (1 - p_fire)).astype(np.float32)
        path = directory / f"demo_{i}_test.pt"
        th.save([Trajectory(obs=obs, acts=acts, infos=None, terminal=False)], path)
        paths.append(path)
    return paths


@pytest.fixture()
def corpus(runs_root):
    """A hajime-shaped synthetic corpus under the temporary runs root."""
    demo_dir = runs_root / "hajime_ippo" / "demos"
    make_demos(demo_dir, num_actions=18)
    return demo_dir


class FakeEnv:
    """Stand-in for the *wrapped* (VecEnv) environment used by the recorder tests.

    SB3 VecEnv.step returns a 4-tuple ``(obs, rewards, dones, infos)``.
    """

    def __init__(self, num_actions: int = 18, frames: int = 4, width: int = 128):
        self.num_actions = num_actions
        self.shape = (frames, width, width)
        self.rng = np.random.default_rng(0)
        self.steps = 0

    def reset(self):
        return self.rng.integers(0, 256, size=(1,) + self.shape, dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        obs = self.rng.integers(0, 256, size=(1,) + self.shape, dtype=np.uint8)
        return obs, np.array([0.0]), np.array([False]), [{}]

    def close(self):
        pass

    def get_attr(self, *a, **k):
        return []


@pytest.fixture()
def fake_env():
    return FakeEnv()
