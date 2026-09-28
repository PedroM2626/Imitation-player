"""Shared CLI wiring: profile selection, device resolution, environment building."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Allow `python agent/cli/x.py` as well as `python -m agent.cli.x`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.config import load_profile, num_actions  # noqa: E402
from agent.utils import paths  # noqa: E402
from agent.utils.game_env import GenericGameEnv  # noqa: E402


def add_common(parser: argparse.ArgumentParser, training: bool = False) -> None:
    parser.add_argument("--profile", default=os.environ.get("IMITATION_PROFILE"),
                        help="game profile to use (see agent/config/profiles/) "
                             f"[env IMITATION_PROFILE, default hajime_ippo]")
    parser.add_argument("--runs-root", default=None,
                        help="override the runs/ directory holding demos, models and mlruns")
    if training:
        parser.add_argument("--device", default=None,
                            help="torch device; defaults to cuda when available, else cpu")


def resolve_device(explicit: Optional[str]) -> str:
    """Honour an explicit --device, otherwise pick the best available."""
    if explicit:
        return explicit
    if th.cuda.is_available():
        print(f"[OK] GPU detected: {th.cuda.get_device_name(0)} (CUDA {th.version.cuda})")
        return "cuda"
    print("[!] CUDA not found, using CPU (much slower)")
    return "cpu"


def load(explicit: Optional[str], runs_root: Optional[str] = None) -> Dict[str, Any]:
    profile = load_profile(explicit)
    if runs_root:
        os.environ["IMITATION_RUNS"] = runs_root
    return profile


def build_config(game_config: Dict[str, Any], dummy: bool) -> Dict[str, Any]:
    """Copy a profile's GAME_CONFIG and force the dummy (no-window) flag."""
    config = dict(game_config)
    config["dummy"] = dummy
    return config


def wrapped_env(config: Dict[str, Any]) -> DummyVecEnv:
    """DummyVecEnv -> VecTransposeImage -> VecFrameStack(4): (1,128,128) -> (4,128,128)."""
    env = GenericGameEnv(config)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def action_names(profile: Dict[str, Any]) -> list:
    return [m.get("name", f"ACT_{i}") for i, m in enumerate(profile["GAME_CONFIG"]["actions"]["mappings"])]


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70 + "\n")


def cli_entry(func):
    """Turn expected operational failures into a clean message, not a traceback."""

    def wrapper(argv=None):
        from agent.utils.demos import DemoError
        from agent.utils.game_env import WindowNotFoundError

        try:
            return func(argv)
        except (DemoError, WindowNotFoundError) as exc:
            raise SystemExit(f"[!] {exc}") from exc

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper


__all__ = ["add_common", "resolve_device", "load", "build_config", "wrapped_env",
           "action_names", "header", "cli_entry", "num_actions", "paths", "np"]
