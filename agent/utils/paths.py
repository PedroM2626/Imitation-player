"""
Where a profile's data lives.

Everything produced by a run is rooted at ``runs/<profile>/`` so that two games
never share a demonstration folder, and so that the code is independent of the
current working directory. ``IMITATION_RUNS`` overrides the root.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return Path(os.environ.get("IMITATION_RUNS", REPO_ROOT / "runs"))


def profile_root(profile: str) -> Path:
    return runs_root() / profile


def demos_dir(profile: str) -> Path:
    return profile_root(profile) / "demos"


def models_dir(profile: str) -> Path:
    return profile_root(profile) / "models"


def mlruns_dir(profile: str) -> Path:
    return profile_root(profile) / "mlruns"


def logs_dir(profile: str) -> Path:
    return profile_root(profile) / "logs"


def ensure_dirs(profile: str, *dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
