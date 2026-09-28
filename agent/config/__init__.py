"""
Profile loader.

A *profile* is one game's configuration: which window to capture, how big the
observation is, how actions are emitted, and what the action vector means.
Profiles live in ``agent.config.profiles``; the active one is chosen by the
``--profile`` flag of every CLI script, or by the ``IMITATION_PROFILE``
environment variable.

Machine-specific paths (emulator executable, ROM) do not belong in a profile,
because they are not portable. Put them in ``agent/config/local.py`` (copied
from ``local.example.py``, git-ignored); anything there overrides the profile.
"""

from __future__ import annotations

import copy
import importlib
import os
from typing import Any

DEFAULT_PROFILE = "hajime_ippo"
_PROFILE_PACKAGE = "agent.config.profiles"
_LOCAL_MODULE = "agent.config.local"

#: Keys a local override may replace, at the top level of GAME_CONFIG.
_LOCAL_KEYS = ("exe_path", "rom_path", "process_name")


def available_profiles() -> list:
    from agent.config import profiles as _pkg

    directory = os.path.dirname(_pkg.__file__)
    return sorted(
        name[:-3]
        for name in os.listdir(directory)
        if name.endswith(".py") and not name.startswith("_")
    )


def active_profile_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("IMITATION_PROFILE", DEFAULT_PROFILE)


def _apply_local_overrides(game_config: dict[str, Any]) -> dict[str, Any]:
    try:
        local = importlib.import_module(_LOCAL_MODULE)
    except ImportError:
        return game_config

    overrides = getattr(local, "LOCAL_OVERRIDES", {}) or {}
    for profile_name, values in overrides.items():
        if profile_name == game_config.get("profile"):
            for key, value in values.items():
                if key in _LOCAL_KEYS:
                    game_config[key] = value
    return game_config


def load_profile(name: str | None = None) -> dict[str, Any]:
    """Return ``{"name", "GAME_CONFIG", "TRAINING_CONFIG"}`` for one game."""
    profile = active_profile_name(name)
    try:
        module = importlib.import_module(f"{_PROFILE_PACKAGE}.{profile}")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Unknown profile {profile!r}. Available: {', '.join(available_profiles())}"
        ) from exc

    game_config = copy.deepcopy(module.GAME_CONFIG)
    game_config.setdefault("profile", profile)
    game_config = _apply_local_overrides(game_config)

    return {
        "name": profile,
        "GAME_CONFIG": game_config,
        "TRAINING_CONFIG": copy.deepcopy(getattr(module, "TRAINING_CONFIG", {})),
    }


def game_config(name: str | None = None) -> dict[str, Any]:
    return load_profile(name)["GAME_CONFIG"]


def num_actions(config: dict[str, Any]) -> int:
    actions = config.get("actions", {})
    declared = int(actions.get("num_actions", len(actions.get("mappings", []))))
    mappings = int(len(actions.get("mappings", [])))
    if declared != mappings:
        raise ValueError(
            f"profile {config.get('profile')!r}: actions.num_actions is {declared} "
            f"but actions.mappings has {mappings} entries; they must agree"
        )
    return declared
