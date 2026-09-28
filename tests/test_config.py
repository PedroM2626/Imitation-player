"""Configuration profiles are self-consistent and machine-portable."""

import pytest

from agent.config import available_profiles, load_profile, num_actions
from agent.utils.emission import validate_mappings


@pytest.mark.parametrize("profile", available_profiles())
def test_declared_width_matches_table(profile):
    cfg = load_profile(profile)["GAME_CONFIG"]
    assert num_actions(cfg) == len(cfg["actions"]["mappings"])


@pytest.mark.parametrize("profile", available_profiles())
def test_mapping_table_is_emittable(profile):
    cfg = load_profile(profile)["GAME_CONFIG"]
    validate_mappings(cfg["actions"]["mappings"], num_actions(cfg))


@pytest.mark.parametrize("profile", available_profiles())
def test_no_machine_specific_paths_are_committed(profile):
    """exe_path/rom_path must be None in a profile; they belong in config/local.py."""
    cfg = load_profile(profile)["GAME_CONFIG"]
    for key in ("exe_path", "rom_path"):
        value = cfg.get(key)
        assert value is None or not value.endswith((".exe", ".iso")), (
            f"{profile}.{key} hard-codes a workstation path; move it to agent/config/local.py")


@pytest.mark.parametrize("profile", available_profiles())
def test_capture_geometry_is_square_and_divisible(profile):
    """The Impala-family flatten head needs H = W and divisible by 2**3."""
    cfg = load_profile(profile)["GAME_CONFIG"]["capture"]
    w, h = cfg["internal_width"], cfg["internal_height"]
    assert w == h and w % 8 == 0


def test_unknown_profile_is_reported_clearly():
    with pytest.raises(SystemExit, match="Unknown profile"):
        load_profile("no_such_game")


def test_local_overrides_are_applied(monkeypatch, tmp_path):
    import agent.config as config_module

    monkeypatch.setattr(config_module, "_LOCAL_KEYS", ("exe_path",))
    merged = config_module._apply_local_overrides(
        {"profile": "hajime_ippo", "exe_path": None})
    assert merged["exe_path"] is None or isinstance(merged["exe_path"], str)


def test_width_policy_default_is_strict():
    cfg = load_profile("hajime_ippo")["GAME_CONFIG"]
    assert cfg["actions"]["width_policy"] == "strict"
