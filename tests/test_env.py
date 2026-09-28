"""The dummy environment is importable and usable without a window or a pad."""

import gymnasium as gym
import numpy as np
import pytest

from agent.cli.common import build_config, wrapped_env
from agent.config import load_profile
from agent.utils.game_env import GenericGameEnv


@pytest.fixture()
def dummy_config():
    return build_config(load_profile("hajime_ippo")["GAME_CONFIG"], dummy=True)


def test_spaces_are_declared_from_the_profile(dummy_config):
    env = GenericGameEnv(dummy_config)
    assert env.observation_space.shape == (128, 128, 1)
    assert env.observation_space.dtype == np.uint8
    assert isinstance(env.action_space, gym.spaces.MultiBinary)
    assert env.action_space.n == 18


def test_dummy_step_refuses_to_actuate(dummy_config):
    """Stepping a dummy env must be loud: it has no device to write to."""
    env = GenericGameEnv(dummy_config)
    with pytest.raises(RuntimeError, match="cannot actuate"):
        env.step(np.zeros(18, dtype=np.float32))


def test_dummy_reset_returns_a_black_frame(dummy_config):
    env = GenericGameEnv(dummy_config)
    obs, info = env.reset()
    assert obs.shape == (128, 128, 1)
    assert obs.dtype == np.uint8
    assert np.all(obs == 0)
    assert info == {}


def test_action_width_is_validated(dummy_config):
    env = GenericGameEnv(dummy_config)
    with pytest.raises(ValueError, match="width 9"):
        env.step(np.zeros(9, dtype=np.float32))


def test_wrapped_observation_is_four_stacked_channels(dummy_config):
    vec = wrapped_env(dummy_config)
    obs = vec.reset()
    assert obs.shape == (1, 4, 128, 128)
    assert vec.observation_space.shape == (4, 128, 128)


def test_window_timeout_raises_rather_than_degrading(monkeypatch, dummy_config):
    """A missing window used to warn and leave a half-built environment alive."""
    from agent.utils import game_env as ge

    monkeypatch.setattr(GenericGameEnv, "find_window_by_process_name", staticmethod(lambda n: None))
    config = dict(dummy_config)
    config["dummy"] = False
    config["window_timeout_s"] = 0
    config["exe_path"] = None
    env = ge.GenericGameEnv.__new__(ge.GenericGameEnv)
    with pytest.raises((ge.WindowNotFoundError, Exception)) as excinfo:
        ge.GenericGameEnv.__init__(env, config)
    assert (
        excinfo.typename in ("WindowNotFoundError", "RuntimeError")
        or "window" in str(excinfo.value).lower()
    )


def test_windows_flags_are_booleans():
    from agent.utils import windows

    for flag in (windows.HAS_WIN32, windows.HAS_DXCAM, windows.HAS_VGAMEPAD, windows.HAS_MSS):
        assert isinstance(flag, bool)


def test_recorder_and_deploy_import_without_a_display():
    """These modules used to be unimportable off Windows at module scope."""
    import importlib

    for module in ("agent.cli.record", "agent.cli.deploy", "agent.cli.dagger"):
        importlib.import_module(module)
