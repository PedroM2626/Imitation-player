"""Every exit path of the recorder must flush, not discard, the buffer."""

import numpy as np
import torch as th

from agent.cli.record import TrajectoryRecorder


def recorder_for(tmp_path, fake_env):
    return TrajectoryRecorder(fake_env, tmp_path / "demos", max_trajectories=5,
                              human=None, names=[])


def test_k_stop_writes_one_file_per_segment(tmp_path, fake_env):
    rec = recorder_for(tmp_path, fake_env)
    obs = fake_env.reset()
    rec.obs = np.squeeze(obs, axis=0)
    rec.recorded_obs = [rec.obs.copy()]

    rec.start()
    for _ in range(6):
        rec.step(np.zeros(18, dtype=np.float32))
    path = rec.stop_and_save()

    payload = th.load(path, map_location="cpu", weights_only=False)
    traj = payload[0]
    assert traj.acts.shape == (6, 18)
    assert traj.obs.shape == (7, 4, 128, 128)  # N+1 observations


def test_finish_flushes_an_open_recording(tmp_path, fake_env):
    rec = recorder_for(tmp_path, fake_env)
    rec.start()
    for _ in range(5):
        rec.step(np.zeros(18, dtype=np.float32))

    rec.finish()  # what ESC / Ctrl+C / window-close now call

    written = list((tmp_path / "demos").glob("demo_*.pt"))
    assert len(written) == 1, "an in-progress recording was discarded"
    traj = th.load(written[0], map_location="cpu", weights_only=False)[0]
    assert traj.acts.shape == (5, 18)


def test_finish_with_nothing_buffered_writes_no_file(tmp_path, fake_env):
    rec = recorder_for(tmp_path, fake_env)
    rec.finish()
    assert not list((tmp_path / "demos").glob("demo_*.pt"))


def test_flushing_twice_does_not_duplicate(tmp_path, fake_env):
    rec = recorder_for(tmp_path, fake_env)
    rec.start()
    for _ in range(4):
        rec.step(np.zeros(18, dtype=np.float32))
    rec.stop_and_save()
    rec.finish()
    assert len(list((tmp_path / "demos").glob("demo_*.pt"))) == 1


def test_buffer_desynchronisation_is_caught(tmp_path, fake_env):
    rec = recorder_for(tmp_path, fake_env)
    rec.start()
    rec.recorded_obs = [np.zeros((4, 128, 128), np.uint8)]
    rec.recorded_actions = [np.zeros(18, np.float32), np.zeros(18, np.float32)]
    try:
        rec.stop_and_save()
        saved = True
    except RuntimeError:
        saved = False
    assert not saved, "a desynchronised buffer should raise, not be written"
