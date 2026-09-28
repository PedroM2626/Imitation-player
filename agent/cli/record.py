"""
Record human demonstrations.

    python -m agent.cli.record --profile hajime_ippo

``K`` starts and stops a recording; each stop writes one file. ``ESC``, the
window close button and Ctrl+C all **flush the in-progress buffer before
exiting** -- previously they discarded it, which is how recordings were lost
while the overlay promised "Save & Exit".
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch as th
from imitation.data.types import Trajectory

from agent.cli.common import (
    action_names,
    add_common,
    build_config,
    cli_entry,
    header,
    load,
    wrapped_env,
)
from agent.utils import paths
from agent.utils.input_map import HumanInput


class TrajectoryRecorder:
    """Buffers (observation, action) pairs and writes them out per stop."""

    def __init__(
        self,
        env,
        demo_dir: Path,
        max_trajectories: int = 10,
        human: HumanInput = None,
        names: list[str] = None,
    ):
        self.env = env
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self.max_trajectories = int(max_trajectories)
        self.human = human
        self.names = names or []

        self.is_recording = False
        self.recorded_obs: list[np.ndarray] = []
        self.recorded_actions: list[np.ndarray] = []
        self.count_record = 0
        self.saved_files: list[Path] = []

    def start(self) -> None:
        obs = self.env.reset()
        self.obs = np.squeeze(obs, axis=0)
        self.recorded_obs = [self.obs.copy()]
        self.recorded_actions = []
        self.is_recording = True
        print(f"Recording trajectory {self.count_record}... (press K to stop and save)")

    def stop_and_save(self) -> Path:
        """Flush the buffer to one file. Safe to call with an empty buffer."""
        self.is_recording = False
        if len(self.recorded_actions) < 2:
            print("Nothing to save (buffer shorter than two frames).")
            self.recorded_obs, self.recorded_actions = [], []
            return None

        obs_uint8 = np.stack([np.asarray(o, dtype=np.uint8) for o in self.recorded_obs], axis=0)
        acts = np.asarray(self.recorded_actions, dtype=np.float32)
        if obs_uint8.shape[0] != acts.shape[0] + 1:
            raise RuntimeError(
                f"buffer desynchronised: {obs_uint8.shape[0]} observations for "
                f"{acts.shape[0]} actions (expected N+1)"
            )

        traj = Trajectory(obs=obs_uint8, acts=acts, infos=None, terminal=False)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.demo_dir / f"demo_{self.count_record}_{stamp}.pt"
        th.save([traj], path)
        print(f"[OK] Saved {acts.shape[0]} frames to {path}")

        self.saved_files.append(path)
        self.recorded_obs, self.recorded_actions = [], []
        self.count_record += 1
        return path

    def step(self, action: np.ndarray) -> np.ndarray:
        obs, _, _, _ = self.env.step([action])
        obs = np.squeeze(obs, axis=0)
        if self.is_recording:
            self.recorded_obs.append(obs)
            self.recorded_actions.append(action.copy())
        return obs

    def finish(self) -> None:
        """Guaranteed flush, used by every exit path."""
        if self.is_recording:
            print("Flushing in-progress recording before exiting...")
            self.stop_and_save()


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(parser)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="stop after this many saved files (default: profile's recording.max_trajectories)",
    )
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    n_act = int(config["actions"]["num_actions"])
    demo_dir = paths.demos_dir(profile["name"])

    header("RECORDING DEMONSTRATIONS")
    print(f"profile      : {profile['name']}")
    print(f"input mode   : {config['actions'].get('input_mode', 'gamepad')}")
    print(f"action width : {n_act}")
    print(f"destination  : {demo_dir}")

    env = wrapped_env(build_config(config, dummy=False))
    human = HumanInput(config)
    recorder = TrajectoryRecorder(
        env,
        demo_dir,
        max_trajectories=args.max_trajectories
        or config.get("recording", {}).get("max_trajectories", 10),
        human=human,
        names=action_names(profile),
    )

    try:
        import keyboard
        import pygame
    except ImportError as exc:
        print(f"Missing input/display dependency: {exc}. Install keyboard and pygame.")
        return 1

    pygame.init()
    screen = pygame.display.set_mode((300, 340))
    pygame.display.set_caption("Imitation Player - Capture")
    font = pygame.font.SysFont(None, 22)

    fps_start, fps_frames, fps_avg = time.time(), 0, 0.0
    toggle_debounce = 0.0

    obs = env.reset()
    recorder.obs = np.squeeze(obs, axis=0)
    recorder.recorded_obs = [recorder.obs.copy()]

    exit_reason = "user"
    try:
        while recorder.count_record < recorder.max_trajectories:
            loop_start = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            if keyboard.is_pressed("esc"):
                exit_reason = "esc"
                break

            if keyboard.is_pressed("k") and time.time() - toggle_debounce > 0.3:
                toggle_debounce = time.time()
                if recorder.is_recording:
                    recorder.stop_and_save()
                else:
                    recorder.start()

            action = recorder.human.read()
            obs = recorder.step(action)
            frame = np.squeeze(obs, axis=0) if obs.ndim == 3 else obs

            fps_frames += 1
            if time.time() - fps_start >= 1.0:
                fps_avg = fps_frames / (time.time() - fps_start)
                fps_start, fps_frames = time.time(), 0

            screen.fill((0, 0, 0))
            small = pygame.transform.scale(
                pygame.surfarray.make_surface(np.asarray(frame, dtype=np.uint8).T), (256, 256)
            )
            screen.blit(small, (22, 10))
            status = "RECORDING" if recorder.is_recording else "IDLE"
            for i, line in enumerate(
                [
                    f"{status}   fps: {fps_avg:.0f}",
                    f"Demos: {recorder.count_record}/{recorder.max_trajectories}",
                    "[K] toggle record   [ESC] save & exit",
                    f"action bits on: {np.flatnonzero(action).tolist()}",
                ]
            ):
                screen.blit(font.render(line, True, (255, 255, 255)), (10, 276 + i * 16))
            pygame.display.flip()

            budget = 1.0 / float(config["capture"].get("target_fps", 60))
            remaining = budget - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        exit_reason = "ctrl-c"
    finally:
        recorder.finish()
        pygame.quit()
        env.close()
        print(
            f"\nExited ({exit_reason}). {len(recorder.saved_files)} file(s) written to {demo_dir}"
        )
        for p in recorder.saved_files:
            print("   ", p.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
