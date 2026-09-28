"""
``GenericGameEnv``: a Gymnasium environment backed by a real window.

Observation = one grayscale frame grabbed from the game window.
Action = a binary vector emitted to a virtual gamepad or as synthesised
keyboard/mouse events. Reward and termination are deliberately degenerate
(constant 0, never done): behavioural cloning ignores both, and a black-box
game gives no way to define them. That choice is what blocks every episodic
metric in this project -- see docs/README section 5.3.

The environment is importable on any platform. Live capture and actuation are
not: they need Windows, and the gamepad mode additionally needs the ViGEmBus
driver. ``config["dummy"] = True`` skips both, which is how training and the
test suite run.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from agent.utils import windows
from agent.utils.emission import build_emitter


class WindowNotFoundError(RuntimeError):
    pass


class GenericGameEnv(gym.Env):
    """Screen-capture environment for an arbitrary PC game or emulator."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            from agent.config import game_config

            config = game_config()

        self.config = config
        self.process_name = config.get("process_name", "game")
        self.exe_path = config.get("exe_path")
        self.hide_window = bool(config.get("hide_window", False))

        capture = config.get("capture", {})
        self.internal_width = int(capture.get("internal_width", 128))
        self.internal_height = int(capture.get("internal_height", 128))
        self.target_fps = float(capture.get("target_fps", 60))
        self.buffer_len = int(capture.get("buffer_len", 1))
        self.window_offset = config.get("window_offset", {})

        actions = config.get("actions", {})
        self.num_actions = int(actions.get("num_actions", 18))
        self.input_mode = actions.get("input_mode", "gamepad")
        self.mappings = actions.get("mappings", [])

        self.action_space = gym.spaces.MultiBinary(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.internal_height, self.internal_width, 1),
            dtype=np.uint8,
        )

        self.dummy = bool(config.get("dummy", False))
        self.hwnd: int | None = None
        self.img: np.ndarray | None = None
        self.emitter = None
        self.camera = None
        self.mss_sct = None
        self.region = self._default_region()
        self._last_step_time = 0.0
        self.stale_frames = 0
        self.dropped_frames = 0

        if self.dummy:
            return

        self.hwnd = self.find_window_by_process_name(self.process_name)
        if not self.hwnd and self.exe_path:
            command = [self.exe_path]
            if config.get("rom_path"):
                command.append(config["rom_path"])
            print(f"Launching process: {command}")
            self.pid = subprocess.Popen(command).pid
        else:
            self.pid = None

        self.wait_start(timeout_s=float(config.get("window_timeout_s", 120)))

        self.emitter = build_emitter(config, windows.vg)

        self._open_capture()

    # ---------------- observation ----------------

    def _default_region(self) -> tuple[int, int, int, int]:
        left = int(self.window_offset.get("left", 0))
        top = int(self.window_offset.get("top", 0))
        return (left, top, left + self.internal_width, top + self.internal_height)

    def _window_region(self) -> tuple[int, int, int, int]:
        left, top, right, bottom = windows.win32gui.GetWindowRect(self.hwnd)
        return (
            left + int(self.window_offset.get("left", 0)),
            top + int(self.window_offset.get("top", 0)),
            right - int(self.window_offset.get("right", 0)),
            bottom - int(self.window_offset.get("bottom", 0)),
        )

    def _open_capture(self) -> None:
        """Prefer GPU desktop duplication, fall back to a CPU grab."""
        self.region = self._window_region()
        if windows.HAS_DXCAM:
            try:
                self.camera = windows.dxcam.create(
                    output_color="GRAY", max_buffer_len=self.buffer_len
                )
                self.camera.start(region=self.region, target_fps=self.target_fps)
                return
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[capture] DXCam unavailable ({exc.__class__.__name__}: {exc}); "
                    f"falling back to mss. This is typical on dual-GPU laptops."
                )
        if not windows.HAS_MSS:
            raise RuntimeError(
                "No screen-capture backend is available: install dxcam or mss. "
                "For offline training use a dummy environment."
            )
        self.mss_sct = windows.mss.mss()

    def _grab(self) -> np.ndarray | None:
        if self.camera is not None:
            return self.camera.get_latest_frame()
        if self.mss_sct is not None:
            left, top, right, bottom = self._window_region()
            monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
            try:
                return cv2.cvtColor(np.array(self.mss_sct.grab(monitor)), cv2.COLOR_BGRA2GRAY)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _get_observation(self) -> np.ndarray:
        if self.dummy:
            if self.img is None:
                self.img = np.zeros(self.observation_space.shape, dtype=np.uint8)
            return self.img

        frame = self._grab()
        if frame is None:
            # Counted rather than silent: a frozen frame is otherwise
            # indistinguishable from a genuinely static game screen.
            self.dropped_frames += 1
            if self.img is not None:
                return self.img
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        resized = cv2.resize(
            frame, (self.internal_width, self.internal_height), interpolation=cv2.INTER_NEAREST
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        self.img = resized
        return resized

    # ---------------- action ----------------

    def step(self, actions: np.ndarray):
        frame_start = time.perf_counter()

        actions = np.asarray(actions)
        if actions.ndim > 1:
            actions = actions.reshape(-1)[-self.num_actions :]
        if actions.shape[-1] != self.num_actions:
            raise ValueError(
                f"action vector has width {actions.shape[-1]}, environment expects "
                f"{self.num_actions}"
            )

        if self.dummy:
            raise RuntimeError(
                "A dummy environment cannot actuate a game. Use it only for offline "
                "training on recorded demonstrations."
            )

        self.emitter.apply(actions)
        observation = self._get_observation()

        # Honour capture.target_fps: wait out the remainder of the frame budget
        # instead of spinning as fast as the GPU allows.
        budget = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        elapsed = time.perf_counter() - frame_start
        if budget > elapsed:
            time.sleep(budget - elapsed)

        return observation, 0.0, False, False, {}

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if self.emitter is not None:
            self.emitter.release_all()
        return self._get_observation(), {}

    def render(self, mode="human"):
        return self.img

    def close(self) -> None:
        if self.emitter is not None:
            self.emitter.close()
            self.emitter = None
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:  # noqa: BLE001
                pass
            self.camera = None
        if self.mss_sct is not None:
            try:
                self.mss_sct.close()
            except Exception:  # noqa: BLE001
                pass
            self.mss_sct = None

    # ---------------- window discovery ----------------

    @staticmethod
    def find_window_by_process_name(process_name: str) -> int | None:
        if not windows.HAS_WIN32:
            return None
        found = []

        def callback(hwnd, _extra):
            if not windows.win32gui.IsWindowVisible(hwnd):
                return True
            import psutil

            try:
                _, pid = windows.win32process.GetWindowThreadProcessId(hwnd)
                if process_name.lower() in psutil.Process(pid).name().lower():
                    found.append(hwnd)
            except Exception:  # noqa: BLE001 - protected processes
                pass
            return True

        windows.win32gui.EnumWindows(callback, None)
        return found[0] if found else None

    def wait_start(self, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.hwnd = self.find_window_by_process_name(self.process_name)
            if self.hwnd:
                print(f"Found window HWND: {self.hwnd}.")
                if self.hide_window:
                    left, top, right, bottom = windows.win32gui.GetWindowRect(self.hwnd)
                    w, h = right - left, bottom - top
                    windows.win32gui.MoveWindow(self.hwnd, -w, -h, w, h, True)
                    print("Window moved off-screen; capture region follows it.")
                return
            print("Waiting for game window...")
            time.sleep(1.0)
        raise WindowNotFoundError(
            f"No visible window belonging to a process matching {self.process_name!r} "
            f"appeared within {timeout_s:.0f}s. Start the game first, or set "
            f"GAME_CONFIG['process_name'] / 'exe_path' correctly."
        )
