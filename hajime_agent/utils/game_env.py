"""
Generic game environment for Imitation Learning.
Based on resident_requiem.py, but adapted for any game.
"""

import gymnasium as gym
import subprocess
import win32gui
import numpy as np
import win32process
import time
import psutil
import vgamepad as vg
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from typing import Tuple, Optional, List
import dxcam
from collections import deque
from typing import Dict, Any


class GenericGameEnv(gym.Env):
    """
    Generic environment for games via emulator or PC executable.
    
    Captures the game screen and sends commands through a virtual gamepad (DualShock 4/Xbox).
    Compatible with PS3 (RPCS3), PS2 (PCSX2) emulators, and PC games.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the environment.
        
        Args:
            config: Dictionary with the game settings. 
                   If None, uses the 're9' process (Resident Evil Requiem).
        """
        super().__init__()
        
        # Configuration
        if config is None:
            try:
                import sys
                import os
                # Add parent dir to sys.path to resolve config
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from config.game_config import GAME_CONFIG
                config = GAME_CONFIG
            except Exception as e:
                print(f"Warning: Could not load config.game_config: {e}")
                config = {
                    "process_name": "rpcs3",
                    "exe_path": None,
                    "capture": {
                        "width": 854,
                        "height": 480,
                        "internal_width": 128,
                        "internal_height": 128,
                        "target_fps": 240,
                        "buffer_len": 1,
                    },
                    "window_offset": {"left": 20, "top": 100, "right": 0, "bottom": 0},
                }
        
        self.config = config
        self.process_name = config.get("process_name", "re9")
        self.exe_path = config.get("exe_path")
        self.hide_window = config.get("hide_window", False)
        
        # Capture
        cap = config.get("capture", {})
        self.width = cap.get("width", 854)
        self.height = cap.get("height", 480)
        self.internal_width = cap.get("internal_width", 128)
        self.internal_height = cap.get("internal_height", 128)
        self.target_fps = cap.get("target_fps", 240)
        self.buffer_len = cap.get("buffer_len", 1)
        
        # Window offsets
        self.win_off = config.get("window_offset", {"left": 20, "top": 100, "right": 0, "bottom": 0})
        
        # Action space (18 binary actions by default)
        self.num_actions = config.get("actions", {}).get("num_actions", 18)
        self.action_space = gym.spaces.MultiBinary(self.num_actions)
        
        # Observation space (Grayscale image)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.internal_height, self.internal_width, 1),
            dtype=np.uint8,
        )
        
        self.dummy = config.get("dummy", False)
        if self.dummy:
            self.hwnd = None
            self.pid = None
            self.img = None
            self.frame_time = 1.0 / self.target_fps
            return
            
        # Find/Open the game
        self.hwnd = self.find_window_by_process_name(self.process_name)
        self.pid = None
        
        if not self.hwnd and self.exe_path:
            cmd = [self.exe_path]
            rom_path = self.config.get("rom_path")
            if rom_path:
                cmd.append(rom_path)
            print(f"Launching process: {cmd}")
            process = subprocess.Popen(cmd)
            self.pid = process.pid
        
        self.wait_start()
        
        # Virtual gamepad (Xbox 360 / XInput)
        self.gamepad = vg.VX360Gamepad()
        self.prev_keys = set()
        
        # Camera (GPU-accelerated screen capture)
        self.camera = None
        self.mss_sct = None
        try:
            self.camera = dxcam.create(output_color="GRAY", max_buffer_len=self.buffer_len)
            self.region = self._get_window_region()
            self.camera.start(region=self.region, target_fps=self.target_fps)
        except Exception as e:
            print("\n" + "="*80)
            print("WARNING: DXCam failed to initialize. Falling back to mss.")
            print("This usually happens on laptops with dual GPUs.")
            print("="*80 + "\n")
            self.camera = None
            import mss
            self.mss_sct = mss.mss()
        
        self.img = None
        self.frame_time = 1.0 / self.target_fps
    
    def _get_window_region(self) -> tuple:
        """Return the window region to capture."""
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        return (
            left + self.win_off.get("left", 20),
            top + self.win_off.get("top", 100),
            right - self.win_off.get("right", 0),
            bot - self.win_off.get("bottom", 0),
        )
    
    def render(self, mode="human"):
        """Return the current captured frame."""
        return self.img
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self.prev_keys = set()
        observation = self._get_observation()
        return observation, {}
    
    def step(self, actions: np.ndarray):
        """Execute an action in the game."""
        frame_start = time.perf_counter()
        
        if not len(actions) > 1:
            actions = actions[0]
        
        # --- Convert vector actions to the virtual gamepad ---
        current = set()
        x_value = 0
        y_value = 0
        right_x = 0
        right_y = 0
        
        # D-Pad / Left Stick
        if actions[0] > 0:  # UP
            current.add(-2)
            y_value = -1.0
        if actions[1] > 0:  # DOWN
            current.add(-2)
            y_value = 1.0
        if actions[2] > 0:  # LEFT
            current.add(-2)
            x_value = -1.0
        if actions[3] > 0:  # RIGHT
            current.add(-2)
            x_value = 1.0
        
        # Face buttons
        if actions[4] > 0:
            current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if actions[5] > 0:
            current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        if actions[6] > 0:
            current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        
        # Triggers
        if abs(actions[7]) > 0:
            current.add(-4)
        if abs(actions[8]) > 0:
            current.add(-5)
        
        # L3
        if actions[9] > 0:
            current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
        
        # Right Stick X (camera)
        if np.any(actions[10:14] == 1):
            current.add(-3)
            if actions[10]: right_x = 0.5
            if actions[11]: right_x = 1
            if actions[12]: right_x = -0.5
            if actions[13]: right_x = -1
        
        # Right Stick Y (camera)
        if np.any(actions[14:18] == 1):
            current.add(-3)
            if actions[14]: right_y = -0.5
            if actions[15]: right_y = -1
            if actions[16]: right_y = 0.5
            if actions[17]: right_y = 1
        
        # --- Apply commands to the gamepad ---
        for b in current - self.prev_keys:
            if b > 0:
                self.gamepad.press_button(b)
        
        if -2 in current:
            self.gamepad.left_joystick_float(x_value_float=x_value, y_value_float=y_value)
        if -3 in current:
            self.gamepad.right_joystick_float(x_value_float=right_x, y_value_float=right_y)
        if -4 in current:
            self.gamepad.left_trigger(value=255)
        if -5 in current:
            self.gamepad.right_trigger(value=255)
        
        # Release buttons not pressed
        for b in self.prev_keys - current:
            if b > 0:
                self.gamepad.release_button(b)
            elif b == -2:
                self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            elif b == -3:
                self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            elif b == -4:
                self.gamepad.left_trigger(value=0)
            elif b == -5:
                self.gamepad.right_trigger(value=0)
        
        self.gamepad.update()
        self.prev_keys = current.copy()
        
        # Get new observation
        observation = self._get_observation()
        return observation, 0.0, False, False, {}
    
    def _get_observation(self) -> np.ndarray:
        """Capture the game screen and resize it."""
        frame = None
        if self.camera is not None:
            frame = self.camera.get_latest_frame()
        elif self.mss_sct is not None:
            left, top, right, bot = self._get_window_region()
            monitor = {"top": top, "left": left, "width": right - left, "height": bot - top}
            try:
                sct_img = self.mss_sct.grab(monitor)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2GRAY)
            except Exception:
                frame = None

        if frame is None:
            return self.img if self.img is not None else np.zeros(
                (self.internal_height, self.internal_width, 1), dtype=np.uint8
            )
        
        resized = cv2.resize(frame, (self.internal_width, self.internal_height), interpolation=cv2.INTER_NEAREST)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        self.img = resized
        return resized
    
    def find_window_by_process_name(self, process_name: str) -> Optional[int]:
        """Find the game window by process name."""
        def callback(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                tid, win_pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(win_pid)
                    if process_name.lower() in process.name().lower():
                        result.append(hwnd)
                except Exception:
                    pass
            return True
        
        result = []
        win32gui.EnumWindows(callback, result)
        return result[0] if result else None
    
    def wait_start(self) -> None:
        """Wait for the game to start, up to a 2 minute timeout."""
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.hwnd = self.find_window_by_process_name(self.process_name)
            if self.hwnd:
                print(f"Window founded HWND: {self.hwnd}.")
                
                # Hide window (optional)
                if self.hide_window:
                    print("Hiding window...")
                    rect = win32gui.GetWindowRect(self.hwnd)
                    x, y, w, h = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
                    win32gui.MoveWindow(self.hwnd, -w, -h, w, h, True)
                
                # Update capture region
                self.region = self._get_window_region()
                break
            print("Waiting for game window...")
            time.sleep(1)
        else:
            print(f"WARNING: Window for '{self.process_name}' not found after {timeout}s!")


# ============================================================
# TEMPORAL LSTM FEATURE EXTRACTOR
# ============================================================

class SpatialAttention(nn.Module):
    """Spatial attention to focus on the relevant elements of the screen."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        attention = th.sigmoid(self.conv(x))
        return x * attention


class TemporalAttentionLSTM(BaseFeaturesExtractor):
    """
    Feature extractor with CNN + LSTM + temporal attention.
    Lets the AI remember what happened in the last 10 frames.
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 debug: bool = False):
        super().__init__(observation_space, features_dim)
        
        self.n_frames = observation_space.shape[0]
        self.frame_height = observation_space.shape[1]
        self.frame_width = observation_space.shape[2]
        self.debug = debug
        
        # CNN to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_frames, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # Bidirectional LSTM to capture temporal context
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Temporal attention (which frames are the most important?)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size * 2, 1)
        )
        
        # Final linear layer
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )
        
        self.hidden_state = None
        self.hidden_reset = True
        self.window_size = 10
        self.feature_buffer = deque(maxlen=self.window_size)
    
    def repackage_hidden(self, h):
        """Detach the computation history to avoid a memory leak."""
        if isinstance(h, th.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        
        # Normalize image
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        # Extract CNN features
        cnn_features = self.cnn(x)
        
        # Update temporal buffer
        if batch_size == 1:
            self.feature_buffer.append(cnn_features.detach())
        else:
            self.feature_buffer.append(cnn_features)
            for i in range(len(self.feature_buffer) - 1):
                self.feature_buffer[i] = self.feature_buffer[i].detach()
        
        # Fill the buffer until it reaches the window size
        if len(self.feature_buffer) == 1:
            while len(self.feature_buffer) < self.window_size:
                self.feature_buffer.append(self.feature_buffer[-1])
        
        # Stack the temporal sequence
        sequence = th.stack(list(self.feature_buffer), dim=1)
        sequence.requires_grad_(True)
        
        # Reset hidden state if needed
        should_reset = (
            self.hidden_reset or 
            self.hidden_state is None or 
            self.hidden_state[0].shape[1] != batch_size
        )
        
        if should_reset:
            num_directions = 2
            hidden_size_total = self.lstm_num_layers * num_directions
            h0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            c0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            current_hidden = (h0, c0)
            self.hidden_reset = False
        else:
            current_hidden = self.repackage_hidden(self.hidden_state)
        
        # Pass through the LSTM
        lstm_out, last_hidden = self.lstm(sequence, current_hidden)
        self.hidden_state = last_hidden
        
        # Apply temporal attention
        attention_weights = th.softmax(self.attention(lstm_out), dim=1)
        context = th.sum(attention_weights * lstm_out, dim=1)
        
        # Map to final features
        features = self.linear(context)
        return features
    
    def reset_hidden(self, dones=None):
        """Reset the LSTM state (must be called between episodes)."""
        self.hidden_state = None
        self.hidden_reset = True
        self.feature_buffer.clear()
