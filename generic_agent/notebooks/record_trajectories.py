"""
Human trajectory recording script for generic games.
Adapted from imitatation-record.ipynb to be simpler and more generic.
"""

import time
import cv2
import numpy as np
import keyboard
import os
import pygame
import threading
import ctypes
from pathlib import Path
import torch as th
import os
import sys

# Adjust the path so the modules can be found
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from imitation.data.types import Trajectory
from game_env import GenericGameEnv
from config.game_config import GAME_CONFIG


class TrajectoryRecorder:
    """Trajectory recorder for Imitation Learning."""
    
    def __init__(self, env, demo_path='demos/', max_traj=10, screen_size=(854, 480)):
        self.env = env
        self.demo_path = demo_path
        self.max_traj = max_traj
        self.screen_width, self.screen_height = screen_size
        self.trajectories = []
        self.recorded_obs = []
        self.recorded_actions = []
        self.is_recording = False
        self.count_record = 0
        
        os.makedirs(demo_path, exist_ok=True)
        
        # Gamepad state
        self.state = {k: 0 for k in ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "X", 
                                       "START", "CAM_X", "CAM_Y", "LT", "RT", "L3"]}
        self.lock = threading.Lock()
        
        # Renderer
        self.renderer = None
        
    def start_recording(self):
        """Start the recording process."""
        print("Done! Press 'K' to start/stop the record.")
        
        # Initialize pygame
        pygame.init()
        
        # Initialize XInput (ctypes)
        try:
            self.xinput = ctypes.windll.xinput1_4
        except AttributeError:
            try:
                self.xinput = ctypes.windll.xinput1_3
            except AttributeError:
                self.xinput = None
                
        class XINPUT_GAMEPAD(ctypes.Structure):
            _fields_ = [('wButtons', ctypes.c_ushort), ('bLeftTrigger', ctypes.c_ubyte), ('bRightTrigger', ctypes.c_ubyte), ('sThumbLX', ctypes.c_short), ('sThumbLY', ctypes.c_short), ('sThumbRX', ctypes.c_short), ('sThumbRY', ctypes.c_short)]
        class XINPUT_STATE(ctypes.Structure):
            _fields_ = [('dwPacketNumber', ctypes.c_ulong), ('Gamepad', XINPUT_GAMEPAD)]
        self.XINPUT_STATE = XINPUT_STATE
            
        window = pygame.display.set_mode((self.screen_width, self.screen_height), 
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AI Agent Capture")
        font = pygame.font.SysFont("Arial", 24)
        
        is_running = False
        num_actions = GAME_CONFIG["actions"].get("num_actions", 18)
        action = np.zeros(num_actions, dtype=np.float32)
        actual_fps = 0
        fps_start_time = time.time()
        fps_avg = 0
        
        while self.count_record < self.max_traj:
            loop_start = time.time()
            
            # Pygame event pumping (keeps the window from freezing on Windows) and exit check
            force_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    force_exit = True
            
            if keyboard.is_pressed('esc') or force_exit:
                print("\nExiting and saving trajectories...")
                break
            
            # Check the recording toggle
            if keyboard.is_pressed('k'):
                is_running = not is_running
                time.sleep(0.3)
                print(f"Recording: {is_running}")
            
            # Capture keyboard and joystick input
            action.fill(0)
            
            input_mode = GAME_CONFIG["actions"].get("input_mode", "gamepad")
            
            if input_mode == "keyboard_mouse":
                import mouse
                for i, mapping in enumerate(GAME_CONFIG["actions"]["mappings"]):
                    if mapping["type"] == "key":
                        if keyboard.is_pressed(mapping["key"]):
                            action[i] = 1
                    elif mapping["type"] == "mouse_button":
                        if mouse.is_pressed(mapping["button"]):
                            action[i] = 1
            else:
                # Legacy keyboard mapping for gamepad mode
                if keyboard.is_pressed('up'): action[0] = 1
                if keyboard.is_pressed('down'): action[1] = 1
                if keyboard.is_pressed('left'): action[2] = 1
                if keyboard.is_pressed('right'): action[3] = 1
                if keyboard.is_pressed('i'): action[4] = 1
                if keyboard.is_pressed('o'): action[5] = 1
                if keyboard.is_pressed('p'): action[6] = 1
                
                # XInput (Global Background Controller Capture)
                if hasattr(self, 'xinput') and self.xinput:
                    # Read only the physical gamepad (Slot 0) to avoid the feedback loop
                    # of reading the buttons pressed on the AI's own virtual gamepad
                    for j_idx in range(1):
                        state = self.XINPUT_STATE()
                        if self.xinput.XInputGetState(j_idx, ctypes.byref(state)) == 0:
                            buttons = state.Gamepad.wButtons
                            # D-Pad
                            if buttons & 0x0001: action[0] = 1 # UP
                            if buttons & 0x0002: action[1] = 1 # DOWN
                            if buttons & 0x0004: action[2] = 1 # LEFT
                            if buttons & 0x0008: action[3] = 1 # RIGHT
                            
                            # Left analog stick
                            if state.Gamepad.sThumbLY > 16000: action[0] = 1 # UP
                            if state.Gamepad.sThumbLY < -16000: action[1] = 1 # DOWN
                            if state.Gamepad.sThumbLX < -16000: action[2] = 1 # LEFT
                            if state.Gamepad.sThumbLX > 16000: action[3] = 1 # RIGHT
                            
                            # Face buttons (A=0x1000, B=0x2000, X=0x4000, Y=0x8000)
                            if buttons & 0x1000: action[4] = 1 # A / Cross
                            if buttons & 0x2000: action[5] = 1 # B / Circle
                            if buttons & 0x4000: action[6] = 1 # X / Square
            
            # Other buttons can be mapped here
            
            # Run the action in the environment
            action_input = [action.reshape(1, -1)]
            obs, _, _, _ = self.env.step(action_input)
            
            # Render
            if obs is not None:
                # Shape is (4, 84, 84). The last channel is the current GRAY frame.
                img_gray = obs[0, -1, :, :].copy()
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                
                img_resized = cv2.resize(img_rgb, (self.screen_width, self.screen_height), 
                                        interpolation=cv2.INTER_NEAREST)
                color = (0, 255, 0) if is_running else (0, 0, 255)
                
                # Use surfarray as in run_ai.py to avoid flatten() color bugs
                surface = pygame.surfarray.make_surface(img_resized.swapaxes(0, 1))
                window.blit(surface, (0, 0))
                
                # Status text
                fps_text = font.render(f"FPS: {int(actual_fps)}", True, (255, 255, 255))
                count_text = font.render(f"Demos: {self.count_record}/{self.max_traj}", True, (255, 255, 255))
                status_text = font.render("RECORDING" if is_running else "IDLE", True, color)
                instructions_text = font.render("[K] Toggle Record   [ESC] Save & Exit", True, (200, 200, 200))
                
                window.blit(fps_text, (10, 10))
                window.blit(count_text, (10, 40))
                window.blit(status_text, (10, 70))
                window.blit(instructions_text, (10, 100))
                
                pygame.display.flip()
            
            # Record trajectory
            if is_running:
                # np.squeeze drops the batch dimension (batch=1) coming from DummyVecEnv
                obs_to_save = np.squeeze(obs, axis=0)
                if not self.is_recording:
                    self.recorded_obs.append(obs_to_save)
                self.recorded_obs.append(obs_to_save)
                self.recorded_actions.append(action.copy())
                self.is_recording = True
            elif self.is_recording:
                print(f"Finalizing trajectory {self.count_record}...")
                obs_uint8 = np.stack([o.astype(np.uint8) for o in self.recorded_obs], axis=0)
                traj = Trajectory(obs=obs_uint8, acts=np.array(self.recorded_actions), 
                                  infos=None, terminal=False)
                
                # Save the trajectory to its own file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_file = os.path.join(self.demo_path, f"demo_{self.count_record}_{timestamp}.pt")
                th.save([traj], save_file)
                print(f"[OK] Trajectory saved successfully to: {save_file}")
                
                # Increment the count and clear memory
                self.trajectories.append(traj) # Added only for the main loop count
                self.recorded_obs, self.recorded_actions = [], []
                self.count_record += 1
                self.is_recording = False
            
            # FPS counter
            fps_avg += 1
            if time.time() - fps_start_time >= 1.0:
                actual_fps = fps_avg
                fps_avg = 0
                fps_start_time = time.time()
        
        # Saving now happens in main()'s finally block
        pass


def main():
    """Main function to start recording."""
    def make_env():
        return GenericGameEnv(GAME_CONFIG)
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    
    # Record trajectories
    recorder = TrajectoryRecorder(env, demo_path='./demos/', max_traj=10)
    
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    finally:
        # Trajectories are already saved incrementally! No need to save in the finally block
        print("\nRecording process completed.")
        
        try:
            pygame.quit()
        except:
            pass


if __name__ == "__main__":
    main()
