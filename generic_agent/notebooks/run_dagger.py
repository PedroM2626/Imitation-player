"""
Script de DAgger Interativo para jogos genericos.
Alterna dinamicamente entre IA e Humano para corrigir erros da IA.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import time
import cv2
import numpy as np
import keyboard
import pygame
import threading
import torch as th
from pathlib import Path
from imitation.data.types import Trajectory

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.policies import ActorCriticPolicy
from game_env import GenericGameEnv
from utils import get_last_index, LSTMWrapper
from config.game_config import GAME_CONFIG

# --- CONFIGURACAO ---
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
SCREEN_WIDTH = 854
SCREEN_HEIGHT = 480
MAX_FPS = 30
MAX_TRAJ = 10
DEMO_PATH = './demos/'

MODEL_PATH = "./models/"
STEPS_PATH = MODEL_PATH + "steps"

os.makedirs(DEMO_PATH, exist_ok=True)

class RendererThread(threading.Thread):
    def __init__(self, width=854, height=480):
        super().__init__(daemon=True)
        self.frame = None
        self.color = (0, 0, 255)
        self.fps = 0
        self.running = True
        self.count = 0
        self.action_from_ai = False
        self.lock = threading.Lock()
        self.width = width
        self.height = height
    
    def update_data(self, frame, color, count, fps, action_from_ai):
        with self.lock:
            self.frame = frame
            self.color = color
            self.count = count
            self.fps = fps
            self.action_from_ai = action_from_ai
    
    def run(self):
        pygame.init()
        window = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("HG-DAgger Capture")
        font = pygame.font.SysFont("Arial", 22)
        
        while self.running:
            pygame.event.pump()
            
            with self.lock:
                if self.frame is not None:
                    img_view = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    img_rgb = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
                    
                    surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                    window.blit(surf, (0, 0))
                    
                    # Status text
                    fps_txt = font.render(f"FPS: {int(self.fps)}", True, (255, 255, 255))
                    count_txt = font.render(f"Demos: {self.count}/{MAX_TRAJ}", True, (255, 255, 255))
                    status_color = (0, 255, 0) if self.color == (0, 255, 0) else (255, 0, 0)
                    rec_txt = font.render("RECORDING" if self.color == (0, 255, 0) else "IDLE", True, status_color)
                    ai_txt = font.render("CONTROLE: IA" if self.action_from_ai else "CONTROLE: HUMANO", True, (200, 200, 255))
                    instr_txt = font.render("[K] Rec  [L] Toggle IA/Humano  [ESC] Exit", True, (200, 200, 200))
                    
                    window.blit(fps_txt, (10, 10))
                    window.blit(count_txt, (10, 40))
                    window.blit(rec_txt, (10, 70))
                    window.blit(ai_txt, (10, 100))
                    window.blit(instr_txt, (10, 130))
                    
                    pygame.display.flip()
            time.sleep(0.01)

def main():
    env = GenericGameEnv(GAME_CONFIG)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()
    
    # Load Model
    main_model_file = os.path.join(os.path.dirname(MODEL_PATH), "bc_policy.zip")
    if not os.path.exists(main_model_file):
        main_model_file = os.path.join(MODEL_PATH, "bc_policy.zip")
    
    if os.path.exists(main_model_file):
        print(f"Loading main model: {main_model_file}")
        policy = ActorCriticPolicy.load(main_model_file, device=DEVICE)
    else:
        last_idx = get_last_index(STEPS_PATH, "bc_policy", ".zip")
        if last_idx < 0:
            print("No trained models found! Train an initial model first using train_agent.py.")
            return
        model_file = os.path.join(STEPS_PATH, f"bc_policy{last_idx}.zip")
        print(f"Loading checkpoint model: {model_file}")
        policy = ActorCriticPolicy.load(model_file, device=DEVICE)
    
    has_lstm = (hasattr(policy, 'lstm') or (hasattr(policy, 'features_extractor') and hasattr(policy.features_extractor, 'lstm')))
    if has_lstm:
        policy = LSTMWrapper(policy)
        policy.reset()
        print("LSTM model loaded")

    renderer = RendererThread(SCREEN_WIDTH, SCREEN_HEIGHT)
    renderer.start()
    
    is_running = False
    is_recording = False
    count_record = 0
    trajectories = []
    recorded_obs = []
    recorded_actions = []
    
    fps_count = 0
    fps_start = time.time()
    actual_fps = 0
    
    action_from_ai = True
    print("Done! Press 'K' to start/stop the record. Press 'L' to toggle AI vs Human control.")
    
    try:
        while count_record < MAX_TRAJ:
            loop_start = time.time()
            
            # Toggles
            if keyboard.is_pressed('k'):
                is_running = not is_running
                time.sleep(0.3)
                print(f"Recording: {is_running}")
                
            if keyboard.is_pressed('l'):
                action_from_ai = not action_from_ai
                time.sleep(0.3)
                print(f"Control from AI: {action_from_ai}")
                if action_from_ai and has_lstm:
                    policy.reset()
                    
            if keyboard.is_pressed('esc'):
                break
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            num_actions = GAME_CONFIG["actions"].get("num_actions", 18)
            action = np.zeros((1, num_actions), dtype=np.float32)
            
            # Pegar acao manual
            manual_action = np.zeros((1, num_actions), dtype=np.float32)
            input_mode = GAME_CONFIG["actions"].get("input_mode", "gamepad")
            if input_mode == "keyboard_mouse":
                import mouse
                for i, mapping in enumerate(GAME_CONFIG["actions"]["mappings"]):
                    if mapping["type"] == "key":
                        if keyboard.is_pressed(mapping["key"]): manual_action[0][i] = 1
                    elif mapping["type"] == "mouse_button":
                        if mouse.is_pressed(mapping["button"]): manual_action[0][i] = 1
            else:
                # Fallback legacy
                if keyboard.is_pressed('up'): manual_action[0][0] = 1
                if keyboard.is_pressed('down'): manual_action[0][1] = 1
                if keyboard.is_pressed('left'): manual_action[0][2] = 1
                if keyboard.is_pressed('right'): manual_action[0][3] = 1
                if keyboard.is_pressed('i'): manual_action[0][4] = 1
                if keyboard.is_pressed('o'): manual_action[0][5] = 1
                if keyboard.is_pressed('p'): manual_action[0][6] = 1

            if is_recording:
                # Previsao da IA
                aggressiveness = GAME_CONFIG.get("aggressiveness", 1.0)
                if aggressiveness == 1.0:
                    pred_act, _ = policy.predict(obs, deterministic=False)
                else:
                    from stable_baselines3.common.utils import obs_as_tensor
                    obs_tensor = obs_as_tensor(obs, policy.device)
                    with th.no_grad():
                        features = policy.extract_features(obs_tensor)
                        latent_pi, _ = policy.mlp_extractor(features)
                        action_logits = policy.action_net(latent_pi)
                    probs = th.sigmoid(action_logits).cpu().numpy()[0]
                    mappings = GAME_CONFIG["actions"].get("mappings", [])
                    for i, m in enumerate(mappings):
                        if i < len(probs) and m.get("type") == "mouse_button":
                            probs[i] = min(probs[i] * aggressiveness, 1.0)
                    pred_act = (np.random.rand(len(probs)) < probs).astype(np.float32)

            if is_recording and action_from_ai:
                action = [pred_act]
            else:
                action = manual_action

            obs, _, _, _ = env.step(action)
            
            # Renderer update
            if obs is not None:
                frame_gray = obs[0, -1, :, :].copy().astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            else:
                w, h = env.observation_space.shape[0], env.observation_space.shape[1]
                frame_bgr = np.zeros((w, h, 3), dtype=np.uint8)
                
            color = (0, 255, 0) if is_running else (0, 0, 255)
            renderer.update_data(frame_bgr, color, count_record, actual_fps, action_from_ai)
            
            if is_running:
                if not is_recording:
                    action_from_ai = True
                    if has_lstm: policy.reset()
                    recorded_obs.append(obs)
                recorded_obs.append(obs)
                recorded_actions.append(action[0])
                is_recording = True
            elif is_recording:
                print(f"Finalizando trajetoria {count_record}...")
                obs_to_save = []
                for o in recorded_obs:
                    if o.ndim == 4: # SB3 Vectorized output
                        obs_to_save.append(np.squeeze(o, axis=0).astype(np.uint8))
                    else:
                        obs_to_save.append(o.astype(np.uint8))
                        
                obs_uint8 = np.stack(obs_to_save, axis=0)
                traj = Trajectory(obs=obs_uint8, acts=np.array(recorded_actions), infos=None, terminal=False)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_file = os.path.join(DEMO_PATH, f"dagger_demo_{count_record}_{timestamp}.pt")
                th.save([traj], save_file)
                print(f"[OK] DAgger Trajetoria salva com sucesso em: {save_file}")
                
                trajectories.append(traj)
                recorded_obs, recorded_actions = [], []
                count_record += 1
                is_recording = False
                action_from_ai = False
            
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                actual_fps = fps_count
                fps_count = 0
                fps_start = time.time()
                
    except KeyboardInterrupt:
        pass
    finally:
        renderer.running = False
        pygame.quit()
        print("Processo concluido. Os dados DAgger foram salvos na pasta demos/ e serao usados no proximo treino!")

if __name__ == "__main__":
    main()
