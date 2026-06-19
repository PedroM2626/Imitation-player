"""
Notebook para rodar a IA treinada (variante Transformer) em qualquer jogo.
Adaptado de run_ai.py para utilizar o VisionTransformerExtractor como
feature extractor em vez da NatureCNN padrao.

A estrutura de execucao e identica ao run_ai.py:
    - Toggle K para ativar/desativar a IA
    - Escape para sair
    - Renderer em thread separada via pygame
    - Modo manual quando a IA esta desativada
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
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from game_env import GenericGameEnv
from utils import get_last_index, LSTMWrapper
from vision_transformer import VisionTransformerExtractor
from config.game_config import GAME_CONFIG


# --- CONFIGURACAO ---
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
SCREEN_WIDTH = 854
SCREEN_HEIGHT = 480
MAX_FPS = 120

# Ajuste o caminho dos modelos treinados
MODEL_PATH = "./models/"
STEPS_PATH = MODEL_PATH + "steps"


class RendererThread(threading.Thread):
    """Thread de renderizacao para visualizar a IA jogando."""
    
    def __init__(self, width=854, height=480):
        super().__init__(daemon=True)
        self.frame = None
        self.info = ""
        self.color = (0, 0, 255)
        self.fps = 0
        self.running = True
        self.current_epoch = 0
        self.lock = threading.Lock()
        self.width = width
        self.height = height
    
    def update_data(self, frame, info, color, fps, current_epoch):
        with self.lock:
            self.frame = frame
            self.info = info
            self.color = color
            self.fps = fps
            self.current_epoch = current_epoch
    
    def run(self):
        pygame.init()
        window = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AI Transformer Agent")
        font = pygame.font.SysFont("Arial", 22)
        
        while self.running:
            pygame.event.pump()
            
            with self.lock:
                if self.frame is not None:
                    # Redimensionar frame para a tela
                    img_view = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    img_rgb = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
                    
                    # Criar surface do pygame
                    surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                    window.blit(surf, (0, 0))
                    
                    # Texto de status
                    txt = font.render(
                        f"{self.info} | FPS: {int(self.fps)} | Epoch: {self.current_epoch}",
                        True, (255, 255, 255)
                    )
                    pygame.draw.circle(window, self.color, (30, 30), 10)
                    window.blit(txt, (50, 20))
                    
                    pygame.display.flip()
            time.sleep(0.01)


class AITransformerPlayer:
    """Agente de IA que joga o jogo usando um modelo Transformer pre-treinado."""
    
    def __init__(self, env, model_path, device='cuda'):
        self.env = env
        self.model_path = model_path
        self.device = device
        self.policy = None
        self.current_epoch = 0
        
    def load_latest_model(self):
        """Carrega o modelo Transformer mais recente do diretorio de modelos."""
        # 1. Tentar carregar o modelo principal bc_policy_transformer.zip na raiz de models ou do path de steps
        main_model_file = os.path.join(os.path.dirname(self.model_path), "bc_policy_transformer.zip")
        if not os.path.exists(main_model_file):
            main_model_file = os.path.join(self.model_path, "bc_policy_transformer.zip")
            
        if os.path.exists(main_model_file):
            print(f"Loading main Transformer model: {main_model_file}")
            self.policy = ActorCriticCnnPolicy.load(
                main_model_file,
                device=self.device,
                custom_objects={
                    'features_extractor_class': VisionTransformerExtractor,
                }
            )
            self.current_epoch = "final"
        else:
            # Fallback para o get_last_index
            last_idx = get_last_index(self.model_path, "bc_policy_transformer", ".zip")
            if last_idx < 0:
                print("No trained Transformer models found!")
                return False
            
            model_file = os.path.join(self.model_path, f"bc_policy_transformer{last_idx}.zip")
            print(f"Loading checkpoint Transformer model: {model_file}")
            self.policy = ActorCriticCnnPolicy.load(
                model_file,
                device=self.device,
                custom_objects={
                    'features_extractor_class': VisionTransformerExtractor,
                }
            )
            self.current_epoch = last_idx
        
        # Verificar se tem LSTM e envolver o wrapper
        has_lstm = (hasattr(self.policy, 'lstm') or 
                   (hasattr(self.policy, 'features_extractor') and 
                    hasattr(self.policy.features_extractor, 'lstm')))
        
        if has_lstm:
            self.policy = LSTMWrapper(self.policy)
            self.policy.reset()
            print("LSTM model loaded")
        
        return True
    
    def play(self, manual_mode=False, max_steps=None):
        """Executa a IA Transformer no jogo."""
        obs = self.env.reset()
        renderer = RendererThread(SCREEN_WIDTH, SCREEN_HEIGHT)
        renderer.start()
        
        fps_count = 0
        fps_start = time.time()
        actual_fps = 0
        steps = 0
        ai_active = False
        
        print("--- System started (Press K to toggle AI, Escape to exit) ---")
        
        try:
            while True:
                loop_start = time.time()
                
                # Toggle AI
                if keyboard.is_pressed('k'):
                    ai_active = not ai_active
                    if ai_active:
                        print("AI ACTIVATED")
                        # Resetar LSTM ao ativar
                        if isinstance(self.policy, LSTMWrapper):
                            self.policy.reset()
                    else:
                        print("MANUAL MODE")
                    time.sleep(0.3)
                    
                # Exit
                if keyboard.is_pressed('esc'):
                    break
                
                # Pygame Event Pump e Manual Override
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                
                num_actions = GAME_CONFIG["actions"].get("num_actions", 18)
                action = np.zeros((1, num_actions), dtype=np.float32)
                
                if ai_active and self.policy:
                    aggressiveness = GAME_CONFIG.get("aggressiveness", 1.0)
                    
                    if aggressiveness == 1.0:
                        pred_act, _ = self.policy.predict(obs, deterministic=False)
                    else:
                        from stable_baselines3.common.utils import obs_as_tensor
                        obs_tensor = obs_as_tensor(obs, self.policy.device)
                        with th.no_grad():
                            features = self.policy.extract_features(obs_tensor)
                            latent_pi, _ = self.policy.mlp_extractor(features)
                            action_logits = self.policy.action_net(latent_pi)
                            
                        # Extrair a porcentagem exata que a IA tem "certeza" de cada botao
                        probs = th.sigmoid(action_logits).cpu().numpy()[0]
                        
                        # Aplicar fator de agressividade nos cliques de mouse (geralmente ataques)
                        mappings = GAME_CONFIG["actions"].get("mappings", [])
                        for i, m in enumerate(mappings):
                            if i < len(probs) and m.get("type") == "mouse_button":
                                probs[i] = min(probs[i] * aggressiveness, 1.0)
                        
                        # Amostragem usando a probabilidade hackeada
                        pred_act = (np.random.rand(len(probs)) < probs).astype(np.float32)
                        
                    action = [pred_act]
                else:
                    # Modo manual
                    input_mode = GAME_CONFIG["actions"].get("input_mode", "gamepad")
                    if input_mode == "keyboard_mouse":
                        import mouse
                        for i, mapping in enumerate(GAME_CONFIG["actions"]["mappings"]):
                            if mapping["type"] == "key":
                                if keyboard.is_pressed(mapping["key"]): action[0][i] = 1
                            elif mapping["type"] == "mouse_button":
                                if mouse.is_pressed(mapping["button"]): action[0][i] = 1
                    else:
                        if keyboard.is_pressed('up'): action[0][0] = 1
                        if keyboard.is_pressed('down'): action[0][1] = 1
                        if keyboard.is_pressed('left'): action[0][2] = 1
                        if keyboard.is_pressed('right'): action[0][3] = 1
                        if keyboard.is_pressed('i'): action[0][4] = 1
                        if keyboard.is_pressed('o'): action[0][5] = 1
                        if keyboard.is_pressed('p'): action[0][6] = 1
                
                # Executar acao
                obs, _, _, _ = self.env.step(action)
                
                # Atualizar renderer
                status = "AI ACTIVE" if ai_active else "MANUAL"
                color = (0, 255, 0) if ai_active else (0, 0, 255)
                if obs is not None:
                    # obs[0] has shape (4, W, H). The last channel is the current GRAY frame.
                    frame_gray = obs[0, -1, :, :].copy().astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                else:
                    w, h = self.env.observation_space.shape[0], self.env.observation_space.shape[1]
                    frame_bgr = np.zeros((w, h, 3), dtype=np.uint8)
                renderer.update_data(frame_bgr, status, color, actual_fps, self.current_epoch)
                
                # FPS counter
                fps_count += 1
                if time.time() - fps_start >= 1.0:
                    actual_fps = fps_count
                    fps_count = 0
                    fps_start = time.time()
                
                steps += 1
                if max_steps and steps >= max_steps:
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            renderer.running = False
            pygame.quit()


def main():
    # Criar ambiente
    env = GenericGameEnv(GAME_CONFIG)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    
    # Criar player Transformer e carregar modelo
    player = AITransformerPlayer(env, STEPS_PATH, device=DEVICE)
    
    if player.load_latest_model():
        player.play()
    else:
        print("No Transformer model to load. Train first using the Transformer training script.")


if __name__ == "__main__":
    main()
