"""
Script de gravacao de trajetorias humanas para jogos genericos.
Adaptado de imitatation-record.ipynb para ser mais simples e generico.
"""

import time
import cv2
import numpy as np
import keyboard
import pygame
import threading
from pathlib import Path
import torch as th
import os
import sys

# Ajuste o path para encontrar os modulos
sys.path.insert(0, os.path.abspath("../utils"))

from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
from game_env import GenericGameEnv


class TrajectoryRecorder:
    """Gravador de trajetorias para Imitation Learning."""
    
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
        
        # Estado do controle
        self.state = {k: 0 for k in ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "X", 
                                       "START", "CAM_X", "CAM_Y", "LT", "RT", "L3"]}
        self.lock = threading.Lock()
        
        # Renderer
        self.renderer = None
        
    def start_recording(self):
        """Inicia o processo de gravacao."""
        print("Done! Press 'K' to start/stop the record.")
        
        # Inicializar pygame
        pygame.init()
        window = pygame.display.set_mode((self.screen_width, self.screen_height), 
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AI Agent Capture")
        font = pygame.font.SysFont("Arial", 24)
        
        is_running = False
        action = np.zeros(18, dtype=np.float32)
        actual_fps = 0
        fps_start_time = time.time()
        fps_avg = 0
        
        while self.count_record < self.max_traj:
            loop_start = time.time()
            
            # Verificar toggle de gravacao
            if keyboard.is_pressed('k'):
                is_running = not is_running
                time.sleep(0.3)
                print(f"Recording: {is_running}")
            
            # Capturar entrada do teclado
            if keyboard.is_pressed('up'): action[0] = 1
            else: action[0] = 0
            if keyboard.is_pressed('down'): action[1] = 1
            else: action[1] = 0
            if keyboard.is_pressed('left'): action[2] = 1
            else: action[2] = 0
            if keyboard.is_pressed('right'): action[3] = 1
            else: action[3] = 0
            if keyboard.is_pressed('i'): action[4] = 1
            else: action[4] = 0
            if keyboard.is_pressed('o'): action[5] = 1
            else: action[5] = 0
            if keyboard.is_pressed('p'): action[6] = 1
            else: action[6] = 0
            
            # Outros botoes podem ser mapeados aqui
            
            # Executar acao no ambiente
            action_input = [action.reshape(1, -1)]
            obs, _, _, _ = self.env.step(action_input)
            
            # Renderizar
            if obs is not None:
                img = obs[0, :, :, -1]
                img_resized = cv2.resize(img, (self.screen_width, self.screen_height), 
                                        interpolation=cv2.INTER_NEAREST)
                color = (0, 255, 0) if is_running else (0, 0, 255)
                
                # Converter para pygame surface
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB) if len(img_resized.shape) == 2 else cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                surface = pygame.image.frombuffer(img_rgb.flatten(), 
                                                  (self.screen_width, self.screen_height), 'RGB')
                
                window.blit(surface, (0, 0))
                
                # Texto de status
                fps_text = font.render(f"FPS: {int(actual_fps)}", True, (255, 255, 255))
                count_text = font.render(f"Demos: {self.count_record}/{self.max_traj}", True, (255, 255, 255))
                status_text = font.render("RECORDING" if is_running else "IDLE", True, color)
                
                window.blit(fps_text, (10, 10))
                window.blit(count_text, (10, 40))
                window.blit(status_text, (10, 70))
                
                pygame.display.flip()
            
            # Gravar trajetoria
            if is_running:
                if not self.is_recording:
                    self.recorded_obs.append(obs)
                self.recorded_obs.append(obs)
                self.recorded_actions.append(action.copy())
                self.is_recording = True
            elif self.is_recording:
                print(f"Finish trajectory {self.count_record}")
                obs_uint8 = np.stack([o.astype(np.uint8) for o in self.recorded_obs], axis=0)
                print(f"Trajectory shape: {obs_uint8.shape}")
                self.trajectories.append(
                    Trajectory(obs=obs_uint8, acts=np.array(self.recorded_actions), 
                              infos=None, terminal=False)
                )
                self.recorded_obs, self.recorded_actions = [], []
                self.count_record += 1
                self.is_recording = False
            
            # FPS counter
            fps_avg += 1
            if time.time() - fps_start_time >= 1.0:
                actual_fps = fps_avg
                fps_avg = 0
                fps_start_time = time.time()
        
        # Salvar trajetorias
        print("Saving to file...")
        th.save(self.trajectories, os.path.join(self.demo_path, f"demos{self.count_record}.pt"))
        print("Save completed!")
        pygame.quit()


def main():
    """Funcao principal para iniciar gravacao."""
    # Configuracao do jogo (ajuste conforme necessario)
    config = {
        "process_name": "re9",  # Altere para o processo do seu jogo
        "exe_path": None,  # R"C:\Caminho\Para\Jogo.exe" ou None
    }
    
    # Criar ambiente
    def make_env():
        def _init():
            env = GenericGameEnv(config)
            return env
        return _init
    
    env = make_vec_env(make_env(), n_envs=1, vec_env_cls=DummyVecEnv)
    env.reset()
    
    # Gravar trajetorias
    recorder = TrajectoryRecorder(env, demo_path='./demos/', max_traj=10)
    recorder.start_recording()


if __name__ == "__main__":
    main()
