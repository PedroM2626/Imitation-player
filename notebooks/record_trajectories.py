"""
Script de gravacao de trajetorias humanas para jogos genericos.
Adaptado de imitatation-record.ipynb para ser mais simples e generico.
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

# Ajuste o path para encontrar os modulos
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from imitation.data.types import Trajectory
from game_env import GenericGameEnv
from config.game_config import GAME_CONFIG


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
        
        # Inicializar XInput (ctypes)
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
        action = np.zeros(18, dtype=np.float32)
        actual_fps = 0
        fps_start_time = time.time()
        fps_avg = 0
        
        while self.count_record < self.max_traj:
            loop_start = time.time()
            
            # Bombeamento de eventos Pygame (evita que a janela trave no Windows) e check de saida
            force_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    force_exit = True
            
            if keyboard.is_pressed('esc') or force_exit:
                print("\nExiting and saving trajectories...")
                break
            
            # Verificar toggle de gravacao
            if keyboard.is_pressed('k'):
                is_running = not is_running
                time.sleep(0.3)
                print(f"Recording: {is_running}")
            
            # Capturar entrada do teclado e joystick
            action.fill(0)
            
            # Teclado
            if keyboard.is_pressed('up'): action[0] = 1
            if keyboard.is_pressed('down'): action[1] = 1
            if keyboard.is_pressed('left'): action[2] = 1
            if keyboard.is_pressed('right'): action[3] = 1
            if keyboard.is_pressed('i'): action[4] = 1
            if keyboard.is_pressed('o'): action[5] = 1
            if keyboard.is_pressed('p'): action[6] = 1
            
            # XInput (Global Background Controller Capture)
            if hasattr(self, 'xinput') and self.xinput:
                for j_idx in range(4):
                    state = self.XINPUT_STATE()
                    if self.xinput.XInputGetState(j_idx, ctypes.byref(state)) == 0:
                        buttons = state.Gamepad.wButtons
                        # D-Pad
                        if buttons & 0x0001: action[0] = 1 # UP
                        if buttons & 0x0002: action[1] = 1 # DOWN
                        if buttons & 0x0004: action[2] = 1 # LEFT
                        if buttons & 0x0008: action[3] = 1 # RIGHT
                        
                        # Analogico esquerdo
                        if state.Gamepad.sThumbLY > 16000: action[0] = 1 # UP
                        if state.Gamepad.sThumbLY < -16000: action[1] = 1 # DOWN
                        if state.Gamepad.sThumbLX < -16000: action[2] = 1 # LEFT
                        if state.Gamepad.sThumbLX > 16000: action[3] = 1 # RIGHT
                        
                        # Botoes de face (A=0x1000, B=0x2000, X=0x4000, Y=0x8000)
                        if buttons & 0x1000: action[4] = 1 # A / Cross
                        if buttons & 0x2000: action[5] = 1 # B / Circle
                        if buttons & 0x4000: action[6] = 1 # X / Square
            
            # Outros botoes podem ser mapeados aqui
            
            # Executar acao no ambiente
            action_input = [action.reshape(1, -1)]
            obs, _, _, _ = self.env.step(action_input)
            
            # Renderizar
            if obs is not None:
                # Selecionar o frame mais recente (o ultimo no stack)
                # Como VecTransposeImage converte para (C, H, W) e VecFrameStack empilha no canal,
                # o formato e (12, 84, 84). Os ultimos 3 canais sao o frame RGB atual.
                img_rgb = obs[0, -3:, :, :]
                img_rgb = np.transpose(img_rgb, (1, 2, 0)) # (84, 84, 3)
                
                img_resized = cv2.resize(img_rgb, (self.screen_width, self.screen_height), 
                                        interpolation=cv2.INTER_NEAREST)
                color = (0, 255, 0) if is_running else (0, 0, 255)
                
                # Converter para pygame surface
                surface = pygame.image.frombuffer(img_resized.flatten(), 
                                                  (self.screen_width, self.screen_height), 'RGB')
                
                window.blit(surface, (0, 0))
                
                # Texto de status
                fps_text = font.render(f"FPS: {int(actual_fps)}", True, (255, 255, 255))
                count_text = font.render(f"Demos: {self.count_record}/{self.max_traj}", True, (255, 255, 255))
                status_text = font.render("RECORDING" if is_running else "IDLE", True, color)
                instructions_text = font.render("[K] Toggle Record   [ESC] Save & Exit", True, (200, 200, 200))
                
                window.blit(fps_text, (10, 10))
                window.blit(count_text, (10, 40))
                window.blit(status_text, (10, 70))
                window.blit(instructions_text, (10, 100))
                
                pygame.display.flip()
            
            # Gravar trajetoria
            if is_running:
                # np.squeeze remove a dimensao de lote (batch=1) vinda do DummyVecEnv
                obs_to_save = np.squeeze(obs, axis=0)
                if not self.is_recording:
                    self.recorded_obs.append(obs_to_save)
                self.recorded_obs.append(obs_to_save)
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
        
        # O salvamento agora ocorre no bloco finally da main()
        pass


def main():
    """Funcao principal para iniciar gravacao."""
    def make_env():
        return GenericGameEnv(GAME_CONFIG)
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    
    # Gravar trajetorias
    recorder = TrajectoryRecorder(env, demo_path='./demos/', max_traj=10)
    
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    finally:
        # Garante que as trajetorias sejam salvas mesmo se o script for abortado (Ctrl+C)
        if len(recorder.trajectories) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            print(f"\nSaving {len(recorder.trajectories)} trajectories to file...")
            save_file = os.path.join(recorder.demo_path, f"demos_{len(recorder.trajectories)}_{timestamp}.pt")
            th.save(recorder.trajectories, save_file)
            print(f"Save completed at: {save_file}")
        else:
            print("\nNo complete trajectories were recorded. Nothing to save.")
        
        try:
            pygame.quit()
        except:
            pass


if __name__ == "__main__":
    main()
