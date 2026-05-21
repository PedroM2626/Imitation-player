"""
Script de treinamento generico para imitation learning.
Baseado em trainning-imitation.ipynb.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path

from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory, TrajectoryWithRew
from imitation.data.serialize import load_trajectory
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from game_env import GenericGameEnv, TemporalAttentionLSTM


class ImitationLearner:
    """
    Treina um agente de Imitation Learning usando Behavioral Cloning (BC)
    e DAgger (Dataset Aggregation).
    """
    
    def __init__(self, config, demo_path='./demos/'):
        self.config = config
        self.demo_path = demo_path
        self.env = self._create_env()
        self.trajectories = []
        
        os.makedirs(demo_path, exist_ok=True)
    
    def _create_env(self):
        """Cria o ambiente envolvido."""
        env = GenericGameEnv(self.config)
        env = WarpFrame(env, width=128, height=128)  # Resizing para 128x128
        env = DummyVecEnv([lambda: env])
        # env = VecFrameStack(env, 4)  # Opcional: frame stacking
        return env
    
    def load_demos(self):
        """Carrega as trajetorias salvas."""
        demo_files = glob.glob(os.path.join(self.demo_path, 'demos*.pt'))
        
        trajectories = []
        for f in sorted(demo_files):
            print(f"Loading: {f}")
            try:
                data = th.load(f, map_location=th.device('cpu'))
                # Suporte tanto para listas quanto para trajetorias individuais
                if isinstance(data, list):
                    trajectories.extend(data)
                else:
                    trajectories.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.trajectories = trajectories
        print(f"Loaded {len(self.trajectories)} trajectories")
        return trajectories
    
    def train(self, epochs=100, batch_size=384, save_path='./models/'):
        """
        Treina a politica usando Behavioral Cloning.
        
        Args:
            epochs: Numero de epocas de treinamento
            batch_size: Tamanho do batch
            save_path: Caminho para salvar os modelos
        """
        if not self.trajectories:
            print("No trajectories to train!")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Criar objeto BC (Behavioral Cloning)
        rng = np.random.default_rng()
        
        bc_trainer = BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=rng,
            batch_size=batch_size,
        )
        
        print("Starting training...")
        bc_trainer.train(n_epochs=epochs)
        
        # Salvar modelo
        model_path = os.path.join(save_path, "bc_policy.zip")
        bc_trainer.policy.save(model_path)
        print(f"Model saved to {model_path}")
        
        return bc_trainer
    
    def dagger_iteration(self):
        """
        Executa uma iteracao de DAGGER:
        1. A politica atual joga e gera trajetorias
        2. O humano (implicitamente) corrige e gera novos dados
        3. Retreina com dados combinados
        """
        # TODO: DAGGER implementacao completa necessita de loop humano-in-the-loop
        pass


def main():
    # Configuracao do jogo
    config = {
        "process_name": "re9",  # Altere para o seu jogo
        "exe_path": None,
        "capture": {
            "width": 854,
            "height": 480,
            "internal_width": 128,
            "internal_height": 128,
            "target_fps": 240,
        },
        "window_offset": {"left": 20, "top": 100, "right": 0, "bottom": 0},
        "actions": {"num_actions": 18},
    }
    
    # Criar e treinar
    learner = ImitationLearner(config, demo_path='./demos/')
    learner.load_demos()
    learner.train(epochs=100, save_path='./models/')


if __name__ == "__main__":
    main()
