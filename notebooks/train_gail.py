"""
Script para treinar um agente usando GAIL (Generative Adversarial Imitation Learning).
Diferente do BC, o GAIL requer que o jogo esteja rodando para treinar o gerador (PPO).
"""

import os
import sys
import glob
import argparse
from typing import List, Dict, Any
import time

# Ajustar path para encontrar os modulos
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th

# Bibliotecas de Machine Learning
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat

# Modulos locais
from game_env import GenericGameEnv
from config.game_config import GAME_CONFIG

import mlflow

class MLflowOutputFormat(KVWriter):
    """Custom KVWriter to log metrics directly to MLflow."""
    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        if mlflow.active_run():
            for key, value in key_values.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(key, float(value), step=step)

    def close(self) -> None:
        pass


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


class DataManager:
    """Gerencia os dados de treinamento (trajetorias)."""
    
    def __init__(self, demo_path: str = './demos/'):
        self.demo_path = demo_path
        self.trajectories: List[Trajectory] = []
    
    def load_demos(self) -> List[Trajectory]:
        print_header("1. CARREGANDO DADOS (GABARITO HUMANO)")
        demo_files = sorted(glob.glob(os.path.join(self.demo_path, 'demos*.pt')))
        
        if not demo_files:
            raise FileNotFoundError(f"Nenhum arquivo de demo encontrado em {self.demo_path}")
        
        print(f"Encontrados {len(demo_files)} arquivos de demo:")
        
        all_trajectories = []
        for i, demo_file in enumerate(demo_files):
            try:
                print(f"  [{i+1}/{len(demo_files)}] Carregando {os.path.basename(demo_file)}...", end=" ")
                data = th.load(demo_file, map_location='cpu')
                
                if isinstance(data, list):
                    for t in data:
                        obs = t.obs
                        if obs.ndim == 5 and obs.shape[1] == 1:
                            obs = np.squeeze(obs, axis=1)
                        t = Trajectory(obs=obs, acts=t.acts, infos=t.infos, terminal=t.terminal)
                        all_trajectories.append(t)
                    print(f"({len(data)} traj.)")
                else:
                    obs = data.obs
                    if obs.ndim == 5 and obs.shape[1] == 1:
                        obs = np.squeeze(obs, axis=1)
                    data = Trajectory(obs=obs, acts=data.acts, infos=data.infos, terminal=data.terminal)
                    all_trajectories.append(data)
                    print("(1 traj.)")
                    
            except Exception as e:
                print(f" ERRO: {e}")
                continue
        
        self.trajectories = all_trajectories
        return all_trajectories


class GAILTrainer:
    def __init__(self, config: Dict[str, Any], device='cuda', total_timesteps=100000):
        self.config = config
        self.device = th.device(device)
        self.total_timesteps = total_timesteps
        
        print_header("2. INICIANDO AMBIENTE DO JOGO")
        print("Aguardando jogo ser detectado na tela... GAIL precisa jogar o jogo para aprender!")
        self.env = self._create_env()
        self.gail_trainer = None
        self.learner = None

    def _create_env(self):
        train_config = self.config.copy()
        train_config["dummy"] = False  # MUITO IMPORTANTE: O jogo DEVE estar rodando fisicamente!
        env = GenericGameEnv(train_config)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
        return env

    def setup_trainer(self, trajectories: List[Trajectory], model_path: str = None):
        print_header("3. CONFIGURANDO REDE ADVERSARIA (GAIL)")
        
        rng = np.random.default_rng(seed=42)
        log_dir = os.path.join("./models/imitation/", "gail_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        output_formats = [HumanOutputFormat(sys.stdout), CSVOutputFormat(os.path.join(log_dir, "progress.csv")), MLflowOutputFormat()]
        custom_sb3_logger = Logger(folder=log_dir, output_formats=output_formats)

        # 1. Agente (Gerador) usando PPO
        if model_path and os.path.exists(model_path):
            print(f"[+] Transfer Learning ativado! Carregando PPO base: {model_path}")
            self.learner = PPO.load(model_path, env=self.env, device=self.device)
        else:
            if model_path:
                print(f"[!] Aviso: Modelo '{model_path}' nao encontrado. Treinando PPO do zero.")
            self.learner = PPO(
                env=self.env,
                policy=ActorCriticCnnPolicy,
                batch_size=64,
                ent_coef=0.01,
                learning_rate=3e-4,
                gamma=0.99,
                n_steps=1024,
                device=self.device,
            )

        # 2. Rede de Recompensa (Discriminador)
        reward_net = BasicRewardNet(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            normalize_input_layer=RunningNorm,
        )

        # 3. GAIL Trainer
        self.gail_trainer = GAIL(
            demonstrations=trajectories,
            demo_batch_size=64,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=self.env,
            gen_algo=self.learner,
            reward_net=reward_net,
            custom_logger=custom_sb3_logger,
        )
        print("[OK] GAIL Trainer configurado com PPO e Discriminador.")

    def train(self, save_path: str = './models/'):
        print_header("4. INICIANDO TREINAMENTO ADVERSARIO (JOGO ATIVO)")
        print(f"O agente vai jogar por {self.total_timesteps} passos.")
        
        os.makedirs(save_path, exist_ok=True)
        
        mlflow.set_experiment("Hajime_no_Ippo_Imitation_Learning")
        with mlflow.start_run(run_name="GAIL_Run"):
            mlflow.log_params({"algorithm": "GAIL", "timesteps": self.total_timesteps, "algo_base": "PPO"})
            
            try:
                self.gail_trainer.train(total_timesteps=self.total_timesteps)
                print("\n[OK] Treinamento GAIL concluido!")
                
                # Salvar a politica do PPO gerada pelo GAIL
                policy_path = os.path.join(save_path, "gail_policy")
                self.learner.policy.save(policy_path)
                mlflow.log_artifact(policy_path + ".zip")
                print(f"[OK] Modelo salvo em {policy_path}.zip")
                
            except Exception as e:
                print(f"\n[ERRO] Falha no treinamento GAIL: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Treinamento GAIL.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Passos totais de jogo para treinamento")
    parser.add_argument("--model_path", type=str, default=None, help="Caminho para modelo (.zip) pre-treinado (Transfer Learning)")
    args = parser.parse_args()

    dm = DataManager()
    try:
        trajectories = dm.load_demos()
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    device = "cuda" if th.cuda.is_available() else "cpu"
    trainer = GAILTrainer(GAME_CONFIG, device=device, total_timesteps=args.timesteps)
    trainer.setup_trainer(trajectories, model_path=args.model_path)
    trainer.train()

if __name__ == "__main__":
    main()
