"""
Script to train an agent using GAIL (Generative Adversarial Imitation Learning).
Unlike BC, GAIL requires the game to be running so the generator (PPO) can be trained.
"""

import os
import sys
import glob
import argparse
from typing import List, Dict, Any
import time

# Adjust the path so the modules can be found
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th

# Machine Learning libraries
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat

# Local modules
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
    """Manages the training data (trajectories)."""
    
    def __init__(self, demo_path: str = './demos/'):
        self.demo_path = demo_path
        self.trajectories: List[Trajectory] = []
    
    def load_demos(self) -> List[Trajectory]:
        print_header("1. LOADING DATA (HUMAN DEMOS)")
        demo_files = sorted(glob.glob(os.path.join(self.demo_path, 'demo*.pt')))
        
        if not demo_files:
            raise FileNotFoundError(f"No demo files found in {self.demo_path}")
        
        print(f"Found {len(demo_files)} demo files:")
        
        all_trajectories = []
        for i, demo_file in enumerate(demo_files):
            try:
                print(f"  [{i+1}/{len(demo_files)}] Loading {os.path.basename(demo_file)}...", end=" ")
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
                print(f" ERROR: {e}")
                continue
        
        self.trajectories = all_trajectories
        return all_trajectories


class GAILTrainer:
    def __init__(self, config: Dict[str, Any], device='cuda', total_timesteps=100000):
        self.config = config
        self.device = th.device(device)
        self.total_timesteps = total_timesteps
        
        print_header("2. STARTING THE GAME ENVIRONMENT")
        print("Waiting for the game to be detected on screen... GAIL needs to play the game to learn!")
        self.env = self._create_env()
        self.gail_trainer = None
        self.learner = None

    def _create_env(self):
        train_config = self.config.copy()
        train_config["dummy"] = False  # CRITICAL: the game MUST actually be running!
        env = GenericGameEnv(train_config)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
        return env

    def setup_trainer(self, trajectories: List[Trajectory], model_path: str = None):
        print_header("3. SETTING UP THE ADVERSARIAL NETWORK (GAIL)")
        
        rng = np.random.default_rng(seed=42)
        log_dir = os.path.join("./models/imitation/", "gail_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        output_formats = [HumanOutputFormat(sys.stdout), CSVOutputFormat(os.path.join(log_dir, "progress.csv")), MLflowOutputFormat()]
        custom_sb3_logger = Logger(folder=log_dir, output_formats=output_formats)

        # 1. Agent (Generator) using PPO
        if model_path and os.path.exists(model_path):
            print(f"[+] Transfer Learning enabled! Loading base PPO: {model_path}")
            self.learner = PPO.load(model_path, env=self.env, device=self.device)
        else:
            if model_path:
                print(f"[!] Warning: model '{model_path}' not found. Training PPO from scratch.")
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

        # 2. Reward network (Discriminator)
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
        print("[OK] GAIL Trainer configured with PPO and Discriminator.")

    def train(self, save_path: str = './models/'):
        print_header("4. STARTING ADVERSARIAL TRAINING (GAME ACTIVE)")
        print(f"The agent will play for {self.total_timesteps} steps.")
        
        os.makedirs(save_path, exist_ok=True)
        
        mlflow.set_experiment("Hajime_no_Ippo_Imitation_Learning")
        with mlflow.start_run(run_name="GAIL_Run"):
            mlflow.log_params({"algorithm": "GAIL", "timesteps": self.total_timesteps, "algo_base": "PPO"})
            
            try:
                self.gail_trainer.train(total_timesteps=self.total_timesteps)
                print("\n[OK] GAIL training complete!")
                
                # Save the PPO policy generated by GAIL
                policy_path = os.path.join(save_path, "gail_policy")
                self.learner.policy.save(policy_path)
                mlflow.log_artifact(policy_path + ".zip")
                print(f"[OK] Model saved to {policy_path}.zip")
                
            except Exception as e:
                print(f"\n[ERROR] GAIL training failed: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="GAIL training.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total gameplay steps for training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pretrained model (.zip) (Transfer Learning)")
    args = parser.parse_args()

    dm = DataManager()
    try:
        trajectories = dm.load_demos()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    device = "cuda" if th.cuda.is_available() else "cpu"
    trainer = GAILTrainer(GAME_CONFIG, device=device, total_timesteps=args.timesteps)
    trainer.setup_trainer(trajectories, model_path=args.model_path)
    trainer.train()

if __name__ == "__main__":
    main()
