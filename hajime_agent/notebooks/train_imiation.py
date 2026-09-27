"""
Generic training script for imitation learning.
Based on trainning-imitation.ipynb.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path

from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory, TrajectoryWithRew
from imitation.data.serialize import load_trajectory
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat
from game_env import GenericGameEnv, TemporalAttentionLSTM
from config.game_config import GAME_CONFIG, TRAINING_CONFIG

import mlflow

class MLflowOutputFormat(KVWriter):
    """
    Custom KVWriter to log metrics directly to MLflow.
    """
    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        if mlflow.active_run():
            for key, value in key_values.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(key, float(value), step=step)

    def close(self) -> None:
        pass


class ImitationLearner:
    """
    Trains an Imitation Learning agent using Behavioral Cloning (BC)
    and DAgger (Dataset Aggregation).
    """
    
    def __init__(self, config, demo_path='./demos/'):
        self.config = config
        self.demo_path = demo_path
        self.env = self._create_env()
        self.trajectories = []
        
        os.makedirs(demo_path, exist_ok=True)
    
    def _create_env(self):
        """Creates the wrapped environment."""
        train_config = self.config.copy()
        train_config["dummy"] = True
        env = GenericGameEnv(train_config)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        return env
    
    def load_demos(self):
        """Loads the saved trajectories."""
        demo_files = glob.glob(os.path.join(self.demo_path, 'demos*.pt'))
        
        trajectories = []
        for f in sorted(demo_files):
            print(f"Loading: {f}")
            try:
                data = th.load(f, map_location=th.device('cpu'))
                # Support both lists and individual trajectories
                if isinstance(data, list):
                    for t in data:
                        obs = t.obs
                        if obs.ndim == 5 and obs.shape[1] == 1:
                            obs = np.squeeze(obs, axis=1)
                        if obs.shape[-1] == 3 or obs.shape[-1] == 1:
                            obs = np.transpose(obs, (0, 3, 1, 2))
                        trajectories.append(Trajectory(obs=obs, acts=t.acts, infos=t.infos, terminal=t.terminal))
                else:
                    obs = data.obs
                    if obs.ndim == 5 and obs.shape[1] == 1:
                        obs = np.squeeze(obs, axis=1)
                    if obs.shape[-1] == 3 or obs.shape[-1] == 1:
                        obs = np.transpose(obs, (0, 3, 1, 2))
                    trajectories.append(Trajectory(obs=obs, acts=data.acts, infos=data.infos, terminal=data.terminal))
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.trajectories = trajectories
        print(f"Loaded {len(self.trajectories)} trajectories")
        return trajectories
    
    def train(self, epochs=100, batch_size=384, save_path='./models/'):
        """
        Trains the policy using Behavioral Cloning.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Path to save the models
        """
        if not self.trajectories:
            print("No trajectories to train!")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Create the BC (Behavioral Cloning) object
        rng = np.random.default_rng()
        
        # Create a custom logger for MLflow, console and CSV
        log_dir = os.path.join(save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        output_formats = [
            HumanOutputFormat(sys.stdout),
            CSVOutputFormat(os.path.join(log_dir, "progress.csv")),
            MLflowOutputFormat()
        ]
        custom_sb3_logger = Logger(folder=log_dir, output_formats=output_formats)
        
        # Create the CNN policy
        from stable_baselines3.common.policies import ActorCriticCnnPolicy
        policy = ActorCriticCnnPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            lr_schedule=lambda _: 1e-4,
        )
        
        bc_trainer = BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=rng,
            policy=policy,
            batch_size=batch_size,
            demonstrations=self.trajectories,
            custom_logger=custom_sb3_logger,
        )
        
        print("Starting training...")
        bc_trainer.train(n_epochs=epochs)
        
        # Save model
        model_path = os.path.join(save_path, "bc_policy.zip")
        bc_trainer.policy.save(model_path)
        print(f"Model saved to {model_path}")
        
        return bc_trainer
    
    def dagger_iteration(self):
        """
        Runs one DAgger iteration:
        1. The current policy plays and generates trajectories
        2. The human (implicitly) corrects them and produces new data
        3. Retrains on the combined data
        """
        # TODO: a full DAgger implementation requires a human-in-the-loop
        pass


def main():
    # ===== MLFLOW SETUP =====
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Hajime_no_Ippo_Imitation_Learning")
    
    with mlflow.start_run(run_name="BC_Training_Simple") as run:
        # Log hyperparameters
        mlflow.log_param("epochs", TRAINING_CONFIG.get("epochs", 100))
        mlflow.log_param("batch_size", TRAINING_CONFIG.get("batch_size", 384))
        mlflow.log_param("learning_rate", TRAINING_CONFIG.get("learning_rate", 1e-4))
        mlflow.log_param("model_type", "imitation_learning_bc_simple")
        
        # Log emulator/ROM parameters
        mlflow.log_param("process_name", GAME_CONFIG.get("process_name"))
        mlflow.log_param("rom_path", GAME_CONFIG.get("rom_path"))
        mlflow.log_param("exe_path", GAME_CONFIG.get("exe_path"))
        
        # Create and train
        learner = ImitationLearner(GAME_CONFIG, demo_path='./demos/')
        trajectories = learner.load_demos()
        
        # Log dataset metadata
        mlflow.log_param("dataset_num_trajectories", len(trajectories))
        mlflow.log_param("dataset_total_frames", sum(len(t.obs) for t in trajectories) if trajectories else 0)
        
        learner.train(
            epochs=TRAINING_CONFIG.get("epochs", 100),
            batch_size=TRAINING_CONFIG.get("batch_size", 384),
            save_path='./models/'
        )
        
        # Log artifacts
        mlflow.log_artifact('./models/bc_policy.zip', artifact_path="model")


if __name__ == "__main__":
    main()
