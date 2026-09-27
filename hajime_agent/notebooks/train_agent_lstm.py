"""
Tutorial: Training an Agent with a CNN+LSTM+Attention Architecture
============================================================

This script uses TemporalAttentionLSTM as the feature extractor,
which combines a CNN for spatial vision with a bidirectional LSTM for
temporal memory and Attention to focus on the most relevant frames.

Unlike the default train_agent.py, here the model
has MEMORY: it remembers what happened in the last 10 frames,
making it possible to understand speed, movement direction and
temporal attack patterns of the enemy.

HOW TO USE:
    python train_agent_lstm.py --epochs 50 --batch 384 --lr 1e-4

The model is saved to models/bc_policy_lstm.zip
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Optional, Dict, Any

# Adjust path to find the modules
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
import pickle

# Machine Learning libraries
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat

# Local modules
from game_env import GenericGameEnv, TemporalAttentionLSTM
from config.game_config import GAME_CONFIG, TRAINING_CONFIG

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
    """Print a formatted header to the console."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_cuda() -> str:
    """Check if CUDA is available and return the device."""
    if th.cuda.is_available():
        device = "cuda"
        print(f"[OK] GPU detected: {th.cuda.get_device_name(0)}")
        print(f"     CUDA version: {th.version.cuda}")
    else:
        device = "cpu"
        print("[!] CUDA not found, using CPU (slower!)")
    return device


class DataManager:
    """Manages the training data (trajectories)."""
    
    def __init__(self, demo_path: str = './demos/'):
        self.demo_path = Path(demo_path)
        self.trajectories: List[Trajectory] = []
        self.stats = {}
    
    def load_demos(self, verbose: bool = True) -> List[Trajectory]:
        """Loads all trajectories from the demos/ directory."""
        print_header("1. LOADING DATA")
        
        if not self.demo_path.exists():
            raise FileNotFoundError(
                f"Directory {self.demo_path} not found!\n"
                "Record trajectories first with record_trajectories.py"
            )
        
        demo_files = sorted(self.demo_path.glob('demo*.pt'))
        
        if not demo_files:
            raise FileNotFoundError(
                f"No demo files found in {self.demo_path}\n"
                "Run record_trajectories.py first!"
            )
        
        print(f"Found {len(demo_files)} demo files:")
        
        all_trajectories = []
        for i, demo_file in enumerate(demo_files):
            try:
                print(f"  [{i+1}/{len(demo_files)}] Loading {demo_file.name}...", end=" ")
                data = th.load(demo_file, map_location='cpu')
                
                expected_num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
                
                if isinstance(data, list):
                    for t in data:
                        obs = t.obs
                        if obs.ndim == 5 and obs.shape[1] == 1:
                            obs = np.squeeze(obs, axis=1)
                        
                        acts = t.acts
                        if acts.shape[-1] > expected_num_actions:
                            acts = acts[..., :expected_num_actions]
                        elif acts.shape[-1] < expected_num_actions:
                            pad_width = [(0, 0)] * (acts.ndim - 1) + [(0, expected_num_actions - acts.shape[-1])]
                            acts = np.pad(acts, pad_width, mode='constant', constant_values=0)
                            
                        t = Trajectory(obs=obs, acts=acts, infos=t.infos, terminal=t.terminal)
                        all_trajectories.append(t)
                    print(f"({len(data)} traj.)")
                else:
                    obs = data.obs
                    if obs.ndim == 5 and obs.shape[1] == 1:
                        obs = np.squeeze(obs, axis=1)
                        
                    acts = data.acts
                    if acts.shape[-1] > expected_num_actions:
                        acts = acts[..., :expected_num_actions]
                    elif acts.shape[-1] < expected_num_actions:
                        pad_width = [(0, 0)] * (acts.ndim - 1) + [(0, expected_num_actions - acts.shape[-1])]
                        acts = np.pad(acts, pad_width, mode='constant', constant_values=0)
                        
                    data = Trajectory(obs=obs, acts=acts, infos=data.infos, terminal=data.terminal)
                    all_trajectories.append(data)
                    print("(1 traj.)")
                    
            except Exception as e:
                print(f" ERROR: {e}")
                continue
        
        self.trajectories = all_trajectories
        
        total_frames = sum(len(t.obs) for t in all_trajectories)
        self.stats = {
            'num_trajectories': len(all_trajectories),
            'total_frames': total_frames,
            'avg_frames_per_traj': total_frames / len(all_trajectories) if all_trajectories else 0,
        }
        
        if verbose:
            print(f"\n[OK] Data loaded:")
            print(f"     - {self.stats['num_trajectories']} trajectories")
            print(f"     - {self.stats['total_frames']} total frames")
            print(f"     - {self.stats['avg_frames_per_traj']:.1f} frames/trajectory (mean)")
        
        return all_trajectories
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats
    
    def describe_actions(self) -> None:
        """Print the action distribution (usage of each button)."""
        if not self.trajectories:
            print("No data loaded!")
            return
        
        num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
        print(f"\nAction distribution (0-{num_actions-1}):")
        action_counts = np.zeros(num_actions)
        action_total = 0
        
        for traj in self.trajectories:
            acts = traj.acts
            if len(acts.shape) == 1:
                acts = acts.reshape(-1, 1)
            action_total += len(acts)
            for frame_acts in acts:
                action_counts += frame_acts
        
        action_names = []
        mappings = GAME_CONFIG.get("actions", {}).get("mappings", [])
        if len(mappings) >= num_actions:
            action_names = [m.get("name", f"ACT_{i}") for i, m in enumerate(mappings[:num_actions])]
        else:
            action_names = [
                "UP", "DOWN", "LEFT", "RIGHT",
                "X(CROSS)", "O(CIRCLE)", "[](SQUARE)",
                "L2", "R2", "L3",
                "CAM_R1", "CAM_R2", "CAM_L1", "CAM_L2",
                "CAM_U1", "CAM_U2", "CAM_D1", "CAM_D2"
            ][:num_actions]
        
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            pct = (count / action_total) * 100 if action_total > 0 else 0
            bar = "\u2588" * int(pct / 2)
            print(f"  [{i:2d}] {name:12s}: {bar} {pct:.1f}% ({int(count)}x)")
        
        print(f"\nTotal: {action_total} frames analyzed")


class LSTMTrainer:
    """Manages agent training using CNN+LSTM+Attention."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 batch_size: int = 384,
                 epochs: int = 100,
                 model_save_path: str = './models/'):
        
        self.config = config
        self.device = th.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_path = model_save_path
        
        self.env = self._create_env()
        self.bc_trainer = None
        self.policy = None
    
    def _create_env(self):
        """Create the base Stable Baselines3 environment."""
        train_config = self.config.copy()
        train_config["dummy"] = True
        env = GenericGameEnv(train_config)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
        return env
    
    def setup_bc_trainer(self, trajectories: List[Trajectory], model_path: str = None):
        """Set up the Behavioral Cloning trainer with LSTM."""
        print_header("2. SETTING UP THE TRAINER (CNN+LSTM+Attention)")
        
        rng = np.random.default_rng(seed=42)
        
        log_dir = os.path.join(self.model_save_path, "imitation", "bc_lstm_logs")
        os.makedirs(log_dir, exist_ok=True)
        output_formats = [HumanOutputFormat(sys.stdout), CSVOutputFormat(os.path.join(log_dir, "progress.csv")), MLflowOutputFormat()]
        custom_sb3_logger = Logger(folder=log_dir, output_formats=output_formats)
        
        # Configure TemporalAttentionLSTM as the feature extractor
        policy_kwargs = dict(
            features_extractor_class=TemporalAttentionLSTM,
            features_extractor_kwargs=dict(
                features_dim=512,
                lstm_hidden_size=256,
                lstm_num_layers=2,
                debug=False,
            ),
        )
        
        policy = None
        if model_path and os.path.exists(model_path):
            print(f"[+] Transfer Learning enabled! Loading: {model_path}")
            policy = ActorCriticCnnPolicy.load(model_path, device=self.device)
        elif model_path:
            print(f"[!] Warning: model '{model_path}' not found. Training from scratch.")
        
        if policy is None:
            print("[*] Creating policy with TemporalAttentionLSTM...")
            policy = ActorCriticCnnPolicy(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                lr_schedule=lambda _: self.learning_rate,
                **policy_kwargs,
            )
        
        self.bc_trainer = BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=trajectories,
            rng=rng,
            device=self.device,
            policy=policy,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.learning_rate},
            custom_logger=custom_sb3_logger,
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in policy.parameters())
        print(f"[OK] BC Trainer configured (CNN+LSTM+Attention):")
        print(f"     Batch size: {self.batch_size}")
        print(f"     Learning rate: {self.learning_rate}")
        print(f"     Device: {self.device}")
        print(f"     Parameters: {num_params:,}")
    
    def train(self, save_path: str = './models/'):
        """Run the full training."""
        print_header("3. TRAINING - CNN+LSTM+Attention Behavioral Cloning")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.bc_trainer.train(
            n_epochs=self.epochs,
            progress_bar=True,
        )
        
        print(f"\n[OK] Training complete!")
        
        model_path = os.path.join(save_path, "bc_policy_lstm.zip")
        self.bc_trainer.policy.save(model_path)
        print(f"[OK] Model saved to: {model_path}")
        
        self.policy = self.bc_trainer.policy
        return self.policy


def main():
    parser = argparse.ArgumentParser(
        description="Train an agent with a CNN+LSTM+Attention architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_agent_lstm.py --epochs 50 --lr 1e-4
  python train_agent_lstm.py --batch 256 --device cpu
        """)
    
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG.get("epochs", 100), help='Number of epochs')
    parser.add_argument('--batch', type=int, default=TRAINING_CONFIG.get("batch_size", 384), help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG.get("learning_rate", 1e-4), help='Learning rate')
    parser.add_argument("--device", type=str, default="cuda", help="Device for training (cuda/cpu)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pretrained model (.zip) for Transfer Learning")
    
    args = parser.parse_args()
    
    print_header("IMITATION LEARNING - TRAINING (CNN+LSTM+Attention)")
    
    device = args.device or check_cuda()
    print(f"\n[CONFIG] Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}, Device: {device}")
    
    # MLflow
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Hajime_no_Ippo_LSTM_IL")
    
    with mlflow.start_run(run_name="BC_LSTM_Training") as run:
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("device", str(device))
        mlflow.log_param("model_type", "imitation_learning_bc")
        mlflow.log_param("architecture", "cnn_lstm_attention")
        mlflow.log_param("lstm_hidden_size", 256)
        mlflow.log_param("lstm_num_layers", 2)
        mlflow.log_param("lstm_bidirectional", True)
        mlflow.log_param("window_size", 10)
        
        try:
            # Step 1: Load data
            data_manager = DataManager(demo_path='./demos/')
            trajectories = data_manager.load_demos(verbose=True)
            data_manager.describe_actions()
            
            # Step 2: Set up the trainer
            trainer = LSTMTrainer(
                config=GAME_CONFIG,
                device=device,
                learning_rate=args.lr,
                batch_size=args.batch,
                epochs=args.epochs
            )
            
            # Step 3: Train
            trainer.setup_bc_trainer(trajectories, model_path=args.model_path)
            trained_policy = trainer.train(save_path='./models/')
            
            # Log artifacts
            mlflow.log_artifact('./models/bc_policy_lstm.zip', artifact_path="model")
            
            print_header("TRAINING COMPLETE (CNN+LSTM+Attention)")
            print("Next steps:")
            print("  1. Test the model: python run_ai_lstm.py")
            print("  2. Compare with other models: python compare_models.py")
            
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

