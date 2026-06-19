"""
Tutorial: Treinar Agente com Arquitetura CNN+LSTM+Attention
============================================================

Este script usa o TemporalAttentionLSTM como feature extractor,
que combina CNN para visao espacial com LSTM bidirecional para
memoria temporal e Attention para focar nos frames mais relevantes.

A diferenca para o train_agent.py padrao e que aqui o modelo
tem MEMORIA: ele lembra o que aconteceu nos ultimos 10 frames,
permitindo entender velocidade, direcao de movimento e padroes
temporais de ataque do inimigo.

COMO USAR:
    python train_agent_lstm.py --epochs 50 --batch 384 --lr 1e-4

O modelo sera salvo em models/bc_policy_lstm.zip
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Optional, Dict, Any

# Ajustar path para encontrar os modulos
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
import pickle

# Bibliotecas de Machine Learning
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat

# Modulos locais
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
    """Imprime um cabecalho formatado na tela."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_cuda() -> str:
    """Verifica se CUDA esta disponivel e retorna o device."""
    if th.cuda.is_available():
        device = "cuda"
        print(f"[OK] GPU detectada: {th.cuda.get_device_name(0)}")
        print(f"     CUDA version: {th.version.cuda}")
    else:
        device = "cpu"
        print("[!] CUDA nao encontrada, usando CPU (mais lento!)")
    return device


class DataManager:
    """Gerencia os dados de treinamento (trajetorias)."""
    
    def __init__(self, demo_path: str = './demos/'):
        self.demo_path = Path(demo_path)
        self.trajectories: List[Trajectory] = []
        self.stats = {}
    
    def load_demos(self, verbose: bool = True) -> List[Trajectory]:
        """Carrega todas as trajetorias do diretorio demos/."""
        print_header("1. CARREGANDO DADOS")
        
        if not self.demo_path.exists():
            raise FileNotFoundError(
                f"Diretorio {self.demo_path} nao encontrado!\n"
                "Grave trajetorias primeiro com record_trajectories.py"
            )
        
        demo_files = sorted(self.demo_path.glob('demo*.pt'))
        
        if not demo_files:
            raise FileNotFoundError(
                f"Nenhum arquivo de demo encontrado em {self.demo_path}\n"
                "Execute record_trajectories.py primeiro!"
            )
        
        print(f"Encontrados {len(demo_files)} arquivos de demo:")
        
        all_trajectories = []
        for i, demo_file in enumerate(demo_files):
            try:
                print(f"  [{i+1}/{len(demo_files)}] Carregando {demo_file.name}...", end=" ")
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
                print(f" ERRO: {e}")
                continue
        
        self.trajectories = all_trajectories
        
        total_frames = sum(len(t.obs) for t in all_trajectories)
        self.stats = {
            'num_trajectories': len(all_trajectories),
            'total_frames': total_frames,
            'avg_frames_per_traj': total_frames / len(all_trajectories) if all_trajectories else 0,
        }
        
        if verbose:
            print(f"\n[OK] Dados carregados:")
            print(f"     - {self.stats['num_trajectories']} trajetorias")
            print(f"     - {self.stats['total_frames']} frames totais")
            print(f"     - {self.stats['avg_frames_per_traj']:.1f} frames/trajetoria (media)")
        
        return all_trajectories
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats
    
    def describe_actions(self) -> None:
        """Imprime distribuicao das acoes (uso de cada botao)."""
        if not self.trajectories:
            print("Nenhum dado carregado!")
            return
        
        num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
        print(f"\nDistribuicao das acoes (0-{num_actions-1}):")
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
        
        print(f"\nTotal: {action_total} frames analisados")


class LSTMTrainer:
    """Gerencia o treinamento do agente usando CNN+LSTM+Attention."""
    
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
        """Cria o ambiente base do Stable Baselines3."""
        train_config = self.config.copy()
        train_config["dummy"] = True
        env = GenericGameEnv(train_config)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
        return env
    
    def setup_bc_trainer(self, trajectories: List[Trajectory], model_path: str = None):
        """Configura o treinador de Behavioral Cloning com LSTM."""
        print_header("2. CONFIGURANDO O TREINADOR (CNN+LSTM+Attention)")
        
        rng = np.random.default_rng(seed=42)
        
        log_dir = os.path.join(self.model_save_path, "imitation", "bc_lstm_logs")
        os.makedirs(log_dir, exist_ok=True)
        output_formats = [HumanOutputFormat(sys.stdout), CSVOutputFormat(os.path.join(log_dir, "progress.csv")), MLflowOutputFormat()]
        custom_sb3_logger = Logger(folder=log_dir, output_formats=output_formats)
        
        # Configurar o TemporalAttentionLSTM como feature extractor
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
            print(f"[+] Transfer Learning ativado! Carregando: {model_path}")
            policy = ActorCriticCnnPolicy.load(model_path, device=self.device)
        elif model_path:
            print(f"[!] Aviso: Modelo '{model_path}' nao encontrado. Treinando do zero.")
        
        if policy is None:
            print("[*] Criando policy com TemporalAttentionLSTM...")
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
        
        # Contar parametros
        num_params = sum(p.numel() for p in policy.parameters())
        print(f"[OK] BC Trainer configurado (CNN+LSTM+Attention):")
        print(f"     Batch size: {self.batch_size}")
        print(f"     Learning rate: {self.learning_rate}")
        print(f"     Device: {self.device}")
        print(f"     Parametros: {num_params:,}")
    
    def train(self, save_path: str = './models/'):
        """Executa o treinamento completo."""
        print_header("3. TREINAMENTO - CNN+LSTM+Attention Behavioral Cloning")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.bc_trainer.train(
            n_epochs=self.epochs,
            progress_bar=True,
        )
        
        print(f"\n[OK] Treinamento concluido!")
        
        model_path = os.path.join(save_path, "bc_policy_lstm.zip")
        self.bc_trainer.policy.save(model_path)
        print(f"[OK] Modelo salvo em: {model_path}")
        
        self.policy = self.bc_trainer.policy
        return self.policy


def main():
    parser = argparse.ArgumentParser(
        description="Treina agente com arquitetura CNN+LSTM+Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python train_agent_lstm.py --epochs 50 --lr 1e-4
  python train_agent_lstm.py --batch 256 --device cpu
        """)
    
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG.get("epochs", 100), help='Numero de epocas')
    parser.add_argument('--batch', type=int, default=TRAINING_CONFIG.get("batch_size", 384), help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG.get("learning_rate", 1e-4), help='Learning rate')
    parser.add_argument("--device", type=str, default="cuda", help="Device para treinamento (cuda/cpu)")
    parser.add_argument("--model_path", type=str, default=None, help="Caminho para modelo (.zip) pre-treinado (Transfer Learning)")
    
    args = parser.parse_args()
    
    print_header("IMITATION LEARNING - TREINAMENTO (CNN+LSTM+Attention)")
    
    device = args.device or check_cuda()
    print(f"\n[CONFIG] Epocas: {args.epochs}, Batch: {args.batch}, LR: {args.lr}, Device: {device}")
    
    # MLflow
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Generic_Agent_LSTM_IL")
    
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
            # Passo 1: Carregar dados
            data_manager = DataManager(demo_path='./demos/')
            trajectories = data_manager.load_demos(verbose=True)
            data_manager.describe_actions()
            
            # Passo 2: Configurar treinador
            trainer = LSTMTrainer(
                config=GAME_CONFIG,
                device=device,
                learning_rate=args.lr,
                batch_size=args.batch,
                epochs=args.epochs
            )
            
            # Passo 3: Treinar
            trainer.setup_bc_trainer(trajectories, model_path=args.model_path)
            trained_policy = trainer.train(save_path='./models/')
            
            # Registrar artefatos
            mlflow.log_artifact('./models/bc_policy_lstm.zip', artifact_path="model")
            
            print_header("TREINAMENTO CONCLUIDO (CNN+LSTM+Attention)")
            print("Proximos passos:")
            print("  1. Teste o modelo: python run_ai_lstm.py")
            print("  2. Compare com outros modelos: python compare_models.py")
            
        except Exception as e:
            print(f"\n[ERRO] Falha no treinamento: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
