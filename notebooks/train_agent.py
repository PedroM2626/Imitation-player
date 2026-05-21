"""
Tutorial Completo: Como Treinar um Agente de Imitation Learning
================================================================

Este script guia voce passo-a-passo pelo processo de treinamento
de um agente que joga qualquer jogo usando Imitation Learning.

ETAPAS DO TREINAMENTO:
=====================
1. COLETA DE DADOS (gravar voce jogando)
2. PREPARACAO DOS DADOS
3. TREINAMENTO INICIAL (Behavioral Cloning)
4. TESTE DO AGENTE
5. REFINAMENTO (DAGGER - opcional mas recomendado)
6. RETREINAMENTO

REQUISITOS:
============
- Python 3.9+ (3.11 recomendado)
- GPU NVIDIA com CUDA (opcional mas acelera MUITO)
- Dados de demonstracao (trajetorias)

COMO USAR:
==========
1. Grabe trajetorias usando record_trajectories.py
2. Confira que os arquivos .pt estao em demos/
3. Execute: python train_agent.py
4. O modelo sera salvo em models/bc_policy.zip

COMANDO:
    python train_agent.py --epochs 50 --batch 384 --lr 1e-4

ARGUMENTOS:
    --epochs    : Numero de epocas (default: 100)
    --batch     : Tamanho do batch (default: 384)
    --lr        : Learning rate (default: 1e-4)
    --dagger    : Rodar iteracao DAgger apos treino
    --device    : 'cuda' ou 'cpu' (default: auto)
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Optional, Dict, Any

# Ajustar path para encontrar os modulos
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
import pickle

# Bibliotecas de Machine Learning
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Modulos locais
from game_env import GenericGameEnv, TemporalAttentionLSTM


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
    """
    Gerencia os dados de treinamento (trajetorias).
    """
    
    def __init__(self, demo_path: str = './demos/'):
        self.demo_path = Path(demo_path)
        self.trajectories: List[Trajectory] = []
        self.stats = {}
    
    def load_demos(self, verbose: bool = True) -> List[Trajectory]:
        """
        Carrega todas as trajetorias do diretorio demos/.
        
        Returns:
            Lista de objetos Trajectory
        """
        print_header("1. CARREGANDO DADOS")
        
        if not self.demo_path.exists():
            raise FileNotFoundError(
                f"Diretorio {self.demo_path} nao encontrado!\n"
                "Grave trajetorias primeiro com record_trajectories.py"
            )
        
        # Encontrar todos os arquivos de demo
        demo_files = sorted(self.demo_path.glob('demos*.pt'))
        
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
                
                if isinstance(data, list):
                    all_trajectories.extend(data)
                    print(f"({len(data)} traj.")
                else:
                    all_trajectories.append(data)
                    print("(1 traj.)")
                    
            except Exception as e:
                print(f" ERRO: {e}")
                continue
        
        self.trajectories = all_trajectories
        
        # Calcular estatisticas
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
        """Retorna estatisticas dos dados."""
        return self.stats
    
    def describe_actions(self) -> None:
        """Imprime distribuicao das acoes (uso de cada botao)."""
        if not self.trajectories:
            print("Nenhum dado carregado!")
            return
        
        print("\nDistribuicao das acoes (0-17):")
        action_counts = np.zeros(18)
        action_total = 0
        
        for traj in self.trajectories:
            acts = traj.acts
            if len(acts.shape) == 1:
                acts = acts.reshape(-1, 1)
            action_total += len(acts)
            for frame_acts in acts:
                action_counts += frame_acts
        
        action_names = [
            "UP", "DOWN", "LEFT", "RIGHT",
            "X(CROSS)", "O(CIRCLE)", "[](SQUARE)",
            "L2", "R2", "L3",
            "CAM_R1", "CAM_R2", "CAM_L1", "CAM_L2",
            "CAM_U1", "CAM_U2", "CAM_D1", "CAM_D2"
        ]
        
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            pct = (count / action_total) * 100 if action_total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  [{i:2d}] {name:12s}: {bar} {pct:.1f}% ({int(count)}x)")
        
        print(f"\nTotal: {action_total} frames analisados")


class Trainer:
    """
    Gerencia o treinamento do agente usando Imitation Learning.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 batch_size: int = 384,
                 epochs: int = 100):
        
        self.config = config
        self.device = th.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Ambiente para as politicas do SB3
        self.env = self._create_env()
        
        # Trainer de Behavioral Cloning
        self.bc_trainer = None
        self.policy = None
    
    def _create_env(self):
        """Cria o ambiente base do Stable Baselines3."""
        env = GenericGameEnv(self.config)
        env = WarpFrame(env, width=128, height=128)
        env = DummyVecEnv([lambda: env])
        return env
    
    def setup_bc_trainer(self, trajectories: List[Trajectory]):
        """
        Configura o treinador de Behavioral Cloning.
        
        Args:
            trajectories: Lista de trajetorias para treinar
        """
        print_header("2. CONFIGURANDO O TREINADOR")
        
        # Gerador de numeros aleatorios
        rng = np.random.default_rng(seed=42)
        
        # Criar o trainer de BC
        self.bc_trainer = BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            rng=rng,
            batch_size=self.batch_size,
            expert_data=trajectories,  # Trajetorias como 'expert data'
           环境整治=self.learning_rate,
        )
        
        print(f"[OK] BC Trainer configurado:")
        print(f"     Batch size: {self.batch_size}")
        print(f"     Learning rate: {self.learning_rate}")
        print(f"     Device: {self.device}")
    
    def train(self, save_path: str = './models/'):
        """
        Executa o treinamento completo.
        
        Returns:
            O modelo treinado (ActorCriticPolicy)
        """
        print_header("3. TREINAMENTO - Behavioral Cloning")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Callback para monitorar o treinamento
        def progress_callback(epoch: int, batch_num: int, batch_size: int):
            if batch_num % 10 == 0:
                print(f"  Epoch {epoch}, batch {batch_num}: training...", end="\r")
        
        self.bc_trainer.train(
            n_epochs=self.epochs,
            progress_bar=True,  # Mostra barra de progresso
            # log_interval=5,
        )
        
        print(f"\n[OK] Treinamento concluido!")
        
        # Salvar o modelo
        model_path = os.path.join(save_path, "bc_policy.zip")
        self.bc_trainer.policy.save(model_path)
        print(f"[OK] Modelo salvo em: {model_path}")
        
        self.policy = self.bc_trainer.policy
        return self.policy
    
    def save_checkpoint(self, name: str = "checkpoint", save_dir: str = './models/checkpoints/'):
        """Salva um checkpoint (copia do modelo)."""
        os.makedirs(save_dir, exist_ok=True)
        if self.policy:
            path = os.path.join(save_dir, f"{name}.zip")
            self.policy.save(path)
            return path
        return None


def run_dagger_iteration(config: Dict[str, Any], 
                         current_model_path: str, 
                         demo_path: str = './demos/dagger/'):
    """
    Executa uma iteracao de DAGGER (Dataset Aggregation):
    
    1. A policy treinada joga e gera predicoes
    2. O humano observa e intervendo quando necessario (voz efetiva)
    3. As trajetorias produzidas pelas predicao sao 'rotuladas' com as acoes corretas
    4. Dados novos sao adicionados ao dataset
    5. Retreina com os dados combinados
    
    Esta funcao e um placeholder - usa a implementacao do imitation-hg-dagger.ipynb
    para a logica real.
    """
    print("\n[DAGGER] Iteracao de refinamento...")
    print("  1. Policy atual joga e gera predicoes")
    print("  2. Humanos labels as predicoes (manualmente ou semi-auto)")
    print("  3. Novos dados adicionados ao dataset")
    print("  4. Retranging com dados combinados")
    
    # TODO: Implementar logica DAGGER completa
    pass


def main():
    # ===== PARSING DE ARGUMENTOS =====
    parser = argparse.ArgumentParser(
        description="Treina um agente de Imitation Learning para jogar qualquer jogo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python train_agent.py --epochs 50 --lr 1e-4
  python train_agent.py --batch 256 --device cpu
  python train_agent.py --dagger  # Inclui iteracao DAgger
        """)
    
    parser.add_argument('--epochs', type=int, default=100, help='Numero de epocas')
    parser.add_argument('--batch', type=int, default=384, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dagger', action='store_true', help='Executar DAgger apos treino')
    parser.add_argument('--device', type=str, default=None, help="'cuda' ou 'cpu'")
    
    args = parser.parse_args()
    
    print_header("IMITATION LEARNING - TREINAMENTO")
    
    # ===== VERIFICACOES INICIAIS =====
    device = args.device or check_cuda()
    print(f"\n[CONFIG] Epocas: {args.epochs}, Batch: {args.batch}, LR: {args.lr}, Device: {device}")
    
    # ===== CONFIGURACAO DO JOGO =====
    config = {
        "process_name": "re9",  # ALTERE PARA O SEU JOGO!
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
    
    try:
        # ===== PASSO 1: CARREGAR DADOS =====
        data_manager = DataManager(demo_path='./demos/')
        trajectories = data_manager.load_demos(verbose=True)
        data_manager.describe_actions()
        
        # ===== PASSO 2: CONFIGURAR TREINADOR =====
        trainer = Trainer(
            config=config,
            device=device,
            learning_rate=args.lr,
            batch_size=args.batch,
            epochs=args.epochs
        )
        
        # ===== PASSO 3: TREINAR =====
        trainer.setup_bc_trainer(trajectories)
        trained_policy = trainer.train(save_path='./models/')
        
        # ===== PASSO 4: (OPCIONAL) DAGGER =====
        if args.dagger:
            run_dagger_iteration(config, './models/bc_policy.zip')
        
        print_header("TREINAMENTO CONCLUIDO")
        print("Proximos passos:")
        print("  1. Teste o modelo: python run_ai.py")
        print("  2. Se precisar melhorar, use DAGGER: imitation-hg-dagger.ipynb")
        print("  3. Quando satisfeito, use o modelo final para jogar automaticamente!")
        
    except Exception as e:
        print(f"\n[ERRO] Falha no treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
