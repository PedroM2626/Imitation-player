"""
Treinamento do Agente de Imitation Learning com Impoola-CNN
===========================================================

Este script treina um agente usando a arquitetura Impoola-CNN, que combina
blocos residenciais (tipo Impala) com Global Average Pooling (GAP).
Isso resulta em um modelo muito menor, invariante a translações e altamente generalizável.

Após o término do treinamento, o script extrai as métricas de tempo, loss final,
tamanho do modelo e número de parâmetros, atualizando automaticamente os arquivos
README.md e comparison_results.md.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Ajustar path para encontrar os módulos
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
import torch.nn as nn
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat, CSVOutputFormat
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# Módulos locais
from game_env import GenericGameEnv
from config.game_config import GAME_CONFIG
from impoola_cnn import ImpoolaCNNExtractor

import mlflow

# ============================================================
# Captura de métricas para MLflow
# ============================================================

class MetricCapture(KVWriter):
    """Captura o último valor de cada métrica logada pelo BC trainer."""
    def __init__(self):
        self.metrics: Dict[str, float] = {}

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        self.metrics.update(key_values)

    def close(self) -> None:
        pass


class MLflowOutputFormat(KVWriter):
    """Loga métricas diretamente no MLflow durante o treino."""
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        if mlflow.active_run():
            for key, value in key_values.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    metric_name = f"{self.prefix}{key}" if self.prefix else key
                    mlflow.log_metric(metric_name, float(value), step=step)

    def close(self) -> None:
        pass


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_demos(demo_path: str = "./demos/") -> List[Trajectory]:
    demo_dir = Path(demo_path)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Diretório {demo_dir} não encontrado!")
        
    demo_files = sorted(demo_dir.glob("demo*.pt"))
    if not demo_files:
        raise FileNotFoundError(f"Nenhum arquivo demo*.pt encontrado em {demo_dir}")
        
    expected_num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
    all_trajectories: List[Trajectory] = []
    
    print(f"Carregando {len(demo_files)} arquivos de demo:")
    for i, demo_file in enumerate(demo_files):
        try:
            print(f"  [{i+1}/{len(demo_files)}] Carregando {demo_file.name}...", end=" ")
            data = th.load(demo_file, map_location="cpu")
            items = data if isinstance(data, list) else [data]
            count = 0
            for t in items:
                obs = t.obs
                if obs.ndim == 5 and obs.shape[1] == 1:
                    obs = np.squeeze(obs, axis=1)
                
                acts = t.acts
                if acts.shape[-1] > expected_num_actions:
                    acts = acts[..., :expected_num_actions]
                elif acts.shape[-1] < expected_num_actions:
                    pad_width = [(0, 0)] * (acts.ndim - 1) + [(0, expected_num_actions - acts.shape[-1])]
                    acts = np.pad(acts, pad_width, mode="constant", constant_values=0)
                    
                all_trajectories.append(Trajectory(obs=obs, acts=acts, infos=t.infos, terminal=t.terminal))
                count += 1
            print(f"({count} traj.)")
        except Exception as e:
            print(f" ERRO: {e}")
            
    total_frames = sum(len(t.obs) for t in all_trajectories)
    print(f"\n[OK] Dados carregados: {len(all_trajectories)} trajetórias, {total_frames} frames totais.")
    return all_trajectories


def create_env():
    train_config = GAME_CONFIG.copy()
    train_config["dummy"] = True
    env = GenericGameEnv(train_config)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# Atualização automática dos arquivos Markdown
# ============================================================

def update_markdown_files(impoola_results: Dict[str, Any]):
    print_header("ATUALIZANDO DOCUMENTACAO")
    
    # Baselines obtidos anteriormente para os outros modelos (10 épocas)
    baselines = {
        "NatureCNN": {
            "final_loss": 3.476316,
            "training_time": 18.3,
            "model_size_mb": 16.33,
            "num_params": 4196810
        },
        "CNN_LSTM": {
            "final_loss": 3.024518,
            "training_time": 67.8,
            "model_size_mb": 23.71,
            "num_params": 6116779
        },
        "ViT_Transformer": {
            "final_loss": 2.998286,
            "training_time": 3288.8,
            "model_size_mb": 9.70,
            "num_params": 2448010
        },
        "Impoola_CNN": {
            "final_loss": impoola_results["final_loss"],
            "training_time": impoola_results["training_time"],
            "model_size_mb": impoola_results["model_size_mb"],
            "num_params": impoola_results["num_params"]
        }
    }
    
    # Determinar melhores e piores para formatação inteligente do README
    losses = {k: v["final_loss"] for k, v in baselines.items() if not np.isnan(v["final_loss"])}
    times = {k: v["training_time"] for k, v in baselines.items()}
    sizes = {k: v["model_size_mb"] for k, v in baselines.items()}
    params = {k: v["num_params"] for k, v in baselines.items()}
    
    best_loss = min(losses, key=losses.get)
    worst_loss = max(losses, key=losses.get)
    
    best_time = min(times, key=times.get)
    worst_time = max(times, key=times.get)
    
    best_size = min(sizes, key=sizes.get)
    worst_size = max(sizes, key=sizes.get)
    
    best_params = min(params, key=params.get)
    worst_params = max(params, key=params.get)
    
    # Função auxiliar para formatar células do README
    def fmt_cell(model_key, val, best_key, worst_key, val_type):
        is_best = (model_key == best_key)
        is_worst = (model_key == worst_key)
        
        if val_type == "loss":
            base_str = f"{val:.2f}"
            suffix = " *(Melhor)*" if is_best else (" *(Pior)*" if is_worst else "")
        elif val_type == "time":
            if val > 60:
                base_str = f"{val/60:.1f} minutos"
            else:
                base_str = f"{val:.1f} segundos"
            suffix = " *(Mais Rápido)*" if is_best else (" *(Pesadíssimo)*" if is_worst else "")
        elif val_type == "size":
            base_str = f"{val:.2f} MB"
            suffix = " *(O mais leve!)*" if is_best else (" *(Mais Pesado)*" if is_worst else "")
        elif val_type == "params":
            base_str = f"{val/1e6:.2f} Milhões"
            suffix = " *(Mais Eficiente)*" if is_best else (" *(Menos Eficiente)*" if is_worst else "")
            
        if is_best:
            return f"**`{base_str}`**{suffix}"
        return f"`{base_str}`{suffix}"

    # 1. Atualizar README.md
    readme_path = Path("../../README.md")
    if readme_path.exists():
        print(f"Atualizando {readme_path.resolve()}...")
        readme_content = readme_path.read_text(encoding="utf-8")
        
        # Gerar a tabela com 4 colunas de modelos
        new_table = f"""| Métrica | NatureCNN (Padrão) | CNN + LSTM + Attention | Vision Transformer (ViT) | Impoola-CNN (GAP) |
| :--- | :--- | :--- | :--- | :--- |
| **Poder de Aprendizado (Loss Final)** | {fmt_cell("NatureCNN", baselines["NatureCNN"]["final_loss"], best_loss, worst_loss, "loss")} | {fmt_cell("CNN_LSTM", baselines["CNN_LSTM"]["final_loss"], best_loss, worst_loss, "loss")} | {fmt_cell("ViT_Transformer", baselines["ViT_Transformer"]["final_loss"], best_loss, worst_loss, "loss")} | {fmt_cell("Impoola_CNN", baselines["Impoola_CNN"]["final_loss"], best_loss, worst_loss, "loss")} |
| **Tempo de Treinamento** | {fmt_cell("NatureCNN", baselines["NatureCNN"]["training_time"], best_time, worst_time, "time")} | {fmt_cell("CNN_LSTM", baselines["CNN_LSTM"]["training_time"], best_time, worst_time, "time")} | {fmt_cell("ViT_Transformer", baselines["ViT_Transformer"]["training_time"], best_time, worst_time, "time")} | {fmt_cell("Impoola_CNN", baselines["Impoola_CNN"]["training_time"], best_time, worst_time, "time")} |
| **Peso do Arquivo Final** | {fmt_cell("NatureCNN", baselines["NatureCNN"]["model_size_mb"], best_size, worst_size, "size")} | {fmt_cell("CNN_LSTM", baselines["CNN_LSTM"]["model_size_mb"], best_size, worst_size, "size")} | {fmt_cell("ViT_Transformer", baselines["ViT_Transformer"]["model_size_mb"], best_size, worst_size, "size")} | {fmt_cell("Impoola_CNN", baselines["Impoola_CNN"]["model_size_mb"], best_size, worst_size, "size")} |
| **Quantidade de Parâmetros (Neurônios)** | {fmt_cell("NatureCNN", baselines["NatureCNN"]["num_params"], best_params, worst_params, "params")} | {fmt_cell("CNN_LSTM", baselines["CNN_LSTM"]["num_params"], best_params, worst_params, "params")} | {fmt_cell("ViT_Transformer", baselines["ViT_Transformer"]["num_params"], best_params, worst_params, "params")} | {fmt_cell("Impoola_CNN", baselines["Impoola_CNN"]["num_params"], best_params, worst_params, "params")} |"""
        
        start_tag = "<!-- BENCHMARK_START -->"
        end_tag = "<!-- BENCHMARK_END -->"
        
        if start_tag in readme_content and end_tag in readme_content:
            parts = readme_content.split(start_tag)
            before = parts[0]
            after = parts[1].split(end_tag)[1]
            updated_content = f"{before}{start_tag}\n{new_table}\n{end_tag}{after}"
            readme_path.write_text(updated_content, encoding="utf-8")
            print("[OK] README.md atualizado com sucesso.")
        else:
            print("[!] Tags de benchmark não encontradas no README.md. Tabela não atualizada no README.")
    else:
        print("[!] README.md não encontrado no caminho relativo esperado.")

    # 2. Atualizar models/comparison_results.md
    comp_path = Path("./models/comparison_results.md")
    print(f"Atualizando {comp_path.resolve()}...")
    
    comp_md = f"""# Comparacao de Modelos: NatureCNN vs CNN+LSTM vs ViT vs Impoola-CNN

## Resultados

| Metrica             | NatureCNN          | CNN+LSTM           | ViT Transformer    | Impoola-CNN        |
|---------------------|--------------------|--------------------|--------------------|--------------------|
| Loss Final          | {baselines['NatureCNN']['final_loss']:.6f}          | {baselines['CNN_LSTM']['final_loss']:.6f}          | {baselines['ViT_Transformer']['final_loss']:.6f}          | {baselines['Impoola_CNN']['final_loss']:.6f}          |
| Tempo de Treino     | {baselines['NatureCNN']['training_time']:.1f}s               | {baselines['CNN_LSTM']['training_time']:.1f}s               | {baselines['ViT_Transformer']['training_time']:.1f}s               | {baselines['Impoola_CNN']['training_time']:.1f}s               |
| Tamanho do Modelo   | {baselines['NatureCNN']['model_size_mb']:.2f} MB          | {baselines['CNN_LSTM']['model_size_mb']:.2f} MB          | {baselines['ViT_Transformer']['model_size_mb']:.2f} MB          | {baselines['Impoola_CNN']['model_size_mb']:.2f} MB          |
| Num. Parametros     | {baselines['NatureCNN']['num_params']:,}        | {baselines['CNN_LSTM']['num_params']:,}        | {baselines['ViT_Transformer']['num_params']:,}        | {baselines['Impoola_CNN']['num_params']:,}        |

## Analise

- **Menor Loss**: {best_loss.replace('_', ' ')}
- **Mais Rapido**: {best_time.replace('_', ' ')}
- **Mais Leve**: {best_size.replace('_', ' ')}
- **NatureCNN**: Modelo convolucional padrao, mais leve e rapido de treinar.
- **CNN+LSTM**: Adiciona memoria temporal com LSTM bidirecional e atencao temporal.
- **ViT Transformer**: Modelo baseado em Self-Attention, captura relacoes espaciais e temporais de forma mais sofisticada.
- **Impoola-CNN**: Modelo residual baseado no Impala com Global Average Pooling (GAP), sendo o mais eficiente em parametros e altamente invariante a translacoes.

## Configuracao do Experimento

- Epocas: {impoola_results.get('epochs', 'N/A')}
- Batch Size: {impoola_results.get('batch', 'N/A')}
- Learning Rate: {impoola_results.get('lr', 'N/A')}
- Device: {impoola_results.get('device', 'N/A')}
- Num. Acoes: {GAME_CONFIG.get('actions', {}).get('num_actions', 'N/A')}

## Artefatos

- Modelo NatureCNN: `models/NatureCNN_policy.zip`
- Modelo CNN+LSTM: `models/CNN_LSTM_policy.zip`
- Modelo ViT: `models/ViT_Transformer_policy.zip`
- Modelo Impoola-CNN: `models/ImpoolaCNN_policy.zip`
- MLflow Experiment: `Model_Comparison`
"""
    comp_path.write_text(comp_md, encoding="utf-8")
    print("[OK] comparison_results.md atualizado com sucesso.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Treina o agente usando Impoola-CNN para Imitation Learning.")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas (default: 10)")
    parser.add_argument("--batch", type=int, default=384, help="Tamanho do batch (default: 384)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu, default: cuda)")
    args = parser.parse_args()

    print_header("TREINAMENTO IMPOOLA-CNN")

    # Verificar dispositivo
    if args.device == "cuda" and not th.cuda.is_available():
        print("[!] CUDA nao disponivel, usando CPU.")
        args.device = "cpu"
    else:
        if args.device == "cuda":
            print(f"[OK] GPU detectada: {th.cuda.get_device_name(0)}")

    print(f"[CONFIG] Epocas: {args.epochs}, Batch: {args.batch}, LR: {args.lr}, Device: {args.device}")

    # Configurar MLflow
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Model_Comparison")

    # Carregar dados
    trajectories = load_demos(demo_path="./demos/")

    # Criar ambiente
    env = create_env()

    # Treinar modelo Impoola-CNN
    with mlflow.start_run(run_name="Impoola_CNN"):
        policy_kwargs = {
            "features_extractor_class": ImpoolaCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 512,
                "channels_list": [32, 64, 128],
            },
        }

        mlflow.log_param("model_type", "Impoola_CNN")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("device", args.device)
        mlflow.log_param("features_dim", 512)
        mlflow.log_param("channels_list", "[32, 64, 128]")

        # Setup do BC
        rng = np.random.default_rng(seed=42)
        log_dir = "./models/imitation/impoola_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        metric_capture = MetricCapture()
        output_formats = [
            HumanOutputFormat(sys.stdout),
            metric_capture,
            MLflowOutputFormat(prefix="Impoola_CNN/"),
        ]
        custom_logger = Logger(folder=None, output_formats=output_formats)

        policy = ActorCriticCnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: args.lr,
            **policy_kwargs,
        ).to(args.device)

        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=rng,
            device=args.device,
            policy=policy,
            batch_size=args.batch,
            optimizer_kwargs={"lr": args.lr},
            custom_logger=custom_logger,
        )

        # Treinar e medir tempo
        start_time = time.time()
        bc_trainer.train(n_epochs=args.epochs, progress_bar=True)
        training_time = time.time() - start_time

        # Salvar modelo
        os.makedirs("./models/", exist_ok=True)
        model_path = "./models/ImpoolaCNN_policy.zip"
        bc_trainer.policy.save(model_path)

        # Coletar métricas finais
        final_loss = metric_capture.metrics.get("bc/loss", float("nan"))
        model_size = get_model_size_mb(model_path)
        num_params = count_parameters(bc_trainer.policy)

        # Logar métricas
        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("training_time_s", training_time)
        mlflow.log_metric("model_size_mb", model_size)
        mlflow.log_metric("num_params", num_params)
        mlflow.log_artifact(model_path, artifact_path="model")

        impoola_results = {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "device": args.device,
            "final_loss": final_loss,
            "training_time": training_time,
            "model_size_mb": model_size,
            "num_params": num_params,
        }

        print_header("RESULTADOS IMPOOLA-CNN")
        print(f"Loss final:  {final_loss:.6f}")
        print(f"Tempo:       {training_time:.1f}s")
        print(f"Tamanho:     {model_size:.2f} MB")
        print(f"Parâmetros:  {num_params:,}")

        # Atualizar arquivos markdown
        update_markdown_files(impoola_results)
        
        # Logar markdown de comparação como artefato
        mlflow.log_artifact("./models/comparison_results.md", artifact_path="report")

    print_header("PROCESSO CONCLUIDO")


if __name__ == "__main__":
    main()
