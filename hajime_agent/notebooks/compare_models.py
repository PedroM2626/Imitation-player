"""
Comparacao de Modelos: NatureCNN vs Vision Transformer (ViT)
=============================================================

Este script treina ambos os modelos no mesmo conjunto de dados de demonstracao
e compara metricas de desempenho (loss, tempo de treino, tamanho, parametros).

Todos os resultados sao registrados no MLflow sob o experimento "Hajime_Model_Comparison".

COMO USAR:
    python compare_models.py --epochs 30 --batch 384 --lr 1e-4 --device cuda

ARGUMENTOS:
    --epochs    : Numero de epocas (default: 30)
    --batch     : Tamanho do batch (default: 384)
    --lr        : Learning rate (default: 1e-4)
    --device    : 'cuda' ou 'cpu' (default: cuda)
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Ajustar path para encontrar os modulos
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from game_env import GenericGameEnv, TemporalAttentionLSTM
from config.game_config import GAME_CONFIG
from vision_transformer import VisionTransformerExtractor

import mlflow


# ============================================================
# Captura de metricas
# ============================================================

class MetricCapture(KVWriter):
    """Captura o ultimo valor de cada metrica logada pelo BC trainer."""

    def __init__(self):
        self.metrics: Dict[str, float] = {}

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        self.metrics.update(key_values)

    def close(self) -> None:
        pass


class MLflowOutputFormat(KVWriter):
    """Loga metricas diretamente no MLflow durante o treino."""

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


# ============================================================
# Funcoes utilitarias
# ============================================================

def print_header(title: str) -> None:
    """Imprime um cabecalho formatado na tela."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_demos(demo_path: str = "./demos/") -> List[Trajectory]:
    """
    Carrega todas as trajetorias dos arquivos demo*.pt no diretorio informado.

    Aplica a correcao de shape de acoes (mesmo comportamento do DataManager
    em train_agent.py).

    Returns:
        Lista de objetos Trajectory prontos para treino.
    """
    demo_dir = Path(demo_path)

    if not demo_dir.exists():
        raise FileNotFoundError(
            f"Diretorio {demo_dir} nao encontrado!\n"
            "Grave trajetorias primeiro com record_trajectories.py"
        )

    demo_files = sorted(demo_dir.glob("demo*.pt"))

    if not demo_files:
        raise FileNotFoundError(
            f"Nenhum arquivo de demo encontrado em {demo_dir}\n"
            "Execute record_trajectories.py primeiro!"
        )

    expected_num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
    all_trajectories: List[Trajectory] = []

    print(f"Encontrados {len(demo_files)} arquivos de demo:")

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

                # Correcao de shape de acoes
                acts = t.acts
                if acts.shape[-1] > expected_num_actions:
                    acts = acts[..., :expected_num_actions]
                elif acts.shape[-1] < expected_num_actions:
                    pad_width = (
                        [(0, 0)] * (acts.ndim - 1)
                        + [(0, expected_num_actions - acts.shape[-1])]
                    )
                    acts = np.pad(acts, pad_width, mode="constant", constant_values=0)

                all_trajectories.append(
                    Trajectory(obs=obs, acts=acts, infos=t.infos, terminal=t.terminal)
                )
                count += 1

            print(f"({count} traj.)")
        except Exception as e:
            print(f" ERRO: {e}")
            continue

    total_frames = sum(len(t.obs) for t in all_trajectories)
    print(f"\n[OK] Dados carregados:")
    print(f"     - {len(all_trajectories)} trajetorias")
    print(f"     - {total_frames} frames totais")

    return all_trajectories


def create_env():
    """
    Cria o ambiente dummy (sem abrir janela do jogo).

    Pipeline: GenericGameEnv(dummy) -> DummyVecEnv -> VecTransposeImage -> VecFrameStack(4)
    """
    train_config = GAME_CONFIG.copy()
    train_config["dummy"] = True
    env = GenericGameEnv(train_config)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def count_parameters(policy) -> int:
    """Retorna o numero total de parametros treinaveis da policy."""
    return sum(p.numel() for p in policy.parameters())


def get_model_size_mb(path: str) -> float:
    """Retorna o tamanho do arquivo de modelo em MB."""
    return os.path.getsize(path) / (1024 * 1024)


def train_model(
    name: str,
    env,
    trajectories: List[Trajectory],
    policy_kwargs: dict,
    epochs: int,
    batch: int,
    lr: float,
    device: str,
) -> Dict[str, Any]:
    """
    Treina um modelo BC e retorna as metricas coletadas.

    Args:
        name: Identificador do modelo (ex: "NatureCNN", "ViT_Transformer").
        env: Ambiente vetorizado (VecFrameStack).
        trajectories: Lista de trajetorias para treino.
        policy_kwargs: Kwargs para a policy. Se None, usa NatureCNN padrao.
        epochs: Numero de epocas.
        batch: Tamanho do batch.
        lr: Learning rate.
        device: Dispositivo de treino ('cuda' ou 'cpu').

    Returns:
        Dicionario com metricas: final_loss, training_time, model_size_mb, num_params.
    """
    print_header(f"TREINANDO: {name}")

    rng = np.random.default_rng(seed=42)

    # Configurar captura de metricas
    metric_capture = MetricCapture()
    output_formats = [
        HumanOutputFormat(sys.stdout),
        metric_capture,
        MLflowOutputFormat(prefix=f"{name}/"),
    ]
    custom_logger = Logger(folder=None, output_formats=output_formats)

    # Criar policy com ou sem kwargs customizados
    policy = None
    if policy_kwargs is not None:
        policy = ActorCriticCnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: lr,
            **policy_kwargs,
        )
        policy = policy.to(device)

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=rng,
        device=device,
        policy=policy,
        batch_size=batch,
        optimizer_kwargs={"lr": lr},
        custom_logger=custom_logger,
    )

    # Treinar e medir tempo
    start_time = time.time()
    bc_trainer.train(n_epochs=epochs, progress_bar=True)
    training_time = time.time() - start_time

    # Salvar modelo para medir tamanho
    os.makedirs("./models/", exist_ok=True)
    model_path = f"./models/{name}_policy.zip"
    bc_trainer.policy.save(model_path)

    # Coletar metricas
    final_loss = metric_capture.metrics.get("bc/loss", float("nan"))
    model_size = get_model_size_mb(model_path)
    num_params = count_parameters(bc_trainer.policy)

    results = {
        "name": name,
        "final_loss": final_loss,
        "training_time": training_time,
        "model_size_mb": model_size,
        "num_params": num_params,
    }

    print(f"\n[OK] {name} concluido:")
    print(f"     Loss final:  {final_loss:.6f}")
    print(f"     Tempo:       {training_time:.1f}s")
    print(f"     Tamanho:     {model_size:.2f} MB")
    print(f"     Parametros:  {num_params:,}")

    return results


# ============================================================
# Funcoes de comparacao e relatorio
# ============================================================

def print_comparison_table(cnn_results: Dict[str, Any], lstm_results: Dict[str, Any], vit_results: Dict[str, Any]) -> str:
    """
    Imprime e retorna uma tabela comparativa entre os tres modelos.

    Returns:
        String formatada com a tabela.
    """
    header = (
        "=" * 85 + "\n"
        "  COMPARACAO DE MODELOS\n"
        + "=" * 85
    )

    table_lines = [
        header,
        "",
        f"{'Metrica':<22}| {'NatureCNN':<18}| {'CNN+LSTM':<18}| {'ViT Transformer':<18}",
        "-" * 22 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 18,
        (
            f"{'Loss Final':<22}| "
            f"{cnn_results['final_loss']:<17.6f}| "
            f"{lstm_results['final_loss']:<17.6f}| "
            f"{vit_results['final_loss']:<17.6f}"
        ),
        (
            f"{'Tempo de Treino':<22}| "
            f"{cnn_results['training_time']:<14.1f} s  | "
            f"{lstm_results['training_time']:<14.1f} s  | "
            f"{vit_results['training_time']:<14.1f} s"
        ),
        (
            f"{'Tamanho do Modelo':<22}| "
            f"{cnn_results['model_size_mb']:<14.2f} MB | "
            f"{lstm_results['model_size_mb']:<14.2f} MB | "
            f"{vit_results['model_size_mb']:<14.2f} MB"
        ),
        (
            f"{'Num. Parametros':<22}| "
            f"{cnn_results['num_params']:<17,}| "
            f"{lstm_results['num_params']:<17,}| "
            f"{vit_results['num_params']:<17,}"
        ),
        "",
    ]

    table_str = "\n".join(table_lines)
    print("\n" + table_str)
    return table_str


def save_results_markdown(
    cnn_results: Dict[str, Any],
    lstm_results: Dict[str, Any],
    vit_results: Dict[str, Any],
    output_path: str = "./models/comparison_results.md",
) -> None:
    """Salva os resultados da comparacao em formato Markdown."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determinar melhor modelo por loss
    losses = {"NatureCNN": cnn_results["final_loss"], "CNN+LSTM": lstm_results["final_loss"], "ViT Transformer": vit_results["final_loss"]}
    best_loss = min(losses, key=losses.get)

    times = {"NatureCNN": cnn_results["training_time"], "CNN+LSTM": lstm_results["training_time"], "ViT Transformer": vit_results["training_time"]}
    fastest = min(times, key=times.get)

    md = f"""# Comparacao de Modelos: NatureCNN vs CNN+LSTM vs ViT Transformer

## Resultados

| Metrica             | NatureCNN          | CNN+LSTM           | ViT Transformer    |
|---------------------|--------------------|--------------------|--------------------|
| Loss Final          | {cnn_results['final_loss']:.6f}          | {lstm_results['final_loss']:.6f}          | {vit_results['final_loss']:.6f}          |
| Tempo de Treino     | {cnn_results['training_time']:.1f}s               | {lstm_results['training_time']:.1f}s               | {vit_results['training_time']:.1f}s               |
| Tamanho do Modelo   | {cnn_results['model_size_mb']:.2f} MB          | {lstm_results['model_size_mb']:.2f} MB          | {vit_results['model_size_mb']:.2f} MB          |
| Num. Parametros     | {cnn_results['num_params']:,}        | {lstm_results['num_params']:,}        | {vit_results['num_params']:,}        |

## Analise

- **Menor Loss**: {best_loss}
- **Mais Rapido**: {fastest}
- **NatureCNN**: Modelo convolucional padrao, mais leve e rapido de treinar.
- **CNN+LSTM**: Adiciona memoria temporal com LSTM bidirecional e atencao temporal.
- **ViT Transformer**: Modelo baseado em Self-Attention, captura relacoes espaciais e temporais de forma mais sofisticada, mas exige mais recursos.

## Configuracao do Experimento

- Epocas: {cnn_results.get('epochs', 'N/A')}
- Batch Size: {cnn_results.get('batch', 'N/A')}
- Learning Rate: {cnn_results.get('lr', 'N/A')}
- Device: {cnn_results.get('device', 'N/A')}
- Num. Acoes: {GAME_CONFIG.get('actions', {}).get('num_actions', 'N/A')}

## Artefatos

- Modelo NatureCNN: `models/NatureCNN_policy.zip`
- Modelo CNN+LSTM: `models/CNN_LSTM_policy.zip`
- Modelo ViT: `models/ViT_Transformer_policy.zip`
- MLflow Experiment: `Model_Comparison`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[OK] Resultados salvos em: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compara NatureCNN vs Vision Transformer para Imitation Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=30, help="Numero de epocas (default: 30)")
    parser.add_argument("--batch", type=int, default=384, help="Tamanho do batch (default: 384)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu, default: cuda)")
    args = parser.parse_args()

    print_header("COMPARACAO DE MODELOS: NatureCNN vs ViT Transformer")

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
    mlflow.set_experiment("Hajime_Model_Comparison")

    # --- Carregar dados uma unica vez ---
    trajectories = load_demos(demo_path="./demos/")

    # --- Criar ambiente uma unica vez ---
    env = create_env()

    # --- Treinar CNN (NatureCNN padrao) ---
    with mlflow.start_run(run_name="NatureCNN"):
        mlflow.log_param("model_type", "NatureCNN")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("device", args.device)

        cnn_results = train_model(
            name="NatureCNN",
            env=env,
            trajectories=trajectories,
            policy_kwargs=None,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device=args.device,
        )
        cnn_results.update({"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "device": args.device})

        mlflow.log_metric("final_loss", cnn_results["final_loss"])
        mlflow.log_metric("training_time_s", cnn_results["training_time"])
        mlflow.log_metric("model_size_mb", cnn_results["model_size_mb"])
        mlflow.log_metric("num_params", cnn_results["num_params"])
        mlflow.log_artifact("./models/NatureCNN_policy.zip", artifact_path="model")

    # --- Treinar CNN+LSTM ---
    with mlflow.start_run(run_name="CNN_LSTM"):
        lstm_policy_kwargs = {
            "features_extractor_class": TemporalAttentionLSTM,
            "features_extractor_kwargs": {
                "features_dim": 512,
                "lstm_hidden_size": 256,
                "lstm_num_layers": 2,
                "debug": False,
            },
        }

        mlflow.log_param("model_type", "CNN_LSTM")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("device", args.device)
        mlflow.log_param("features_dim", 512)
        mlflow.log_param("lstm_hidden_size", 256)
        mlflow.log_param("lstm_num_layers", 2)

        lstm_results = train_model(
            name="CNN_LSTM",
            env=env,
            trajectories=trajectories,
            policy_kwargs=lstm_policy_kwargs,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device=args.device,
        )
        lstm_results.update({"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "device": args.device})

        mlflow.log_metric("final_loss", lstm_results["final_loss"])
        mlflow.log_metric("training_time_s", lstm_results["training_time"])
        mlflow.log_metric("model_size_mb", lstm_results["model_size_mb"])
        mlflow.log_metric("num_params", lstm_results["num_params"])
        mlflow.log_artifact("./models/CNN_LSTM_policy.zip", artifact_path="model")

    # --- Treinar ViT Transformer ---
    with mlflow.start_run(run_name="ViT_Transformer"):
        vit_policy_kwargs = {
            "features_extractor_class": VisionTransformerExtractor,
            "features_extractor_kwargs": {
                "features_dim": 512,
                "embed_dim": 256,
                "patch_size": 16,
                "num_heads": 4,
                "num_layers": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.1,
            },
        }

        mlflow.log_param("model_type", "ViT_Transformer")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("device", args.device)
        mlflow.log_param("features_dim", 512)
        mlflow.log_param("embed_dim", 256)
        mlflow.log_param("patch_size", 16)
        mlflow.log_param("num_heads", 4)
        mlflow.log_param("num_layers", 4)

        vit_results = train_model(
            name="ViT_Transformer",
            env=env,
            trajectories=trajectories,
            policy_kwargs=vit_policy_kwargs,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device=args.device,
        )
        vit_results.update({"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "device": args.device})

        mlflow.log_metric("final_loss", vit_results["final_loss"])
        mlflow.log_metric("training_time_s", vit_results["training_time"])
        mlflow.log_metric("model_size_mb", vit_results["model_size_mb"])
        mlflow.log_metric("num_params", vit_results["num_params"])
        mlflow.log_artifact("./models/ViT_Transformer_policy.zip", artifact_path="model")

    # --- Comparacao ---
    table_str = print_comparison_table(cnn_results, lstm_results, vit_results)

    # Salvar resultados em Markdown
    save_results_markdown(cnn_results, lstm_results, vit_results, output_path="./models/comparison_results.md")

    # Logar arquivo de comparacao como artefato no MLflow (run pai)
    with mlflow.start_run(run_name="Comparison_Summary"):
        mlflow.log_param("models_compared", "NatureCNN vs CNN_LSTM vs ViT_Transformer")
        mlflow.log_metric("cnn_loss", cnn_results["final_loss"])
        mlflow.log_metric("lstm_loss", lstm_results["final_loss"])
        mlflow.log_metric("vit_loss", vit_results["final_loss"])
        mlflow.log_metric("cnn_time_s", cnn_results["training_time"])
        mlflow.log_metric("lstm_time_s", lstm_results["training_time"])
        mlflow.log_metric("vit_time_s", vit_results["training_time"])
        mlflow.log_metric("cnn_params", cnn_results["num_params"])
        mlflow.log_metric("lstm_params", lstm_results["num_params"])
        mlflow.log_metric("vit_params", vit_results["num_params"])
        mlflow.log_artifact("./models/comparison_results.md", artifact_path="report")

    print_header("COMPARACAO CONCLUIDA")
    print("Artefatos gerados:")
    print("  - models/NatureCNN_policy.zip")
    print("  - models/CNN_LSTM_policy.zip")
    print("  - models/ViT_Transformer_policy.zip")
    print("  - models/comparison_results.md")
    print("  - MLflow experiment: Model_Comparison")


if __name__ == "__main__":
    main()

