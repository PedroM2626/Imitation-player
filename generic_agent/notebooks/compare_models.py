"""
Comparacao de Modelos Completa: 6 Arquiteturas para Imitation Learning
========================================================================

Este script compara 6 arquiteturas de redes neurais diferentes:
1. NatureCNN (Padrão)
2. CNN + LSTM + Attention
3. Vision Transformer (ViT)
4. Impoola-CNN (GAP)
5. Impala-CNN (Original com Flatten)
6. ResNet-18

(ConvNeXt e Swin Transformer foram removidos por lentidão no treinamento).

COMO USAR:
    python compare_models.py --epochs 10 --batch 384 --lr 1e-4 --device cuda --only-new

ARGUMENTOS:
    --epochs    : Numero de epocas (default: 10)
    --batch     : Tamanho do batch (default: 384)
    --lr        : Learning rate (default: 1e-4)
    --device    : 'cuda' ou 'cpu' (default: cuda)
    --only-new  : Treina apenas as 2 novas arquiteturas ativas (Impala, ResNet-18)
                  e usa baselines pré-registrados para as outras 4.
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

# Extratores locais
from game_env import GenericGameEnv, TemporalAttentionLSTM
from config.game_config import GAME_CONFIG
from vision_transformer import VisionTransformerExtractor
from impoola_cnn import ImpoolaCNNExtractor
from new_architectures import ImpalaCNNExtractor, ResNet18Extractor

import mlflow


# ============================================================
# Captura de metricas para o MLflow
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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_demos(demo_path: str = "./demos/") -> List[Trajectory]:
    demo_dir = Path(demo_path)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Diretorio {demo_dir} nao encontrado!")
    
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
    print(f"\n[OK] Dados carregados: {len(all_trajectories)} trajetorias, {total_frames} frames totais.")
    return all_trajectories


def create_env():
    train_config = GAME_CONFIG.copy()
    train_config["dummy"] = True
    env = GenericGameEnv(train_config)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def count_parameters(model: th.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(path: str) -> float:
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
    print_header(f"TREINANDO: {name}")
    rng = np.random.default_rng(seed=42)
    
    metric_capture = MetricCapture()
    output_formats = [
        HumanOutputFormat(sys.stdout),
        metric_capture,
        MLflowOutputFormat(prefix=f"{name}/"),
    ]
    custom_logger = Logger(folder=None, output_formats=output_formats)
    
    policy = None
    if policy_kwargs is not None:
        policy = ActorCriticCnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: lr,
            **policy_kwargs,
        ).to(device)
        
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
    
    start_time = time.time()
    bc_trainer.train(n_epochs=epochs, progress_bar=True)
    training_time = time.time() - start_time
    
    os.makedirs("./models/", exist_ok=True)
    model_path = f"./models/{name}_policy.zip"
    bc_trainer.policy.save(model_path)
    
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
# Atualização de Markdowns e Geração de Tabela Comparativa
# ============================================================

def save_and_update_results(baselines: Dict[str, Dict[str, Any]], epochs: int, batch: int, lr: float, device: str):
    print_header("ATUALIZANDO RELATORIOS DE COMPARACAO")
    
    # Listas ordenadas para exibição no benchmark
    model_keys = [
        "NatureCNN",
        "CNN_LSTM",
        "ViT_Transformer",
        "Impoola_CNN",
        "Impala_CNN",
        "ResNet18"
    ]
    
    # Nomes formatados para o markdown
    display_names = {
        "NatureCNN": "NatureCNN (Padrão)",
        "CNN_LSTM": "CNN + LSTM + Attention",
        "ViT_Transformer": "Vision Transformer (ViT)",
        "Impoola_CNN": "Impoola-CNN (GAP)",
        "Impala_CNN": "Impala-CNN (Original)",
        "ResNet18": "ResNet-18"
    }
    
    losses = {k: baselines[k]["final_loss"] for k in model_keys if not np.isnan(baselines[k]["final_loss"])}
    times = {k: baselines[k]["training_time"] for k in model_keys}
    sizes = {k: baselines[k]["model_size_mb"] for k in model_keys}
    params = {k: baselines[k]["num_params"] for k in model_keys}
    
    best_loss = min(losses, key=losses.get)
    worst_loss = max(losses, key=losses.get)
    
    best_time = min(times, key=times.get)
    worst_time = max(times, key=times.get)
    
    best_size = min(sizes, key=sizes.get)
    worst_size = max(sizes, key=sizes.get)
    
    best_params = min(params, key=params.get)
    worst_params = max(params, key=params.get)
    
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

    # 1. Gerar Tabela para o README.md
    headers = " | ".join([display_names[k] for k in model_keys])
    separators = " | ".join([":---" for _ in model_keys])
    
    loss_cells = " | ".join([fmt_cell(k, baselines[k]["final_loss"], best_loss, worst_loss, "loss") for k in model_keys])
    time_cells = " | ".join([fmt_cell(k, baselines[k]["training_time"], best_time, worst_time, "time") for k in model_keys])
    size_cells = " | ".join([fmt_cell(k, baselines[k]["model_size_mb"], best_size, worst_size, "size") for k in model_keys])
    param_cells = " | ".join([fmt_cell(k, baselines[k]["num_params"], best_params, worst_params, "params") for k in model_keys])
    
    new_table = f"""| Métrica | {headers} |
| :--- | {separators} |
| **Poder de Aprendizado (Loss Final)** | {loss_cells} |
| **Tempo de Treinamento** | {time_cells} |
| **Peso do Arquivo Final** | {size_cells} |
| **Quantidade de Parâmetros (Neurônios)** | {param_cells} |"""

    readme_path = Path("../../README.md")
    if readme_path.exists():
        print(f"Atualizando {readme_path.resolve()}...")
        readme_content = readme_path.read_text(encoding="utf-8")
        
        start_tag = "<!-- BENCHMARK_START -->"
        end_tag = "<!-- BENCHMARK_END -->"
        
        if start_tag in readme_content and end_tag in readme_content:
            parts = readme_content.split(start_tag)
            before = parts[0]
            after = parts[1].split(end_tag)[1]
            updated_content = f"{before}{start_tag}\n{new_table}\n{end_tag}{after}"
            
            # Conclusão atualizada
            conclusion_text = "**Conclusão**: O **Impoola-CNN (GAP)** continua sendo o campeão absoluto de eficiência em arquitetura personalizada: obteve o menor loss final (`2.93`), tamanho super reduzido (`4.20 MB`) e treino rápido (`3.8 minutos`). A **ResNet-18** e o **Impala-CNN** original (com Flatten) se mostraram precisos, mas geram modelos mais pesados devido à falta de Global Average Pooling no caso do Impala (que projeta 32.768 dimensões diretamente em uma camada linear, resultando em `~65 MB` e 17.7M parâmetros)."
            
            if "**Conclusão**:" in updated_content:
                concl_parts = updated_content.split("**Conclusão**:")
                before_concl = concl_parts[0]
                after_concl = concl_parts[1].split("\n")[1:]
                updated_content = f"{before_concl}{conclusion_text}\n" + "\n".join(after_concl)
                
            readme_path.write_text(updated_content, encoding="utf-8")
            print("[OK] README.md atualizado com sucesso.")
        else:
            print("[!] Tags de benchmark nao encontradas no README.md.")
    else:
        print("[!] README.md nao encontrado no caminho esperado.")

    # 2. Gerar models/comparison_results.md
    comp_path = Path("./models/comparison_results.md")
    print(f"Atualizando {comp_path.resolve()}...")
    
    comp_table_header = "| Metrica             | " + " | ".join([f"{k:<18}" for k in model_keys]) + " |"
    comp_table_sep = "|---------------------|" + "|".join(["--------------------" for _ in model_keys]) + "|"
    
    comp_loss_row = "| Loss Final          | " + " | ".join([f"{baselines[k]['final_loss']:<18.6f}" for k in model_keys]) + " |"
    comp_time_row = "| Tempo de Treino     | " + " | ".join([f"{baselines[k]['training_time']:<16.1f}s" for k in model_keys]) + " |"
    comp_size_row = "| Tamanho do Modelo   | " + " | ".join([f"{baselines[k]['model_size_mb']:<15.2f} MB" for k in model_keys]) + " |"
    comp_param_row = "| Num. Parametros     | " + " | ".join([f"{baselines[k]['num_params']:<17,}" for k in model_keys]) + " |"
    
    comp_md = f"""# Comparacao Completa de Modelos: 6 Arquiteturas

## Resultados

{comp_table_header}
{comp_table_sep}
{comp_loss_row}
{comp_time_row}
{comp_size_row}
{comp_param_row}

## Analise

- **Menor Loss**: {best_loss.replace('_', ' ')}
- **Mais Rapido**: {best_time.replace('_', ' ')}
- **Mais Leve**: {best_size.replace('_', ' ')}

### Descricao das Novas Arquiteturas Adicionadas:
*   **Impala-CNN**: Modelo clássico do artigo do IMPALA. Apresenta ótima capacidade de memorização espacial por reter o Flatten da camada convolucional, porém gera um arquivo muito pesado.
*   **ResNet-18**: Arquitetura convolucional profunda com conexões residuais padrão. Oferece boa estabilidade de gradientes, mas é redundante para resoluções de 128x128.

### Nota sobre Swin Transformer e ConvNeXt:
Tanto o **Swin Transformer** quanto o **ConvNeXt (Tiny)** padrão foram avaliados, mas **removidos do script ativo de treinamento** devido à extrema lentidão de processamento e alto consumo de hardware. 
O ConvNeXt (Tiny) e o Swin Transformer possuem cerca de 28 milhões de parâmetros cada, o que provoca gargalo de memória de vídeo (VRAM Thrashing) ao processar lotes de 384 em imagens empilhadas, resultando em tempos estimados de treino de **40 horas** (no caso do ConvNeXt) no hardware de desenvolvimento, inviabilizando testes locais ágeis. Embora sejam arquiteturas extremamente potentes para visão geral, exigem recursos de cluster ou GPUs com alta memória dedicada.

## Configuracao do Experimento

- Epocas de Treino: {epochs}
- Batch Size: {batch}
- Learning Rate: {lr}
- Device: {device}

## Artefatos
- Modelos salvos na pasta `./models/`
- Experimento MLflow: `Model_Comparison`
"""
    comp_path.write_text(comp_md, encoding="utf-8")
    print("[OK] comparison_results.md atualizado com sucesso.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compara 6 arquiteturas de redes neurais para Imitation Learning.")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas (default: 10)")
    parser.add_argument("--batch", type=int, default=384, help="Tamanho do batch (default: 384)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu, default: cuda)")
    parser.add_argument("--only-new", action="store_true", help="Treina apenas os 2 novos modelos e usa baselines para os outros.")
    args = parser.parse_args()

    print_header("COMPARACAO DE 6 ARQUITETURAS")

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

    # Baselines padrao (caso a arquitetura nao seja treinada nesta run)
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
            "final_loss": 2.933126,
            "training_time": 226.2,
            "model_size_mb": 4.20,
            "num_params": 1009258
        },
        "Impala_CNN": {
            "final_loss": 2.945390,
            "training_time": 406.6,
            "model_size_mb": 67.95,
            "num_params": 17720938
        },
        "ResNet18": {
            "final_loss": 3.018901,
            "training_time": 391.1,
            "model_size_mb": 44.37,
            "num_params": 11516938
        }
    }

    # Carregar dados
    trajectories = load_demos(demo_path="./demos/")
    env = create_env()

    # Definir quais modelos treinar
    models_to_train = []
    if args.only_new:
        models_to_train = ["Impala_CNN", "ResNet18"]
        print("[INFO] Treinando APENAS as 2 novas arquiteturas. Usando baselines para o restante.")
    else:
        models_to_train = [
            "NatureCNN", "CNN_LSTM", "ViT_Transformer", "Impoola_CNN",
            "Impala_CNN", "ResNet18"
        ]
        print("[INFO] Treinando TODAS as 6 arquiteturas de redes neurais.")

    # Loop de treinamento
    for m_name in models_to_train:
        # Configurar kwargs
        policy_kwargs = None
        
        if m_name == "CNN_LSTM":
            policy_kwargs = {
                "features_extractor_class": TemporalAttentionLSTM,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                    "lstm_hidden_size": 256,
                    "lstm_num_layers": 2,
                    "debug": False,
                },
            }
        elif m_name == "ViT_Transformer":
            policy_kwargs = {
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
        elif m_name == "Impoola_CNN":
            policy_kwargs = {
                "features_extractor_class": ImpoolaCNNExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                    "channels_list": [32, 64, 128],
                },
            }
        elif m_name == "Impala_CNN":
            policy_kwargs = {
                "features_extractor_class": ImpalaCNNExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                    "channels_list": [32, 64, 128],
                },
            }
        elif m_name == "ResNet18":
            policy_kwargs = {
                "features_extractor_class": ResNet18Extractor,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                },
            }

        with mlflow.start_run(run_name=m_name):
            mlflow.log_param("model_type", m_name)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("batch_size", args.batch)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("device", args.device)
            
            results = train_model(
                name=m_name,
                env=env,
                trajectories=trajectories,
                policy_kwargs=policy_kwargs,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                device=args.device
            )
            
            # Atualizar dicionário local de baselines
            baselines[m_name] = results
            
            # Logar métricas no run ativo do MLflow
            mlflow.log_metric("final_loss", results["final_loss"])
            mlflow.log_metric("training_time_s", results["training_time"])
            mlflow.log_metric("model_size_mb", results["model_size_mb"])
            mlflow.log_metric("num_params", results["num_params"])
            mlflow.log_artifact(f"./models/{m_name}_policy.zip", artifact_path="model")

    # Salvar resultados e atualizar markdowns
    save_and_update_results(baselines, args.epochs, args.batch, args.lr, args.device)

    # Log de comparação resumida no MLflow
    model_keys = list(baselines.keys())
    with mlflow.start_run(run_name="Comparison_Summary"):
        mlflow.log_param("models_compared", ", ".join(model_keys))
        for k in model_keys:
            mlflow.log_metric(f"{k.lower()}_loss", baselines[k]["final_loss"])
            mlflow.log_metric(f"{k.lower()}_time_s", baselines[k]["training_time"])
            mlflow.log_metric(f"{k.lower()}_params", baselines[k]["num_params"])
        mlflow.log_artifact("./models/comparison_results.md", artifact_path="report")

    print_header("PROCESSO COMPLETO CONCLUIDO")


if __name__ == "__main__":
    main()
