"""
Full Model Comparison: 6 Architectures for Imitation Learning
========================================================================

This script compares 6 different neural network architectures:
1. NatureCNN (Default)
2. CNN + LSTM + Attention
3. Vision Transformer (ViT)
4. Impoola-CNN (GAP)
5. Impala-CNN (Original with Flatten)
6. ResNet-18

(ConvNeXt and Swin Transformer were removed because of slow training).

HOW TO USE:
    python compare_models.py --epochs 10 --batch 384 --lr 1e-4 --device cuda --only-new

ARGUMENTS:
    --epochs    : Number of epochs (default: 10)
    --batch     : Batch size (default: 384)
    --lr        : Learning rate (default: 1e-4)
    --device    : 'cuda' or 'cpu' (default: cuda)
    --only-new  : Trains only the 2 active new architectures (Impala, ResNet-18)
                  and uses the pre-recorded baselines for the other 4.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Adjust the path so the modules can be found
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../utils"))

import numpy as np
import torch as th
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# Local extractors
from game_env import GenericGameEnv, TemporalAttentionLSTM
from config.game_config import GAME_CONFIG
from vision_transformer import VisionTransformerExtractor
from impoola_cnn import ImpoolaCNNExtractor
from new_architectures import ImpalaCNNExtractor, ResNet18Extractor

import mlflow


# ============================================================
# Metric capture for MLflow
# ============================================================

class MetricCapture(KVWriter):
    """Captures the last value of each metric logged by the BC trainer."""
    def __init__(self):
        self.metrics: Dict[str, float] = {}

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        self.metrics.update(key_values)

    def close(self) -> None:
        pass


class MLflowOutputFormat(KVWriter):
    """Logs metrics directly to MLflow during training."""
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
# Utility functions
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_demos(demo_path: str = "./demos/") -> List[Trajectory]:
    demo_dir = Path(demo_path)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Directory {demo_dir} not found!")
    
    demo_files = sorted(demo_dir.glob("demo*.pt"))
    if not demo_files:
        raise FileNotFoundError(f"No demo*.pt files found in {demo_dir}")
        
    expected_num_actions = GAME_CONFIG.get("actions", {}).get("num_actions", 18)
    all_trajectories: List[Trajectory] = []
    
    print(f"Loading {len(demo_files)} demo files:")
    for i, demo_file in enumerate(demo_files):
        try:
            print(f"  [{i+1}/{len(demo_files)}] Loading {demo_file.name}...", end=" ")
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
            print(f" ERROR: {e}")
            
    total_frames = sum(len(t.obs) for t in all_trajectories)
    print(f"\n[OK] Data loaded: {len(all_trajectories)} trajectories, {total_frames} total frames.")
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
    print_header(f"TRAINING: {name}")
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
    
    print(f"\n[OK] {name} complete:")
    print(f"     Final loss:  {final_loss:.6f}")
    print(f"     Time:        {training_time:.1f}s")
    print(f"     Size:        {model_size:.2f} MB")
    print(f"     Params:      {num_params:,}")
    return results


# ============================================================
# Markdown Updates and Comparison Table Generation
# ============================================================

def save_and_update_results(baselines: Dict[str, Dict[str, Any]], epochs: int, batch: int, lr: float, device: str):
    print_header("UPDATING COMPARISON REPORTS")
    
    # Ordered lists for benchmark display
    model_keys = [
        "NatureCNN",
        "CNN_LSTM",
        "ViT_Transformer",
        "Impoola_CNN",
        "Impala_CNN",
        "ResNet18"
    ]
    
    # Formatted names for the markdown
    display_names = {
        "NatureCNN": "NatureCNN (Default)",
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
            suffix = " *(Best)*" if is_best else (" *(Worst)*" if is_worst else "")
        elif val_type == "time":
            if val > 60:
                base_str = f"{val/60:.1f} minutes"
            else:
                base_str = f"{val:.1f} seconds"
            suffix = " *(Fastest)*" if is_best else (" *(Slowest)*" if is_worst else "")
        elif val_type == "size":
            base_str = f"{val:.2f} MB"
            suffix = " *(Lightest!)*" if is_best else (" *(Heaviest)*" if is_worst else "")
        elif val_type == "params":
            base_str = f"{val/1e6:.2f} Million"
            suffix = " *(Most Efficient)*" if is_best else (" *(Least Efficient)*" if is_worst else "")
            
        if is_best:
            return f"**`{base_str}`**{suffix}"
        return f"`{base_str}`{suffix}"

    # 1. Generate the table for README.md
    headers = " | ".join([display_names[k] for k in model_keys])
    separators = " | ".join([":---" for _ in model_keys])
    
    loss_cells = " | ".join([fmt_cell(k, baselines[k]["final_loss"], best_loss, worst_loss, "loss") for k in model_keys])
    time_cells = " | ".join([fmt_cell(k, baselines[k]["training_time"], best_time, worst_time, "time") for k in model_keys])
    size_cells = " | ".join([fmt_cell(k, baselines[k]["model_size_mb"], best_size, worst_size, "size") for k in model_keys])
    param_cells = " | ".join([fmt_cell(k, baselines[k]["num_params"], best_params, worst_params, "params") for k in model_keys])
    
    new_table = f"""| Metric | {headers} |
| :--- | {separators} |
| **Learning Power (Final Loss)** | {loss_cells} |
| **Training Time** | {time_cells} |
| **Final File Size** | {size_cells} |
| **Parameter Count (Neurons)** | {param_cells} |"""

    readme_path = Path("../../README.md")
    if readme_path.exists():
        print(f"Updating {readme_path.resolve()}...")
        readme_content = readme_path.read_text(encoding="utf-8")
        
        start_tag = "<!-- BENCHMARK_START -->"
        end_tag = "<!-- BENCHMARK_END -->"
        
        if start_tag in readme_content and end_tag in readme_content:
            parts = readme_content.split(start_tag)
            before = parts[0]
            after = parts[1].split(end_tag)[1]
            updated_content = f"{before}{start_tag}\n{new_table}\n{end_tag}{after}"
            
            # Conclusion derived from the measured values, not hard-coded prose
            def _summary(key: str) -> str:
                b = baselines[key]
                return (f"{display_names[key]} ({b['final_loss']:.2f} loss, "
                        f"{b['model_size_mb']:.2f} MB, {b['num_params'] / 1e6:.2f}M params)")

            conclusion_text = (
                "**Conclusion**: the lowest in-sample training loss was reached by "
                f"{_summary(best_loss)}, and the most size- and parameter-efficient model was "
                f"{_summary(best_params)}. These losses are training objectives measured on the "
                "demonstration corpus itself, so they rank optimisation ease rather than policy "
                "quality; see the threats-to-validity section before treating this as a ranking."
            )
            
            if "**Conclusion**:" in updated_content:
                concl_parts = updated_content.split("**Conclusion**:")
                before_concl = concl_parts[0]
                after_concl = concl_parts[1].split("\n")[1:]
                updated_content = f"{before_concl}{conclusion_text}\n" + "\n".join(after_concl)
                
            readme_path.write_text(updated_content, encoding="utf-8")
            print("[OK] README.md updated successfully.")
        else:
            print("[!] Benchmark tags not found in README.md.")
    else:
        print("[!] README.md not found at the expected path.")

    # 2. Generate models/comparison_results.md
    comp_path = Path("./models/comparison_results.md")
    print(f"Updating {comp_path.resolve()}...")
    
    comp_table_header = "| Metric              | " + " | ".join([f"{k:<18}" for k in model_keys]) + " |"
    comp_table_sep = "|---------------------|" + "|".join(["--------------------" for _ in model_keys]) + "|"
    
    comp_loss_row = "| Final Loss          | " + " | ".join([f"{baselines[k]['final_loss']:<18.6f}" for k in model_keys]) + " |"
    comp_time_row = "| Training Time       | " + " | ".join([f"{baselines[k]['training_time']:<16.1f}s" for k in model_keys]) + " |"
    comp_size_row = "| Model Size          | " + " | ".join([f"{baselines[k]['model_size_mb']:<15.2f} MB" for k in model_keys]) + " |"
    comp_param_row = "| Num. Params         | " + " | ".join([f"{baselines[k]['num_params']:<17,}" for k in model_keys]) + " |"
    
    comp_md = f"""# Full Model Comparison: 6 Architectures

## Results

{comp_table_header}
{comp_table_sep}
{comp_loss_row}
{comp_time_row}
{comp_size_row}
{comp_param_row}

## Analysis

- **Lowest Loss**: {best_loss.replace('_', ' ')}
- **Fastest**: {best_time.replace('_', ' ')}
- **Lightest**: {best_size.replace('_', ' ')}

### Description of the Newly Added Architectures:
*   **Impala-CNN**: Classic model from the IMPALA paper. It has excellent spatial memorization capacity because it keeps the Flatten of the convolutional layer, but it produces a very heavy file.
*   **ResNet-18**: Deep convolutional architecture with standard residual connections. It offers good gradient stability, but it is redundant for 128x128 resolutions.

### Note on Swin Transformer and ConvNeXt:
Both the standard **Swin Transformer** and **ConvNeXt (Tiny)** were evaluated, but **removed from the active training script** due to extreme processing slowness and high hardware consumption. 
The ConvNeXt (Tiny) and the Swin Transformer each have about 28 million parameters, which causes a video memory bottleneck (VRAM Thrashing) when processing batches of 384 on stacked images, resulting in estimated training times of **40 hours** (in the ConvNeXt case) on the development hardware, which rules out agile local testing. Although they are extremely powerful architectures for general vision, they require cluster resources or GPUs with high dedicated memory.

## Experiment Configuration

- Training Epochs: {epochs}
- Batch Size: {batch}
- Learning Rate: {lr}
- Device: {device}

## Artifacts
- Models saved in the `./models/` folder
- MLflow Experiment: `Model_Comparison`
"""
    comp_path.write_text(comp_md, encoding="utf-8")
    print("[OK] comparison_results.md updated successfully.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compares 6 neural network architectures for Imitation Learning.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch", type=int, default=384, help="Batch size (default: 384)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu, default: cuda)")
    parser.add_argument("--only-new", action="store_true", help="Trains only the 2 new models and uses baselines for the others.")
    args = parser.parse_args()

    print_header("COMPARISON OF 6 ARCHITECTURES")

    # Check device
    if args.device == "cuda" and not th.cuda.is_available():
        print("[!] CUDA not available, using CPU.")
        args.device = "cpu"
    else:
        if args.device == "cuda":
            print(f"[OK] GPU detected: {th.cuda.get_device_name(0)}")

    print(f"[CONFIG] Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}, Device: {args.device}")

    # Set up MLflow
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Model_Comparison")

    # Baselines used when an architecture is not retrained in this run.
    # Values are the completed MLflow runs of experiment "Model_Comparison".
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
            "final_loss": 3.027630,
            "training_time": 345.1,
            "model_size_mb": 67.95,
            "num_params": 17720938
        },
        "ResNet18": {
            "final_loss": 2.901681,
            "training_time": 251.3,
            "model_size_mb": 44.37,
            "num_params": 11516938
        }
    }

    # Load data
    trajectories = load_demos(demo_path="./demos/")
    env = create_env()

    # Define which models to train
    models_to_train = []
    if args.only_new:
        models_to_train = ["Impala_CNN", "ResNet18"]
        print("[INFO] Training ONLY the 2 new architectures. Using baselines for the rest.")
    else:
        models_to_train = [
            "NatureCNN", "CNN_LSTM", "ViT_Transformer", "Impoola_CNN",
            "Impala_CNN", "ResNet18"
        ]
        print("[INFO] Training ALL 6 neural network architectures.")

    # Training loop
    for m_name in models_to_train:
        # Configure kwargs
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
            
            # Update the local baselines dictionary
            baselines[m_name] = results
            
            # Log metrics to the active MLflow run
            mlflow.log_metric("final_loss", results["final_loss"])
            mlflow.log_metric("training_time_s", results["training_time"])
            mlflow.log_metric("model_size_mb", results["model_size_mb"])
            mlflow.log_metric("num_params", results["num_params"])
            mlflow.log_artifact(f"./models/{m_name}_policy.zip", artifact_path="model")

    # Save results and update the markdown files
    save_and_update_results(baselines, args.epochs, args.batch, args.lr, args.device)

    # Log the summarized comparison to MLflow
    model_keys = list(baselines.keys())
    with mlflow.start_run(run_name="Comparison_Summary"):
        mlflow.log_param("models_compared", ", ".join(model_keys))
        for k in model_keys:
            mlflow.log_metric(f"{k.lower()}_loss", baselines[k]["final_loss"])
            mlflow.log_metric(f"{k.lower()}_time_s", baselines[k]["training_time"])
            mlflow.log_metric(f"{k.lower()}_params", baselines[k]["num_params"])
        mlflow.log_artifact("./models/comparison_results.md", artifact_path="report")

    print_header("FULL PROCESS COMPLETE")


if __name__ == "__main__":
    main()
