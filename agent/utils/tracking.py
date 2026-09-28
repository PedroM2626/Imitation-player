"""
MLflow plumbing shared by every training script.

Experiment names are derived from the active profile and the script's purpose,
so runs land where a reader expects instead of all reporting into
``Hajime_no_Ippo_Imitation_Learning`` regardless of the package they came from.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.logger import KVWriter

try:
    import mlflow
except ImportError:  # pragma: no cover - mlflow is a hard dependency of the CLI
    mlflow = None

from agent.utils import paths

#: ``bc/l2_loss`` is ``l2_weight * l2_norm`` and ``imitation`` defaults
#: ``l2_weight`` to 0.0, so the key is definitionally dead in this project.
#: It is excluded from the tracking store; ``bc/l2_norm`` is still logged and is
#: the informative one.
DEFAULT_SKIP_KEYS = ("l2_loss",)


class MetricCapture(KVWriter):
    """Keeps the last value seen for each logged metric."""

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}

    def write(self, key_values, key_excluded, step=0):
        for key, value in key_values.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.metrics[key] = float(value)

    def close(self):
        pass


class MLflowOutputFormat(KVWriter):
    """Forwards SB3 logger scalars into the active MLflow run."""

    def __init__(self, prefix: str = "", skip_keys=DEFAULT_SKIP_KEYS) -> None:
        self.prefix = prefix
        self.skip_keys = tuple(skip_keys or ())

    def write(self, key_values, key_excluded, step=0):
        if mlflow is None or not mlflow.active_run():
            return
        for key, value in key_values.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            if any(key.endswith(skip) for skip in self.skip_keys):
                continue
            mlflow.log_metric(self.prefix + key, float(value), step=step)

    def close(self):
        pass


def experiment_name(profile: str, purpose: str) -> str:
    return f"{profile}_{purpose}"


def configure_store(profile: str) -> None:
    if mlflow is None:  # pragma: no cover
        raise RuntimeError("mlflow is not installed")
    store = paths.mlruns_dir(profile).as_posix()
    paths.ensure_dirs(profile, paths.mlruns_dir(profile))
    mlflow.set_tracking_uri(f"file:{store}")


def start_run(profile: str, purpose: str, run_name: str, params: Optional[Dict[str, Any]] = None):
    configure_store(profile)
    mlflow.set_experiment(experiment_name(profile, purpose))
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
    return run


def log_dataset(profile: str, stats: Dict[str, Any]) -> None:
    """Record what the run actually trained on.

    Without this a corpus that lost files to a silent load error is
    indistinguishable from the intended one.
    """
    if mlflow is None or not mlflow.active_run():
        return
    mlflow.log_param("profile", profile)
    mlflow.log_param("corpus_frames", int(stats["frames"]))
    mlflow.log_param("corpus_trajectories", int(stats["trajectories"]))
    mlflow.log_param("num_actions", int(stats["num_actions"]))
    mlflow.log_param("distinct_joint_actions", int(stats["distinct_joint_actions"]))
    mlflow.log_metric("marginal_baseline_nats", float(stats["marginal_baseline_nats"]))
    mlflow.log_metric("uniform_baseline_nats", float(stats["uniform_baseline_nats"]))
    mlflow.log_metric("corpus_joint_entropy_bits", float(stats["joint_entropy_bits"]))
