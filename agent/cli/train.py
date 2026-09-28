"""
Behavioural cloning.

    python -m agent.cli.train --profile hajime_ippo --arch lstm --epochs 300

One entry point for every architecture (the four near-duplicate
``train_agent_*.py`` scripts are gone). Every run logs what it actually trained
on -- frame count, trajectory count, action width and the trivial-predictor
baseline -- so a result can be read against a reference instead of in isolation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from agent.cli.common import (
    action_names,
    add_common,
    build_config,
    cli_entry,
    header,
    load,
    num_actions,
    resolve_device,
    wrapped_env,
)
from agent.utils import demos as demos_mod
from agent.utils import paths, tracking
from agent.utils.architectures import ARCHITECTURES, get, policy_kwargs_for

SEED = 42


def build_policy(
    arch_key: str,
    env,
    device,
    learning_rate: float,
    training_config: dict[str, Any],
    warm_start: str = None,
):
    """An instantiated policy, or None to let BC build the default NatureCNN one."""
    if warm_start:
        path = Path(warm_start)
        if not path.exists():
            raise SystemExit(f"[!] --model_path {path} does not exist")
        print(f"[+] Warm start from {path}")
        return ActorCriticCnnPolicy.load(path, device=device)

    arch = get(arch_key)
    if arch.factory is None:
        return None

    kwargs = policy_kwargs_for(arch, env.observation_space, training_config)
    policy = ActorCriticCnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: learning_rate,
        **kwargs,
    ).to(device)
    params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[+] Encoder {arch.name} -> {params:,} trainable parameters")
    return policy


def train(
    trajectories: list[Trajectory],
    env,
    arch_key: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    profile: dict[str, Any],
    log_dir: Path,
    warm_start: str = None,
    checkpoint: Path = None,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    capture = tracking.MetricCapture()
    logger = Logger(
        folder=str(log_dir),
        output_formats=[
            HumanOutputFormat(sys.stdout),
            CSVOutputFormat(str(log_dir / "progress.csv")),
            tracking.MLflowOutputFormat(prefix=f"{get(arch_key).name}/"),
            capture,
        ],
    )

    policy = build_policy(
        arch_key, env, device, learning_rate, profile["TRAINING_CONFIG"], warm_start
    )

    bc = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=np.random.default_rng(seed=SEED),
        device=device,
        policy=policy,
        batch_size=batch_size,
        optimizer_kwargs={"lr": learning_rate},
        custom_logger=logger,
    )

    start = time.time()
    bc.train(n_epochs=epochs, progress_bar=True)
    elapsed = time.time() - start

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    bc.policy.save(str(checkpoint))

    final_loss = capture.metrics.get("bc/loss", float("nan"))
    prob_true = capture.metrics.get("bc/prob_true_act", float("nan"))
    params = sum(p.numel() for p in bc.policy.parameters() if p.requires_grad)
    size_mb = checkpoint.stat().st_size / (1024 * 1024)

    if tracking.mlflow is not None and tracking.mlflow.active_run():
        for key, value in capture.metrics.items():
            tracking.mlflow.log_metric(f"final/{key.rsplit('/', 1)[-1]}", value)
        tracking.mlflow.log_metrics(
            {
                "final_loss": final_loss,
                "training_time_s": elapsed,
                "num_params": params,
                "model_size_mb": size_mb,
                "prob_true_act": prob_true,
            }
        )
    logger.close()
    return {
        "final_loss": final_loss,
        "training_time": elapsed,
        "num_params": params,
        "model_size_mb": size_mb,
        "prob_true_act": prob_true,
    }


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(parser, training=True)
    parser.add_argument(
        "--arch",
        default="naturecnn",
        choices=sorted(ARCHITECTURES),
        help="feature extractor (default: naturecnn = stable-baselines3 default)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model_path", default=None, help="checkpoint to continue training from")
    parser.add_argument(
        "--width-policy",
        choices=("strict", "coerce"),
        default=None,
        help="override actions.width_policy for a legacy corpus",
    )
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    training_config = profile["TRAINING_CONFIG"]
    n_act = num_actions(config)

    epochs = args.epochs or int(training_config.get("epochs", 100))
    batch = args.batch or int(training_config.get("batch_size", 384))
    lr = args.lr or float(training_config.get("learning_rate", 1e-4))
    device = resolve_device(args.device)
    width_policy = args.width_policy or config["actions"].get("width_policy", "strict")

    header(f"BEHAVIOURAL CLONING - {get(args.arch).name}")
    print(
        f"profile: {profile['name']} | epochs: {epochs} | batch: {batch} | lr: {lr} | device: {device}"
    )

    demo_dir = paths.demos_dir(profile["name"])
    trajectories = demos_mod.load_demos(demo_dir, n_act, policy=width_policy)
    stats = demos_mod.summarise(trajectories, n_act)
    demos_mod.print_summary(stats, action_names(profile))
    print(
        f"\nA frame-independent per-bit predictor scores {stats['marginal_baseline_nats']:.4f} nats "
        f"on this corpus; a run that finishes above it has not learned to condition on the screen."
    )

    models_dir = paths.models_dir(profile["name"])
    paths.ensure_dirs(profile["name"], models_dir, paths.logs_dir(profile["name"]))
    checkpoint = models_dir / f"{get(args.arch).checkpoint_prefix}.zip"
    if args.arch == "naturecnn":
        checkpoint = models_dir / "bc_policy.zip"

    tracking.start_run(
        profile["name"],
        "imitation_bc",
        run_name=args.arch,
        params={
            "arch": get(args.arch).name,
            "epochs": epochs,
            "batch_size": batch,
            "learning_rate": lr,
            "device": device,
            "width_policy": width_policy,
            "seed": SEED,
            "warm_start": args.model_path,
        },
    )
    tracking.log_dataset(profile["name"], stats)
    try:
        result = train(
            trajectories,
            wrapped_env(build_config(config, dummy=True)),
            args.arch,
            device,
            epochs,
            batch,
            lr,
            profile,
            paths.logs_dir(profile["name"]),
            args.model_path,
            checkpoint,
        )
    finally:
        if tracking.mlflow is not None and tracking.mlflow.active_run():
            tracking.mlflow.end_run()

    print(f"\n[OK] {checkpoint}")
    print(
        f"     final loss      : {result['final_loss']:.6f} nats "
        f"(baseline {stats['marginal_baseline_nats']:.4f}, "
        f"margin {result['final_loss'] - stats['marginal_baseline_nats']:+.4f})"
    )
    print(f"     prob_true_act   : {result['prob_true_act']:.6f}")
    print(f"     trained in      : {result['training_time']:.1f}s")
    print(f"     parameters      : {result['num_params']:,.0f}")
    print(f"     checkpoint      : {result['model_size_mb']:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
