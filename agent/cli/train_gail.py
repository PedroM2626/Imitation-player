"""
GAIL: adversarial imitation (Ho & Erlikman, 2016).

    python -m agent.cli.train_gail --profile hajime_ippo --timesteps 100000

A discriminator learns to tell demonstrated frames apart from the agent's own,
and the generator (PPO) is optimised against that signal instead of a hand-made
reward.

**Status: never completed a run in this project, and not testable offline.** The
generator must interact with the real game, so the environment cannot be in
dummy mode: start the title first. Treat this as experimental tooling rather
than a validated pipeline (docs/LIMITATIONS.md B3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from agent.cli.common import add_common, build_config, cli_entry, header, load, resolve_device, wrapped_env
from agent.utils import demos as demos_mod
from agent.utils import paths
from agent.utils import tracking

SEED = 42


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(parser, training=True)
    parser.add_argument("--timesteps", type=int, default=100000, help="generator-environment steps")
    parser.add_argument("--model_path", default=None, help="PPO checkpoint to warm start from")
    parser.add_argument("--demo-batch-size", type=int, default=64)
    parser.add_argument("--gen-batch-capacity", type=int, default=2048)
    parser.add_argument("--disc-updates-per-round", type=int, default=4)
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    n_act = int(config["actions"]["num_actions"])
    device = resolve_device(args.device)

    header("GAIL (adversarial imitation)")
    print("The game must already be running: the generator interacts with the live window.")

    trajectories = demos_mod.load_demos(paths.demos_dir(profile["name"]), n_act,
                                        config["actions"].get("width_policy", "strict"))
    stats = demos_mod.summarise(trajectories, n_act)
    demos_mod.print_summary(stats)

    env = wrapped_env(build_config(config, dummy=False))
    log_dir = paths.logs_dir(profile["name"]) / "gail"
    log_dir.mkdir(parents=True, exist_ok=True)

    tracking.start_run(profile["name"], "gail", run_name="gail", params={
        "total_timesteps": args.timesteps, "demo_batch_size": args.demo_batch_size,
        "gen_replay_buffer_capacity": args.gen_batch_capacity,
        "n_disc_updates_per_round": args.disc_updates_per_round,
        "generator": "PPO(batch_size=64, lr=3e-4, n_steps=1024, ent_coef=0.01, gamma=0.99)",
        "device": device, "seed": SEED, "corpus_frames": stats["frames"],
    })
    tracking.log_dataset(profile["name"], stats)

    if args.model_path and Path(args.model_path).exists():
        learner = PPO.load(args.model_path, env=env, device=device)
        print(f"[+] Warm start from {args.model_path}")
    else:
        learner = PPO(env=env, policy=ActorCriticCnnPolicy, batch_size=64, ent_coef=0.01,
                      learning_rate=3e-4, gamma=0.99, n_steps=1024, device=device)

    logger = Logger(folder=str(log_dir), output_formats=[
        HumanOutputFormat(sys.stdout),
        CSVOutputFormat(str(log_dir / "progress.csv")),
        tracking.MLflowOutputFormat(prefix="gail/"),
    ])

    reward_net = BasicRewardNet(observation_space=env.observation_space,
                                action_space=env.action_space,
                                normalize_input_layer=RunningNorm)
    trainer = GAIL(demonstrations=trajectories, demo_batch_size=args.demo_batch_size,
                   gen_replay_buffer_capacity=args.gen_batch_capacity,
                   n_disc_updates_per_round=args.disc_updates_per_round, venv=env,
                   gen_algo=learner, reward_net=reward_net,
                   allow_variable_horizon=True, custom_logger=logger)

    print(f"[OK] training for {args.timesteps} environment steps...")
    trainer.train(args.timesteps)

    out = paths.models_dir(profile["name"]) / "gail_policy.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    learner.save(str(out))
    if tracking.mlflow is not None and tracking.mlflow.active_run():
        tracking.mlflow.log_artifact(str(log_dir / "progress.csv"))
        tracking.mlflow.end_run()
    logger.close()
    env.close()
    print(f"\n[OK] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
