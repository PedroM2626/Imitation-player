"""
Interactive DAgger correction collection.

    python -m agent.cli.dagger --profile hajime_ippo --rounds 3

The agent plays; ``L`` (or holding it) transfers authority to you, and the
frames you steer are written as ordinary demonstrations, so the next
``agent.cli.train`` run aggregates them automatically. This is the collection
half of DAgger (Ross et al., 2011).

The old scripts advertised ``train.py --dagger``, whose handler was a ``pass``
statement. That flag is gone: aggregation is simply "retrain over demos/", and
this script now drives the full loop - collect, retrain, redeploy - for
``--rounds`` iterations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from agent.cli.common import (
    add_common,
    build_config,
    cli_entry,
    header,
    load,
    num_actions,
    resolve_device,
    wrapped_env,
)
from agent.cli.deploy import choose_checkpoint
from agent.utils import demos as demos_mod
from agent.utils import paths
from agent.utils.architectures import ARCHITECTURES, get
from agent.utils.input_map import HumanInput
from agent.utils.utils import PolicyRunner, load_policy


def collect_round(
    env, runner, human, demo_dir: Path, profile_name: str, seconds: float, num_actions: int
) -> int:
    """Let the agent play, hand over on ``L``, and record what the human does."""
    import keyboard
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((280, 60))
    font = pygame.font.SysFont(None, 22)

    buffer_obs, buffer_acts = [], []
    started = time.time()
    saved = 0
    obs = env.reset()
    human_prev_active = False

    try:
        while time.time() - started < seconds:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
            if keyboard.is_pressed("esc"):
                break

            human_active = keyboard.is_pressed("l")
            if human_active:
                action = np.zeros((1, num_actions), dtype=np.float32)
                action[0] = human.read()
                buffer_obs.append(np.squeeze(obs, axis=0).copy())
                buffer_acts.append(action[0].copy())
            else:
                action, _ = runner.predict(obs, deterministic=False)

            if human_prev_active and not human_active and len(buffer_acts) > 1:
                saved += _flush(buffer_obs, buffer_acts, demo_dir, profile_name)
                buffer_obs, buffer_acts = [], []
                runner.reset()
            human_prev_active = human_active

            obs, _, _, _ = env.step(action)

            screen.fill((0, 0, 0))
            state = "HUMAN (recording)" if human_active else "AI"
            left = seconds - (time.time() - started)
            for i, line in enumerate(
                [f"{state}   {left:4.1f}s left", "[L] take over   [ESC] stop"]
            ):
                screen.blit(font.render(line, True, (255, 255, 255)), (8, 6 + i * 20))
            pygame.display.flip()
    finally:
        if len(buffer_acts) > 1:
            saved += _flush(buffer_obs, buffer_acts, demo_dir, profile_name)
        pygame.quit()
        env.close()
    return saved


def _flush(buffer_obs, buffer_acts, demo_dir: Path, profile_name: str) -> int:
    import torch as th
    from imitation.data.types import Trajectory

    obs = np.stack(buffer_obs).astype(np.uint8)
    acts = np.asarray(buffer_acts, dtype=np.float32)
    # Match the recorder's N+1 observation convention.
    obs = np.concatenate([obs, obs[-1:]], axis=0)
    traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=False)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = demo_dir / f"demo_dagger_{profile_name}_{stamp}.pt"
    th.save([traj], path)
    print(f"[dagger] wrote {acts.shape[0]} correction frames -> {path.name}")
    return 1


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(parser, training=True)
    parser.add_argument("--arch", default="naturecnn", choices=sorted(ARCHITECTURES))
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="collect-then-retrain iterations (default: TRAINING_CONFIG.dagger_iterations)",
    )
    parser.add_argument("--seconds", type=float, default=120.0, help="collection time per round")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--collect-only", action="store_true", help="do not retrain afterwards")
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    training_config = profile["TRAINING_CONFIG"]
    rounds = (
        args.rounds if args.rounds is not None else int(training_config.get("dagger_iterations", 3))
    )
    n_act = num_actions(config)
    demo_dir = paths.demos_dir(profile["name"])

    header("DAGGER CORRECTION COLLECTION")
    print(f"profile : {profile['name']} | rounds: {rounds} | {args.seconds:.0f}s each")
    print("Press and HOLD L to take control; release it to hand back and save that segment.")

    device = resolve_device(args.device)
    total_saved = 0
    for round_index in range(rounds):
        print(f"\n--- round {round_index + 1}/{rounds} ---")
        checkpoint = choose_checkpoint(
            paths.models_dir(profile["name"]),
            "bc_policy" if args.arch == "naturecnn" else get(args.arch).checkpoint_prefix,
        )
        env = wrapped_env(build_config(config, dummy=False))
        runner = PolicyRunner(load_policy(checkpoint, device=device))
        human = HumanInput(config)
        total_saved += collect_round(
            env, runner, human, demo_dir, profile["name"], args.seconds, n_act
        )

        if args.collect_only:
            break

        from agent.cli import train as train_cli

        train_args = ["--profile", profile["name"], "--arch", args.arch, "--device", device]
        for flag, value in (("--epochs", args.epochs), ("--batch", args.batch), ("--lr", args.lr)):
            if value is not None:
                train_args += [flag, str(value)]
        stats = demos_mod.summarise(
            demos_mod.load_demos(demo_dir, n_act, config["actions"].get("width_policy", "strict")),
            n_act,
        )
        print(
            f"[dagger] aggregated corpus is now {stats['frames']} frames "
            f"(baseline {stats['marginal_baseline_nats']:.4f} nats); retraining..."
        )
        train_cli.main(train_args)

    print(f"\n[OK] {total_saved} correction file(s) written to {demo_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
