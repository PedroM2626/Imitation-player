"""
Play the game with a trained policy.

    python -m agent.cli.deploy --profile hajime_ippo --arch lstm

``K`` hands control to the agent or back to you, ``ESC`` quits. Unlike the old
scripts, ``deploy.fps`` is now actually enforced: the loop sleeps for the
remaining frame budget instead of spinning as fast as the GPU allows.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agent.cli.common import (
    add_common,
    build_config,
    cli_entry,
    header,
    load,
    num_actions,
    wrapped_env,
)
from agent.utils import paths
from agent.utils.architectures import ARCHITECTURES, get
from agent.utils.checkpoints import resolve_checkpoint
from agent.utils.utils import PolicyRunner, load_policy


def choose_checkpoint(models_dir, prefix: str):
    path = resolve_checkpoint(models_dir, prefix)
    if path is None:
        raise SystemExit(
            f"No checkpoint matching {prefix}*.zip in {models_dir}. Train one first: "
            f"python -m agent.cli.train --arch <name>"
        )
    return path


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common(parser, training=True)
    parser.add_argument("--arch", default="naturecnn", choices=sorted(ARCHITECTURES))
    parser.add_argument("--model", default=None, help="explicit checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--start-manual", action="store_true", help="begin with the human in control"
    )
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    deploy_cfg = config.get("deploy", {})
    fps = float(deploy_cfg.get("fps", 30))
    aggressiveness = float(deploy_cfg.get("aggressiveness", 1.0))
    n_act = num_actions(config)

    models_dir = paths.models_dir(profile["name"])
    prefix = "bc_policy" if args.arch == "naturecnn" else get(args.arch).checkpoint_prefix
    checkpoint = args.model or choose_checkpoint(models_dir, prefix)

    header("DEPLOYING POLICY")
    print(f"checkpoint    : {checkpoint}")
    print(f"profile       : {profile['name']}  (input mode {config['actions'].get('input_mode')})")
    print(
        f"inference cap : {fps:.0f} fps"
        + (f"  aggressiveness={aggressiveness}" if aggressiveness != 1.0 else "  (no sharpening)")
    )

    env = wrapped_env(build_config(config, dummy=False))
    device = "cuda" if args.device == "cuda" else ("cuda" if _cuda_ok() else "cpu")
    runner = PolicyRunner(load_policy(checkpoint, device=device))

    try:
        import keyboard
    except ImportError:
        print("The 'keyboard' package is required to toggle control.")
        return 1

    from agent.utils.input_map import HumanInput

    human = HumanInput(config)
    attack_bits = [
        i
        for i, m in enumerate(config["actions"]["mappings"])
        if m.get("name") in deploy_cfg.get("attack_buttons", ())
    ]

    ai_active = not args.start_manual
    obs = env.reset()
    steps = 0
    budget = 1.0 / fps if fps > 0 else 0.0
    try:
        while args.max_steps is None or steps < args.max_steps:
            loop_start = time.perf_counter()

            if keyboard.is_pressed("esc"):
                break
            if _toggled("k"):
                ai_active = not ai_active
                runner.reset()
                print(f"Control -> {'AI' if ai_active else 'HUMAN'}")

            if ai_active:
                if aggressiveness != 1.0 and attack_bits:
                    action = runner.sharpen(obs[0], aggressiveness, attack_bits)
                else:
                    action, _ = runner.predict(obs, deterministic=False)
            else:
                action = np.zeros((1, n_act), dtype=np.float32)
                action[0] = human.read()

            obs, _, _, _ = env.step(
                action if np.ndim(action) == 2 else [np.asarray(action).reshape(-1)[:n_act]]
            )
            steps += 1

            remaining = budget - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nStopped after {steps} steps.")
        env.close()
    return 0


_TOGGLED_LAST = {"k": 0.0}


def _toggled(key: str, debounce: float = 0.3) -> bool:
    import keyboard

    if not keyboard.is_pressed(key):
        return False
    now = time.time()
    if now - _TOGGLED_LAST[key] < debounce:
        return False
    _TOGGLED_LAST[key] = now
    return True


def _cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
