"""
Demonstration loading, statistics and the trivial-predictor baseline.

Loading is where the old code silently corrupted data: files with an unexpected
action width were truncated or zero-padded inside a per-file ``try/except``, and
nothing recorded how many frames a run actually saw. The default policy is now
``strict`` (refuse to train on a corpus that does not match the profile), with
``coerce`` available as an explicit, loudly-warned opt-in for legacy corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch as th
from imitation.data.types import Trajectory

DEMO_GLOB = "demo*.pt"


class DemoError(RuntimeError):
    pass


def _reconcile_width(acts: np.ndarray, expected: int) -> np.ndarray:
    width = acts.shape[-1]
    if width == expected:
        return acts
    if width > expected:
        return acts[..., :expected]
    pad = [(0, 0)] * (acts.ndim - 1) + [(0, expected - width)]
    return np.pad(acts, pad, mode="constant", constant_values=0)


def load_demos(
    directory: Path,
    num_actions: int,
    policy: str = "strict",
    glob: str = DEMO_GLOB,
) -> List[Trajectory]:
    """Read ``demo*.pt`` files into ``Trajectory`` objects.

    ``policy``: ``strict`` raises if a file's action width differs from
    ``num_actions``; ``coerce`` truncates/pads and warns per file.
    """
    directory = Path(directory)
    if not directory.exists():
        raise DemoError(f"demonstration directory {directory} does not exist")

    files = sorted(directory.glob(glob))
    if not files:
        raise DemoError(
            f"no files matching {glob!r} in {directory}. Record some first "
            f"(python -m agent.cli.record --profile <name>)."
        )

    trajectories: List[Trajectory] = []
    failures = []
    for path in files:
        try:
            payload = th.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {exc.__class__.__name__}: {exc}")
            continue

        items = payload if isinstance(payload, list) else [payload]
        for traj in items:
            obs = np.asarray(traj.obs)
            acts = np.asarray(traj.acts)
            if obs.ndim == 5 and obs.shape[1] == 1:
                obs = np.squeeze(obs, axis=1)

            if acts.shape[-1] != num_actions:
                message = (f"{path.name}: actions have width {acts.shape[-1]} but the profile "
                           f"declares num_actions={num_actions}")
                if policy == "strict":
                    raise DemoError(message + ". Re-record, choose the right profile, or set "
                                              "actions.width_policy='coerce' to accept truncation.")
                print(f"[demo] WARNING coercing width: {message}")
                acts = _reconcile_width(acts, num_actions)

            trajectories.append(Trajectory(obs=obs, acts=acts.astype(np.float32),
                                           infos=None, terminal=False))

    if failures:
        raise DemoError(
            f"{len(failures)} demonstration file(s) could not be read:\n  "
            + "\n  ".join(failures)
        )
    if not trajectories:
        raise DemoError(f"no usable trajectories found in {directory}")

    return trajectories


def summarise(trajectories: Sequence[Trajectory], num_actions: int) -> Dict[str, object]:
    """Corpus statistics, including the trivial-predictor baseline."""
    acts = np.concatenate([np.asarray(t.acts).reshape(-1, num_actions) for t in trajectories])
    marginals = acts.mean(axis=0)
    probs = np.clip(marginals, 1e-12, 1 - 1e-12)
    per_bit_entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))

    joint = acts.astype(np.int8)
    unique, counts = np.unique(joint, axis=0, return_counts=True)
    share = counts / counts.sum()
    joint_entropy_bits = float(-(share * np.log2(share)).sum())

    return {
        "frames": int(acts.shape[0]),
        "trajectories": len(list(trajectories)),
        "num_actions": int(num_actions),
        "marginals": marginals,
        "per_bit_entropy_nats": per_bit_entropy,
        # NLL of the best frame-independent product-of-sigmoids predictor.
        # Any model above this number has not learned to condition on the screen.
        "marginal_baseline_nats": float(per_bit_entropy.sum()),
        "uniform_baseline_nats": float(num_actions * np.log(2)),
        "distinct_joint_actions": int(unique.shape[0]),
        "joint_entropy_bits": joint_entropy_bits,
        "modal_action_share": float(share.max()),
        "top5_action_share": float(np.sort(share)[::-1][:5].sum()),
        "zero_action_share": float((acts.sum(axis=1) == 0).mean()),
    }


def print_summary(stats: Dict[str, object], names: Optional[Sequence[str]] = None) -> None:
    print(f"\nCorpus: {stats['trajectories']} trajectories, {stats['frames']} frames, "
          f"{stats['num_actions']} action bits")
    print(f"  distinct joint actions : {stats['distinct_joint_actions']} "
          f"(of {2 ** int(stats['num_actions'])} possible)")
    print(f"  modal action share     : {stats['modal_action_share']:.4f}")
    print(f"  top-5 action share     : {stats['top5_action_share']:.4f}")
    print(f"  all-zero action share  : {stats['zero_action_share']:.4f}")
    print(f"  marginal baseline NLL  : {stats['marginal_baseline_nats']:.4f} nats "
          f"(uniform would be {stats['uniform_baseline_nats']:.4f})")
    print("  per-bit firing rate:")
    marginals = stats["marginals"]  # type: ignore[assignment]
    for i, value in enumerate(marginals):  # type: ignore[union-attr]
        label = names[i] if names and i < len(names) else f"ACT_{i}"
        print(f"    [{i:>2}] {label:<16} {value:.4f}")
    dead = [i for i, value in enumerate(marginals) if value == 0]  # type: ignore[union-attr]
    if dead:
        print(f"  NOTE: bits {dead} never fire in this corpus; their contribution to "
              f"the loss is identically zero.")
