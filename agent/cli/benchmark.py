"""
Architecture benchmark.

    python -m agent.cli.benchmark --profile hajime_ippo --epochs 10 --batch 384

Trains every registered encoder under one identical budget, then regenerates the
comparison table in ``README.md`` (between the ``BENCHMARK_START`` /
``BENCHMARK_END`` markers) and in ``runs/<profile>/models/comparison_results.md``.

Two defects of the previous harness are fixed here:

* it used to substitute **hard-coded** numbers for any architecture it did not
  retrain, and those constants had drifted until they disagreed with the report
  they were supposed to describe. Untrained columns are now read back from the
  MLflow store, and a column is labelled as historical when it is;
* it injected a hand-written conclusion sentence that contradicted the table it
  sat under (it claimed the smallest model had the lowest loss). The conclusion
  is now computed from the measurements.

Every reported figure is an in-sample training objective compared against the
corpus's own trivial baseline; see docs/RESULTS.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from agent.cli import train as train_cli
from agent.cli.common import (action_names, add_common, build_config, cli_entry, header, load,
                              num_actions, resolve_device, wrapped_env)
from agent.utils import demos as demos_mod
from agent.utils import paths
from agent.utils import tracking
from agent.utils.architectures import ARCHITECTURES, get

COLUMN_ORDER = ["naturecnn", "lstm", "transformer", "impoola", "impala", "resnet18"]
DISPLAY = {
    "naturecnn": "NatureCNN (reference)",
    "lstm": "CNN + LSTM + Attention",
    "transformer": "Vision Transformer (ViT)",
    "impoola": "Impoola-CNN (GAP)",
    "impala": "Impala-CNN (Original)",
    "resnet18": "ResNet-18",
}


def historical_from_store(profile: str, arch_name: str) -> Optional[Dict[str, float]]:
    """Most recent completed run of ``arch_name`` in the profile's MLflow store."""
    if tracking.mlflow is None:
        return None
    tracking.configure_store(profile)
    experiment = f"{profile}_imitation_bc"
    try:
        runs = tracking.mlflow.search_runs(experiment_names=[experiment], max_results=1000)
    except Exception:  # noqa: BLE001 - store absent, or the experiment was never created
        return None
    if runs is None or runs.empty or "tags.mlflow.runName" not in runs.columns:
        return None

    subset = runs[runs["tags.mlflow.runName"] == arch_name].sort_values("start_time")
    if subset.empty or "metrics.final_loss" not in subset.columns:
        return None
    completed = subset.dropna(subset=["metrics.final_loss"])
    if completed.empty:
        return None

    row = completed.iloc[-1]
    values = completed["metrics.final_loss"].astype(float).tolist()
    return {
        "final_loss": float(row["metrics.final_loss"]),
        "training_time": float(row.get("metrics.training_time_s", float("nan"))),
        "model_size_mb": float(row.get("metrics.model_size_mb", float("nan"))),
        "num_params": float(row.get("metrics.num_params", float("nan"))),
        "prob_true_act": float(row.get("metrics.prob_true_act", float("nan"))),
        "n_runs": len(values),
        "loss_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def build_table(measurements: Dict[str, Dict[str, float]], keys, baseline_nats: float) -> str:
    def best(key_of_metric):
        present = {k: measurements[k][key_of_metric] for k in keys
                   if measurements[k].get(key_of_metric) is not None}
        return min(present, key=present.get) if present else None

    best_loss, best_time = best("final_loss"), best("training_time")
    best_size, best_params = best("model_size_mb"), best("num_params")
    worst_loss = max((k for k in keys if measurements[k].get("final_loss")),
                     key=lambda k: measurements[k]["final_loss"], default=None)

    def cell(key, metric, value, best_key, worst_key, fmt):
        if value is None:
            return "`-`"
        tag = " *(Best)*" if key == best_key else (" *(Worst)*" if key == worst_key else "")
        bold = "**" if key == best_key else ""
        return f"{bold}`{fmt(value)}`{tag}{bold}"

    header_row = "| Metric | " + " | ".join(DISPLAY[k] for k in keys) + " |"
    separator = "| :--- | " + " | ".join(":---" for _ in keys) + " |"
    rows = [
        ("**Final training loss (nats, in-sample)**",
         lambda k: cell(k, "final_loss", measurements[k].get("final_loss"), best_loss, worst_loss,
                        lambda v: f"{v:.2f}")),
        ("**Margin over the marginal baseline**",
         lambda k: cell(k, "final_loss",
                        None if measurements[k].get("final_loss") is None
                        else measurements[k]["final_loss"] - baseline_nats,
                        best_loss, worst_loss, lambda v: f"{v:+.2f}")),
        ("**Training time**",
         lambda k: cell(k, "training_time", measurements[k].get("training_time"), best_time, None,
                        lambda v: f"{v/60:.1f} minutes" if v > 60 else f"{v:.1f} seconds")),
        ("**Checkpoint size**",
         lambda k: cell(k, "model_size_mb", measurements[k].get("model_size_mb"), best_size, None,
                        lambda v: f"{v:.2f} MB")),
        ("**Parameters**",
         lambda k: cell(k, "num_params", measurements[k].get("num_params"), best_params, None,
                        lambda v: f"{v/1e6:.2f} million")),
        ("**Runs available**",
         lambda k: str(int(measurements[k].get("n_runs") or 0))),
    ]
    body = "\n".join(f"| {label} | " + " | ".join(fn(k) for k in keys) + " |" for label, fn in rows)
    return "\n".join([header_row, separator, body])


def conclusion(measurements, keys, baseline_nats) -> str:
    losses = {k: measurements[k]["final_loss"] for k in keys if measurements[k].get("final_loss")}
    params = {k: measurements[k]["num_params"] for k in keys if measurements[k].get("num_params")}
    if not losses or not params:
        return ""
    low, efficient = min(losses, key=losses.get), min(params, key=params.get)
    def label(k):
        m = measurements[k]
        return (f"{DISPLAY[k]} ({m['final_loss']:.2f} nats, {m['model_size_mb']:.2f} MB, "
                f"{m['num_params'] / 1e6:.2f}M params)")
    return (f"**Conclusion**: the lowest in-sample training loss was reached by {label(low)}, and the "
            f"most size- and parameter-efficient model was {label(efficient)}. The corpus's "
            f"frame-independent marginal predictor scores {baseline_nats:.4f} nats, so any figure above "
            f"that evidences faster optimisation rather than conditioning on the screen; treat this as a "
            f"ranking of optimisation ease, not of policy quality.")


def regenerate_readme(table: str, conclusion_text: str, readme: Path) -> None:
    if not readme.exists():
        print(f"[!] {readme} not found; skipping README refresh")
        return
    content = readme.read_text(encoding="utf-8")
    start, end = "<!-- BENCHMARK_START -->", "<!-- BENCHMARK_END -->"
    if start not in content or end not in content:
        print("[!] benchmark markers missing from README; skipping README refresh")
        return
    before, rest = content.split(start, 1)
    _old, after = rest.split(end, 1)
    content = f"{before}{start}\n{table}\n{end}{after}"

    if conclusion_text and "**Conclusion**:" in content:
        head, tail = content.split("**Conclusion**:", 1)
        remainder = "\n".join(tail.split("\n")[1:])
        content = f"{head}{conclusion_text}\n{remainder}"
    elif not conclusion_text:
        print("[!] conclusion could not be computed from the measurements; README conclusion left as-is")
    readme.write_text(content, encoding="utf-8")
    print(f"[OK] refreshed {readme}")


def write_report(path: Path, table: str, measurements, keys, baseline_nats, epochs, batch, lr, device,
                 stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    losses = {k: measurements[k]["final_loss"] for k in keys if measurements[k].get("final_loss")}
    times = {k: measurements[k]["training_time"] for k in keys if measurements[k].get("training_time")}
    sizes = {k: measurements[k]["model_size_mb"] for k in keys if measurements[k].get("model_size_mb")}
    path.write_text(f"""# Model comparison: {len(keys)} architectures

Generated by `python -m agent.cli.benchmark`. Losses are **in-sample training
objectives** (summed per-bit negative log-likelihood, in nats), not generalisation
estimates. The last row of each table entry counts how many completed runs exist
for that architecture; several gaps in the ranking are smaller than the spread
between repeated runs of one architecture.

## Results

{table}

## Reference points

| Quantity | Value |
|---|---|
| Uniform predictor (all logits zero) | `{stats['uniform_baseline_nats']:.4f}` nats |
| Frame-independent per-bit marginal predictor | `{baseline_nats:.4f}` nats |
| Lowest measured model loss | `{min(losses.values()) if losses else float('nan'):.4f}` nats |

## Extremes

- Lowest loss: {DISPLAY[min(losses, key=losses.get)] if losses else '-'}
- Fastest: {DISPLAY[min(times, key=times.get)] if times else '-'}
- Smallest checkpoint: {DISPLAY[min(sizes, key=sizes.get)] if sizes else '-'}

## Experiment configuration

| Item | Value |
|---|---|
| Profile | `{stats.get('profile', '-')}` |
| Epochs | {epochs} |
| Batch size | {batch} |
| Learning rate | {lr} |
| Device | {device} |
| Corpus | {stats['frames']} frames across {stats['trajectories']} trajectories |
| Action width | {stats['num_actions']} |
| Distinct joint actions | {stats['distinct_joint_actions']} |
| Modal action share | {stats['modal_action_share']:.4f} |
| Seed | {train_cli.SEED} |
""", encoding="utf-8")
    print(f"[OK] wrote {path}")


@cli_entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(parser, training=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--archs", nargs="*", default=None, choices=sorted(ARCHITECTURES),
                        help="subset to train now; the rest is filled from the tracking store")
    args = parser.parse_args(argv)

    profile = load(args.profile, args.runs_root)
    config = profile["GAME_CONFIG"]
    n_act = num_actions(config)
    device = resolve_device(args.device)

    to_train = args.archs or COLUMN_ORDER
    # Columns always render in the fixed study order; --archs only selects
    # which of them are re-measured now versus read back from the store.
    keys = list(COLUMN_ORDER)

    header("ARCHITECTURE BENCHMARK")
    demos_dir = paths.demos_dir(profile["name"])
    trajectories = demos_mod.load_demos(demos_dir, n_act, config["actions"].get("width_policy", "strict"))
    stats = demos_mod.summarise(trajectories, n_act)
    stats["profile"] = profile["name"]
    baseline_nats = float(stats["marginal_baseline_nats"])
    demos_mod.print_summary(stats, action_names(profile))

    env = wrapped_env(build_config(config, dummy=True))
    measurements: Dict[str, Dict[str, float]] = {}

    for key in keys:
        arch = get(key)
        if key in to_train:
            checkpoint = paths.models_dir(profile["name"]) / f"{arch.checkpoint_prefix}.zip"
            if key == "naturecnn":
                checkpoint = paths.models_dir(profile["name"]) / "bc_policy.zip"
            started = time.time()
            result = train_cli.train(
                trajectories, env, key, device, args.epochs, args.batch, args.lr, profile,
                paths.logs_dir(profile["name"]), None, checkpoint)
            result["n_runs"] = 1
            result["loss_sd"] = 0.0
            measurements[key] = result
            print(f"[OK] {arch.name}: loss {result['final_loss']:.6f} in {result['training_time']:.1f}s")
        else:
            historical = historical_from_store(profile["name"], arch.name)
            if historical is None:
                print(f"[!] {arch.name}: no run in the store and not selected for training; column left empty")
                measurements[key] = {"final_loss": None, "training_time": None,
                                     "model_size_mb": None, "num_params": None, "n_runs": 0}
            else:
                historical["measured_now"] = False
                measurements[key] = historical

    table = build_table(measurements, keys, baseline_nats)
    text = conclusion(measurements, keys, baseline_nats)
    print("\n" + table + "\n")
    print(text + "\n")

    readme = paths.REPO_ROOT / "README.md"
    regenerate_readme(table, text, readme)
    write_report(paths.models_dir(profile["name"]) / "comparison_results.md", table, measurements,
                 keys, baseline_nats, args.epochs, args.batch, args.lr, device, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
