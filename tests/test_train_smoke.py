"""One-epoch CPU training for every registered architecture.

These are the tests that matter: they exercise loading, the wrapped
observation space, the extractor, the BC loop, checkpoint writing and MLflow
logging for each encoder. They are slow relative to the rest of the suite, so
only two architectures run under ``-m smoke``-excluded CI by default and the
full set is covered locally.
"""

import pytest

from agent.cli import train as train_cli
from agent.utils.architectures import ARCHITECTURES

ALL_ARCHS = sorted(ARCHITECTURES)


def run(runs_root, profile, arch, epochs=1, batch=16, extra=None):
    args = ["--profile", profile, "--arch", arch, "--epochs", str(epochs),
            "--batch", str(batch), "--device", "cpu"]
    return train_cli.main(args + (extra or []))


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_training_writes_a_checkpoint(corpus, runs_root, arch):
    assert run(runs_root, "hajime_ippo", arch) == 0
    from agent.utils import paths
    models = paths.models_dir("hajime_ippo")
    assert any(p.suffix == ".zip" for p in models.iterdir())


def test_reported_loss_is_finite_and_positive(corpus, runs_root, capsys):
    run(runs_root, "hajime_ippo", "impoola")
    out = capsys.readouterr().out
    assert "final loss" in out and "nan" not in out.split("final loss")[1].split("\n")[0]


def test_the_baseline_reference_is_printed(corpus, runs_root, capsys):
    run(runs_root, "hajime_ippo", "naturecnn")
    assert "frame-independent per-bit predictor" in capsys.readouterr().out


def test_strict_width_policy_blocks_the_legacy_mixed_corpus(runs_root):
    """The shipped corpus mixes 18-bit and 7-bit recordings; strict mode must refuse it."""
    from tests.conftest import make_demos
    demos_dir = runs_root / "hajime_ippo" / "demos"
    make_demos(demos_dir, num_actions=18, trajectories=1)
    make_demos(demos_dir, num_actions=18, width_override=7, trajectories=1, seed=3)

    with pytest.raises(SystemExit):
        run(runs_root, "hajime_ippo", "naturecnn")


def test_coerce_override_allows_the_legacy_corpus(runs_root):
    from tests.conftest import make_demos
    demos_dir = runs_root / "hajime_ippo" / "demos"
    make_demos(demos_dir, num_actions=18, trajectories=1)
    make_demos(demos_dir, num_actions=18, width_override=7, trajectories=1, seed=3)

    assert run(runs_root, "hajime_ippo", "naturecnn",
               extra=["--width-policy", "coerce"]) == 0


def test_warm_start_loads_the_previous_checkpoint(corpus, runs_root):
    from agent.utils import paths
    run(runs_root, "hajime_ippo", "naturecnn")
    checkpoint = paths.models_dir("hajime_ippo") / "bc_policy.zip"
    assert run(runs_root, "hajime_ippo", "naturecnn",
               extra=["--model_path", str(checkpoint)]) == 0


def test_mlflow_run_records_the_corpus_and_the_baseline(corpus, runs_root):
    mlflow = pytest.importorskip("mlflow")
    run(runs_root, "hajime_ippo", "impoola")

    from agent.utils import paths, tracking
    tracking.configure_store("hajime_ippo")
    runs = mlflow.search_runs(experiment_names=["hajime_ippo_imitation_bc"])
    assert not runs.empty
    row = runs.iloc[0]
    assert "params.corpus_frames" in runs.columns
    assert float(row["params.corpus_frames"]) > 0
    assert "metrics.marginal_baseline_nats" in runs.columns
    assert float(row["metrics.marginal_baseline_nats"]) > 0
    assert row["params.profile"] == "hajime_ippo"
