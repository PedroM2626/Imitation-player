"""Demonstration loading: no silent corruption, correct statistics."""

import numpy as np
import pytest

from agent.utils import demos


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(demos.DemoError, match="does not exist"):
        demos.load_demos(tmp_path / "nope", 18)


def test_empty_directory_names_the_fix(corpus):
    (corpus / "demo_0_test.pt").unlink()
    (corpus / "demo_1_test.pt").unlink()
    with pytest.raises(demos.DemoError, match="agent.cli.record"):
        demos.load_demos(corpus, 18)


def test_strict_policy_refuses_a_mismatched_width(runs_root):
    demos_dir = runs_root / "hajime_ippo" / "demos"
    from tests.conftest import make_demos
    make_demos(demos_dir, num_actions=18, trajectories=1)
    make_demos(demos_dir, num_actions=18, width_override=7, trajectories=1, seed=5)

    with pytest.raises(demos.DemoError, match="width 7"):
        demos.load_demos(demos_dir, 18, policy="strict")


def test_coerce_policy_pads_and_warns(runs_root, capsys):
    demos_dir = runs_root / "hajime_ippo" / "demos"
    from tests.conftest import make_demos
    make_demos(demos_dir, num_actions=18, trajectories=1)
    make_demos(demos_dir, num_actions=18, width_override=7, trajectories=1, seed=5)

    trajectories = demos.load_demos(demos_dir, 18, policy="coerce")
    assert all(t.acts.shape[1] == 18 for t in trajectories)
    assert "WARNING coercing width" in capsys.readouterr().out


def test_corrupt_file_raises_instead_of_shrinking_the_corpus(corpus):
    (corpus / "demo_broken.pt").write_bytes(b"not a torch archive")
    with pytest.raises(demos.DemoError, match="could not be read"):
        demos.load_demos(corpus, 18)


def test_observation_and_action_lengths_stay_synchronised(corpus):
    trajectories = demos.load_demos(corpus, 18)
    for t in trajectories:
        assert t.obs.shape[0] == t.acts.shape[0] + 1


def test_marginal_baseline_matches_a_brute_force_computation(corpus):
    trajectories = demos.load_demos(corpus, 18)
    stats = demos.summarise(trajectories, 18)

    acts = np.concatenate([t.acts for t in trajectories])
    probs = acts.mean(axis=0)
    expected = float(-(probs * np.log(probs) + (1 - probs) * np.log(1 - probs)).sum())
    assert stats["marginal_baseline_nats"] == pytest.approx(expected, abs=1e-9)


def test_uniform_baseline_is_nats_over_bits(corpus):
    stats = demos.summarise(demos.load_demos(corpus, 18), 18)
    assert stats["uniform_baseline_nats"] == pytest.approx(18 * np.log(2))


def test_dead_bits_are_detected(corpus):
    trajectories = demos.load_demos(corpus, 18)
    # Zero out the upper half of every action vector.
    zeroed = [type(t)(obs=t.obs, acts=np.concatenate(
        [t.acts[:, :9], np.zeros((t.acts.shape[0], 9), np.float32)], axis=1),
        infos=None, terminal=False) for t in trajectories]
    stats = demos.summarise(zeroed, 18)
    assert all(v == 0.0 for v in stats["marginals"][9:])
    # The entropy is computed on probabilities clipped away from exactly 0/1,
    # so a dead bit contributes ~1e-11 rather than a hard zero.
    assert pytest.approx(stats["per_bit_entropy_nats"][9], abs=1e-9) == 0.0


def test_summary_prints_the_reference_point(corpus, capsys):
    stats = demos.summarise(demos.load_demos(corpus, 18), 18)
    demos.print_summary(stats, ["BTN"] * 18)
    out = capsys.readouterr().out
    assert "marginal baseline NLL" in out
    assert "distinct joint actions" in out
