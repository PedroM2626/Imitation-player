"""Checkpoint discovery orders numerically and prefers the final artifact."""

from agent.utils.checkpoints import get_last_index, list_checkpoints, resolve_checkpoint


def touch(tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_nine_does_not_outrank_ten(tmp_path):
    touch(tmp_path, "bc_policy9.zip")
    touch(tmp_path, "bc_policy10.zip")
    assert get_last_index(tmp_path, "bc_policy") == 10


def test_final_checkpoint_wins_over_numbered(tmp_path):
    touch(tmp_path, "bc_policy.zip")
    touch(tmp_path, "bc_policy10.zip")
    assert resolve_checkpoint(tmp_path).name == "bc_policy.zip"


def test_numbered_fallback_when_no_final(tmp_path):
    touch(tmp_path, "bc_policy2.zip")
    touch(tmp_path, "bc_policy15.zip")
    assert resolve_checkpoint(tmp_path).name == "bc_policy15.zip"


def test_list_is_sorted_by_numeric_index(tmp_path):
    """Numbered checkpoints descend by index; the final artifact carries index -1."""
    for n in (2, 10, 1):
        touch(tmp_path, f"bc_policy{n}.zip")
    touch(tmp_path, "bc_policy.zip")
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["bc_policy10.zip", "bc_policy2.zip", "bc_policy1.zip", "bc_policy.zip"]


def test_missing_directory_resolves_to_nothing(tmp_path):
    assert get_last_index(tmp_path / "absent", "bc_policy") == -1
    assert resolve_checkpoint(tmp_path / "absent") is None


def test_unrelated_files_are_ignored(tmp_path):
    touch(tmp_path, "comparison_results.md")
    touch(tmp_path, "other_policy99.zip")
    assert resolve_checkpoint(tmp_path, "bc_policy") is None
