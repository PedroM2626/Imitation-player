"""The generated tables must never fabricate a column it did not measure."""

from agent.cli import benchmark as b


def measurement(loss=None, time_=None, size=None, params=None, n=1):
    return {"final_loss": loss, "training_time": time_, "model_size_mb": size,
            "num_params": params, "n_runs": n, "loss_sd": 0.0}


MEASURED = {
    "naturecnn": measurement(3.48, 18.3, 16.33, 4196810),
    "lstm": measurement(3.02, 67.8, 23.71, 6116779),
    "transformer": measurement(3.00, 3288.8, 9.70, 2448010),
    "impoola": measurement(2.93, 226.2, 4.20, 1009258),
    "impala": measurement(3.03, 345.1, 67.95, 17720938),
    "resnet18": measurement(2.90, 251.3, 44.37, 11516938),
}
BASELINE = 2.6066


def test_columns_follow_the_study_order():
    table = b.build_table(MEASURED, b.COLUMN_ORDER, BASELINE)
    header = table.splitlines()[0]
    assert header.index("NatureCNN") < header.index("Impoola") < header.index("ResNet-18")


def test_best_and_worst_are_annotated():
    table = b.build_table(MEASURED, b.COLUMN_ORDER, BASELINE)
    assert "`3.48` *(Worst)*" in table
    assert "**`2.90`" in table


def test_margin_row_is_relative_to_the_baseline():
    table = b.build_table(MEASURED, b.COLUMN_ORDER, BASELINE)
    row = [l for l in table.splitlines() if "Margin over" in l][0]
    assert "+0.87" in row   # 3.4763 - 2.6066


def test_unmeasured_architectures_render_as_empty_not_invented():
    partial = dict(MEASURED)
    partial["resnet18"] = measurement(n=0)
    table = b.build_table(partial, b.COLUMN_ORDER, BASELINE)
    row = [l for l in table.splitlines() if "Final training loss" in l][0]
    assert row.rstrip("| ").endswith("`-`")
    runs_row = [l for l in table.splitlines() if "Runs available" in l][0]
    assert runs_row.rstrip("| ").endswith("0")


def test_conclusion_names_the_measured_extremes():
    text = b.conclusion(MEASURED, b.COLUMN_ORDER, BASELINE)
    assert text.startswith("**Conclusion**:")
    assert "ResNet-18" in text and "Impoola-CNN (GAP)" in text
    assert "2.6066" in text


def test_conclusion_does_not_contradict_the_table():
    """Regression: the hand-written sentence claimed the smallest model won on loss."""
    text = b.conclusion(MEASURED, b.COLUMN_ORDER, BASELINE)
    claim = text.split("lowest in-sample training loss was reached by ")[1].split("(")[0].strip()
    lowest = min(MEASURED, key=lambda k: MEASURED[k]["final_loss"])
    assert claim == b.DISPLAY[lowest]


def test_conclusion_is_empty_when_nothing_was_measured():
    empty = {k: measurement(n=0) for k in b.COLUMN_ORDER}
    assert b.conclusion(empty, b.COLUMN_ORDER, BASELINE) == ""


def test_regenerate_readme_preserves_everything_outside_the_markers(tmp_path):
    original = (
        "# Doc\n\nprologue\n\n<!-- BENCHMARK_START -->\nold table\n<!-- BENCHMARK_END -->\n\n"
        "epilogue\n\n**Conclusion**: stale sentence that is one line\n\ntrailing section\n")
    readme = tmp_path / "README.md"
    readme.write_text(original, encoding="utf-8")

    b.regenerate_readme("| new |\n| - |", "**Conclusion**: fresh sentence", readme)
    new = readme.read_text(encoding="utf-8")

    assert new.count("<!-- BENCHMARK_START -->") == 1
    assert new.count("**Conclusion**:") == 1
    assert "prologue" in new and "epilogue" in new and "trailing section" in new
    assert "| new |" in new and "old table" not in new
    assert "fresh sentence" in new and "stale sentence" not in new


def test_regenerate_readme_never_blanks_the_conclusion(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- BENCHMARK_START -->\nx\n<!-- BENCHMARK_END -->\n"
                      "**Conclusion**: keep me\n", encoding="utf-8")
    b.regenerate_readme("| y |", "", readme)
    assert "keep me" in readme.read_text(encoding="utf-8")
