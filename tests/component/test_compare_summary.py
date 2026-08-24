"""Comparing a re-run's summary against the stored one.

The guard has to be shown able to fail: a comparison that reports agreement on
anything it is handed would let a re-run that did not reproduce pass as one that
did.
"""

from __future__ import annotations

import json

from compare_summary import compare, main


def _summary(**overrides):
    base = {
        "n_learners": 160,
        "seed": 20260811,
        "arms": {
            "coupled": "20260824T124927_calculus_paired_coupled_9c95bba3",
            "decoupled": "20260824T124930_calculus_paired_decoupled_8bf94985",
        },
        "mean_items": {"coupled": 44.63125, "decoupled": 44.63125},
        "results": [
            {"outcome": "gain", "mean_diff": -0.0096, "ties": 133},
            {"outcome": "goals_mastered", "mean_diff": 1.525, "ties": 15},
        ],
        "dose_matched": {
            "run_id": "20260824T124933_calculus_paired_dosematched_coupled_4ccce872",
            "budget": 32,
        },
    }
    base.update(overrides)
    return base


class TestCompare:
    def test_identical_summaries_agree(self) -> None:
        differences, checked = compare(_summary(), _summary())
        assert differences == []
        # Guards against agreement reached by comparing nothing at all.
        assert checked > 5

    def test_run_ids_and_timestamps_are_not_differences(self) -> None:
        # The one thing two runs of the same experiment always differ in.
        other = _summary()
        other["arms"] = {"coupled": "20260101T000000_x_coupled_aaaa", "decoupled": "b"}
        other["dose_matched"]["run_id"] = "20260101T000000_x_dosematched_coupled_bbbb"
        assert compare(other, _summary())[0] == []

    def test_a_changed_number_is_reported(self) -> None:
        differences, _ = compare(_summary(mean_items={"coupled": 99.0}), _summary())
        assert any("mean_items.coupled" in line for line in differences)

    def test_a_change_inside_a_list_is_reported(self) -> None:
        # The outcomes are a list, and a difference in one of them is the whole
        # point of running the comparison.
        moved = _summary()
        moved["results"][1]["mean_diff"] = 0.0
        differences, _ = compare(moved, _summary())
        assert any("results[1].mean_diff" in line for line in differences)

    def test_a_missing_field_is_reported(self) -> None:
        short = _summary()
        del short["seed"]
        assert any("seed" in line for line in compare(short, _summary())[0])

    def test_tolerance_admits_a_last_digit_and_no_more(self) -> None:
        near = _summary(mean_items={"coupled": 44.63125 + 1e-12, "decoupled": 44.63125})
        assert compare(near, _summary())[0] == []
        far = _summary(mean_items={"coupled": 44.64, "decoupled": 44.63125})
        assert compare(far, _summary())[0] != []


class TestIgnore:
    """A branch the re-run deliberately did not produce.

    The propagation study's model-backed condition is left out when there is no
    model, so the stored summary has a branch nothing can match. Reporting it
    every time would bury a difference that means something.
    """

    def test_an_ignored_branch_is_not_a_difference(self) -> None:
        without = _summary()
        del without["dose_matched"]
        assert compare(without, _summary(), ignore=["dose_matched"])[0] == []

    def test_ignoring_one_branch_does_not_hide_another(self) -> None:
        moved = _summary(mean_items={"coupled": 99.0, "decoupled": 44.63125})
        del moved["dose_matched"]
        differences, _ = compare(moved, _summary(), ignore=["dose_matched"])
        assert any("mean_items.coupled" in line for line in differences)

    def test_a_prefix_does_not_match_a_longer_sibling_name(self) -> None:
        # 'dose' must not silence 'dose_matched'.
        without = _summary()
        del without["dose_matched"]
        assert compare(without, _summary(), ignore=["dose"])[0] != []


class TestCommandLine:
    def test_exit_status_reports_the_verdict(self, tmp_path, monkeypatch, capsys) -> None:
        stored = tmp_path / "stored.json"
        rerun = tmp_path / "rerun.json"
        stored.write_text(json.dumps(_summary()))
        rerun.write_text(json.dumps(_summary()))

        monkeypatch.setattr(
            "sys.argv", ["compare_summary", "--rerun", str(rerun), "--stored", str(stored)]
        )
        assert main() == 0
        assert "reproduces" in capsys.readouterr().out

        rerun.write_text(json.dumps(_summary(seed=1)))
        assert main() == 1
        assert "differs" in capsys.readouterr().out
