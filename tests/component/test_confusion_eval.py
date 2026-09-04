"""Scoring the confusion detector, and the properties that make its figure mean
something.

The detector had a hand-labelled set and a pytest gate long before it had a way
to produce a number anyone could store or cite. What is tested here is the
scoring, not the detector: whether the report carries the floor, whether the two
halves stay apart, and whether a set that stopped being balanced would be
noticed rather than quietly reported as though it still were.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_newton.core.evaluation.confusion import (
    ConfusionCase,
    ConfusionGold,
    load_gold,
    score,
)

GOLD = Path("tests/fixtures/gold/calculus_confusion_cases.yaml")


class _Always:
    """Fires on everything. The failure the floor exists to make visible."""

    def confused(self, concept_id: str, text: str) -> str | None:
        return text


class _Never:
    """Fires on nothing — the model-free detector's behaviour."""

    def confused(self, concept_id: str, text: str) -> str | None:
        return None


def _gold(*pairs: tuple[str, bool]) -> ConfusionGold:
    return ConfusionGold(
        cases=tuple(
            ConfusionCase(
                concept_id="c", text=text, confused=flag, source="written for a test"
            )
            for text, flag in pairs
        )
    )


class TestTheGoldSetLoads:
    def test_it_reads_the_real_fixture(self) -> None:
        gold = load_gold(GOLD)
        assert gold.cases
        assert all(case.source.strip() for case in gold.cases)

    def test_the_real_fixture_is_balanced(self) -> None:
        # Asserted here as well as in the calibration gate, because the scoring
        # reports a floor derived from it and a report is only readable while
        # this holds.
        assert load_gold(GOLD).balanced

    def test_an_empty_set_is_refused(self, tmp_path: Path) -> None:
        # A set with no cases would score 0/0 and print as a clean 0.0%.
        path = tmp_path / "empty.yaml"
        path.write_text("cases: []\n")
        with pytest.raises(ValueError, match="no cases"):
            load_gold(path)


class TestTheFloorIsReported:
    """A constant answer must be visibly worth exactly half.

    This is the property that makes the agreement figure readable at all, so it
    is asserted rather than left to the fixture's header.
    """

    def test_a_detector_that_always_fires_scores_the_floor(self) -> None:
        gold = _gold(("a", True), ("b", True), ("c", False), ("d", False))
        report = score(gold, _Always(), "always")
        assert report.agreement == report.floor == 0.5
        assert not report.beats_the_floor

    def test_a_detector_that_never_fires_also_scores_the_floor(self) -> None:
        gold = _gold(("a", True), ("b", True), ("c", False), ("d", False))
        report = score(gold, _Never(), "never")
        assert report.agreement == 0.5
        assert not report.beats_the_floor

    def test_the_summary_cannot_travel_without_the_floor(self) -> None:
        # The figure and the number qualifying it are written together, so a
        # stored summary cannot be quoted without it.
        stored = score(_gold(("a", True), ("b", False)), _Never(), "never").as_dict()
        assert "floor" in stored and "agreement" in stored
        assert "beats_the_floor" in stored


class TestTheTwoHalvesStayApart:
    """Pooling them would hide the failure that actually matters.

    A detector firing on everything gets the confused half perfectly. The half
    that must *not* fire — hedging, uncertainty, "this was confusing" — is the
    hard one, and it is the one a single pooled accuracy would bury.
    """

    def test_always_firing_is_perfect_on_one_half_and_useless_on_the_other(self) -> None:
        gold = _gold(("a", True), ("b", True), ("c", False), ("d", False))
        report = score(gold, _Always(), "always")
        assert report.detected_confusion == 1.0
        assert report.left_work_alone == 0.0

    def test_a_disagreement_says_which_direction_it_went(self) -> None:
        gold = _gold(("fires when it should not", False), ("misses one", True))
        kinds = {d.kind for d in score(gold, _Always(), "always").disagreements}
        assert kinds == {"false positive"}
        kinds = {d.kind for d in score(gold, _Never(), "never").disagreements}
        assert kinds == {"false negative"}

    def test_a_false_positive_keeps_what_was_read_that_way(self) -> None:
        # A firing is only arguable if you can see the words it read, which is
        # why the detector returns the quote rather than a bool.
        gold = _gold(("I think I made an arithmetic slip", False))
        (only,) = score(gold, _Always(), "always").disagreements
        assert only.got == "I think I made an arithmetic slip"


class TestTheBalanceCheckCanFail:
    """A guard that cannot fail proves nothing.

    If the fixture ever stopped being balanced, the floor would move and the
    agreement would stop meaning what every quotation of it assumes. This is the
    shape that must be detected.
    """

    def test_an_unbalanced_set_reports_itself_as_such(self) -> None:
        gold = _gold(("a", True), ("b", True), ("c", True), ("d", False))
        assert not gold.balanced
        assert gold.floor == 0.75

    def test_and_a_constant_answer_then_beats_a_real_detector(self) -> None:
        # Why it matters, rather than only that it is detected: on an unbalanced
        # set, answering "confused" to everything scores 75% and would read as a
        # good result against a floor nobody printed.
        gold = _gold(("a", True), ("b", True), ("c", True), ("d", False))
        report = score(gold, _Always(), "always")
        assert report.agreement == 0.75
        assert not report.beats_the_floor
