"""Cohort metrics derived from a run's outcomes.

The diagnostic rate is the one with a trap in it. It is computed over the pairs
``(injected label, inferred label)`` collected on every step the diagnostic saw,
and a step can reach it carrying no injected label at all — a person has none on
any step, and a simulated learner has none on a step no misconception of theirs
produced. Scoring those charges the diagnostic for the absence of a question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from run_cohort import _diagnostic_accuracy, _unlabelled_errors


@dataclass
class Outcome:
    """Only the field the two metrics read."""

    diagnoses: tuple[tuple[str | None, str | None], ...] = field(default=())


class TestTheDiagnosticRate:
    def test_it_scores_agreement(self) -> None:
        outcomes = [Outcome((("a", "a"), ("b", "b"), ("c", "x")))]
        assert _diagnostic_accuracy(outcomes) == 2 / 3

    def test_nothing_diagnosed_is_none_rather_than_zero(self) -> None:
        assert _diagnostic_accuracy([Outcome(())]) is None

    def test_a_step_with_no_injected_label_is_not_scored(self) -> None:
        # The pair would otherwise count as a miss, so the rate would fall
        # because the learner erred without holding a bug — not because the
        # diagnostic got anything wrong.
        labelled = [Outcome((("a", "a"), ("b", "b")))]
        with_slip = [Outcome((("a", "a"), ("b", "b"), (None, "c")))]
        assert _diagnostic_accuracy(labelled) == 1.0
        assert _diagnostic_accuracy(with_slip) == 1.0

    def test_only_unlabelled_steps_is_none(self) -> None:
        # Nothing was measurable, which is not the same as measuring zero.
        assert _diagnostic_accuracy([Outcome(((None, "a"), (None, None)))]) is None

    def test_an_oracle_still_reads_exactly_one(self) -> None:
        # The guard the metric exists for: an oracle is perfect by construction,
        # and it is handed None on a slip, so it returns None and agrees.
        assert _diagnostic_accuracy([Outcome((("a", "a"), (None, None)))]) == 1.0


class TestUnlabelledErrorsAreCounted:
    """Excluded from the rate, but reported — or the exclusion hides them."""

    def test_it_counts_the_steps_the_rate_skipped(self) -> None:
        outcomes = [Outcome((("a", "a"), (None, "b"), (None, None)))]
        assert _unlabelled_errors(outcomes) == 2

    def test_it_is_zero_when_every_step_carried_a_label(self) -> None:
        assert _unlabelled_errors([Outcome((("a", "a"), ("b", "c")))]) == 0

    def test_it_sums_across_learners(self) -> None:
        assert _unlabelled_errors([Outcome(((None, "a"),)), Outcome(((None, "b"),))]) == 2
