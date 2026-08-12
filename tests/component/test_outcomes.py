"""Outcome measurement.

These are the figures a sitting is judged by, and each one here exists because
the figure beside it could not answer the question it was being asked. A raw
percentage cannot say whether the training reached anything that needed
teaching; a score with unreadable answers in its denominator charges the
verifier's failure to the learner; and a gain of five points means one thing
when there were ninety available and another when there were nine.
"""

from __future__ import annotations

import pytest

from agent_newton.core.evaluation.outcomes import (
    ItemResult,
    SessionOutcome,
    # Aliased: pytest tries to collect any module-level name starting
    # with Test, and warns that this one has a constructor.
    TestResult as HeldOut,
    dose_by_concept,
    dose_on_gap,
    per_concept_change,
)
from agent_newton.core.state.schema import AuditRecord
from agent_newton.domains.base import Verdict


def bank(*pairs: tuple[str, Verdict]) -> HeldOut:
    """A held-out bank whose items resolve to these verdicts, in order."""
    results = tuple(
        ItemResult(f"i_{concept}", concept, verdict) for concept, verdict in pairs
    )
    return HeldOut(
        correct=sum(1 for _, v in pairs if v is Verdict.CORRECT),
        total=len(pairs),
        unmeasurable=sum(1 for _, v in pairs if v is Verdict.UNPARSEABLE),
        per_item=results,
    )


def outcome(pretest: HeldOut, posttest: HeldOut) -> SessionOutcome:
    return SessionOutcome(
        learner_id="L1",
        arm="coupled",
        pretest=pretest,
        posttest=posttest,
        items_attempted=0,
        items_to_exhaustion=None,
        remediation_ratio=None,
        unmeasurable_steps=0,
    )


def observed(*concepts: str) -> list[AuditRecord]:
    return [
        AuditRecord(
            version=n + 1,
            cause="observation",
            summary="",
            evidence={"concept_id": concept, "item_id": f"i_{concept}"},
        )
        for n, concept in enumerate(concepts)
    ]


class TestUnreadableAnswersAreNotWrongAnswers:
    """The invariant the learner model has always held, applied to the score.

    ``is_evidence`` keeps an unreadable step out of BKT and out of the error
    trace, and ``concepts_missed`` keeps it out of what a learner is told they
    got wrong. The score counted every one of them as a failure, beneath a panel
    saying they were not counted against anyone.
    """

    def test_the_raw_score_still_counts_everything_administered(self) -> None:
        result = bank(("a", Verdict.CORRECT), ("b", Verdict.UNPARSEABLE))
        assert result.score == 0.5

    def test_the_measured_score_leaves_them_out(self) -> None:
        result = bank(("a", Verdict.CORRECT), ("b", Verdict.UNPARSEABLE))
        assert result.measured_score == 1.0
        assert result.measured == 1

    def test_a_wrong_answer_still_counts(self) -> None:
        # The distinction must bite only where the verifier failed. An error is
        # still an error.
        result = bank(("a", Verdict.CORRECT), ("b", Verdict.INCORRECT))
        assert result.measured_score == 0.5

    def test_a_bank_nobody_could_read_scores_nothing_either_way(self) -> None:
        result = bank(("a", Verdict.UNPARSEABLE), ("b", Verdict.UNPARSEABLE))
        assert result.measured == 0
        assert result.measured_score == 0.0

    def test_the_reported_gain_uses_the_measured_score(self) -> None:
        # The demonstrated distortion: a post-test with five wrong answers and
        # ten unreadable ones reported 0%, which reads as a learner who got
        # nothing right rather than as a verifier that could not read them.
        before = bank(*[("c%d" % n, Verdict.INCORRECT) for n in range(15)])
        after = bank(
            *[("c%d" % n, Verdict.CORRECT) for n in range(5)],
            *[("c%d" % n, Verdict.UNPARSEABLE) for n in range(5, 15)],
        )
        assert after.score == 5 / 15
        assert outcome(before, after).gain == pytest.approx(1.0)


class TestNormalisedGain:
    """Raw gain is bounded by what the learner did not already know."""

    def test_it_is_the_share_of_what_was_available(self) -> None:
        before = bank(*[("c%d" % n, Verdict.CORRECT) for n in range(9)],
                      *[("c%d" % n, Verdict.INCORRECT) for n in range(9, 10)])
        after = bank(*[("c%d" % n, Verdict.CORRECT) for n in range(10)])
        assert outcome(before, after).normalised_gain == pytest.approx(1.0)

    def test_half_the_available_room_reads_as_half(self) -> None:
        before = bank(("a", Verdict.CORRECT), ("b", Verdict.INCORRECT))
        after = bank(("a", Verdict.CORRECT), ("b", Verdict.INCORRECT))
        assert outcome(before, after).normalised_gain == pytest.approx(0.0)

        improved = bank(("a", Verdict.CORRECT), ("b", Verdict.CORRECT))
        assert outcome(before, improved).normalised_gain == pytest.approx(1.0)

    def test_a_learner_at_ceiling_is_unavailable_not_zero(self) -> None:
        # Nothing was available to gain, so the ratio is undefined. A zero would
        # report the absence of room as an absence of teaching, and averaging
        # those in is what makes a cohort look untaught.
        perfect = bank(("a", Verdict.CORRECT))
        assert outcome(perfect, perfect).normalised_gain is None

    def test_it_is_unavailable_when_a_bank_was_skipped(self) -> None:
        skipped = HeldOut(correct=0, total=0)
        assert outcome(skipped, skipped).normalised_gain is None

    def test_losing_ground_reads_as_negative(self) -> None:
        before = bank(("a", Verdict.CORRECT), ("b", Verdict.INCORRECT))
        after = bank(("a", Verdict.INCORRECT), ("b", Verdict.INCORRECT))
        normalised = outcome(before, after).normalised_gain
        assert normalised is not None and normalised < 0


class TestPerConceptChange:
    """What an aggregate cannot say.

    A sitting once read −13% and looked like the system harming the learner. It
    had fixed nothing and lost two, and every one of its training steps had gone
    somewhere else — visible only concept by concept.
    """

    def test_it_names_each_kind_of_movement(self) -> None:
        before = bank(
            ("fixed_one", Verdict.INCORRECT),
            ("lost_one", Verdict.CORRECT),
            ("stuck_one", Verdict.INCORRECT),
            ("kept_one", Verdict.CORRECT),
        )
        after = bank(
            ("fixed_one", Verdict.CORRECT),
            ("lost_one", Verdict.INCORRECT),
            ("stuck_one", Verdict.INCORRECT),
            ("kept_one", Verdict.CORRECT),
        )
        states = {c.concept_id: c.state for c in per_concept_change(before, after)}
        assert states == {
            "fixed_one": "fixed",
            "lost_one": "lost",
            "stuck_one": "still_wrong",
            "kept_one": "still_right",
        }

    def test_an_unreadable_end_is_not_a_learning_outcome(self) -> None:
        # Neither a failure nor a success: nothing was measured. Folding it into
        # "still wrong" would report a failure to parse as a failure to learn.
        before = bank(("a", Verdict.INCORRECT))
        after = bank(("a", Verdict.UNPARSEABLE))
        assert [c.state for c in per_concept_change(before, after)] == ["unmeasured"]

    def test_a_concept_in_only_one_bank_is_unmeasured(self) -> None:
        before = bank(("a", Verdict.INCORRECT))
        after = bank(("b", Verdict.CORRECT))
        states = {c.concept_id: c.state for c in per_concept_change(before, after)}
        assert states == {"a": "unmeasured", "b": "unmeasured"}


class TestDose:
    """Where the training time went — the one process measure a person gets.

    There is no ground-truth profile for a human, so ``remediation_ratio``
    reports unavailable. This does not need one: it is read from the audit log.
    """

    def test_steps_are_counted_per_concept(self) -> None:
        assert dose_by_concept(observed("a", "a", "b")) == {"a": 2, "b": 1}

    def test_seeding_is_not_instruction(self) -> None:
        # A held-out test moves posteriors without the learner having practised
        # anything. Counting it would report the model changing its mind as time
        # spent teaching.
        log = observed("a") + [
            AuditRecord(version=9, cause="seed", summary="",
                        evidence={"concept_id": "b", "verdict": "correct"})
        ]
        assert dose_by_concept(log) == {"a": 1}

    def test_decay_is_not_instruction_either(self) -> None:
        log = observed("a") + [
            AuditRecord(version=9, cause="decay", summary="",
                        evidence={"concept_id": "b", "elapsed_days": 30})
        ]
        assert dose_by_concept(log) == {"a": 1}

    def test_the_share_aimed_at_a_gap_is_reported(self) -> None:
        pretest = bank(("a", Verdict.INCORRECT), ("b", Verdict.CORRECT))
        assert dose_on_gap(observed("a", "a", "b", "b"), pretest) == pytest.approx(0.5)

    def test_the_failure_it_was_built_to_expose(self) -> None:
        # 21 of 24 steps on concepts the pre-test had already shown were fine.
        pretest = bank(("gap", Verdict.INCORRECT), ("known", Verdict.CORRECT))
        log = observed(*(["known"] * 21 + ["gap"] * 3))
        assert dose_on_gap(log, pretest) == pytest.approx(3 / 24)

    def test_it_is_unavailable_when_there_was_nothing_to_aim_at(self) -> None:
        # A share of zero would read as a failure to aim rather than as an
        # absence of targets.
        perfect = bank(("a", Verdict.CORRECT))
        assert dose_on_gap(observed("a"), perfect) is None

    def test_it_is_unavailable_without_a_pretest(self) -> None:
        assert dose_on_gap(observed("a"), HeldOut(correct=0, total=0)) is None

    def test_it_is_unavailable_when_no_training_happened(self) -> None:
        pretest = bank(("a", Verdict.INCORRECT))
        assert dose_on_gap([], pretest) is None
