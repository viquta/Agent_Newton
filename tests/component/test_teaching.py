"""The teaching record.

What the system did about one skill, across sittings. It exists because a
learner who never grasps a concept despite sustained appropriate teaching is an
ordinary pedagogical case, and the only thing that can be established either way
is whether the instruction was appropriate — a claim about the system, not the
learner, and therefore one a simulated learner and a person can both support.

The test that matters most here is that ``not_attempted`` can be non-empty. A
record that could only ever report success would be a log, and the claim it
carries would be unfalsifiable.
"""

from __future__ import annotations

import json

import pytest

from agent_newton.core.evaluation.teaching import (
    TUTOR_CAUSE,
    records,
    repertoire,
    summarise,
)
from agent_newton.core.state.schema import AuditRecord, LearnerState
from agent_newton.store import LearnerStore


def observation(concept: str, verdict: str, before: float, after: float, item: str = "i1"):
    return AuditRecord(
        version=1,
        cause="observation",
        summary="",
        evidence={
            "item_id": item, "concept_id": concept, "verdict": verdict,
            "mastery_before": before, "mastery_after": after,
        },
    )


def turn(concept: str, move: str, level: str, targets: str | None = None):
    return AuditRecord(
        version=1,
        cause=TUTOR_CAUSE,
        summary="",
        evidence={
            "item_id": "i1", "concept_id": concept,
            "move": move, "level": level, "targets": targets, "text": "…",
        },
    )


def unreadable(concept: str, item: str = "i1"):
    # No mastery keys, on purpose: nothing moved, so nothing is recorded.
    return AuditRecord(
        version=1, cause="observation", summary="",
        evidence={"item_id": item, "concept_id": concept, "verdict": "unparseable"},
    )


@pytest.fixture
def store(tmp_path):
    with LearnerStore(tmp_path / "learners.db") as opened:
        opened.ensure_learner("L1", "simulated", "calculus")
        yield opened


def sit(store, *log, seq_gap: float = 0.0, learner: str = "L1", arm: str = "coupled"):
    """Write one sitting with this audit log into the store."""
    session_id = store.open_session(
        learner_id=learner, arm=arm, config_hash="h", elapsed_days=seq_gap
    )
    store.close_session(
        session_id, state=LearnerState(learner_id=learner, seed=1), audit_log=log
    )


class TestTheRepertoireComesFromTheCode:
    def test_it_names_every_level_and_move(self) -> None:
        # Derived from the enums so it cannot drift from what the tutor can do.
        assert "level:nudge" in repertoire()
        assert "level:worked_step" in repertoire()
        assert "move:reflect" in repertoire()
        assert len(repertoire()) == 6


class TestItCanSayTheSystemDidNotTryEverything:
    """The property that makes this a measure rather than a log.

    A learner stuck on a concept who never received a worked step is a failure
    of the system. If this could not be reported, the record could only ever
    say the teaching was appropriate.
    """

    def test_a_level_never_reached_is_reported(self, store) -> None:
        sit(
            store,
            observation("power_rule", "incorrect", 0.15, 0.10),
            turn("power_rule", "hint", "nudge"),
        )
        [found] = records(store, "L1", "coupled")
        assert found.not_attempted is not None
        assert "level:worked_step" in found.not_attempted
        assert "level:targeted" in found.not_attempted
        assert found.exhausted_repertoire is False

    def test_reaching_everything_leaves_nothing_outstanding(self, store) -> None:
        sit(
            store,
            observation("power_rule", "incorrect", 0.15, 0.10),
            *[turn("power_rule", m, level) for m in ("hint", "reflect", "remediate")
              for level in ("nudge", "targeted", "worked_step")],
        )
        [found] = records(store, "L1", "coupled")
        assert found.not_attempted == frozenset()
        assert found.exhausted_repertoire is True

    def test_what_was_used_is_reported_too(self, store) -> None:
        sit(
            store,
            observation("power_rule", "incorrect", 0.15, 0.10),
            turn("power_rule", "remediate", "worked_step", targets="power_rule_forgets_decrement"),
        )
        [found] = records(store, "L1", "coupled")
        assert "move:remediate" in found.attempted
        assert found.remediation_targets == {"power_rule_forgets_decrement": 1}


class TestUnrecordedIsNotUntried:
    """Turns were not kept before 2026-08-12, and five sittings predate that.

    Reporting every move as unattempted for those would be a claim about the
    teaching when the truth is that the teaching was not written down — the same
    distinction a skipped test bank draws against a bank scored zero.
    """

    def test_a_history_with_no_turns_reports_unavailable(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10))
        [found] = records(store, "L1", "coupled")
        assert found.instruction_recorded is False
        assert found.not_attempted is None
        assert found.exhausted_repertoire is None

    def test_one_recorded_sitting_makes_the_history_measurable(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10))
        sit(store, observation("power_rule", "incorrect", 0.10, 0.08),
            turn("power_rule", "hint", "nudge"))
        [found] = records(store, "L1", "coupled")
        assert found.instruction_recorded is True
        assert found.not_attempted is not None
        # The unrecorded sitting is still marked, so a reader can see which half
        # of the history the repertoire figure is based on.
        assert [s.instruction_recorded for s in found.sittings] == [False, True]


class TestOnlyInstructionCounts:
    """Seeding and decay move the estimate without anyone being taught."""

    def test_seeding_is_not_an_attempt(self, store) -> None:
        sit(
            store,
            AuditRecord(version=1, cause="seed", summary="", evidence={
                "concept_id": "power_rule", "verdict": "correct",
                "mastery_before": 0.15, "mastery_after": 0.94}),
        )
        [found] = records(store, "L1", "coupled")
        assert found.attempts == 0
        assert found.sittings[0].seeded == 1
        # But the movement is still visible — a concept that rose without being
        # taught is exactly what a reader needs to see.
        assert found.movement == pytest.approx(0.79)

    def test_decay_is_not_an_attempt(self, store) -> None:
        sit(
            store,
            AuditRecord(version=1, cause="decay", summary="", evidence={
                "concept_id": "power_rule", "elapsed_days": 30.0,
                "mastery_before": 0.94, "mastery_after": 0.33}),
        )
        [found] = records(store, "L1", "coupled")
        assert found.attempts == 0
        assert found.sittings[0].decayed == 1

    def test_an_unreadable_step_is_not_a_wrong_answer(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10),
            unreadable("power_rule"))
        [found] = records(store, "L1", "coupled")
        assert found.attempts == 2
        assert found.correct == 0
        assert found.sittings[0].unmeasurable == 1


class TestAcrossSittings:
    """The case this exists for: sustained teaching, nothing moving."""

    def test_the_trajectory_spans_the_history(self, store) -> None:
        sit(store, observation("product_rule", "incorrect", 0.15, 0.12))
        sit(store, observation("product_rule", "incorrect", 0.12, 0.11), seq_gap=30.0)
        sit(store, observation("product_rule", "incorrect", 0.11, 0.10), seq_gap=30.0)

        [found] = records(store, "L1", "coupled")
        assert found.sittings_spanned == 3
        assert found.attempts == 3
        assert [seq for seq, _ in found.trajectory] == [0, 1, 2]
        assert found.movement == pytest.approx(-0.05)

    def test_the_first_gap_is_not_time_spent_teaching(self, store) -> None:
        # A learner returning after 30 days and being taught something new has
        # not been taught it for 30 days.
        sit(store, observation("product_rule", "incorrect", 0.15, 0.12), seq_gap=30.0)
        sit(store, observation("product_rule", "incorrect", 0.12, 0.11), seq_gap=7.0)
        [found] = records(store, "L1", "coupled")
        assert found.days_spanned == 7.0

    def test_effort_and_movement_are_reported_separately(self, store) -> None:
        # The figure this affords: far right, flat. Much teaching, no movement.
        for _ in range(20):
            sit(store, observation("product_rule", "incorrect", 0.15, 0.15))
        [found] = records(store, "L1", "coupled")
        assert found.attempts == 20
        assert found.movement == pytest.approx(0.0)


class TestScoping:
    def test_a_concept_never_touched_has_no_record(self, store) -> None:
        # Not an empty record: the absence of teaching and a record of no
        # teaching are different claims.
        sit(store, observation("power_rule", "correct", 0.15, 0.40))
        assert [r.concept_id for r in records(store, "L1", "coupled")] == ["power_rule"]

    def test_it_can_be_narrowed_to_one_concept(self, store) -> None:
        sit(store, observation("power_rule", "correct", 0.15, 0.40),
            observation("chain_rule", "incorrect", 0.15, 0.10))
        found = records(store, "L1", "coupled", concepts=["chain_rule"])
        assert [r.concept_id for r in found] == ["chain_rule"]

    def test_arms_are_separate_histories(self, store) -> None:
        # The same learner under two architectures is two histories; mixing them
        # would let one arm inherit the other's teaching.
        sit(store, observation("power_rule", "correct", 0.15, 0.40), arm="coupled")
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10), arm="decoupled")
        assert records(store, "L1", "coupled")[0].correct == 1
        assert records(store, "L1", "decoupled")[0].correct == 0


class TestTheSummary:
    def test_it_carries_the_repertoire_and_the_gaps(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10),
            turn("power_rule", "hint", "nudge"))
        summary = summarise(records(store, "L1", "coupled"))
        assert summary["concepts"] == 1
        assert summary["instruction_recorded"] is True
        assert summary["repertoire"] == list(repertoire())
        row = summary["per_concept"]["power_rule"]
        assert "level:worked_step" in row["not_attempted"]
        assert row["exhausted_repertoire"] is False

    def test_an_unrecorded_history_summarises_as_unavailable(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10))
        summary = summarise(records(store, "L1", "coupled"))
        assert summary["instruction_recorded"] is False
        assert summary["per_concept"]["power_rule"]["not_attempted"] is None

    def test_it_serialises(self, store) -> None:
        sit(store, observation("power_rule", "incorrect", 0.15, 0.10))
        json.dumps(summarise(records(store, "L1", "coupled")))
