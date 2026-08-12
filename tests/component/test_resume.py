"""Picking a learner up where they were left.

The property the whole sequence design rests on: **persistence must not change
behaviour, only span it.** A session resumed from stored state has to behave
exactly as one that never stopped, or every multi-session result is measuring
the act of stopping rather than the architecture.
"""

from __future__ import annotations

import pytest

from agent_newton.config import Config, SimulatorConfig
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.simulator import sample_profile
from agent_newton.core.state import bkt
from agent_newton.core.state.store import new_blackboard, resumed_blackboard
from agent_newton.domains import registry
from agent_newton.domains.base import VerificationResult, Verdict


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def config_for(arm: str = "coupled", **overrides) -> Config:
    return Config.model_validate(
        {
            "domain": "toy_algebra",
            "arm": arm,
            "simulator": {"surface": "symbolic"},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed"},
            },
            **overrides,
        }
    )


class TestResumingCarriesTheRightThings:
    def test_the_state_is_the_one_handed_in(self, toy) -> None:
        config = config_for()
        first = new_blackboard("L1", 1, toy.concepts, config)
        first.record_observation(
            item_id="i1",
            concept_id="distribute",
            result=VerificationResult(Verdict.CORRECT, "a"),
        )

        second = resumed_blackboard(first.state, toy.concepts, config)
        assert second.state.mastery == first.state.mastery
        assert second.state.outcomes == first.state.outcomes
        assert second.state.t == first.state.t

    def test_the_audit_log_starts_fresh(self, toy) -> None:
        # It is the account of one sitting. Carrying it would make every run's
        # log grow without bound and stop being readable as a session.
        config = config_for()
        first = new_blackboard("L1", 1, toy.concepts, config)
        first.record_observation(
            item_id="i1",
            concept_id="distribute",
            result=VerificationResult(Verdict.CORRECT, "a"),
        )

        second = resumed_blackboard(first.state, toy.concepts, config)
        assert len(second.audit_log) == 1
        assert second.audit_log[0].summary.startswith("resumed learner")

    def test_the_version_counter_continues(self, toy) -> None:
        # So a version still orders events across a learner's whole history,
        # not just within one sitting.
        config = config_for()
        first = new_blackboard("L1", 1, toy.concepts, config)
        first.record_observation(
            item_id="i1",
            concept_id="distribute",
            result=VerificationResult(Verdict.CORRECT, "a"),
        )
        before = first.version

        second = resumed_blackboard(first.state, toy.concepts, config)
        assert second.version > before

    def test_the_outcome_stream_is_learner_state(self, toy) -> None:
        # It is the whole of what the decoupled view carries. If it lived on the
        # blackboard, the decoupled arm would restart its run of consecutive
        # answers every session while the coupled arm resumed with everything —
        # one arm handed persistence, which is not the manipulation.
        config = config_for(arm="decoupled")
        first = new_blackboard("L1", 1, toy.concepts, config)
        for _ in range(3):
            first.record_observation(
                item_id="i1",
                concept_id="distribute",
                result=VerificationResult(Verdict.CORRECT, "a"),
            )

        second = resumed_blackboard(first.state, toy.concepts, config)
        assert second.view("decoupled").consecutive_correct() == 3


class TestResumingDoesNotChangeBehaviour:
    """The load-bearing property, checked end to end on both arms."""

    def _uninterrupted(self, toy, arm: str, max_items: int):
        config = config_for(arm, cohort={"max_items": max_items, "administer_tests": False})
        session = build_session("L0007", 20260812, toy, config)
        session.run()
        return session.board.state

    def _split(self, toy, arm: str, first: int, second: int):
        from agent_newton.core.agents.base import Resumable

        config = config_for(arm, cohort={"max_items": first, "administer_tests": False})
        one = build_session("L0007", 20260812, toy, config)
        one.run()

        # Three things travel together, and the split fails if any is left out:
        # the state is what the system believes, the profile is what is true,
        # and the planner snapshot is where the decoupled walk had got to.
        planner_state = (
            one.planner.snapshot() if isinstance(one.planner, Resumable) else None
        )
        resumed = build_session(
            "L0007",
            20260812,
            toy,
            config_for(arm, cohort={"max_items": second, "administer_tests": False}),
            state=one.board.state,
            profile=one.learner.profile,  # type: ignore[attr-defined]
            planner_state=planner_state,
        )
        resumed.run()
        return resumed.board.state

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_a_split_session_matches_an_uninterrupted_one(self, toy, arm: str) -> None:
        whole = self._uninterrupted(toy, arm, 8)
        split = self._split(toy, arm, 4, 4)

        assert split.mastery == pytest.approx(whole.mastery)
        assert split.outcomes == whole.outcomes
        assert split.t == whole.t

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_the_same_work_was_done(self, toy, arm: str) -> None:
        whole = self._uninterrupted(toy, arm, 8)
        split = self._split(toy, arm, 4, 4)
        assert [e.item_id for e in split.error_trace] == [
            e.item_id for e in whole.error_trace
        ]

    def test_the_check_would_notice_a_difference(self, toy) -> None:
        # A guard that cannot fail proves nothing: two genuinely different
        # amounts of work must not compare equal.
        assert self._uninterrupted(toy, "coupled", 8).mastery != pytest.approx(
            self._uninterrupted(toy, "coupled", 2).mastery
        )


class TestResumingWithAGap:
    def test_no_gap_leaves_the_model_alone(self, toy) -> None:
        # A sequence with no elapsed time must reproduce sessions run back to
        # back, or decay is manufacturing an effect.
        config = config_for(
            cohort={"max_items": 4, "administer_tests": False},
            decay={"half_life_days": 30.0},
        )
        first = build_session("L0007", 20260812, toy, config)
        first.run()
        before = dict(first.board.state.mastery)

        second = build_session(
            "L0007", 20260812, toy, config,
            state=first.board.state,
            profile=first.learner.profile,  # type: ignore[attr-defined]
            elapsed_days=0.0,
        )
        assert second.board.apply_decay(0.0) == 0
        assert dict(second.board.state.mastery) == before

    def test_a_gap_ages_the_model_before_anything_is_measured(self, toy) -> None:
        config = config_for(
            cohort={"max_items": 4, "administer_tests": False},
            decay={"half_life_days": 10.0},
        )
        first = build_session("L0007", 20260812, toy, config)
        first.run()
        worked = [c for c, p in first.board.state.mastery.items() if p > 0.5]
        assert worked, "the first session mastered nothing to decay"

        second = build_session(
            "L0007", 20260812, toy, config,
            state=first.board.state,
            profile=first.learner.profile,  # type: ignore[attr-defined]
            elapsed_days=60.0,
        )
        before = dict(second.board.state.mastery)
        second.run()

        decayed = [r for r in second.board.audit_log if r.cause == "decay"]
        assert decayed, "the gap moved nothing"
        # The decay entries precede every observation, so what the pre-test
        # measures is the learner as they are now.
        first_observation = next(
            (i for i, r in enumerate(second.board.audit_log) if r.cause == "observation"),
            len(second.board.audit_log),
        )
        assert all(
            i < first_observation
            for i, r in enumerate(second.board.audit_log)
            if r.cause == "decay"
        )
        # Asserted on the decay records, not on mastery after the run: the
        # session's own practice moves the estimates back up, so comparing
        # before and after would conflate ageing with teaching and could pass
        # for the wrong reason.
        assert all(
            r.evidence["mastery_after"] < r.evidence["mastery_before"]
            for r in decayed
            if r.evidence["mastery_before"] > bkt.initial(config.bkt)
        )
        assert {r.evidence["concept_id"] for r in decayed} >= set(worked)
        assert before is not None

    def test_the_observer_is_told_only_when_something_moved(self, toy) -> None:
        from agent_newton.core.orchestration.session import Watching

        class Watcher(Watching):
            def __init__(self) -> None:
                self.resumed: list[tuple[float, int]] = []

            def session_resumed(self, elapsed_days, concepts_decayed) -> None:  # noqa: ANN001
                self.resumed.append((elapsed_days, concepts_decayed))

        config = config_for(cohort={"max_items": 4, "administer_tests": False})
        watcher = Watcher()
        build_session("L0007", 20260812, toy, config, observer=watcher).run()
        assert watcher.resumed == [], "a first session has no gap to report"


class TestAProfileMustTravelWithTheState:
    def test_a_resumed_profile_keeps_its_remediation(self, toy) -> None:
        # Resuming the belief without the ground truth would put a model that
        # remembers alongside a learner who starts over.
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        label = next(iter(profile.firing))
        profile.remediate(label, 0.5)

        session = build_session(
            "L1", 1, toy, config_for(cohort={"max_items": 1, "administer_tests": False}),
            profile=profile,
        )
        assert session.learner.profile is profile  # type: ignore[attr-defined]
        assert session.learner.profile.firing[label] == profile.firing[label]  # type: ignore[attr-defined]

    def test_without_one_a_fresh_profile_is_drawn(self, toy) -> None:
        # Which is what a first session must do, and is why single-session runs
        # are unchanged by any of this.
        config = config_for(cohort={"max_items": 1, "administer_tests": False})
        drawn = build_session("L1", 1, toy, config).learner.profile  # type: ignore[attr-defined]
        expected = sample_profile("L1", 1, toy.misconceptions, config.simulator)
        assert drawn.firing == expected.firing
