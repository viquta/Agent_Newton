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


class TestAReturningLearnerRechecksTheRoute:
    """The same fifteen questions, every sitting, before anything else happens.

    Three cadences were possible and each fails somewhere. Sitting the whole
    bank once never re-measures what a gap between sittings costs — which is the
    thing a returning learner came back to find out. Sitting it every time is
    what happened, and it lets familiarity accumulate on exactly the items the
    outcome is read from, on top of being the longest part of a sitting that a
    person said was already too long. Narrowing it to the route measures what
    the sitting can actually move, and its cost is stated rather than hidden: a
    gap outside the route is not looked for, so it is not found.
    """

    def _resumed(self, toy, mastery: dict[str, float], **cohort):
        """A learner picking up with these posteriors already on the board."""
        config = config_for(
            cohort={"administer_tests": True, "max_items": 1, **cohort}
        )
        first = build_session("L0007", 20260812, toy, config)
        first.board.state.mastery.update(mastery)
        return build_session(
            "L0007",
            20260812,
            toy,
            config,
            state=first.board.state,
            profile=first.learner.profile,  # type: ignore[attr-defined]
        )

    @staticmethod
    def _route(toy, mastery: dict[str, float], config) -> set[str]:
        from agent_newton.core.state import route

        prior = bkt.initial(config.bkt)
        goal = route.next_goal(toy.concepts.goals(), mastery, config.zpd, prior)
        assert goal is not None
        return set(route.remaining(goal, mastery, toy.concepts, config.zpd, prior))

    def test_the_default_is_the_whole_bank(self) -> None:
        # Every measured result was produced under it, and a cohort must keep
        # it: an arm that sat a different instrument is not comparable.
        assert Config().cohort.pretest_scope == "full"

    def test_a_first_sitting_sits_all_of_it(self, toy) -> None:
        # There is no history to narrow against, and the baseline is what
        # everything after it is read against.
        config = config_for(
            cohort={"administer_tests": True, "max_items": 1, "pretest_scope": "route"}
        )
        outcome = build_session("L0007", 20260812, toy, config).run()
        assert outcome.pretest.total == len(toy.items.bank("pretest"))

    def test_a_returning_one_sits_only_the_route(self, toy) -> None:
        mastery = {"integer_arithmetic": 0.97, "combine_like_terms": 0.95}
        session = self._resumed(toy, mastery, pretest_scope="route")
        outcome = session.run()

        assert outcome.pretest.total < len(toy.items.bank("pretest"))
        assert outcome.pretest.covered <= self._route(toy, mastery, session.config)

    def test_what_it_leaves_out_is_what_was_already_mastered(self, toy) -> None:
        # The claim underneath the shorter bank: it is short because the learner
        # did the work, not because the measurement was trimmed arbitrarily.
        mastery = {"integer_arithmetic": 0.97, "combine_like_terms": 0.95}
        outcome = self._resumed(toy, mastery, pretest_scope="route").run()
        assert "integer_arithmetic" not in outcome.pretest.covered
        assert "combine_like_terms" not in outcome.pretest.covered

    def test_the_same_setting_off_sits_everything(self, toy) -> None:
        # The guard can fail: resuming is not on its own what shortens the bank.
        mastery = {"integer_arithmetic": 0.97, "combine_like_terms": 0.95}
        outcome = self._resumed(toy, mastery, pretest_scope="full").run()
        assert outcome.pretest.total == len(toy.items.bank("pretest"))

    def test_both_ends_measure_the_same_concepts(self, toy) -> None:
        # The load-bearing one. Training moves mastery, so a route recomputed
        # after it would narrow to something else — and the gain would be a
        # difference between two different instruments.
        mastery = {"integer_arithmetic": 0.97, "combine_like_terms": 0.95}
        outcome = self._resumed(
            toy, mastery, pretest_scope="route", max_items=6
        ).run()
        assert outcome.pretest.covered == outcome.posttest.covered
        assert outcome.pretest.total == outcome.posttest.total

    def test_a_learner_with_no_goal_left_sits_the_whole_bank(self, toy) -> None:
        # There is no route to narrow to, and what someone who has mastered
        # everything comes back to find out is what has gone stale.
        mastery = {concept_id: 0.99 for concept_id in toy.concepts.ids()}
        outcome = self._resumed(toy, mastery, pretest_scope="route").run()
        assert outcome.pretest.total == len(toy.items.bank("pretest"))

    def test_a_narrowing_that_would_empty_the_bank_is_ignored(self, toy) -> None:
        # An empty bank scores zero out of zero, which reads as "not
        # administered" — the measurement would disappear rather than shorten.
        from agent_newton.core.orchestration.session import Session

        session = self._resumed(toy, {}, pretest_scope="route")
        result = Session._administer(session, "pretest", frozenset({"no_such_concept"}))
        assert result.total == len(toy.items.bank("pretest"))

    def test_the_recheck_is_what_the_seeding_then_uses(self, toy) -> None:
        # Seeding folds the pre-test into the model. With a narrowed bank it
        # touches only the concepts sat, and the carried estimates for
        # everything else stay as the last sitting left them.
        mastery = {"integer_arithmetic": 0.97, "combine_like_terms": 0.95}
        session = self._resumed(
            toy, mastery, pretest_scope="route", seed_from_pretest=True
        )
        outcome = session.run()
        seeded = {
            r.evidence["concept_id"]
            for r in session.board.audit_log
            if r.cause == "seed"
        }
        assert seeded <= outcome.pretest.covered
        assert "integer_arithmetic" not in seeded


class TestTheBanksFollowTheGoalTheSittingWillWalk:
    """A request moves the goal, so it has to move the re-check with it.

    Otherwise the banks measure the route to one goal while the training walks
    the route to another: the gain is computed over concepts the sitting never
    touched, and every step it did teach falls outside `dose_on_gap`.
    """

    def _covered(self, toy, requested: list[str]) -> set[str]:
        config = config_for(
            cohort={
                "administer_tests": True,
                "max_items": 1,
                "pretest_scope": "route",
            }
        )
        first = build_session("L0007", 20260812, toy, config)
        first.board.state.mastery.update({"integer_arithmetic": 0.97})
        session = build_session(
            "L0007", 20260812, toy, config,
            state=first.board.state,
            profile=first.learner.profile,  # type: ignore[attr-defined]
        )
        session.board.record_request(requested)
        return set(session.run().pretest.covered)

    def test_a_request_that_changes_nothing_leaves_the_bank_alone(self, toy) -> None:
        # toy_algebra declares one goal, so nothing can move — which makes this
        # the control rather than the claim.
        assert self._covered(toy, ["distribute"]) == self._covered(toy, [])

    def test_the_bank_covers_the_route_that_will_be_walked(self, toy) -> None:
        from agent_newton.core.state import route

        covered = self._covered(toy, ["distribute"])
        goal = toy.concepts.goals()[0]
        assert covered <= route.relevant(goal, toy.concepts)
        assert covered, "the re-check measured nothing"


class TestAFinishedWalkIsNotAFinishedLearner:
    """⚠️ A resumed decoupled learner was told it had reached every goal.

    Its walk position is the only progress signal it has, so a session picking
    up past the last goal gets no goal at all from its planner — whatever the
    learner actually knows. The session read that as "every goal reached", so a
    learner who attempted nothing was reported as having finished the syllabus
    while having mastered none of it.

    Only one of the two planners can tell the difference, which is why the claim
    is checked against the state instead. ``goals_mastered`` is derived from
    mastery and means the same in both arms; a planner's opinion of its own
    position does not.
    """

    def _walk_past_the_end(self, toy, arm: str = "decoupled"):
        """A learner who knows nothing, whose planner has nowhere left to go."""
        config = config_for(arm, cohort={"max_items": 4, "administer_tests": False})
        session = build_session(
            "L0007", 20260812, toy, config, planner_state={"position": 99}
        )
        return session, session.run()

    def test_it_is_not_reported_as_having_finished(self, toy) -> None:
        session, outcome = self._walk_past_the_end(toy)
        assert outcome.goals_mastered == 0, "this learner should know nothing"
        assert outcome.stop_reason != "every_goal_reached"
        del session

    def test_the_log_says_what_actually_happened(self, toy) -> None:
        session, _ = self._walk_past_the_end(toy)
        assert any("run out of syllabus" in r.summary for r in session.board.audit_log)

    def test_a_learner_who_really_has_finished_still_says_so(self, toy) -> None:
        # The guard can fail: the claim must survive where it is true. Every
        # goal above the band before the session starts, so the first thing the
        # planner is asked returns nothing — for the right reason this time.
        config = config_for(cohort={"max_items": 4, "administer_tests": False})
        finished = build_session("L0007", 20260812, toy, config)
        for goal in toy.concepts.goals():
            finished.board.state.mastery[goal] = 0.99

        session = build_session(
            "L0007", 20260812, toy, config,
            state=finished.board.state,
            profile=finished.learner.profile,  # type: ignore[attr-defined]
        )
        outcome = session.run()
        assert outcome.goals_mastered == len(toy.concepts.goals())
        assert outcome.stop_reason == "every_goal_reached"
