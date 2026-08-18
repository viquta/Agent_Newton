"""The milestone: a full session with no model involved, on both domains.

This is the checkpoint that validates the harness — loop, state, verifier,
pedagogy rules, outcome measurement — before any model noise enters. Everything
here must hold without inference, and fast enough to run in CI.
"""

from __future__ import annotations

from collections import Counter

import pytest

from agent_newton.config import Config
from agent_newton.core.state import bkt
from agent_newton.core.orchestration.session import (
    NotImplementedForModels,
    build_session,
)
from agent_newton.domains import registry

DOMAINS = ("toy_algebra", "calculus")


def config_for(domain: str, arm: str, **overrides) -> Config:
    return Config.model_validate(
        {
            "domain": domain,
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


def run(domain_name: str, arm: str, learner_id: str = "L0000", **overrides):
    config = config_for(domain_name, arm, **overrides)
    domain = registry.load_domain(domain_name)
    session = build_session(learner_id, config.seed, domain, config)
    return session, session.run()


def run_cohort(domain_name: str, arm: str, n: int = 6, **overrides):
    """Several learners.

    Needed because a single learner's misconceptions may sit deep in the
    syllabus and never be reached within the item budget — legitimate
    behaviour, but it makes any assertion about errors vacuous for that
    learner.
    """
    return [run(domain_name, arm, f"L{i:04d}", **overrides) for i in range(n)]


class TestRunsWithoutAModel:
    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_a_session_completes(self, domain_name: str, arm: str) -> None:
        config = config_for(domain_name, arm)
        assert not config.uses_llm(), "this configuration must invoke no model"
        _, outcome = run(domain_name, arm)
        assert outcome.items_attempted > 0

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_session_ends_at_budget_or_by_mastery(self, domain_name: str) -> None:
        # A planner that runs out of *unseen* items must repeat rather than
        # stop, since mastery takes several correct answers. Stopping early is
        # legitimate only when there is genuinely nothing left to teach.
        config = config_for(domain_name, "coupled")
        _, outcome = run(domain_name, "coupled")
        if outcome.items_attempted < config.cohort.max_items:
            assert outcome.items_to_exhaustion is not None, (
                "stopped early without the frontier emptying"
            )

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_tests_are_administered_at_both_ends(self, domain_name: str) -> None:
        _, outcome = run(domain_name, "coupled")
        assert outcome.pretest.total > 0
        assert outcome.posttest.total > 0

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_no_step_is_unmeasurable(self, domain_name: str) -> None:
        # Every response the simulator produces comes from a buggy rule or is
        # the stated answer, so all of it must be readable. Anything else means
        # the verifier and the domain content disagree.
        _, outcome = run(domain_name, "coupled")
        assert outcome.unmeasurable_steps == 0


class TestTheLoopBehaves:
    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_audit_log_records_the_session(self, domain_name: str) -> None:
        session, _ = run(domain_name, "coupled")
        log = session.board.audit_log
        assert log
        assert any(r.cause == "observation" for r in log)
        assert all(r.version > 0 for r in log)

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_no_pedagogy_rule_is_violated(self, domain_name: str) -> None:
        # The tutor is driven by the rules rather than checked against them, so
        # a violation here means that wiring has come apart.
        session, _ = run(domain_name, "coupled")
        assert not [r for r in session.board.audit_log if "violation" in r.summary]

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_oracle_diagnostic_is_perfect(self, domain_name: str) -> None:
        # Anything else means the ground-truth channel is wired wrongly, which
        # would silently corrupt every diagnostic measurement downstream.
        pairs = [
            pair for _, outcome in run_cohort(domain_name, "coupled")
            for pair in outcome.diagnoses
        ]
        assert pairs, "no learner in the cohort made a diagnosable error"
        assert all(injected == inferred for injected, inferred in pairs)

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_remediation_happens(self, domain_name: str) -> None:
        ratios = [o.remediation_ratio for _, o in run_cohort(domain_name, "coupled")]
        # A simulated learner always has a profile, so a None here would mean the
        # cohort ran something that cannot be measured.
        assert all(r is not None for r in ratios)
        assert any(r > 0.0 for r in ratios if r is not None)

    def test_a_misdiagnosing_tutor_remediates_less(self) -> None:
        # The causal chain the whole experiment rests on: diagnostic error means
        # misaimed hints, and misaimed hints do no work.
        _, accurate = run("toy_algebra", "coupled")
        _, noisy = run(
            "toy_algebra",
            "coupled",
            agents={
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "noised_oracle", "noise_rate": 0.9},
                "planner": {"impl": "goal_directed"},
            },
        )
        assert noisy.remediation_ratio is not None
        assert accurate.remediation_ratio is not None
        assert noisy.remediation_ratio < accurate.remediation_ratio


class TestTheDwellingCap:
    """Off by default, and it must stay exactly off.

    ``consolidate`` ranks by recent errors, so a learner who keeps failing a
    concept keeps being given it. That is what consolidation means; what it
    lacks is a floor. With the pre-test now skipping demonstrated concepts there
    are fewer places left to be moved along to, and a verification run put every
    one of its sixty steps on a single concept.
    """

    def _concepts_worked(self, session) -> dict[str, int]:  # noqa: ANN001
        from agent_newton.core.evaluation.outcomes import dose_by_concept

        return dose_by_concept(session.board.audit_log)

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_the_default_changes_nothing(self, domain_name: str, arm: str) -> None:
        # The whole point of a swept parameter whose default is today: every
        # number already measured must be reproduced by the code that has it.
        _, capped = run(domain_name, arm, cohort={"max_visits_per_concept": None})
        _, plain = run(domain_name, arm)
        assert capped.items_attempted == plain.items_attempted
        assert capped.remediation_ratio == plain.remediation_ratio
        assert capped.goals_mastered == plain.goals_mastered
        assert capped.posttest.correct == plain.posttest.correct

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_nothing_is_set_aside_without_a_cap(self, domain_name: str) -> None:
        session, _ = run(domain_name, "coupled")
        assert session.board.weaknesses == frozenset()

    def test_a_stuck_learner_is_moved_along(self) -> None:
        # A person who answers nothing correctly. Uncapped, consolidation keeps
        # returning to whatever they last failed.
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.simulator.human import HumanLearner

        def stuck(**cohort):
            config = config_for(
                "calculus", "coupled",
                cohort={"max_items": 30, "administer_tests": False, **cohort},
            )
            domain = registry.load_domain("calculus")
            learner = HumanLearner(lambda item, attempt: "999")
            session = build_session(
                "L0000", config.seed, domain, config, learner=learner
            )

            class Nothing:
                def diagnose(self, item, response, domain):  # noqa: ANN001
                    return Diagnosis(None)

            session.diagnostic = Nothing()
            session.run()
            return session

        uncapped = self._concepts_worked(stuck())
        capped = self._concepts_worked(stuck(max_visits_per_concept=3))
        assert max(uncapped.values()) > max(capped.values())
        assert len(capped) > len(uncapped), "the cap reached no new material"

    def test_the_weakness_is_recorded_and_auditable(self) -> None:
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.simulator.human import HumanLearner

        config = config_for(
            "calculus", "coupled",
            cohort={
                "max_items": 30, "administer_tests": False,
                "max_visits_per_concept": 3,
            },
        )
        domain = registry.load_domain("calculus")
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session("L0000", config.seed, domain, config, learner=learner)

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        session.diagnostic = Nothing()
        session.run()

        assert session.board.weaknesses
        noted = [r for r in session.board.audit_log if r.evidence.get("weakness")]
        assert noted
        for record in noted:
            assert record.evidence["visits"] >= 3
            assert record.evidence["concept_id"] in session.board.weaknesses

    def test_the_set_aside_concept_is_still_reachable_at_the_end(self) -> None:
        # Set aside, not withdrawn. A learner whose only remaining work is the
        # thing they are stuck on must still be given work rather than none.
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.simulator.human import HumanLearner

        config = config_for(
            "calculus", "coupled",
            cohort={
                "max_items": 60, "administer_tests": False,
                "max_visits_per_concept": 2,
            },
        )
        domain = registry.load_domain("calculus")
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session("L0000", config.seed, domain, config, learner=learner)

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        session.diagnostic = Nothing()
        outcome = session.run()
        # The budget is what ends it, not an empty frontier: every concept
        # having been set aside must not read as nothing left to teach.
        assert outcome.stop_reason == "budget_spent"
        assert outcome.items_attempted == 60


class TestTheHeldOutBanksAreAlwaysReadable:
    """What makes the measured score inert for every simulated result.

    ``gain`` is taken over what the verifier could read rather than over what
    was administered. For a person the two differ; for a simulated learner they
    cannot, because ``domain validate`` admits no item whose stated answer fails
    to verify correct and no buggy rule whose output fails to verify incorrect —
    so the only two responses a simulated learner can give are both readable.

    Asserted rather than reasoned about. If the bank ever drifts from what was
    validated, this is what says so, and until then it is the proof that a
    change to the denominator moved no cohort number.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_no_test_answer_is_unreadable(self, domain_name: str, arm: str) -> None:
        for _, outcome in run_cohort(domain_name, arm):
            assert outcome.pretest.unmeasurable == 0
            assert outcome.posttest.unmeasurable == 0

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_so_the_two_scores_agree(self, domain_name: str) -> None:
        for _, outcome in run_cohort(domain_name, "coupled"):
            assert outcome.pretest.score == outcome.pretest.measured_score
            assert outcome.posttest.score == outcome.posttest.measured_score

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_normalised_gain_is_available_for_someone(self, domain_name: str) -> None:
        # It is None only at ceiling. A cohort where it is None for everyone
        # would mean the pre-test has no room in it at all, and the gain outcome
        # would be measuring nothing.
        normalised = [o.normalised_gain for _, o in run_cohort(domain_name, "coupled")]
        assert any(g is not None for g in normalised)


class TestTheTutorsTurnsAreKept:
    """What the system said back is half of a sitting, and it was not recorded.

    A transcript held every answer the learner gave and nothing the tutor
    replied, so there was no way to read back what a person had been taught —
    including for the sittings whose whole purpose was to judge the tutoring.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_every_turn_reaches_the_audit_log(self, domain_name: str) -> None:
        # A cohort rather than one learner. L0000's misconceptions may sit deep
        # enough in the syllabus never to be reached inside the item budget, so
        # the tutor is never called and the assertion becomes vacuous — which is
        # what `run_cohort` exists for, and what happened when a sixteenth
        # catalogue entry redrew every profile.
        turns = [
            r
            for session, _ in run_cohort(domain_name, "coupled")
            for r in session.board.audit_log
            if r.cause == "tutor"
        ]
        assert turns, "the tutor spoke and none of it was recorded"
        for record in turns:
            assert record.evidence["text"]
            assert record.evidence["move"] in ("hint", "reflect", "remediate")
            assert record.evidence["level"] in ("nudge", "targeted", "worked_step")

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_a_turn_moves_no_estimate(self, domain_name: str) -> None:
        # A hint is instruction. Whether it worked shows up in the next graded
        # step, not in the record of having said it.
        session, _ = run(domain_name, "coupled")
        for record in session.board.audit_log:
            if record.cause == "tutor":
                assert "mastery_after" not in record.evidence

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_only_remediation_carries_a_target(self, domain_name: str) -> None:
        # The error-first rule's price: a reflective prompt costs a turn and
        # teaches nothing, so it aims at nothing.
        session, _ = run(domain_name, "coupled")
        for record in session.board.audit_log:
            if record.cause == "tutor" and record.evidence["move"] != "remediate":
                assert record.evidence["targets"] is None

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_recording_them_is_not_counted_as_a_replan(self, domain_name: str) -> None:
        # `_trigger_counts` reads the audit log by cause, and the threshold
        # sweep reads those counts.
        session, outcome = run(domain_name, "coupled")
        replans = sum(
            1 for r in session.board.audit_log if r.cause == "replan"
        )
        assert sum(outcome.triggers.values()) == replans

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_recording_them_changes_no_outcome(self, domain_name: str, arm: str) -> None:
        """Turns bump the state version, and nothing may depend on its value.

        The frontier cache keys on the version, and a plan records the version
        it was set at. Neither is read as a quantity, but the whole comparison
        would be worthless if one were — so it is asserted rather than assumed.
        """
        from agent_newton.core.state.store import Blackboard

        original = Blackboard.record_turn
        try:
            Blackboard.record_turn = lambda self, **kwargs: None  # type: ignore[assignment]
            _, silent = run(domain_name, arm)
        finally:
            Blackboard.record_turn = original  # type: ignore[assignment]
        _, recorded = run(domain_name, arm)

        assert recorded.items_attempted == silent.items_attempted
        assert recorded.remediation_ratio == silent.remediation_ratio
        assert recorded.goals_mastered == silent.goals_mastered
        assert recorded.distance_to_goal == silent.distance_to_goal
        assert recorded.pretest.correct == silent.pretest.correct
        assert recorded.posttest.correct == silent.posttest.correct
        assert recorded.triggers == silent.triggers


class TestTheArmsDiffer:
    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_they_select_different_work(self, domain_name: str) -> None:
        # If the two planners produced identical sequences the comparison would
        # be measuring nothing, however clean the rest of the machinery is.
        sequences = {}
        for arm in ("coupled", "decoupled"):
            session, _ = run(domain_name, arm, cohort={"max_items": 20})
            sequences[arm] = [
                r.evidence.get("item_id")
                for r in session.board.audit_log
                if r.cause == "observation"
            ]
        assert sequences["coupled"] != sequences["decoupled"]

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_both_face_the_same_learner(self, domain_name: str) -> None:
        # Pairing: same seed, same profile, so the comparison is within-learner.
        from agent_newton.core.simulator import SimulatedLearner

        coupled, _ = run(domain_name, "coupled")
        decoupled, _ = run(domain_name, "decoupled")
        assert isinstance(coupled.learner, SimulatedLearner)
        assert isinstance(decoupled.learner, SimulatedLearner)
        assert coupled.learner.profile.initial == decoupled.learner.profile.initial


class TestTheSupportLevelCannotMoveACohort:
    """Which rung the ladder stops on changes the words and nothing else.

    Worth proving rather than assuming, because it is what makes the scaffolding
    rule safe to correct: a hint reaches a simulated learner through
    ``receive_hint``, which is handed the misconception the hint *targets* and
    not the level it was pitched at. So remediation — the only channel by which
    tutoring changes a simulated outcome — is level-independent.

    Forced to either extreme rather than compared against a stored baseline: a
    baseline would only say the numbers did not move this time, while this says
    the level cannot move them at all.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_forcing_it_high_or_low_changes_nothing(
        self, domain_name: str, arm: str, monkeypatch
    ) -> None:
        from agent_newton.core.agents import tutor as tutor_module
        from agent_newton.core.pedagogy import HintLevel

        def run_at(level: HintLevel):
            monkeypatch.setattr(
                tutor_module, "hint_level", lambda mastery, prior, band, **_: level
            )
            return [
                run(domain_name, arm, f"L{i:04d}")[1] for i in range(4)
            ]

        lowest = run_at(HintLevel.NUDGE)
        highest = run_at(HintLevel.WORKED_STEP)
        for quiet, loud in zip(lowest, highest):
            assert quiet.items_attempted == loud.items_attempted
            assert quiet.diagnoses == loud.diagnoses
            assert quiet.remediation_ratio == loud.remediation_ratio
            assert quiet.goals_mastered == loud.goals_mastered
            assert quiet.gain == loud.gain

    def test_the_level_still_reaches_the_record(self) -> None:
        # The other half: it must not be inert *everywhere*, or a sitting could
        # not be read back against the support it was given — which is how both
        # collapses were found.
        # Cohort, for the same reason as above: one learner may never err.
        levels = {
            r.evidence["level"]
            for session, _ in run_cohort("calculus", "coupled", cohort={"max_items": 4})
            for r in session.board.audit_log
            if r.cause == "tutor"
        }
        assert levels, "no tutor turn was recorded"
        assert levels <= {"nudge", "targeted", "worked_step"}


class TestTheUnreadableCapCannotMoveACohort:
    """A budget only a person can reach.

    Separating the attempt budget from unreadable responses changes what a
    session does the moment one occurs — and a simulated learner cannot produce
    one. Every response it gives is a buggy rule's output or the stated answer,
    and ``domain validate`` requires the first to verify incorrect and the
    second correct. So the whole of decision 1 is inert for every measured
    result, which is a claim worth holding rather than asserting once.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_the_cap_changes_nothing(self, domain_name: str, arm: str) -> None:
        for i in range(4):
            tight = run(
                domain_name, arm, f"L{i:04d}", cohort={"max_unreadable_per_item": 1}
            )[1]
            loose = run(
                domain_name, arm, f"L{i:04d}", cohort={"max_unreadable_per_item": 9}
            )[1]
            assert tight.items_attempted == loose.items_attempted
            assert tight.diagnoses == loose.diagnoses
            assert tight.goals_mastered == loose.goals_mastered
            assert tight.gain == loose.gain

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_because_the_cap_is_never_reached(self, domain_name: str) -> None:
        # The reason it is inert, stated separately: if this ever fails the
        # invariance above is a coincidence rather than a consequence.
        for i in range(4):
            session, outcome = run(domain_name, "coupled", f"L{i:04d}")
            assert outcome.unmeasurable_steps == 0
            assert not [
                r for r in session.board.audit_log if "unreadable response(s)" in r.summary
            ]


class TestItemVariantsDoNotMoveTheNumbers:
    """Varying a repeated question is visible to a person and invisible here.

    A variant changes the wording, the numbers and the params — never the id,
    the concept or the probes. The simulated learner fires on ``probes`` and
    rolls on ``item_id``, so which variant it is asked cannot change whether a
    misconception fires; and whichever wrong answer the rule computes, the
    verifier judges it incorrect. Every cohort figure is therefore unchanged by
    the whole feature, which was checked against the re-run and holds exactly.

    Worth a guard rather than a note. A template that later changed ``probes``
    or the id would shift every measured result, and it would not look wrong —
    it would look like the numbers had moved.
    """

    def test_a_cohort_is_unaffected_by_the_templates(self) -> None:
        from dataclasses import replace as replace_field

        config = config_for("calculus", "coupled")
        with_variants = registry.load_domain("calculus")
        without = replace_field(with_variants, templates={})
        assert with_variants.templates, "calculus declares no templates"

        for i in range(6):
            varied = build_session(f"L{i:04d}", config.seed, with_variants, config).run()
            plain = build_session(f"L{i:04d}", config.seed, without, config).run()
            assert varied.items_attempted == plain.items_attempted
            assert varied.remediation_ratio == plain.remediation_ratio
            assert varied.goals_mastered == plain.goals_mastered
            assert varied.distance_to_goal == plain.distance_to_goal
            assert varied.diagnoses == plain.diagnoses

    def test_but_the_learner_is_asked_something_different(self) -> None:
        # The other half of the claim: if the questions were identical the
        # invariance above would be trivial rather than reassuring.
        domain = registry.load_domain("calculus")
        item = domain.items.bank("practice")[0]
        assert domain.variant(item, 1).prompt != item.prompt


class TestModelBackedAgentsAreAvailable:
    """P8 replaced the refusals. Assembly must still not contact a server."""

    @pytest.mark.parametrize("role", ["tutor", "diagnostic", "planner"])
    def test_a_model_backed_role_can_be_built(self, role: str) -> None:
        base = {
            "tutor": {"impl": "template"},
            "diagnostic": {"impl": "oracle"},
            "planner": {"impl": "goal_directed"},
        }
        spec = {"impl": "llm", "provider": "ollama", "model": "gemma4:12b"}
        config = config_for("toy_algebra", "coupled", agents={**base, role: spec})
        session = build_session("L0000", 1, registry.load_domain("toy_algebra"), config)
        assert session is not None

    def test_a_model_backed_surface_is_still_refused(self) -> None:
        # The renderer is the one piece P8 did not build. Refusing loudly beats
        # silently falling back to the symbolic one, which would make a run
        # claim a naturalistic simulator it did not use.
        config = Config.model_validate(
            {
                "domain": "toy_algebra",
                "simulator": {"surface": "llm"},
                "agents": {
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "goal_directed"},
                },
            }
        )
        with pytest.raises(NotImplementedForModels):
            build_session("L0000", 1, registry.load_domain("toy_algebra"), config)


def concepts_worked(session) -> list[str]:
    """The concepts a session actually gave work on, in order, deduplicated."""
    seen = [
        record.evidence.get("concept_id")
        for record in session.board.audit_log
        if record.cause == "observation"
    ]
    return [c for i, c in enumerate(seen) if i == 0 or c != seen[i - 1]]


class TestPlanningIsDirected:
    """The goal is shared state; the route toward it is derived from evidence."""

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_a_goal_is_set_and_recorded(self, domain_name: str) -> None:
        session, outcome = run(domain_name, "coupled")
        assert session.board.plan is not None
        assert outcome.goal == session.board.plan.goal
        assert session.board.plan.goal in session.domain.concepts.goals()

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_goal_change_is_auditable(self, domain_name: str) -> None:
        # A target nobody can account for afterwards is not auditable, and the
        # audit trail is the point.
        session, _ = run(domain_name, "coupled")
        plans = [r for r in session.board.audit_log if r.cause == "plan"]
        assert plans
        for record in plans:
            assert record.evidence["goal"]
            assert record.evidence["emphasis"]

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_setting_a_goal_is_not_counted_as_a_replan_trigger(
        self, domain_name: str
    ) -> None:
        # The threshold analysis reads the trigger breakdown, so a plan record
        # leaking into it would show up as a trigger that does not exist.
        _, outcome = run(domain_name, "coupled")
        assert all("goal set" not in trigger for trigger in outcome.triggers)

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_work_stays_on_the_way_to_the_goal(self, domain_name: str) -> None:
        session, _ = run(domain_name, "coupled")
        graph = session.domain.concepts
        # Goals advance during a session, so anything worked must be relevant to
        # some declared goal rather than to the final one only.
        reachable = set()
        for goal in graph.goals():
            reachable |= graph.all_prerequisites(goal) | {goal}
        for concept in concepts_worked(session):
            assert concept in reachable, f"{concept} is on the way to no goal"

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_distance_to_the_goal_is_reported(self, domain_name: str) -> None:
        _, outcome = run(domain_name, "coupled")
        assert outcome.distance_to_goal is not None
        assert outcome.distance_to_goal >= 0


class TestIntentNeedsTheLearnerModel:
    """The sharpest form of the ablation.

    Acting on ``consolidate`` needs the error trace; acting on ``advance`` needs
    the posteriors. A view carrying neither cannot honour either — not less
    well, but not at all.
    """

    def _paths(self, domain_name: str, arm: str):
        paths = {}
        for emphasis in ("consolidate", "advance"):
            session, _ = run(
                domain_name,
                arm,
                agents={
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "goal_directed", "emphasis": emphasis},
                },
            )
            paths[emphasis] = concepts_worked(session)
        return paths

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_coupled_arm_honours_it(self, domain_name: str) -> None:
        paths = self._paths(domain_name, "coupled")
        assert paths["consolidate"] != paths["advance"]

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_decoupled_arm_cannot(self, domain_name: str) -> None:
        # Identical, because nothing in its view distinguishes the two.
        paths = self._paths(domain_name, "decoupled")
        assert paths["consolidate"] == paths["advance"]

    def test_advancing_reaches_further_than_consolidating(self) -> None:
        # Not a value judgement: the two are asked for different things, and
        # this asserts each does what it was asked.
        far = run(
            "calculus",
            "coupled",
            agents={
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed", "emphasis": "advance"},
            },
        )[0]
        near = run(
            "calculus",
            "coupled",
            agents={
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed", "emphasis": "consolidate"},
            },
        )[0]
        deepest = lambda s: max(  # noqa: E731
            s.domain.concepts.depth(c) for c in concepts_worked(s)
        )
        assert deepest(far) > deepest(near)


class TestOutcomesAreComparableBetweenArms:
    """An outcome compared across arms must come from the state, not an agent.

    The agents are what differ, so a number one of them keeps means something
    different in each arm — and it will not look wrong, because a plausible
    number is exactly what it produces.
    """

    def test_distance_is_measured_against_the_same_goal_in_both_arms(self) -> None:
        # Measured against the planner's own target, the coupled arm was scored
        # against the last declared goal and the decoupled arm against the
        # first, and the two were reported side by side as 0.50 and 0.53.
        sessions = {arm: run("calculus", arm) for arm in ("coupled", "decoupled")}
        graph = sessions["coupled"][0].domain.concepts

        def measured_against(session, outcome):
            from agent_newton.core.state import bkt, route

            return route.next_goal(
                graph.goals(),
                dict(session.board.state.mastery),
                session.config.zpd,
                bkt.initial(session.config.bkt),
            )

        for arm, (session, outcome) in sessions.items():
            target = measured_against(session, outcome)
            if target is None:
                assert outcome.distance_to_goal == 0, arm
                continue
            from agent_newton.core.state import bkt, route

            expected = len(
                route.remaining(
                    target,
                    dict(session.board.state.mastery),
                    graph,
                    session.config.zpd,
                    bkt.initial(session.config.bkt),
                )
            )
            assert outcome.distance_to_goal == expected, arm

    def test_it_does_not_follow_the_planner_target(self) -> None:
        # The decoupled planner advances its target on its own walk position,
        # so if the measure followed the plan the two arms would be scored
        # against different goals.
        session, outcome = run("calculus", "decoupled")
        assert outcome.distance_to_goal is not None
        assert outcome.goal is not None
        # The planner's target and the mastery-derived one may well differ —
        # that difference is exactly why the measure must not use the former.
        assert isinstance(outcome.distance_to_goal, int)

    def test_zero_only_when_every_goal_is_mastered(self) -> None:
        from agent_newton.core.state import bkt, route

        session, outcome = run("calculus", "coupled")
        mastery = dict(session.board.state.mastery)
        remaining = route.next_goal(
            session.domain.concepts.goals(),
            mastery,
            session.config.zpd,
            bkt.initial(session.config.bkt),
        )
        if outcome.distance_to_goal == 0:
            assert remaining is None


class TestALearnerRequestCannotMoveACohort:
    """Nothing but a person sets one, and no cohort has a person in it.

    The request reaches the planner through the view, like the dwelling set, so
    the guard is the same shape: empty by default, and a run with it empty must
    be byte-identical to one from before it existed.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_a_cohort_never_carries_one(self, domain_name: str, arm: str) -> None:
        session, _ = run(domain_name, arm)
        assert session.board.requested == frozenset()
        assert not [
            r for r in session.board.audit_log if "asked to work on" in r.summary
        ]

    def test_the_decoupled_arm_cannot_act_on_one(self) -> None:
        # The same asymmetry `Emphasis` has: honouring a request means choosing
        # a goal whose route reaches it, which needs the posteriors and the
        # graph. A view carrying neither cannot do it — so the selections are
        # identical with and without.
        domain = registry.load_domain("calculus")
        config = config_for("calculus", "decoupled")

        def selections(requested):
            session = build_session("L0000", config.seed, domain, config)
            session.board.record_request(requested)
            session.run()
            return [
                r.evidence["item_id"]
                for r in session.board.audit_log
                if r.cause == "observation"
            ]

        assert selections(["chain_rule"]) == selections([])

    def test_the_coupled_arm_does_act_on_one(self) -> None:
        # And the other half, or the test above passes for the wrong reason.
        domain = registry.load_domain("calculus")
        config = config_for("calculus", "coupled")

        def goal_for(requested):
            session = build_session("L0000", config.seed, domain, config)
            session.board.record_request(requested)
            return session.run().goal

        assert goal_for(["chain_rule"]) != goal_for([])

    def test_the_request_reaches_the_training_and_not_only_the_goal(self) -> None:
        """⚠️ Moving the goal is not enough, and a sitting proved it.

        A concept can be requested, reachable, and on the route, and still never
        come up: the emphasis ranks by difficulty or depth and something else
        wins every time. A person watched one sit in the frontier for a whole
        sitting and said so — *"even the things I chose to work on were not
        entirely there in the session."*

        Set up on a learner who has the prerequisites, because that is when the
        question arises at all. A request for something unreachable moves the
        goal and then walks the route to it, which is the design and is tested
        above.
        """
        from agent_newton.core.evaluation.outcomes import dose_by_concept

        domain = registry.load_domain("calculus")
        config = config_for("calculus", "coupled", cohort={"max_items": 4})
        # Everything early demonstrated, so several concepts sit in the frontier
        # together and the emphasis has a real choice to make.
        known = {
            "limits_of_sequences": 0.95, "average_rate_of_change": 0.95,
            "limit_concept": 0.95, "instantaneous_rate_of_change": 0.95,
            "derivative_from_first_principles": 0.95, "power_rule": 0.95,
        }

        def worked(requested) -> list[str]:
            """The concepts trained, in the order they were reached."""
            session = build_session("L0000", config.seed, domain, config)
            session.board.state.mastery.update(known)
            session.board.record_request(requested)
            session.run()
            order: list[str] = []
            for record in session.board.audit_log:
                if record.cause == "observation":
                    concept = record.evidence["concept_id"]
                    if concept not in order:
                        order.append(concept)
            assert dose_by_concept(session.board.audit_log)
            return order

        unasked = worked([])
        assert len(unasked) > 1, "nothing to choose between; the test proves nothing"
        # Ask for one the emphasis did *not* reach for first.
        second = unasked[1]
        assert worked([second])[0] == second


class TestNoTurnHandsOverTheAnswer:
    """⚠️ The worked step used to, by permission, and a person read it as a leak.

    Checked over the turns a cohort actually produced rather than on constructed
    replies, because the exemption that allowed it lived in the checker — a test
    built from the checker's own rule would have agreed with it.

    Collected through the observer rather than from the audit log, which records
    the item's *id*. A repeated item is asked as a variant with different
    numbers, so looking the id up again yields draw 0 and checks the reply
    against an answer nobody was asked for. That produced a false leak on the
    first run of this test.
    """

    def _turns(self, domain_name: str, n: int = 4):
        from agent_newton.core.orchestration.session import Watching

        from agent_newton.domains.base import Item

        class Watcher(Watching):
            def __init__(self) -> None:
                self.turns: list[tuple[Item, str, str]] = []
                self.unsolved: list[str] = []

            def tutor_replied(self, item, hint) -> None:  # noqa: ANN001
                self.turns.append((item, hint.level.label, hint.text))

            def item_finished(self, item, solved, reason="attempts_spent") -> None:  # noqa: ANN001
                if not solved:
                    self.unsolved.append(item.answer)

        config = config_for(domain_name, "coupled")
        domain = registry.load_domain(domain_name)
        watcher = Watcher()
        for i in range(n):
            build_session(
                f"L{i:04d}", config.seed, domain, config, observer=watcher
            ).run()
        return domain, watcher

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_not_at_any_level(self, domain_name: str) -> None:
        from agent_newton.core.evaluation.tutor import leaks_answer

        domain, watcher = self._turns(domain_name)
        leaked = [
            (level, text)
            for item, level, text in watcher.turns
            if leaks_answer(text, item, domain)
        ]
        assert not leaked, f"a tutor turn gave the answer away: {leaked[:2]}"

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_top_of_the_ladder_was_actually_reached(self, domain_name: str) -> None:
        # Or the assertion above holds for the wrong reason: a cohort that never
        # reaches the top proves nothing about what is said there.
        _, watcher = self._turns(domain_name)
        levels = {level for _, level, _ in watcher.turns}
        assert "worked_step" in levels
        assert len(levels) > 1, "the ladder collapsed to a single level"

    def test_a_nudge_occurs_at_all(self) -> None:
        # The level that was unreachable until the escalation stopped counting
        # the failure it was responding to. Asserted on toy_algebra because it
        # needs a learner holding a concept above `theta_lower` who still fails
        # it — common in a five-concept graph, rare in fifteen, and this is a
        # claim about the rule rather than about either domain.
        _, watcher = self._turns("toy_algebra")
        assert "nudge" in {level for _, level, _ in watcher.turns}

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_the_item_still_reveals_it_when_the_attempts_run_out(
        self, domain_name: str
    ) -> None:
        # The reveal was asked for and is kept — moved to where it costs
        # nothing, since the next question on the concept carries different
        # numbers. It reaches the learner when the item closes, not from the
        # tutor mid-item.
        _, watcher = self._turns(domain_name)
        assert watcher.unsolved, "no item ran out of attempts"


class TestTheLoopAgreesWithTheRulesItCalls:
    """Invariants over a whole session, checked against the state.

    Every defect this project has found in the loop has been a correct rule
    called with the wrong arguments, or two correct things sequenced into a
    wrong outcome — neither of which a unit test on the rule can see. These
    read what a session actually recorded and compare it against the rules
    independently, which is the only place the two can disagree.

    Ported from `research_private/tools/session_probe.py`, which found the
    stop-reason defect this way. The probe is private and exploratory; these
    are the checks worth keeping, so they live where CI runs them.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_every_item_given_was_inside_the_frontier(self, domain_name: str) -> None:
        # The coupled arm only: `FixedOrderPlanner` walks the goals' closure in
        # topological order and consults no frontier — it cannot, its view has
        # neither the posteriors nor the graph. Asserting this against it would
        # report the manipulation as a defect.
        from agent_newton.core.state import bkt, zpd

        session, _ = run(domain_name, "coupled")
        domain = session.domain
        prior = bkt.initial(session.config.bkt)
        band = session.config.zpd

        # Replayed from the audit log: the frontier is derived, so the state at
        # each observation is enough to recompute the zone that was open then.
        mastery: dict[str, float] = {}
        for record in session.board.audit_log:
            if record.cause != "observation":
                continue
            concept = record.evidence["concept_id"]
            frontier = zpd.compute(dict(mastery), domain.concepts, band, prior)
            # A concept with no prior observation is legitimately absent from the
            # replayed mastery; the frontier still admits it at the prior.
            assert concept in frontier or not mastery, (
                f"{record.evidence['item_id']} was given on {concept}, "
                f"outside the frontier {sorted(frontier)}"
            )
            if record.evidence.get("mastery_after") is not None:
                mastery[concept] = record.evidence["mastery_after"]

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_an_unreadable_step_moves_no_estimate(self, domain_name: str) -> None:
        # `UNPARSEABLE` means the verifier could not measure. It must update no
        # posterior and enter no error trace — the distinction the whole state
        # layer is built around.
        session, _ = run(domain_name, "coupled")
        for record in session.board.audit_log:
            if record.cause != "observation":
                continue
            if record.evidence.get("verdict") == "unparseable":
                assert record.evidence.get("delta") in (None, 0, 0.0), (
                    "an unreadable step moved the estimate"
                )

    @pytest.mark.parametrize("arm", ("coupled", "decoupled"))
    def test_the_lifetime_count_moves_once_per_item_given(self, arm: str) -> None:
        # However many steps an item took. It drives the repetition index, which
        # decides both the simulated learner's answer and which template variant
        # is drawn, so a count that moved per *step* would change what the
        # learner faces and silently move every measured number.
        session, outcome = run("calculus", arm)
        assert sum(session.board.state.items_given.values()) == outcome.items_attempted

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_training_finishes_before_the_post_test_starts(self, domain_name: str) -> None:
        # An explanation arriving after the thing it explains is not one: the
        # learner must be told why training ended before the next bank appears.
        from agent_newton.core.orchestration.session import Watching

        class Order(Watching):
            def __init__(self) -> None:
                self.seen: list[str] = []

            def training_finished(self, reason: str, items: int) -> None:
                self.seen.append("training_finished")

            def phase_started(self, phase: str, total: int) -> None:
                self.seen.append(f"phase:{phase}")

        config = config_for(domain_name, "coupled", cohort={"administer_tests": True})
        domain = registry.load_domain(domain_name)
        order = Order()
        build_session("L0000", config.seed, domain, config, observer=order).run()
        assert "training_finished" in order.seen and "phase:posttest" in order.seen
        assert order.seen.index("training_finished") < order.seen.index("phase:posttest")


class TestGoalChangesIsNotGoalsMastered:
    """Why `goals_mastered` exists, asserted rather than left in a comment.

    `goal_changes` counts planner retargets. The decoupled planner cannot see
    mastery, so it retargets on its own position in the syllabus and reports
    goals "reached" that the learner is nowhere near. `outcomes.py` states the
    rule at the top of the module and nothing checked it.
    """

    def _perfect_learner(self, arm: str):
        # Answers the *variant* it is shown, not the written item: the session
        # draws a template variant per repetition, so replying with draw 0's
        # answer would be wrong from the second repetition on.
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.simulator.human import HumanLearner

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = config_for(
            "calculus", arm,
            simulator={"learner": "human", "surface": "symbolic"},
            agents={
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "llm"},
                "planner": {"impl": "goal_directed"},
            },
            cohort={"n_learners": 1, "max_items": 400, "administer_tests": False},
        )
        domain = registry.load_domain("calculus")
        session = build_session(
            "human", config.seed, domain, config,
            learner=HumanLearner(lambda item, attempt: item.answer),
        )
        session.diagnostic = Nothing()
        return session, session.run()

    def test_the_decoupled_arm_reports_goal_changes_it_did_not_earn(self) -> None:
        _, outcome = self._perfect_learner("decoupled")
        assert outcome.goals_mastered == 0
        assert outcome.goal_changes > 0, "the walk did not advance; the test is vacuous"
        # The number a reader would take for progress, against the one derived
        # from the state. Four retargets, nothing learned.
        assert outcome.goal_changes > outcome.goals_mastered

    def test_the_coupled_arm_agrees_only_by_coincidence(self) -> None:
        # ⚠️ 5 == 5 here, and it is not a property. Two off-by-ones cancel: the
        # first plan is not counted (there was no previous goal to complete),
        # and the final retarget *is* counted although the planner had simply
        # run out — `_retarget` returns True whenever a plan existed and none is
        # proposed. Pinned so that fixing either one alone is visible as a
        # change rather than passing quietly.
        _, outcome = self._perfect_learner("coupled")
        assert outcome.goals_mastered == 5
        assert outcome.goal_changes == 5


def turns_of(session):
    """Every tutor turn a session recorded, as its evidence dicts."""
    return [r.evidence for r in session.board.audit_log if r.cause == "tutor"]


#: Passed as `scaffolding=` rather than unpacked, so neither can bind to the
#: positional `learner_id` or `n` that follow the arm.
BANDED_PLUS: dict[str, object] = {"policy": "banded_plus"}
WITH_SUPPORT: dict[str, object] = {
    "policy": "banded_plus",
    "offer_at_presentation": True,
}


class TestTheSilentRegionIsNeverEntered:
    """``hint_level`` defines a region above ``theta_upper``. Selection excludes it.

    Read off the turns a session recorded, never off ``hint_level``. That is
    sitting 7's lesson stated as a test: ``test_the_floor_leaves_the_ladder_intact``
    asserted ``hint_level(0.40, 0) is TARGETED``, passed throughout, and the
    running system meanwhile had one reachable support level — because the
    session was calling the rule with inputs the test never used.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_no_turn_is_ever_pitched_above_the_band(
        self, domain_name: str, arm: str
    ) -> None:
        band = config_for(domain_name, arm, scaffolding=BANDED_PLUS).zpd
        for session, _ in run_cohort(domain_name, arm, scaffolding=BANDED_PLUS):
            for turn in turns_of(session):
                assert turn["mastery"] < band.theta_upper, (
                    f"an item was given on a concept at {turn['mastery']}, at or "
                    f"above theta_upper — the frontier is supposed to exclude it"
                )

    @pytest.mark.parametrize("domain_name", DOMAINS)
    def test_so_the_silent_level_never_reaches_anyone(self, domain_name: str) -> None:
        # The consequence, stated separately: the region is a boundary the rule
        # draws rather than a case a learner meets. It is built because a rule
        # with a hole in it is not a rule, not because anyone will see it.
        for session, _ in run_cohort(domain_name, "coupled", scaffolding=BANDED_PLUS):
            assert all(turn["level"] != "none" for turn in turns_of(session))


class TestTheLadderHasMoreThanOneRung:
    """Sitting 7's collapse, pinned so it cannot come back quietly.

    Every sitting before ``a31ebee`` ran at ``worked_step`` on every turn, and
    the scaffolding figures from all seven describe a system with one support
    level. Nothing measured that, because the only assertions were about the
    function rather than about the turns.
    """

    @pytest.mark.parametrize("policy", ["banded", "banded_plus"])
    def test_a_cohort_reaches_more_than_one_level(self, policy: str) -> None:
        levels = set()
        for session, _ in run_cohort(
            "calculus", "coupled", n=12, scaffolding={"policy": policy}
        ):
            levels |= {turn["level"] for turn in turns_of(session)}
        assert len(levels) > 1, (
            f"every turn in a 12-learner cohort came out at {levels}; the ladder "
            f"has collapsed to one rung again"
        )


class TestSupportAtPresentationReachesTheRecord:
    """An offer has to be in the log, or a sitting cannot be read back."""

    def test_it_is_offered_and_recorded(self) -> None:
        session, _ = run("calculus", "coupled", scaffolding=WITH_SUPPORT)
        offers = [t for t in turns_of(session) if t["move"] == "present"]
        assert offers
        assert {t["level"] for t in offers} <= {"formula", "formula_and_example"}

    def test_it_targets_nothing(self) -> None:
        # `remediation_ratio` counts what a hint aimed at. A resource states the
        # rule and addresses no misconception — none has been observed yet — so
        # a target here would credit it with remediation it did not do.
        session, _ = run("calculus", "coupled", scaffolding=WITH_SUPPORT)
        assert all(
            t["targets"] is None
            for t in turns_of(session)
            if t["move"] == "present"
        )

    def test_nothing_is_offered_when_the_run_does_not_permit_it(self) -> None:
        session, _ = run("calculus", "coupled", scaffolding=BANDED_PLUS)
        assert not [t for t in turns_of(session) if t["move"] == "present"]

    def test_a_domain_with_no_resources_offers_nothing(self) -> None:
        # toy_algebra carries none. The session must run, not raise — optional
        # content has to be genuinely optional.
        session, outcome = run("toy_algebra", "coupled", scaffolding=WITH_SUPPORT)
        assert outcome.items_attempted > 0
        assert not [t for t in turns_of(session) if t["move"] == "present"]

    def test_it_is_offered_once_per_posing_of_the_question(self) -> None:
        """Presentation support, not escalation.

        An item may be posed several times — a concept is worked until its
        posterior clears the band — and each posing gets one offer. Asserted as
        an ordering over the log rather than by counting: two offers on one item
        are correct when a graded step separates them and wrong when nothing
        does, and a count cannot tell those apart.
        """
        session, _ = run("calculus", "coupled", scaffolding=WITH_SUPPORT)
        awaiting: set[str] = set()
        offers = 0
        for record in session.board.audit_log:
            item_id = record.evidence.get("item_id")
            if record.cause == "observation":
                awaiting.discard(item_id)
            elif record.cause == "tutor" and record.evidence["move"] == "present":
                assert item_id not in awaiting, (
                    f"{item_id} was offered support twice with no graded step "
                    f"in between, so the same question carried two offers"
                )
                awaiting.add(item_id)
                offers += 1
        assert offers

    def test_the_offer_fades_as_the_estimate_rises(self) -> None:
        """Fading, read off a real session rather than off the function.

        ``check_support_fading`` proves the rule is monotone. This proves the
        session asks it the right question — which is the distinction sitting 7
        turned on, where the rule was correct throughout and every turn still
        came out at the top of the ladder.
        """
        session, _ = run("calculus", "coupled", scaffolding=WITH_SUPPORT)
        by_item: dict[str, list[tuple[float, str]]] = {}
        for turn in turns_of(session):
            if turn["move"] == "present":
                by_item.setdefault(turn["item_id"], []).append(
                    (turn["mastery"], turn["level"])
                )
        repeated = {k: v for k, v in by_item.items() if len(v) > 1}
        assert repeated, "no item was offered twice, so nothing was faded"
        rank = {"none": 0, "formula": 1, "formula_and_example": 2}
        for offers in repeated.values():
            for (before, low), (after, high) in zip(offers, offers[1:]):
                assert after >= before
                assert rank[high] <= rank[low]


class TestNoneOfThisCanMoveACohort:
    """The property that keeps every measured result valid.

    A simulated learner improves only through ``receive_hint``, which is handed
    the misconception a hint *targets*. A presentation offer targets nothing and
    the ladder changes only how much a reply gives away, so neither can reach
    the one channel that changes an outcome.

    ⚠️ This is a fact about today's simulator, not a guarantee about the design.
    If a mechanism is ever added that responds to how much support was given,
    this test is where it will announce itself — and the config scans are what
    keep a cohort on the original ladder in the meantime.
    """

    @pytest.mark.parametrize("domain_name", DOMAINS)
    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_the_richer_ladder_changes_no_outcome(
        self, domain_name: str, arm: str
    ) -> None:
        plain = [o for _, o in run_cohort(domain_name, arm)]
        richer = [o for _, o in run_cohort(domain_name, arm, scaffolding=WITH_SUPPORT)]
        for before, after in zip(plain, richer):
            assert before.items_attempted == after.items_attempted
            assert before.diagnoses == after.diagnoses
            assert before.remediation_ratio == after.remediation_ratio
            assert before.goals_mastered == after.goals_mastered
            assert before.distance_to_goal == after.distance_to_goal
            assert before.gain == after.gain

    def test_and_the_turns_really_did_change(self) -> None:
        # Otherwise the test above passes because nothing happened at all, which
        # is the shape of a guard that cannot fail.
        plain, _ = run("calculus", "coupled")
        richer, _ = run("calculus", "coupled", scaffolding=WITH_SUPPORT)
        assert len(turns_of(richer)) > len(turns_of(plain))


class TestWhatTheTutorIsToldAboutAnUnseenConcept:
    """``_work_item`` values an unobserved concept at 0.0; everything else uses
    the prior.

    A known inconsistency, recorded as open and deliberately not fixed here —
    it is a decision about what "no evidence yet" should mean, and changing it
    would move support levels in any run with a narrower band. What this pins is
    that the two conventions still *collapse*, so the new tiers did not quietly
    make it live.
    """

    def test_zero_and_the_prior_land_on_the_same_tier(self) -> None:
        from agent_newton.core.pedagogy import hint_level, support_at_presentation

        config = config_for("calculus", "coupled", scaffolding=WITH_SUPPORT)
        prior = bkt.initial(config.bkt)
        for policy in ("banded", "banded_plus"):
            assert hint_level(0.0, 0, config.zpd, policy=policy) is hint_level(
                prior, 0, config.zpd, policy=policy
            )
        assert support_at_presentation(0.0, config.zpd) is support_at_presentation(
            prior, config.zpd
        )

    def test_the_decoupled_arm_is_given_the_most_support_on_every_item(self) -> None:
        """Its view carries no posteriors, so its tutor is handed 0.0 always.

        Worth stating rather than leaving implicit: the arms do not differ in
        *whether* support is offered but in whether it is aimed. The coupled arm
        gives more as the estimate falls; the decoupled arm gives the maximum
        to everyone, forever, because it cannot tell anyone apart.
        """
        session, _ = run("calculus", "decoupled", scaffolding=WITH_SUPPORT)
        offers = [t for t in turns_of(session) if t["move"] == "present"]
        assert offers
        assert {t["level"] for t in offers} == {"formula_and_example"}
        assert {t["mastery"] for t in offers} == {0.0}
