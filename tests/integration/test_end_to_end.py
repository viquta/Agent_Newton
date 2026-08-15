"""The milestone: a full session with no model involved, on both domains.

This is the checkpoint that validates the harness — loop, state, verifier,
pedagogy rules, outcome measurement — before any model noise enters. Everything
here must hold without inference, and fast enough to run in CI.
"""

from __future__ import annotations

import pytest

from agent_newton.config import Config
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
        session, _ = run(domain_name, "coupled")
        turns = [r for r in session.board.audit_log if r.cause == "tutor"]
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
                tutor_module, "hint_level", lambda mastery, prior, band: level
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
        session, _ = run("calculus", "coupled", cohort={"max_items": 4})
        levels = {
            r.evidence["level"] for r in session.board.audit_log if r.cause == "tutor"
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
