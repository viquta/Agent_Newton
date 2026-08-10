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
        assert any(r > 0.0 for r in ratios)

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
        assert noisy.remediation_ratio < accurate.remediation_ratio


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
        coupled, _ = run(domain_name, "coupled")
        decoupled, _ = run(domain_name, "decoupled")
        assert coupled.learner.profile.initial == decoupled.learner.profile.initial


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
