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
                "planner": {"impl": "deterministic"},
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
                "planner": {"impl": "deterministic"},
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
            "planner": {"impl": "deterministic"},
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
                    "planner": {"impl": "deterministic"},
                },
            }
        )
        with pytest.raises(NotImplementedForModels):
            build_session("L0000", 1, registry.load_domain("toy_algebra"), config)
