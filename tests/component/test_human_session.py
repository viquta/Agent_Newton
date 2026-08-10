"""A person in the loop the cohorts run.

The demo drives the real session, so what is tested here is that a human learner
satisfies the same interface a simulated one does and that the session's
handling of *missing ground truth* is honest — a person carries no injected
misconception label, and several things downstream would otherwise report a
number where there is none.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_newton.config import Config
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.simulator.engine import Learner, SimulatedLearner
from agent_newton.core.simulator.human import HumanLearner
from agent_newton.domains import registry


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def human_config(**overrides) -> Config:
    return Config.model_validate(
        {
            "domain": "toy_algebra",
            "arm": "coupled",
            "cohort": {"n_learners": 1, "max_items": 4, "administer_tests": False},
            "simulator": {"learner": "human", "surface": "symbolic"},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "llm"},
                "planner": {"impl": "goal_directed"},
            },
            **overrides,
        }
    )


class TestTheInterfaceIsShared:
    def test_a_human_satisfies_the_learner_protocol(self) -> None:
        assert isinstance(HumanLearner(lambda item, attempt: "1"), Learner)

    def test_so_does_the_simulated_one(self, toy) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import sample_profile

        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        assert isinstance(SimulatedLearner(profile, toy, SimulatorConfig()), Learner)


class TestMissingGroundTruthIsReportedAsMissing:
    def test_a_human_step_carries_no_injected_label(self, toy) -> None:
        # It is what the diagnostic agent must infer. There is none.
        learner = HumanLearner(lambda item, attempt: "7")
        step = learner.answer(toy.items.all()[0])
        assert step.fired is None

    def test_remediation_is_unavailable_not_zero(self) -> None:
        # Zero would read as "nothing was remediated", which is a claim. There
        # is no profile to measure a reduction against at all.
        assert HumanLearner(lambda item, attempt: "7").remediation_ratio() is None

    def test_a_hint_changes_nothing_measurable(self) -> None:
        learner = HumanLearner(lambda item, attempt: "7")
        assert learner.receive_hint("some_misconception") is False

    def test_a_skipped_test_bank_is_distinguishable_from_a_zero_score(self) -> None:
        from agent_newton.core.evaluation.outcomes import TestResult

        skipped = TestResult(correct=0, total=0)
        failed = TestResult(correct=0, total=6)
        assert skipped.score == failed.score == 0.0
        assert not skipped.administered
        assert failed.administered


class TestAnOracleIsRefusedForAHuman:
    """The check that stops a session from looking like it ran."""

    @pytest.mark.parametrize("impl", ["oracle", "noised_oracle"])
    def test_it_is_rejected_at_load(self, impl: str) -> None:
        overrides: dict[str, object] = {"impl": impl}
        if impl == "noised_oracle":
            overrides["noise_rate"] = 0.2
        with pytest.raises(ValidationError, match="injected misconception label"):
            human_config(
                agents={
                    "tutor": {"impl": "template"},
                    "diagnostic": overrides,
                    "planner": {"impl": "goal_directed"},
                }
            )

    def test_the_model_backed_diagnostic_is_accepted(self) -> None:
        assert human_config().agents.diagnostic.impl == "llm"

    def test_a_simulated_learner_may_still_use_an_oracle(self) -> None:
        # The check must bite only where the label is actually absent.
        config = Config.model_validate(
            {
                "domain": "toy_algebra",
                "simulator": {"learner": "simulated", "surface": "symbolic"},
                "agents": {
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "goal_directed"},
                },
            }
        )
        assert config.agents.diagnostic.impl == "oracle"


class TestTheSessionRunsWithAPerson:
    """Driven by a scripted stand-in, so the loop is exercised without a model."""

    def _run(self, toy, answers: list[str]):
        from agent_newton.core.agents.base import Diagnosis

        replies = iter(answers)

        class Nothing:
            """A diagnostic that names nothing — no model, no ground truth."""

            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config()
        learner = HumanLearner(lambda item, attempt: next(replies, "0"))
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return session, session.run()

    def test_it_completes(self, toy) -> None:
        _, outcome = self._run(toy, ["1"] * 20)
        assert outcome.items_attempted > 0
        assert outcome.learner_id == "human"

    def test_the_outcome_reports_no_remediation_figure(self, toy) -> None:
        _, outcome = self._run(toy, ["1"] * 20)
        assert outcome.remediation_ratio is None

    def test_the_tests_were_skipped_not_failed(self, toy) -> None:
        _, outcome = self._run(toy, ["1"] * 20)
        assert not outcome.pretest.administered
        assert not outcome.posttest.administered

    def test_the_blackboard_still_updates(self, toy) -> None:
        # The whole point of the demo: a person's answers move the shared state
        # the same way a simulated learner's do.
        session, _ = self._run(toy, ["1"] * 20)
        assert session.board.state.version > 1
        assert session.board.plan is not None

    def test_every_response_is_kept(self, toy) -> None:
        session, outcome = self._run(toy, ["1"] * 20)
        learner = session.learner
        assert isinstance(learner, HumanLearner)
        assert len(learner.responses) >= outcome.items_attempted


class TestTheObserverOnlyWatches:
    def test_a_session_with_an_observer_behaves_identically(self, toy) -> None:
        # If an observer could change the run, the demo would be showing
        # something other than the system that produced the numbers.
        from agent_newton.core.agents.base import Diagnosis

        class Recorder:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def item_started(self, item, board) -> None:  # noqa: ANN001
                self.calls.append("item")

            def step_graded(self, item, response, result, diagnosis) -> None:  # noqa: ANN001
                self.calls.append("step")

            def tutor_replied(self, item, hint) -> None:  # noqa: ANN001
                self.calls.append("hint")

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        def build(observer):
            replies = iter(["1"] * 30)
            config = human_config()
            learner = HumanLearner(lambda item, attempt: next(replies, "0"))
            session = build_session(
                "human", config.seed, toy, config, learner=learner, observer=observer
            )
            session.diagnostic = Nothing()
            return session.run()

        recorder = Recorder()
        watched = build(recorder)
        plain = build(None)

        assert recorder.calls, "the observer was never called"
        assert watched.items_attempted == plain.items_attempted
        assert watched.goals_mastered == plain.goals_mastered
        assert watched.distance_to_goal == plain.distance_to_goal
