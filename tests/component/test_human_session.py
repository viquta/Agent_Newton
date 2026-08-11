"""A person in the loop the cohorts run.

The demo drives the real session, so what is tested here is that a human learner
satisfies the same interface a simulated one does and that the session's
handling of *missing ground truth* is honest — a person carries no injected
misconception label, and several things downstream would otherwise report a
number where there is none.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from agent_newton.config import Config
from agent_newton.core.orchestration.session import Watching, build_session
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

        # Subclassing `Watching` is how a front end implements only part of
        # the protocol. Implementing it structurally instead would mean a
        # hook added later crashes this observer the first time the session
        # reaches it — mid-run, not at startup.
        class Recorder(Watching):
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


class TestThePreTestCanSteerTheTraining:
    """A person who has just sat a test expects it to have counted.

    Off by default and off in every cohort — it moves the starting frontier, and
    only the coupled arm can route from a frontier. On here so the behaviour is
    tested rather than only configured.
    """

    def _run(self, toy, *, seed_from_pretest: bool, answer: str):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 4,
                "administer_tests": True,
                "seed_from_pretest": seed_from_pretest,
            }
        )
        learner = HumanLearner(lambda item, attempt: answer)
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return session, session.run()

    def test_the_pretest_reports_what_was_missed(self, toy) -> None:
        # Aggregates alone cannot answer "what do I need to work on".
        _, outcome = self._run(toy, seed_from_pretest=False, answer="999")
        assert outcome.pretest.concepts_missed
        assert set(outcome.pretest.concepts_missed) <= set(toy.concepts.ids())

    def test_a_missed_concept_is_estimated_lower_when_seeding_is_on(self, toy) -> None:
        # Read from the audit records rather than from final mastery: practice
        # moves the estimates afterwards, and the claim is about what the
        # pre-test did.
        session, outcome = self._run(toy, seed_from_pretest=True, answer="999")
        seeded = [r for r in session.board.audit_log if r.cause == "seed"]
        assert seeded, "seeding recorded nothing"

        wrong = [r for r in seeded if r.evidence["verdict"] == "incorrect"]
        assert wrong, "the run produced no incorrect pre-test answers to seed from"
        for record in wrong:
            assert record.evidence["mastery_after"] < record.evidence["mastery_before"]
        assert {r.evidence["concept_id"] for r in wrong} >= set(
            outcome.pretest.concepts_missed
        )

    def test_seeding_off_leaves_the_model_untouched_by_the_pretest(self, toy) -> None:
        session, _ = self._run(toy, seed_from_pretest=False, answer="999")
        assert not [r for r in session.board.audit_log if r.cause == "seed"]

    def test_the_pretest_score_is_the_same_either_way(self, toy) -> None:
        # Seeding must not feed back into what the test measured. The banks are
        # administered before anything is folded in, and a score that moved with
        # the flag would mean the measurement had been disturbed.
        _, seeded = self._run(toy, seed_from_pretest=True, answer="999")
        _, plain = self._run(toy, seed_from_pretest=False, answer="999")
        assert seeded.pretest.correct == plain.pretest.correct
        assert seeded.pretest.total == plain.pretest.total


class TestReflectionIsNotAnAnswer:
    """The tutor asks a question in words; the reply is words.

    Before this existed, that reply went to the symbolic verifier, came back
    unreadable, and cost the learner an attempt — the error-first rule asking a
    question it could not receive an answer to.
    """

    def _session(self, toy, reflection: str | None):
        from agent_newton.core.agents.base import Diagnosis

        class AlwaysDiagnoses:
            """Forces the reflect-then-remediate path on every error."""

            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(toy.misconceptions.ids()[0], confidence=1.0)

        config = human_config()
        # Parseable but wrong. Prose would come back UNPARSEABLE, which is
        # never diagnosed, so no reflective prompt would ever be issued.
        learner = HumanLearner(
            lambda item, attempt: "999",
            ask_reflection=(lambda item, prompt: reflection) if reflection else None,
        )
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = AlwaysDiagnoses()
        return session, session.run()

    def test_it_is_recorded_on_the_blackboard(self, toy) -> None:
        session, _ = self._session(toy, "I do not know what a coefficient is")
        said = session.board.state.reflections
        assert [u.text for u in said].count("I do not know what a coefficient is")
        assert all(u.kind == "reflection" for u in said)

    def test_it_keeps_the_concept_it_was_said_about(self, toy) -> None:
        # Dropped once: only the text reached the state, so the tutor read back
        # whatever had been said most recently whatever its subject, and asked a
        # learner differentiating 2/x^2 to revisit their explanation of limits.
        session, _ = self._session(toy, "I am unsure about the second step")
        said = session.board.state.reflections
        assert said
        for utterance in said:
            item = toy.items.get(utterance.item_id)
            assert utterance.concept_id == item.concept_id

    def test_the_tutor_is_only_given_words_about_the_concept_at_hand(self, toy) -> None:
        from agent_newton.core.state.schema import Utterance
        from agent_newton.core.state.views import FullStateView

        session, _ = self._session(toy, "I am unsure about the second step")
        view = session.board.view(arm="coupled")
        assert isinstance(view, FullStateView)

        elsewhere = Utterance(
            text="about something else entirely",
            item_id="other",
            concept_id="a_different_concept",
        )
        view = replace(view, reflections=view.reflections + (elsewhere,))
        for concept_id in {u.concept_id for u in view.reflections}:
            assert all(u.concept_id == concept_id for u in view.said_about(concept_id))
        assert elsewhere not in view.said_about("integer_arithmetic")

    def test_it_is_not_sent_to_the_verifier(self, toy) -> None:
        # The tell: prose reaching the verifier comes back UNPARSEABLE and is
        # counted as a step nobody could measure.
        spoken, _ = self._session(toy, "no idea what this means")
        silent, _ = self._session(toy, None)
        assert spoken.board.unmeasurable == silent.board.unmeasurable

    def test_it_costs_no_attempt(self, toy) -> None:
        spoken, with_words = self._session(toy, "no idea what this means")
        silent, without = self._session(toy, None)
        assert with_words.items_attempted == without.items_attempted

    def test_it_updates_no_estimate(self, toy) -> None:
        # Prose is not evidence about mastery, whatever it says.
        spoken, _ = self._session(toy, "I understand this perfectly")
        silent, _ = self._session(toy, None)
        assert spoken.board.state.mastery == silent.board.state.mastery

    def test_it_reaches_the_audit_log(self, toy) -> None:
        session, _ = self._session(toy, "the notation confuses me")
        said = [r for r in session.board.audit_log if "reflection" in r.summary]
        assert said and said[0].evidence["reflection"] == "the notation confuses me"

    def test_the_coupled_view_carries_it_and_the_decoupled_one_does_not(
        self, toy
    ) -> None:
        # It is something the learner told us about themselves, so it belongs
        # with the rest of the learner model the ablation withholds.
        from agent_newton.core.state.views import FullStateView, ItemCorrectnessView

        session, _ = self._session(toy, "I am unsure about the second step")
        assert isinstance(session.board.view(arm="coupled"), FullStateView)
        assert session.board.view(arm="coupled").reflections  # type: ignore[union-attr]
        assert not hasattr(session.board.view(arm="decoupled"), "reflections")
        assert isinstance(session.board.view(arm="decoupled"), ItemCorrectnessView)

    def test_a_simulated_learner_says_nothing(self, toy) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, toy, SimulatorConfig())
        assert learner.reflect(toy.items.all()[0], "what are you unsure of?") is None


class TestShowingYourWorking:
    """Volunteered steps, on the same terms as a reflection.

    A person who worked a problem on paper has reasoning the final answer does
    not carry. Without a channel for it the tutor can only infer the steps, and
    inferring them is what produced a hint telling someone their division was
    correct when they had multiplied.
    """

    def _session(self, toy, working: str | None):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config()
        learner = HumanLearner(
            lambda item, attempt: "999",
            ask_working=(lambda item, response: working) if working else None,
        )
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return session, session.run()

    def test_it_reaches_the_blackboard_marked_as_working(self, toy) -> None:
        session, _ = self._session(toy, "I multiplied instead of dividing")
        working = [u for u in session.board.state.reflections if u.kind == "working"]
        assert working
        assert working[0].text == "I multiplied instead of dividing"

    def test_it_is_not_sent_to_the_verifier(self, toy) -> None:
        # The tell: prose reaching the verifier comes back UNPARSEABLE and is
        # counted as a step nobody could measure.
        shown, _ = self._session(toy, "first I squared it, then I subtracted")
        silent, _ = self._session(toy, None)
        assert shown.board.unmeasurable == silent.board.unmeasurable

    def test_it_costs_no_attempt(self, toy) -> None:
        _, with_working = self._session(toy, "first I squared it")
        _, without = self._session(toy, None)
        assert with_working.items_attempted == without.items_attempted

    def test_it_updates_no_estimate(self, toy) -> None:
        shown, _ = self._session(toy, "I am certain this is right")
        silent, _ = self._session(toy, None)
        assert shown.board.state.mastery == silent.board.state.mastery

    def test_it_is_kept_apart_from_a_reflection(self, toy) -> None:
        # One is a reply to a question the tutor asked, the other is volunteered.
        # The tutor introduces them differently, so they cannot be merged.
        session, _ = self._session(toy, "I divided by two")
        kinds = {u.kind for u in session.board.state.reflections}
        assert kinds == {"working"}

    def test_a_simulated_learner_shows_none(self, toy) -> None:
        # Its answer comes from a buggy rule, not from steps. Keeping this empty
        # is what keeps the channel out of every cohort number.
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, toy, SimulatorConfig())
        assert learner.show_working(toy.items.all()[0], "3*x") is None


class TestRunningOutOfAttemptsIsReported:
    """Previously indistinguishable from nothing happening.

    The next question simply appeared, and a person had no way to tell whether
    they had got the last one right: *"Why did it go to a new question after
    attempt 3? I don't know if I got the answer right."*
    """

    def _observed(self, toy, answer: str):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        class Watcher(Watching):
            def __init__(self) -> None:
                self.finished: list[tuple[str, bool]] = []

            def item_finished(self, item, solved) -> None:  # noqa: ANN001
                self.finished.append((item.id, solved))

        watcher = Watcher()
        config = human_config()
        learner = HumanLearner(lambda item, attempt: answer)
        session = build_session(
            "human", config.seed, toy, config, learner=learner, observer=watcher
        )
        session.diagnostic = Nothing()
        return watcher, session.run()

    def test_an_exhausted_item_is_reported_unsolved(self, toy) -> None:
        watcher, _ = self._observed(toy, "999")
        assert watcher.finished
        assert all(not solved for _, solved in watcher.finished)

    def test_every_item_worked_is_closed_out_exactly_once(self, toy) -> None:
        # Once per *working* of an item, not once per item: the same item is
        # deliberately given again while the concept is unmastered, and a
        # learner is owed the outcome of each attempt at it.
        watcher, outcome = self._observed(toy, "999")
        assert len(watcher.finished) == outcome.items_attempted


class TestCrossConceptDiagnosesAreCounted:
    def test_a_coherent_diagnosis_is_not_counted(self, toy) -> None:
        from agent_newton.core.agents.base import Diagnosis

        item = next(i for i in toy.items.bank("practice") if i.probes)
        label = item.probes[0]

        class Coherent:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(label, confidence=1.0)

        config = human_config()
        learner = HumanLearner(lambda i, attempt: "wrong")
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Coherent()
        outcome = session.run()
        # The label belongs to the concept it was diagnosed on, so nothing is
        # incoherent even though every answer was wrong.
        assert outcome.cross_concept_diagnoses == 0

    def test_an_incoherent_one_is(self, toy) -> None:
        from agent_newton.core.agents.base import Diagnosis

        # A label belonging to a concept the learner is not working on.
        by_concept = {m.id: m.concept_id for m in toy.misconceptions.all()}
        stray = next(iter(by_concept))

        class Stray:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(stray, confidence=1.0)

        config = human_config()
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Stray()
        outcome = session.run()
        worked = {e.concept_id for e in session.board.state.error_trace}
        if worked - {by_concept[stray]}:
            assert outcome.cross_concept_diagnoses > 0
