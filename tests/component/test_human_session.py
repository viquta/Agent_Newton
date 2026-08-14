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

from agent_newton.config import Config, ZPDConfig
from agent_newton.core.orchestration.session import Watching, build_session
from agent_newton.core.simulator.engine import Learner, SimulatedLearner
from agent_newton.core.simulator.human import HumanLearner
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry
from agent_newton.domains.base import Verdict


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


class TestSeedingLeavesRoomToScaffold:
    """Two correct things combined into a wrong one.

    Weighting a held-out answer up drove a missed concept to about 0.0003, and
    ``hint_level`` gives a worked step below ``theta_lower / 2``. So every
    concept the pre-test flagged received maximum support on its first attempt
    and never a nudge — the ladder was dead on exactly the concepts being
    taught. Neither rule was wrong alone, which is why no test saw it.
    """

    def _seeded(self, toy, *, weight: int, floor: float, verdict: Verdict):
        config = human_config(
            cohort={"seed_from_pretest": True, "pretest_weight": weight,
                    "seed_floor": floor}
        )
        board = new_blackboard("L1", 1, toy.concepts, config)
        board.seed_from_test([("distribute", verdict)], weight=weight, floor=floor)
        return board.probability("distribute")

    def test_without_a_floor_a_missed_concept_lands_at_the_bottom(self, toy) -> None:
        from agent_newton.core.pedagogy import HintLevel, hint_level

        seeded = self._seeded(toy, weight=3, floor=0.0, verdict=Verdict.INCORRECT)
        assert seeded < 0.01
        assert hint_level(seeded, 0, ZPDConfig()) is HintLevel.WORKED_STEP

    def test_the_floor_leaves_the_ladder_intact(self, toy) -> None:
        from agent_newton.core.pedagogy import HintLevel, hint_level

        band = ZPDConfig()
        seeded = self._seeded(toy, weight=3, floor=0.40, verdict=Verdict.INCORRECT)
        assert seeded == pytest.approx(0.40)
        # Starts lower and escalates, which is what the band is for.
        assert hint_level(seeded, 0, band) is HintLevel.TARGETED
        assert hint_level(seeded, 1, band) is HintLevel.WORKED_STEP

    def test_the_concept_is_still_a_priority(self, toy) -> None:
        # Far below theta_upper, so it stays selectable and still sorts ahead of
        # anything the learner has demonstrated.
        seeded = self._seeded(toy, weight=3, floor=0.40, verdict=Verdict.INCORRECT)
        assert seeded < ZPDConfig().theta_upper

    def test_the_floor_never_binds_on_a_correct_answer(self, toy) -> None:
        # It is a floor, not a clamp: a demonstrated concept must still clear
        # the band, or seeding would stop skipping what the learner knows.
        seeded = self._seeded(toy, weight=3, floor=0.40, verdict=Verdict.CORRECT)
        assert seeded > ZPDConfig().theta_upper

    def test_the_default_is_no_floor(self, toy) -> None:
        # Every measured result was produced without one.
        assert Config().cohort.seed_floor == 0.0

    def test_a_floor_that_would_unlock_dependants_is_refused(self) -> None:
        # A prerequisite above theta_lower opens what depends on it, so a floor
        # there would let a *wrong* answer open the material behind it.
        with pytest.raises(ValidationError, match="unlock its dependants"):
            human_config(cohort={"seed_floor": 0.70}, zpd={"theta_lower": 0.70,
                                                           "theta_upper": 0.90})

    def test_the_frontier_is_unchanged_by_a_legal_floor(self, toy) -> None:
        from agent_newton.core.state import zpd

        band = ZPDConfig()
        bare = zpd.compute({"integer_arithmetic": 0.0003}, toy.concepts, band, 0.15)
        floored = zpd.compute({"integer_arithmetic": 0.40}, toy.concepts, band, 0.15)
        assert set(bare) == set(floored)


class TestATestDoesNotNameTheConcept:
    """The banks measure unaided ability, and the heading was naming the method.

    Every question was titled ``calculus · chain_rule``, in the held-out banks
    as well as in training: *"I'm kind of cheating here cause I can see what I
    need to use to solve it."*
    """

    def test_training_still_names_it(self, toy) -> None:
        from agent_newton.demo import question_title

        item = toy.items.bank("practice")[0]
        assert item.concept_id in question_title(toy, item, 0, testing=False)

    def test_a_test_does_not(self, toy) -> None:
        from agent_newton.demo import question_title

        item = toy.items.bank("pretest")[0]
        title = question_title(toy, item, 0, testing=True)
        assert item.concept_id not in title
        assert title == toy.name

    def test_the_attempt_number_is_training_only(self, toy) -> None:
        # A test gives one attempt, so there is nothing to number — and the
        # number would be a second thing the heading leaked.
        from agent_newton.demo import question_title

        item = toy.items.bank("practice")[0]
        assert "attempt 2" in question_title(toy, item, 1, testing=False)
        assert "attempt" not in question_title(toy, item, 1, testing=True)

    def test_the_flag_follows_the_phase(self, toy) -> None:
        # Read from the session's own phase hooks rather than tracked twice.
        from rich.console import Console

        from agent_newton.core.evaluation.outcomes import TestResult
        from agent_newton.demo import DemoObserver

        observer = DemoObserver(
            Console(file=open("/dev/null", "w")), toy, human_config()
        )
        assert observer.testing is False
        observer.phase_started("pretest", 6)
        assert observer.testing is True
        observer.phase_finished("pretest", TestResult(correct=1, total=6))
        assert observer.testing is False


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


class TestAnUnreadableAnswerIsNotAnAttempt:
    """``UNPARSEABLE`` is a failure to measure, and the loop charged for it.

    The learner model has honoured the distinction from the start: an
    unreadable response updates no estimate and enters no error trace. The
    attempt budget did not, so a response nobody could read spent one of the
    three tries at the item — the same category error the test score carried
    until ``measured_score`` was drawn apart from ``score``.

    It cannot be free either. Without a bound of its own, an empty answer would
    be asked for indefinitely, so the two counters are separate and both bind.
    """

    def _run(self, toy, answers: list[str], **cohort):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        replies = iter(answers)
        # The last scripted answer repeats, so a test states only the run it
        # cares about rather than padding to the budget.
        last = answers[-1]
        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 1,
                "administer_tests": False,
                **cohort,
            }
        )
        learner = HumanLearner(lambda item, attempt: next(replies, last))
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        outcome = session.run()
        return session, outcome

    @staticmethod
    def _graded(session) -> list[str]:
        """The verdict of every step the loop actually graded, in order."""
        return [
            record.evidence["verdict"]
            for record in session.board.audit_log
            if record.cause == "observation"
        ]

    def test_a_readable_wrong_answer_still_spends_one(self, toy) -> None:
        # The budget has to keep binding, or the fix below is just a loop that
        # never ends.
        session, _ = self._run(toy, ["999"], max_steps_per_item=3)
        assert self._graded(session) == ["incorrect"] * 3

    def test_an_unreadable_one_does_not(self, toy) -> None:
        # Same budget, one more step: the unreadable answer did not consume any
        # of the three tries, because nothing about the learner was measured.
        session, _ = self._run(toy, ["what does this mean", "999"], max_steps_per_item=3)
        assert self._graded(session) == ["unparseable"] + ["incorrect"] * 3

    def test_a_run_of_them_ends_the_item(self, toy) -> None:
        session, _ = self._run(
            toy, ["no idea"], max_steps_per_item=3, max_unreadable_per_item=3
        )
        assert self._graded(session) == ["unparseable"] * 3
        assert session.board.unmeasurable == 3

    def test_it_is_the_cap_that_ends_it(self, toy) -> None:
        # And the cap is what decides when — not the attempt budget, which is
        # untouched here and would otherwise have stopped it at three.
        session, _ = self._run(
            toy, ["no idea"], max_steps_per_item=3, max_unreadable_per_item=5
        )
        assert self._graded(session) == ["unparseable"] * 5

    def test_the_reason_reaches_the_audit_log(self, toy) -> None:
        # Ending an item having measured nothing is not the same event as
        # running out of attempts, and a log that cannot tell them apart makes a
        # failing verifier look like a failing learner.
        session, _ = self._run(toy, ["no idea"], max_unreadable_per_item=2)
        assert any(
            "unreadable response(s)" in record.summary
            for record in session.board.audit_log
        )

    def test_nothing_it_produced_was_evidence(self, toy) -> None:
        session, _ = self._run(toy, ["no idea"], max_unreadable_per_item=3)
        assert session.board.state.mastery == {}
        assert not session.board.state.error_trace
        assert not session.board.state.outcomes

    def test_the_item_is_still_counted_as_given_exactly_once(self, toy) -> None:
        # The lifetime count drives which variant is asked and which item is
        # least practised. Counting a step rather than a giving would make an
        # unreadable answer look like extra practice.
        session, _ = self._run(toy, ["no idea"], max_unreadable_per_item=3)
        assert set(session.board.state.items_given.values()) == {1}

    def test_support_escalates_even_though_no_attempt_was_spent(self, toy) -> None:
        """The other half of the decision, and the half a person notices.

        A learner who has now not answered twice is stuck, and the one reply
        that is certainly not working is the one already given. Support
        escalates on the step rather than on the attempt.
        """
        from agent_newton.core.agents.base import Diagnosis, Hint
        from agent_newton.core.pedagogy import HintLevel, TutorMove

        class Recording:
            def __init__(self) -> None:
                self.steps: list[int] = []

            def respond(self, item, diagnosis, view, domain, **kwargs):  # noqa: ANN001
                self.steps.append(kwargs["unresolved_steps"])
                return Hint(
                    text=f"turn {len(self.steps)}",
                    move=TutorMove.HINT,
                    level=HintLevel.NUDGE,
                )

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        replies = iter(["no idea"])
        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 1,
                "administer_tests": False,
                "max_unreadable_per_item": 3,
            }
        )
        tutor = Recording()
        learner = HumanLearner(lambda item, attempt: next(replies, "no idea"))
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        session.tutor = tutor
        session.run()

        assert tutor.steps == [1, 2, 3]

    def test_the_tutor_is_given_what_it_already_said(self, toy) -> None:
        # Which is what stops the three identical replies: the prompt is
        # otherwise the same on every one of them.
        from agent_newton.core.agents.base import Diagnosis, Hint
        from agent_newton.core.pedagogy import HintLevel, TutorMove

        class Recording:
            def __init__(self) -> None:
                self.said: list[list[str]] = []

            def respond(self, item, diagnosis, view, domain, **kwargs):  # noqa: ANN001
                self.said.append(list(kwargs["said_this_item"]))
                return Hint(
                    text=f"turn {len(self.said)}",
                    move=TutorMove.HINT,
                    level=HintLevel.NUDGE,
                )

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 1,
                "administer_tests": False,
                "max_unreadable_per_item": 3,
            }
        )
        tutor = Recording()
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(lambda item, attempt: "no idea"),
        )
        session.diagnostic = Nothing()
        session.tutor = tutor
        session.run()

        assert tutor.said == [[], ["turn 1"], ["turn 1", "turn 2"]]


class TestASittingIsKeptHoweverItEnds:
    """The demo returned before writing whenever the person stopped part-way.

    Four human sittings produced no record on disk at all — no transcript, no
    working, no reflections, no audit log — because every one of them was ended
    with :q or Ctrl-C. A sitting is unrepeatable and is the only data of its
    kind this project has, so an interrupted one must still be stored.
    """

    def _session(self, toy, tmp_path):
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.demo import _store

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={"n_learners": 1, "max_items": 3, "administer_tests": False},
            paths={"results_dir": str(tmp_path), "cache_dir": str(tmp_path / "c")},
        )
        learner = HumanLearner(
            lambda item, attempt: "999",
            ask_working=lambda item, response: "I guessed",
        )
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return config, session, learner, _store

    def test_an_interrupted_sitting_is_written(self, toy, tmp_path) -> None:
        import json

        config, session, learner, store = self._session(toy, tmp_path)
        # Stopped part-way: some work happened, then the person quit, so there
        # is no SessionOutcome at all.
        session._work_item(toy.items.bank("practice")[0], [])
        run_dir = store(config, toy, session, learner, None)

        record = json.loads((run_dir / "transcript.json").read_text())
        assert record["completed"] is False
        assert record["responses"], "the answers were lost"
        assert record["said"], "the working was lost"
        assert record["audit_log"], "the audit log was lost"
        assert (run_dir / "manifest.json").exists()

    def test_an_interrupted_sitting_reports_no_outcome_figures(
        self, toy, tmp_path
    ) -> None:
        # Absent, not zero. The session did not finish, so it has no gain and no
        # final distance, and a zero would say the opposite of what it means.
        import json

        config, session, learner, store = self._session(toy, tmp_path)
        session._work_item(toy.items.bank("practice")[0], [])
        run_dir = store(config, toy, session, learner, None)

        record = json.loads((run_dir / "transcript.json").read_text())
        for absent in ("pretest", "posttest", "goals_mastered", "distance_to_goal"):
            assert absent not in record

    def test_the_transcript_carries_what_the_tutor_said(self, toy, tmp_path) -> None:
        # Half of a sitting is the replies. Lifted out of the audit log because
        # a transcript that has to be filtered by cause string to find the
        # teaching is not a record of a conversation.
        import json

        config, session, learner, store = self._session(toy, tmp_path)
        outcome = session.run()
        run_dir = store(config, toy, session, learner, outcome)

        record = json.loads((run_dir / "transcript.json").read_text())
        assert record["turns"], "the tutor's replies were lost"
        for turn in record["turns"]:
            assert turn["text"] and turn["move"] and turn["level"]
            assert turn["item_id"] and turn["concept_id"]

    def test_a_completed_sitting_carries_them(self, toy, tmp_path) -> None:
        import json

        config, session, learner, store = self._session(toy, tmp_path)
        outcome = session.run()
        run_dir = store(config, toy, session, learner, outcome)

        record = json.loads((run_dir / "transcript.json").read_text())
        assert record["completed"] is True
        assert record["stop_reason"] == outcome.stop_reason
        assert record["items_attempted"] == outcome.items_attempted


class TestTheSittingSaysWhetherItTaughtAnything:
    """The question a sitting exists to answer, and could not.

    A pre-to-post percentage cannot say whether the training reached anything
    that needed teaching. One sitting read −13% and looked like the system
    harming the learner; it had spent 21 of its 24 steps on concepts the
    pre-test had already shown were fine. Finding that took a hand count of the
    transcript.
    """

    def _run(self, toy, answer: str = "999"):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={"n_learners": 1, "max_items": 4, "administer_tests": True}
        )
        learner = HumanLearner(lambda item, attempt: answer)
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return config, session, session.run()

    def test_the_panel_renders(self, toy) -> None:
        from rich.console import Console

        from agent_newton.demo import _did_this_teach

        config, session, outcome = self._run(toy)
        panel = _did_this_teach(session, outcome, toy)
        # Rendered rather than merely constructed: a Text built from a concept
        # id that does not resolve raises here and nowhere earlier.
        Console(width=90, file=open("/dev/null", "w")).print(panel)
        assert panel.title == "did this teach you anything"

    def test_the_transcript_carries_what_moved_and_where_the_time_went(
        self, toy, tmp_path
    ) -> None:
        import json

        from agent_newton.demo import _store

        config, session, outcome = self._run(toy)
        config.paths.results_dir = tmp_path
        config.paths.cache_dir = tmp_path / "c"
        run_dir = _store(config, toy, session, learner=session.learner, outcome=outcome)

        record = json.loads((run_dir / "transcript.json").read_text())
        assert record["per_concept_change"]
        assert {c["state"] for c in record["per_concept_change"]} <= {
            "fixed", "lost", "still_wrong", "still_right", "unmeasured"
        }
        assert record["dose_by_concept"]
        assert "dose_on_gap" in record
        assert "normalised_gain" in record

    def test_the_dose_only_counts_training(self, toy) -> None:
        # The banks are administered around the training and update nothing;
        # counting them would report measurement as instruction.
        from agent_newton.core.evaluation.outcomes import dose_by_concept

        _, session, outcome = self._run(toy)
        assert sum(dose_by_concept(session.board.audit_log).values()) <= (
            outcome.items_attempted * 3
        )


class TestTheHistoryAcrossSittings:
    """What has been tried on a concept over a whole history.

    None for a first sitting: there is no history to be across, and an empty
    panel saying so would be noise.
    """

    def _store_with(self, tmp_path, sittings: int):
        from agent_newton.core.state.schema import AuditRecord, LearnerState
        from agent_newton.store import LearnerStore

        store = LearnerStore(tmp_path / "learners.db")
        store.ensure_learner("L1", "human", "toy_algebra")
        for _ in range(sittings):
            session_id = store.open_session(
                learner_id="L1", arm="coupled", config_hash="h", elapsed_days=7.0
            )
            store.close_session(
                session_id,
                state=LearnerState(learner_id="L1", seed=1),
                audit_log=[
                    AuditRecord(
                        version=1, cause="observation", summary="",
                        evidence={
                            "item_id": "ta_dist_p1", "concept_id": "distribute",
                            "verdict": "incorrect",
                            "mastery_before": 0.15, "mastery_after": 0.12,
                        },
                    )
                ],
            )
        return store

    def test_a_first_sitting_has_no_history(self, toy, tmp_path) -> None:
        from agent_newton.demo import _across_sittings

        store = self._store_with(tmp_path, sittings=1)
        assert _across_sittings(store, "L1", human_config(), toy) is None
        store.close()

    def test_a_returning_learner_gets_one(self, toy, tmp_path) -> None:
        from rich.console import Console

        from agent_newton.demo import _across_sittings

        store = self._store_with(tmp_path, sittings=3)
        panel = _across_sittings(store, "L1", human_config(), toy)
        store.close()

        assert panel is not None
        # Rendered, not merely built: a concept id that does not resolve to a
        # name raises here and nowhere earlier.
        Console(width=90, file=open("/dev/null", "w")).print(panel)
        assert panel.title == "across your sittings"


class TestWhyTrainingStopped:
    """The three exits are different events and must be told apart.

    Running out of budget recorded nothing at all, so a sitting that ended
    mid-concept looked to the person like the system giving up on them: the
    post-test simply appeared after a failed item.
    """

    def _run(self, toy, *, max_items: int):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={"n_learners": 1, "max_items": max_items, "administer_tests": False}
        )
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        return session, session.run()

    def test_a_spent_budget_says_so(self, toy) -> None:
        session, outcome = self._run(toy, max_items=3)
        assert outcome.stop_reason == "budget_spent"
        assert any("item budget spent" in r.summary for r in session.board.audit_log)

    def test_a_spent_budget_is_not_an_exhaustion(self, toy) -> None:
        # `items_to_exhaustion` means the material ran out. A capped session did
        # not run out of anything, and reporting it as though it had would read
        # a setting as a result.
        _, outcome = self._run(toy, max_items=3)
        assert outcome.items_to_exhaustion is None

    def test_running_out_of_material_says_something_different(self, toy) -> None:
        # Answering correctly masters concepts until nothing is selectable.
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(cohort={"n_learners": 1, "max_items": 400,
                                      "administer_tests": False})
        learner = HumanLearner(lambda item, attempt: toy.items.get(item.id).answer)
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        outcome = session.run()
        assert outcome.stop_reason in ("every_goal_reached", "nothing_left_to_select")
        assert outcome.items_to_exhaustion is not None

    def test_the_observer_is_told(self, toy) -> None:
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        class Watcher(Watching):
            def __init__(self) -> None:
                self.finished: list[tuple[str, int]] = []

            def training_finished(self, reason, items) -> None:  # noqa: ANN001
                self.finished.append((reason, items))

        watcher = Watcher()
        config = human_config(
            cohort={"n_learners": 1, "max_items": 3, "administer_tests": False}
        )
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session(
            "human", config.seed, toy, config, learner=learner, observer=watcher
        )
        session.diagnostic = Nothing()
        outcome = session.run()

        assert watcher.finished == [("budget_spent", outcome.items_attempted)]

    def test_it_is_reported_before_the_post_test(self, toy) -> None:
        # Order matters for a person: the explanation has to arrive before the
        # thing it explains, or it is not an explanation.
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        class Watcher(Watching):
            def __init__(self) -> None:
                self.order: list[str] = []

            def training_finished(self, reason, items) -> None:  # noqa: ANN001
                self.order.append("training_finished")

            def phase_started(self, phase, total) -> None:  # noqa: ANN001
                self.order.append(f"phase:{phase}")

        watcher = Watcher()
        config = human_config(
            cohort={"n_learners": 1, "max_items": 2, "administer_tests": True}
        )
        learner = HumanLearner(lambda item, attempt: "999")
        session = build_session(
            "human", config.seed, toy, config, learner=learner, observer=watcher
        )
        session.diagnostic = Nothing()
        session.run()

        assert watcher.order == ["phase:pretest", "training_finished", "phase:posttest"]


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
