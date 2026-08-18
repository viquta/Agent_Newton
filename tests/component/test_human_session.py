"""A person in the loop the cohorts run.

The demo drives the real session, so what is tested here is that a human learner
satisfies the same interface a simulated one does and that the session's
handling of *missing ground truth* is honest — a person carries no injected
misconception label, and several things downstream would otherwise report a
number where there is none.
"""

from __future__ import annotations

from collections import Counter
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


class TestMovingOnSaysWhichWayItEnded:
    """"That is all the attempts for this one" — when none had been spent.

    Two ways of not finishing an item, and they had one message between them.
    After three unreadable answers the attempts are all still there and nothing
    about the learner was measured; saying the tries ran out is the same
    category error the attempt budget itself used to carry, told to the person
    it is about.
    """

    def _said(self, toy, reason: str) -> str:
        import io

        from rich.console import Console

        from agent_newton.demo import DemoObserver

        console = Console(file=io.StringIO(), width=90)
        item = toy.items.bank("practice")[0]
        # The pair the session always passes together: `solved` and a reason
        # that agrees with it.
        DemoObserver(console, toy, human_config()).item_finished(
            item, solved=reason == "solved", reason=reason
        )
        printed = console.file.getvalue()  # type: ignore[attr-defined]
        return " ".join(printed.replace("│", " ").split())

    def test_spent_attempts_say_so(self, toy) -> None:
        assert "all the attempts" in self._said(toy, "attempts_spent")

    def test_unreadable_answers_say_something_else(self, toy) -> None:
        said = self._said(toy, "unreadable")
        assert "all the attempts" not in said
        assert "no attempts were used" in said

    def test_both_still_give_the_answer(self, toy) -> None:
        # The item is over either way, and the next question on the concept
        # carries different numbers — which is what makes the reveal teaching
        # rather than a leak.
        answer = toy.items.bank("practice")[0].answer
        assert answer in self._said(toy, "unreadable")
        assert answer in self._said(toy, "attempts_spent")

    def test_a_solved_item_says_nothing(self, toy) -> None:
        assert self._said(toy, "solved").strip() == ""


class TestAShortenedBankSaysWhyItIsShort:
    """A returning learner has sat the long version and would notice.

    And it is not only courtesy: a score out of a re-check is not the same
    measurement as a score out of the whole bank, so the difference has to be
    on the screen where the score is.
    """

    def _said(self, toy, total: int) -> str:
        import io

        from rich.console import Console

        from agent_newton.demo import DemoObserver

        console = Console(file=io.StringIO(), width=90)
        DemoObserver(console, toy, human_config()).phase_started("pretest", total)
        printed = console.file.getvalue()  # type: ignore[attr-defined]
        # Panel borders out, wrapping undone: the assertion is about what was
        # said, not about where rich decided to break the lines.
        return " ".join(printed.replace("│", " ").split())

    def test_a_short_bank_explains_itself(self, toy) -> None:
        whole = len(toy.items.bank("pretest"))
        assert "you have been here before" in self._said(toy, whole - 2)

    def test_the_whole_bank_says_nothing_of_the_kind(self, toy) -> None:
        whole = len(toy.items.bank("pretest"))
        assert "you have been here before" not in self._said(toy, whole)

    def test_it_reads_the_bank_rather_than_the_config(self, toy) -> None:
        # Said from what is actually being administered. Read from the config
        # instead, the two could disagree — and the sentence would be a claim
        # about a setting rather than about the questions in front of someone.
        whole = len(toy.items.bank("pretest"))
        assert f"{whole - 2} of {whole}" in self._said(toy, whole - 2)


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
            ask_working=(lambda item, response, required=False: working) if working else None,
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

            def item_finished(self, item, solved, reason="attempts_spent") -> None:  # noqa: ANN001
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

    @staticmethod
    def _recording():
        """A tutor that keeps the scaffolding rule's two inputs, per turn."""
        from agent_newton.core.agents.base import Hint
        from agent_newton.core.pedagogy import HintLevel, TutorMove

        class Recording:
            def __init__(self) -> None:
                self.failures: list[int] = []
                self.mastery: list[float] = []

            def respond(self, item, diagnosis, view, domain, **kwargs):  # noqa: ANN001
                self.failures.append(kwargs["prior_failures"])
                self.mastery.append(kwargs["mastery"])
                return Hint(
                    text=f"turn {len(self.failures)}",
                    move=TutorMove.HINT,
                    level=HintLevel.NUDGE,
                )

        return Recording()

    def _with_tutor(self, toy, tutor, answers: list[str], **cohort):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        replies = iter(answers)
        last = answers[-1]
        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 1,
                "administer_tests": False,
                **cohort,
            }
        )
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(lambda item, attempt: next(replies, last)),
        )
        session.diagnostic = Nothing()
        session.tutor = tutor
        session.run()
        return session

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

    def test_it_buys_no_support_either(self, toy) -> None:
        """⚠️ This test asserted the opposite, and the opposite was wrong.

        The reasoning was that a learner who has not answered twice is stuck and
        needs more help. True — but the help at the top of the ladder *is the
        answer*, and an unreadable step costs no attempt, so escalating on one
        made the full worked step free and repeatable. A person found it inside
        a minute: type nothing, read the solution, type the solution back, and
        the model records mastery. That is the copying failure the held-out
        banks caught in an earlier sitting, now available on demand.

        So support is earned on work the verifier could read. What stops the
        reply repeating is that the tutor is handed what it already said, which
        gives away nothing that was not earned.
        """
        tutor = self._recording()
        self._with_tutor(toy, tutor, ["no idea"], max_unreadable_per_item=3)
        assert tutor.failures == [0, 0, 0]

    def test_a_readable_failure_is_what_moves_it(self, toy) -> None:
        # The guard can fail: if nothing ever raised the count, the assertion
        # above would hold for a reason that has nothing to do with readability.
        tutor = self._recording()
        self._with_tutor(toy, tutor, ["999"], max_steps_per_item=3)
        assert tutor.failures == [0, 1, 2]

    def test_the_step_being_answered_is_not_counted_against_itself(self, toy) -> None:
        """The first hint is at the mastery baseline, not one rung above it.

        The tutor is only ever called after a failure, so a count that included
        the current step made the baseline unreachable — every first hint came
        out one level high, and ``nudge`` could not occur at all. Two human
        sittings ran entirely at ``worked_step`` on the strength of it.
        """
        tutor = self._recording()
        self._with_tutor(toy, tutor, ["999"], max_steps_per_item=3)
        assert tutor.failures[0] == 0

    def test_the_tutor_is_told_what_was_believed_before_the_answer(self, toy) -> None:
        """And the posterior it is given does not move within the item.

        A wrong answer lowers the estimate immediately, so a tutor reading the
        live view was handed a belief the failure had already revised — the same
        failure then raised the level a second time through the escalation. At
        0.40 one wrong answer lands at 0.26, which is a full worked step on its
        own.
        """
        tutor = self._recording()
        session = self._with_tutor(toy, tutor, ["999"], max_steps_per_item=3)
        assert len(set(tutor.mastery)) == 1, "the baseline moved within the item"

        # ⚠️ The 0.0 branch is not a fallback for "no belief yet" — it is the
        # belief, and it is inconsistent with every other reader of mastery.
        #
        # `_work_item` passes `seen.probability(item.concept_id, 0.0)`, so a
        # concept with no observations is valued at **0.0**. Everywhere else an
        # unobserved concept sits at the BKT **prior**: `route.remaining`,
        # `route.reached` and `zpd.compute` all use `mastery.get(c, prior)`.
        #
        # Inert at the configured band — 0.0 and the 0.15 prior both fall below
        # `theta_lower / 2` (0.35), so both yield a worked step — which is why
        # nothing has noticed. A narrower band separates them, and then the
        # first hint on every fresh concept is pitched from a belief the model
        # does not hold.
        #
        # Kept as a disjunction because both are reachable: 0.0 before the first
        # observation, a real posterior after. Asserting only the second fails
        # here today (0.0 against a board holding 0.228).
        assert tutor.mastery[0] == pytest.approx(0.0) or tutor.mastery[0] in set(
            session.board.state.mastery.values()
        )

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


class TestTheScaffoldingLadderIsReachable:
    """⚠️ Two human sittings ran entirely at ``worked_step``. This is the guard.

    The first was diagnosed as the seeded value: ``pretest_weight: 3`` drove a
    missed concept to 0.0003, under ``theta_lower / 2``, where the rule gives
    maximum support. ``seed_floor`` lifted that to 0.40 — and the second sitting
    came out at ``worked_step`` on five of its six turns anyway, because the
    diagnosis had been incomplete. The failure being responded to was counted
    twice more: once in the posterior, which the tutor read live *after* the
    answer had lowered it, and once in the escalation, which included the
    current step. Either alone put a 0.40 concept at the top of the ladder.

    Asserted on the turns the session recorded rather than on ``hint_level``,
    because ``hint_level`` was correct throughout. It was being called with
    arguments the running system never otherwise produced.
    """

    def _levels(self, toy, answer: str, bank_answer: str = "999", **cohort) -> list[str]:
        """Turn levels on the first training item.

        ``bank_answer`` is answered during the held-out banks and ``answer``
        during training, because the two have to differ for the unreadable case:
        an unreadable pre-test answer measures nothing and therefore seeds
        nothing, so the floor would not bind and the concept would sit at the
        prior — which gives a worked step for a reason that has nothing to do
        with the rule under test.
        """
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={
                "n_learners": 1,
                "max_items": 1,
                "administer_tests": True,
                "seed_from_pretest": True,
                "pretest_weight": 3,
                "seed_floor": 0.40,
                **cohort,
            }
        )
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(
                lambda item, attempt: answer if item.bank == "practice" else bank_answer
            ),
        )
        session.diagnostic = Nothing()
        session.run()

        turns = [r for r in session.board.audit_log if r.cause == "tutor"]
        assert turns, "the sitting produced no tutor turns"
        first = turns[0].evidence["item_id"]
        return [t.evidence["level"] for t in turns if t.evidence["item_id"] == first]

    def test_a_seeded_gap_starts_below_a_worked_step(self, toy) -> None:
        # Everything wrong, so every concept is seeded to the floor. The first
        # hint on the first item is the one both sittings got wrong.
        assert self._levels(toy, "999")[0] == "targeted"

    def test_and_escalates_on_the_next_failure(self, toy) -> None:
        # The other half: starting lower must not mean staying lower. This is
        # the ladder `seed_floor` was introduced to provide.
        assert self._levels(toy, "999")[:2] == ["targeted", "worked_step"]

    def test_an_unreadable_step_never_reaches_a_worked_step(self, toy) -> None:
        # The free answer, from the session end: nothing readable was submitted,
        # so nothing earned the level that states the result.
        levels = self._levels(toy, "no idea", max_unreadable_per_item=3)
        assert levels == ["targeted"] * 3

    def test_a_nudge_is_reachable_at_all(self, toy) -> None:
        # It was not. With the current step counted, the lowest level any
        # learner could be given was one above their baseline, so `nudge`
        # required a mastery no learner in the band could hold.
        from agent_newton.core.pedagogy import HintLevel, hint_level

        assert hint_level(0.96, 0, ZPDConfig()) is HintLevel.NUDGE


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
            ask_working=lambda item, response, required=False: "I guessed",
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

    # --- a learner who masters everything is told the syllabus ran out --------
    #
    # Found by `research_private/tools/session_probe.py`, which drives a real
    # session and checks `stop_reason` against the state rather than against
    # itself: 5 of 5 goals mastered, reported as `nothing_left_to_select`.
    #
    # The disjunction in the test above is why nothing caught it. It accepts
    # either answer, so it passes whichever one the loop gives — which is the
    # shape of test that cannot fail in the direction that matters.
    #
    # The cause is recorded in `11f8bf2`: `_retarget` leaves the last plan on the
    # board when the planner proposes nothing, so `board.plan is None` is only
    # ever true before the first plan is set. `every_goal_reached` is therefore
    # reachable only by a learner who arrives having already mastered everything.

    def _masters_everything(self, toy):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        config = human_config(
            cohort={"n_learners": 1, "max_items": 400, "administer_tests": False}
        )
        learner = HumanLearner(lambda item, attempt: toy.items.get(item.id).answer)
        session = build_session("human", config.seed, toy, config, learner=learner)
        session.diagnostic = Nothing()
        outcome = session.run()
        return session, outcome

    def test_the_state_says_every_goal_was_mastered(self, toy) -> None:
        # The half that is right, and the reason the label is the only defect:
        # nothing about the measurement is wrong, only what the session calls it.
        _, outcome = self._masters_everything(toy)
        assert outcome.goals_mastered == len(list(toy.concepts.goals()))
        assert outcome.distance_to_goal == 0

    def test_but_it_is_reported_as_having_run_out_of_syllabus(self, toy) -> None:
        # ⚠️ Pins the defect, and does not endorse it. Delete this test when the
        # one below stops being xfail.
        _, outcome = self._masters_everything(toy)
        assert outcome.stop_reason == "nothing_left_to_select"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "known: `_retarget` leaves the last plan on the board, so "
            "`every_goal_reached` is unreachable once a plan has been set. "
            "Remove the marker and the pinning test above when fixed."
        ),
    )
    def test_mastering_every_goal_should_say_so(self, toy) -> None:
        _, outcome = self._masters_everything(toy)
        assert outcome.stop_reason == "every_goal_reached"

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


class TestTheReasoningBehindAWrongAnswer:
    """Asked for, insisted on, and asked *first*.

    A wrong answer alone cannot separate a method that is wrong from arithmetic
    that slipped, and an unreadable one carries nothing at all — which is how a
    sitting produced three identical replies to three blanks. The reasoning is
    taken before the verdict is shown and before the diagnostic looks at the
    step: after the verdict it would be an account of a known error rather than
    the thinking that produced it, and after the diagnosis it could not affect
    anything.
    """

    def _run(self, toy, answer: str, working: str | None = "I halved it"):
        from agent_newton.core.agents.base import Diagnosis

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        asked: list[bool] = []

        def ask_working(item, response, required=False):  # noqa: ANN001
            asked.append(required)
            return working or ""

        config = human_config(
            cohort={"n_learners": 1, "max_items": 1, "administer_tests": False}
        )
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(
                lambda item, attempt: answer, ask_working=ask_working
            ),
        )
        session.diagnostic = Nothing()
        session.run()
        return session, asked

    def test_a_wrong_answer_is_asked_for_it(self, toy) -> None:
        _, asked = self._run(toy, "999")
        assert asked and all(asked), "the prompt did not insist on a failed step"

    def test_an_unreadable_answer_is_too(self, toy) -> None:
        # The case with the most to gain: nothing was measured, so the words are
        # the only thing the step produced.
        _, asked = self._run(toy, "no idea at all")
        assert asked and all(asked)

    def test_a_correct_answer_is_asked_but_not_pressed(self, toy) -> None:
        # Kept rather than dropped with the rest of the burden: "I guessed"
        # under a right answer is the one thing that can tell a lucky guess from
        # knowing it, which is the open question about the mastery estimate.
        _, asked = self._run(toy, toy.items.bank("practice")[0].answer)
        # ⚠️ Was `asked == [False] or asked == []`. The second branch is never
        # taken — verified — and it admitted the one failure that matters here:
        # removing the prompt from correct answers entirely would leave `asked`
        # empty and the test would still pass, silently dropping the only
        # mechanism that can tell a lucky guess from knowing it.
        assert asked == [False]

    def test_it_reaches_the_board_before_the_step_is_recorded(self, toy) -> None:
        # Ordering is the whole point: recorded after, it could not have reached
        # the diagnosis of the step it explains.
        session, _ = self._run(toy, "999")
        causes = [
            r.cause
            for r in session.board.audit_log
            if r.cause in ("annotation", "observation")
            and ("reflection" in r.evidence or "verdict" in r.evidence)
        ]
        assert causes[:2] == ["annotation", "observation"]

    def test_declining_is_recorded_as_declining(self, toy) -> None:
        # A refusal is a fact about the sitting — a rising rate means the prompt
        # has become a tax — and it must not read as a step nobody was asked
        # about.
        session, _ = self._run(toy, "999", working=None)
        declined = [r for r in session.board.audit_log if r.evidence.get("declined")]
        assert declined
        assert not [
            u for u in session.board.state.reflections if u.kind == "working"
        ], "a refusal was stored as if it were reasoning"

    def test_a_simulated_learner_is_unaffected(self, toy) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, toy, SimulatorConfig())
        assert learner.show_working(toy.items.all()[0], "3*x", required=True) is None

    def _moves(self, toy, working: str | None) -> list[str]:
        """The tutor's moves when a misconception is confirmed on every step."""
        from agent_newton.core.agents.base import Diagnosis

        class AlwaysDiagnoses:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(toy.misconceptions.ids()[0], confidence=1.0)

        config = human_config(
            cohort={"n_learners": 1, "max_items": 1, "administer_tests": False}
        )
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(
                lambda item, attempt: "999",
                ask_working=lambda item, response, required=False: working or "",
            ),
        )
        session.diagnostic = AlwaysDiagnoses()
        session.run()
        return [
            r.evidence["move"] for r in session.board.audit_log if r.cause == "tutor"
        ]

    def test_explaining_the_step_is_not_asked_for_twice(self, toy) -> None:
        """⚠️ The sitting asked the same question twice in a row.

        *"before I say — how did you get there?"*, answered in full, and then
        *"which specific part are you least confident about?"* — because the
        error-first rule only knew whether a reflective **turn** had been taken.
        What the rule wants is the learner's reasoning between the error and the
        correction, and it was already there.
        """
        assert self._moves(toy, "I forgot the formula")[0] == "remediate"

    def test_declining_still_earns_the_reflective_turn(self, toy) -> None:
        # The guard can fail: the rule must still bite when nothing was said.
        assert self._moves(toy, None)[0] == "reflect"


class TestEndingTrainingEarly:
    """":e" stops the questions and keeps the measurement.

    Distinct from ":q", which ends the sitting where it stands. Someone who has
    had enough of the practice has still done the work, and the held-out bank is
    what turns that into a measured result rather than a transcript.
    """

    def _run(self, toy, stop_after: int):
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.orchestration.session import StopTraining

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        seen = {"n": 0}

        def answer(item, attempt):  # noqa: ANN001
            if item.bank != "practice":
                return "999"
            seen["n"] += 1
            if seen["n"] > stop_after:
                raise StopTraining
            return "999"

        config = human_config(
            cohort={"n_learners": 1, "max_items": 20, "administer_tests": True}
        )
        session = build_session(
            "human", config.seed, toy, config, learner=HumanLearner(answer)
        )
        session.diagnostic = Nothing()
        return session, session.run()

    def test_training_stops_where_the_learner_said(self, toy) -> None:
        _, outcome = self._run(toy, stop_after=2)
        assert outcome.stop_reason == "learner_ended_it"
        assert outcome.items_attempted <= 3

    def test_the_post_test_still_runs(self, toy) -> None:
        # The whole reason this is not `:q`.
        _, outcome = self._run(toy, stop_after=2)
        assert outcome.posttest.administered
        assert outcome.pretest.administered

    def test_it_is_not_an_exhaustion(self, toy) -> None:
        # The material and the budget both had more to give. Reading this back
        # as the system running out would blame the tutoring for the learner
        # having had enough.
        _, outcome = self._run(toy, stop_after=2)
        assert outcome.items_to_exhaustion is None

    def test_it_reaches_the_audit_log(self, toy) -> None:
        session, _ = self._run(toy, stop_after=2)
        assert any(
            "the learner ended training" in r.summary for r in session.board.audit_log
        )

    def test_the_observer_is_told_before_the_post_test(self, toy) -> None:
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.orchestration.session import StopTraining

        class Nothing:
            def diagnose(self, item, response, domain):  # noqa: ANN001
                return Diagnosis(None)

        class Watcher(Watching):
            def __init__(self) -> None:
                self.order: list[str] = []

            def training_finished(self, reason, items) -> None:  # noqa: ANN001
                self.order.append(f"finished:{reason}")

            def phase_started(self, phase, total) -> None:  # noqa: ANN001
                self.order.append(f"phase:{phase}")

        seen = {"n": 0}

        def answer(item, attempt):  # noqa: ANN001
            if item.bank != "practice":
                return "999"
            seen["n"] += 1
            if seen["n"] > 1:
                raise StopTraining
            return "999"

        watcher = Watcher()
        config = human_config(
            cohort={"n_learners": 1, "max_items": 20, "administer_tests": True}
        )
        session = build_session(
            "human", config.seed, toy, config,
            learner=HumanLearner(answer), observer=watcher,
        )
        session.diagnostic = Nothing()
        session.run()

        assert watcher.order == [
            "phase:pretest",
            "finished:learner_ended_it",
            "phase:posttest",
        ]


class TestWhatTheLearnerSawMatchesWhatWasRecorded:
    """The front end against the audit log, which is the only pair worth checking.

    A support offer is decided by the session and rendered by the demo, and
    those are two objects. Asserting either against itself proves nothing: the
    session was right in the sitting that found this and the log recorded it
    correctly, while the screen showed a learner the rule again on a concept
    they had just carried from 0.30 to 0.80.

    So this drives the real session with the real observer and asks the question
    the demo's own prompt asks — ``observer.support_for(item)`` — at exactly the
    point the demo asks it, then counts against the log.
    """

    def _sitting(self, monkeypatch):
        from agent_newton.core.orchestration.session import build_session
        from agent_newton.demo import DemoObserver
        from rich.console import Console

        calculus = registry.load_domain("calculus")
        config = Config.model_validate(
            {
                "domain": "calculus",
                "arm": "coupled",
                "cohort": {
                    "n_learners": 1,
                    "max_items": 12,
                    "max_steps_per_item": 3,
                    "administer_tests": False,
                },
                # Declared simulated, with a HumanLearner injected below —
                # `scaffold_probe.py` does the same and for the same reason. A
                # person carries no injected label, so the config validator
                # refuses an oracle beside one and is right to; every answer
                # here is correct, so nothing is ever diagnosed.
                "simulator": {"learner": "simulated", "surface": "symbolic"},
                "agents": {
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "goal_directed"},
                },
                "scaffolding": {
                    "policy": "banded_plus",
                    "offer_at_presentation": True,
                },
            }
        )
        observer = DemoObserver(Console(quiet=True), calculus, config)
        seen: list[tuple[str, str | None]] = []

        def ask(item, attempt: int) -> str:
            # What `demo.ask` does, at the point it does it.
            if attempt == 0:
                seen.append((item.id, observer.support_for(item)))
            # Right every time, so items are worked through and come round
            # again — which is the situation the defect needed.
            return item.answer

        session = build_session(
            "victor",
            config.seed,
            calculus,
            config,
            learner=HumanLearner(ask),
            observer=observer,
        )
        session.run()
        return session, seen

    def test_a_panel_carries_support_exactly_when_one_was_offered(
        self, monkeypatch
    ) -> None:
        session, seen = self._sitting(monkeypatch)
        offered = [
            record.evidence["item_id"]
            for record in session.board.audit_log
            if record.cause == "tutor" and record.evidence["move"] == "present"
        ]
        shown = [item_id for item_id, support in seen if support is not None]
        assert shown == offered

    def test_the_same_item_asked_again_higher_up_shows_nothing(
        self, monkeypatch
    ) -> None:
        """The defect itself, as a case rather than as a count.

        A learner answering correctly climbs past ``theta_lower``, and a concept
        is worked until it clears ``theta_upper`` — so an item is posed again
        under the same id with the learner no longer below the line. That second
        posing must be bare.
        """
        session, seen = self._sitting(monkeypatch)
        offered = Counter(
            record.evidence["item_id"]
            for record in session.board.audit_log
            if record.cause == "tutor" and record.evidence["move"] == "present"
        )
        posed = Counter(item_id for item_id, _ in seen)
        shown = Counter(item_id for item_id, support in seen if support is not None)

        # The case has to exist or the assertion below is about nothing: an item
        # posed more than once, whose later postings earned their way out of
        # being offered anything.
        outgrown = [
            item_id for item_id in posed
            if posed[item_id] > 1 and offered[item_id] < posed[item_id]
        ]
        assert outgrown, (
            "no item was posed again after its support stopped, so the case "
            "this test exists for was never reached"
        )
        for item_id in outgrown:
            assert shown[item_id] == offered[item_id], (
                f"{item_id} was posed {posed[item_id]} times and offered support "
                f"{offered[item_id]} time(s), but the learner was shown it "
                f"{shown[item_id]} time(s) — the later posings re-rendered an "
                f"offer the session had decided not to make"
            )
