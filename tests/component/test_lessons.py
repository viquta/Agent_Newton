"""Explaining a concept, as opposed to correcting an attempt at one.

Every instructional move the artifact had was a reply to a failed step, so a
learner who had never met a concept and one who held a misconception about it
were answered identically. ``TutorMove.EXPLAIN`` is the move that answers the
first case, and these are the properties it has to hold.

The two that matter most are both about not disturbing the study: a lesson
targets nothing, so it cannot be counted as remediation, and its trigger is
computed from the shared state, so it fires at the same rate in both arms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from agent_newton.config import Config
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.pedagogy import (
    TeachingStyle,
    TutorMove,
    check_move,
    should_explain,
    style_for,
)
from agent_newton.core.simulator.engine import SimulatedStep
from agent_newton.domains import registry
from agent_newton.domains.base import VerificationResult, Verdict


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


def _config(explain_after: int = 3, arm: str = "coupled") -> Config:
    """A model-free calculus run with teaching turned on."""
    return Config.model_validate(
        {
            "domain": "calculus",
            "arm": arm,
            "seed": 20260807,
            "cohort": {"n_learners": 1, "max_items": 8, "administer_tests": False},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed"},
            },
            "teaching": {"explain_after": explain_after},
        }
    )


class AlwaysWrong:
    """A learner who never gets anything right, and remembers being hinted at.

    ``hints`` is what the dangerous test below reads: a lesson must never reach
    ``receive_hint``, because that is what moves a misconception's firing
    probability and therefore the declared primary outcome.
    """

    learner_id = "L_wrong"

    def __init__(self, says: Sequence[str] = ()) -> None:
        self.hints: list[str | None] = []
        #: What this learner says when a concept is explained to them, in order.
        #: Empty is the default and matches `SimulatedLearner`, which returns
        #: None and so ends a lesson at its opening turn — the behaviour every
        #: cohort has whatever the config says.
        self._says = list(says)
        #: Every prompt the tutor put to them during a lesson.
        self.asked: list[str] = []

    #: Parseable, and wrong for every item in the bank. It has to be readable:
    #: an answer the verifier cannot parse is `UNPARSEABLE`, which never enters
    #: the error trace and therefore cannot buy a lesson — which is a separate
    #: property, asserted separately below. Prose here would have made every
    #: test in this file pass for the wrong reason.
    WRONG = "42"

    def answer(self, item, attempt: int = 0, repetition: int = 0) -> SimulatedStep:
        return SimulatedStep(response=self.WRONG, fired=None, correct=False)

    def reflect(self, item, prompt: str) -> str | None:
        return None

    def show_working(self, item, response: str, required: bool = False) -> str | None:
        return None

    def discuss(self, concept_id: str, prompt: str) -> str | None:
        self.asked.append(prompt)
        return self._says.pop(0) if self._says else None

    def receive_hint(self, targeted_misconception: str | None) -> bool:
        self.hints.append(targeted_misconception)
        return False

    def remediation_ratio(self) -> float | None:
        return None


def _board(domain):
    """A fresh blackboard, for the pieces that need one without a session."""
    from agent_newton.core.state.store import new_blackboard

    return new_blackboard("L_probe", 1, domain.concepts, _config())


def _turns_in(board) -> list[dict]:
    """Every turn of every lesson, openings and replies and summaries alike."""
    return [
        record.evidence
        for record in board.audit_log
        if record.cause == "tutor"
        and record.evidence.get("move") == TutorMove.EXPLAIN.value
    ]


def _lessons_in(board) -> list[dict]:
    """One entry per *lesson*, which is the opening turn of each.

    A lesson is a conversation, so counting turns counts exchanges. Everything
    that asks "how many lessons has this learner had" has to count openings —
    see the counting note on `Session._offer_lesson`.
    """
    return [turn for turn in _turns_in(board) if turn.get("opening")]


def _summaries_in(board) -> list[dict]:
    return [turn for turn in _turns_in(board) if turn.get("level") == "summary"]


class TestTheTrigger:
    """When a lesson is owed, expressed as a predicate so it can be argued with."""

    def test_off_is_off_however_much_the_learner_struggles(self) -> None:
        assert not should_explain(999, 0, after=0)

    def test_below_the_threshold_nothing_is_owed(self) -> None:
        assert not should_explain(2, 0, after=3)

    def test_at_the_threshold_a_lesson_is_owed(self) -> None:
        assert should_explain(3, 0, after=3)

    def test_a_lesson_already_given_raises_the_bar_again(self) -> None:
        # Otherwise it would fire on every item after the third error.
        assert not should_explain(3, 1, after=3)
        assert not should_explain(5, 1, after=3)
        assert should_explain(6, 1, after=3)

    def test_errors_across_different_items_count_together(self) -> None:
        # The case the design note gives: three wrong on one question and two on
        # another is five errors on one concept, not two separate runs of
        # nothing. This is why the count comes from the error trace rather than
        # from the per-item attempt counter, which resets.
        assert should_explain(3 + 2, 0, after=5)


class TestExplainIsNotAHintLevel:
    """⚠️ The trap the design note names, kept as an assertion.

    A fourth level above ``WORKED_STEP`` would be reached after one failure in
    the decoupled arm and three in the coupled one, because the decoupled view
    carries no posteriors and its tutor already reads 0.0. The arm defined by
    having less information would receive more teaching, and it would look like
    the coupling advantage disappearing rather than like a defect.
    """

    def test_it_is_a_move_rather_than_a_level(self) -> None:
        from agent_newton.core.pedagogy import HintLevel

        assert TutorMove.EXPLAIN in TutorMove
        assert "EXPLAIN" not in HintLevel.__members__
        assert max(HintLevel) is HintLevel.WORKED_STEP

    def test_the_rules_admit_it_deliberately(self) -> None:
        # Stated as its own branch in `check_move` rather than reached by
        # falling through the remediation test. A check that admits a case by
        # accident reads exactly like one that admits it on purpose, and that
        # disjunction shape has hidden a defect here before.
        assert check_move(TutorMove.EXPLAIN, [], misconception_confirmed=True) is None
        assert check_move(TutorMove.EXPLAIN, [], misconception_confirmed=False) is None

    def test_remediation_is_still_ordered_behind_a_reflection(self) -> None:
        # The neighbouring rule must be untouched: admitting a new move must not
        # quietly admit the one the error-first rule exists to hold back.
        assert (
            check_move(TutorMove.REMEDIATE, [], misconception_confirmed=True) is not None
        )


class TestALessonIsNotRemediation:
    """⚠️ The most consequential property on this branch.

    ``remediation_ratio`` is the declared primary outcome and it counts what a
    hint aimed at. A lesson that carried a target would be credited with
    remediation it did not do, and a lesson that reached ``receive_hint`` would
    move the simulated learner's firing probabilities outright.
    """

    def test_a_lesson_targets_nothing(self, calculus) -> None:
        learner = AlwaysWrong()
        session = build_session(
            "L_wrong", 1, calculus, _config(), learner=learner
        )
        session.run()
        lessons = _lessons_in(session.board)
        assert lessons, "the fixture must actually provoke a lesson"
        assert all(lesson["targets"] is None for lesson in lessons)

    def test_a_lesson_never_reaches_the_learner_as_a_hint(self, calculus) -> None:
        # The direct check. `receive_hint` is what decays a misconception's
        # firing probability, so anything reaching it is remediation whatever it
        # is called.
        learner = AlwaysWrong()
        session = build_session("L_wrong", 1, calculus, _config(), learner=learner)
        session.run()
        assert _lessons_in(session.board), "the fixture must actually provoke a lesson"
        assert all(hint is not None for hint in learner.hints), (
            "a hint with no target reached the learner, which is what a lesson "
            "would look like if it were routed through the remediation branch"
        )

    def test_the_move_recorded_is_its_own(self, calculus) -> None:
        learner = AlwaysWrong()
        session = build_session("L_wrong", 1, calculus, _config(), learner=learner)
        session.run()
        allowed = {s.label for s in TeachingStyle} | {"summary"}
        for turn in _turns_in(session.board):
            assert turn["move"] == "explain"
            # The style stands where a hint records its support level, and the
            # closing summary stands beside it. A lesson has no support level —
            # it is not a quantity of the answer — and recording one would
            # invite it to be read as a rung on the ladder, which it is not.
            assert turn["level"] in allowed


class TestTheTriggerIsArmInvariant:
    """The trigger reads the board, never the view.

    The error trace and the audit log live on the shared state and are written
    identically whichever planner is running; only the view differs. A trigger
    keyed on mastery or the frontier would fire at different rates in the two
    arms, and the result would look like a finding rather than a fault — the
    same shape as the fourth-hint-level trap.
    """

    def _session_with_errors(self, calculus, arm: str, errors: int):
        session = build_session(
            "L_arm", 1, calculus, _config(arm=arm), learner=AlwaysWrong()
        )
        for index in range(errors):
            session.board.record_observation(
                item_id="ca_pow_p1",
                concept_id="power_rule",
                result=VerificationResult(Verdict.INCORRECT, "n x^(n-1)"),
                misconception_label=None,
                confidence=0.0,
                attempt=index,
                response="wrong",
            )
        return session

    def test_both_arms_teach_on_the_same_evidence(self, calculus) -> None:
        for arm in ("coupled", "decoupled"):
            session = self._session_with_errors(calculus, arm, errors=3)
            assert session._offer_lesson("power_rule"), (
                f"the {arm} arm did not teach on evidence the other one did; the "
                f"trigger has picked up something view-mediated"
            )

    def test_both_arms_withhold_on_the_same_evidence(self, calculus) -> None:
        for arm in ("coupled", "decoupled"):
            session = self._session_with_errors(calculus, arm, errors=2)
            assert not session._offer_lesson("power_rule")


class TestWhatCannotBuyALesson:
    def test_an_unreadable_answer_cannot(self, calculus) -> None:
        """⚠️ "The learner keeps failing" and "the verifier keeps failing" are
        different events, and only one of them is about the learner.

        ``UNPARSEABLE`` never enters the error trace, so this holds by
        construction — but it holds by construction only as long as the trigger
        reads the trace rather than counting steps, so it is worth asserting.
        """
        session = build_session(
            "L_unread", 1, calculus, _config(), learner=AlwaysWrong()
        )
        for index in range(6):
            session.board.record_observation(
                item_id="ca_pow_p1",
                concept_id="power_rule",
                result=VerificationResult(Verdict.UNPARSEABLE, "n x^(n-1)"),
                misconception_label=None,
                confidence=0.0,
                attempt=index,
                response="???",
            )
        assert not session._offer_lesson("power_rule")

    def test_a_concept_with_no_lesson_teaches_nothing(self, calculus) -> None:
        # Optional content, legitimately absent. The loop simply has nothing to
        # say, and says nothing rather than saying something empty.
        from dataclasses import replace

        from agent_newton.domains.content import YamlConceptResources

        stripped = replace(
            calculus,
            resources=YamlConceptResources(
                [
                    replace(r, what_it_means="", why_it_works="")
                    for r in calculus.resources.all()
                ]
            ),
        )
        learner = AlwaysWrong()
        session = build_session("L_none", 1, stripped, _config(), learner=learner)
        session.run()
        assert not _lessons_in(session.board)

    def test_off_by_default_nothing_is_taught(self, calculus) -> None:
        learner = AlwaysWrong()
        session = build_session(
            "L_off", 1, calculus, _config(explain_after=0), learner=learner
        )
        session.run()
        assert not _lessons_in(session.board)


class TestTheStyleIsChosenByARule:
    """Not by a prompt, like every other instructional decision here.

    A model asked to "be Socratic if that seems better" can talk itself out of
    it, and a constraint a model can talk itself out of is not one. The same
    reasoning already puts the support level and the move in the rules rather
    than in the tutor's instructions.
    """

    def test_the_first_lesson_is_the_authored_one(self) -> None:
        # PLAIN is the text a person wrote and the validator checked, and it
        # needs no model — so the first lesson on any concept is the same in a
        # model-free run as at a keyboard.
        assert style_for(0) is TeachingStyle.PLAIN

    def test_a_repeat_lesson_is_told_differently(self) -> None:
        """The ideas note's point, and the reason the rotation exists.

        *After receiving teaching-point_z, they still seem to misunderstand it*
        — a learner who did not understand the plain account is unlikely to be
        helped by the plain account again.
        """
        assert style_for(1) is not TeachingStyle.PLAIN
        assert style_for(2) not in {style_for(0), style_for(1)}

    def test_it_comes_back_round_rather_than_running_out(self) -> None:
        # A learner who exhausts the repertoire is still owed a lesson.
        assert style_for(3) is style_for(0)

    def test_what_the_learner_asked_for_wins(self) -> None:
        # Precedence stated explicitly, because two rules that can disagree
        # eventually will. A stated preference is not overridden by "you had
        # that one last time" — the rotation is a guess and the preference is
        # not.
        for given in range(4):
            assert (
                style_for(given, chosen=TeachingStyle.SOCRATIC)
                is TeachingStyle.SOCRATIC
            )

    def test_it_is_learner_input_and_not_learner_model(self, calculus) -> None:
        """Which is what would make it fair to hand to both arms.

        The same footing as ``Emphasis`` and a stated request: a thing a person
        said about themselves, not an inference about what they know. It lives
        beside ``Emphasis`` on the state for that reason.
        """
        from agent_newton.core.state.schema import Emphasis, TeachingStyle as OnTheState

        assert OnTheState is TeachingStyle
        assert Emphasis.__module__ == TeachingStyle.__module__

    def test_the_board_records_the_choice(self, calculus) -> None:
        session = build_session("L_style", 1, calculus, _config(), learner=AlwaysWrong())
        assert session.board.teaching_style is None
        session.board.record_teaching_style(TeachingStyle.REAL_WORLD)
        assert session.board.teaching_style is TeachingStyle.REAL_WORLD
        assert [
            r for r in session.board.audit_log
            if r.evidence.get("teaching_style") == "real_world"
        ], "a choice the learner made must be readable back from the sitting"

    def test_the_chosen_style_is_what_gets_recorded(self, calculus) -> None:
        session = build_session("L_style", 1, calculus, _config(), learner=AlwaysWrong())
        session.board.record_teaching_style(TeachingStyle.REAL_WORLD)
        session.run()
        lessons = _lessons_in(session.board)
        assert lessons
        assert all(lesson["level"] == "real_world" for lesson in lessons)
        # And the summary is the authored account whatever style was chosen —
        # the style is how it was talked about, not what is left behind.
        assert _summaries_in(session.board)


class TestTheTemplateTutorIgnoresTheStyle:
    """⚠️ The cohorts run it, so its output must not vary with anything.

    A lesson whose wording moved with the style would make every measured number
    depend on something outside the manipulation — the same reasoning that keeps
    ``respond`` from varying with the learner's response.
    """

    def test_every_style_produces_the_same_text(self, calculus) -> None:
        from agent_newton.config import ZPDConfig
        from agent_newton.core.agents.tutor import TemplateTutor

        tutor = TemplateTutor(ZPDConfig())
        resource = calculus.resources.get("power_rule")
        texts = {tutor.explain(resource, style) for style in TeachingStyle}
        assert len(texts) == 1

    def test_and_it_is_the_authored_lesson(self, calculus) -> None:
        # Ignoring the style is not a degraded lesson. PLAIN *is* the authored
        # text, and it is the one thing every domain offering resources has.
        from agent_newton.config import ZPDConfig
        from agent_newton.core.agents.tutor import TemplateTutor

        resource = calculus.resources.get("power_rule")
        assert (
            TemplateTutor(ZPDConfig()).explain(resource, TeachingStyle.SOCRATIC)
            == resource.lesson()
        )


class TestAModelMayRevoiceALessonButNotWriteOne:
    """The mathematics a learner is taught is authored and validated.

    A model is handed text that ``domain validate`` has already checked is plain
    text and does not answer any item on the concept at any template draw, and
    is asked to say the same thing differently. Generating the mathematics fresh
    at a keyboard would throw those guarantees away.
    """

    class Replies:
        label = "fake/model"

        def __init__(self, text: str) -> None:
            self._text = text
            self.calls = 0

        def generate(self, prompt, schema, system):  # noqa: ANN001
            from agent_newton.llm.base import Completion
            import json

            self.calls += 1
            return Completion(
                text=json.dumps({"text": self._text}),
                model="fake",
                provider="fake",
            )

    def _tutor(self, provider):
        from agent_newton.config import ZPDConfig
        from agent_newton.core.agents.llm import LLMTutor

        return LLMTutor(provider, ZPDConfig())

    def test_every_style_opens_by_talking(self, calculus) -> None:
        """⚠️ Changed deliberately, and it is the point of the revision.

        `PLAIN` used to return the authored text untouched and call no model.
        That made it an exposition rather than an opening, and a learner said
        what the whole thing then reads like: *"I really thought that would be
        more of a dialogue between me and the Tutor."* Every style now opens
        with something the learner can reply to.

        Nothing is lost by it. The authored text is still what the learner is
        left holding — it is the summary, and the summary calls no model.
        """
        for style in TeachingStyle:
            provider = self.Replies("So — what do you think a power does?")
            self._tutor(provider).explain(calculus.resources.get("power_rule"), style)
            assert provider.calls == 1, f"{style.label} did not open a conversation"

    def test_a_lesson_turn_is_not_bounded_by_a_hint_shaped_schema(self) -> None:
        """⚠️ Found by driving a sitting, and it is one defect in a third place.

        A field description goes into the JSON schema, and the schema is what
        constrains decoding — so `HintReply`'s "Two sentences at most" is not
        documentation, it is an instruction. A lesson opening stopped at
        "have you ever worked with the concept of", mid-sentence, because of it.

        `_TUTOR_SYSTEM` once demanded two sentences globally while `WORKED_STEP`
        asked for the step to be worked through; that was fixed by moving the
        budget from the prompt to the level. It survived one layer further down,
        where nothing reads like a length budget at all.
        """
        from agent_newton.core.agents.schemas import HintReply, LessonReply

        assert LessonReply is not HintReply
        hint = HintReply.model_json_schema()["properties"]["text"]["description"]
        lesson = LessonReply.model_json_schema()["properties"]["text"]["description"]
        assert "two sentences" in hint.lower()
        assert "two sentences" not in lesson.lower()

    def test_a_turn_that_stops_mid_sentence_is_refused(self) -> None:
        """⚠️ Observed at a keyboard, not hypothesised.

        An opening came back as *"...have you ever worked with the concept of "*
        — valid JSON, right shape, schema-clean, cut off mid-phrase, and it
        reached the learner looking like a question that had been asked.
        Deterministic at temperature zero, so it recurred identically on every
        re-run: not noise a retry outruns.

        Refusing it here rather than at the call site is what puts it through
        `complete()`'s repair loop, which shows the model its own reply and asks
        again. On the real case that produced a finished question.
        """
        import pydantic

        from agent_newton.core.agents.schemas import LessonReply

        with pytest.raises(pydantic.ValidationError):
            LessonReply(text="have you ever worked with the concept of ")

    def test_a_finished_turn_is_accepted(self) -> None:
        # And the guard can pass, which is the other half of it meaning
        # anything.
        from agent_newton.core.agents.schemas import LessonReply

        for good in (
            "What do you think a rate is?",
            "Try it and see.",
            "Work it through (carefully).",
            'The word "rate" means speed.',
        ):
            assert LessonReply(text=good).text == good

    def test_a_tutor_that_cannot_finish_falls_back_to_the_authored_account(
        self, calculus
    ) -> None:
        # The existing `ProviderError` path, reached through the repair loop
        # giving up. A lesson that cannot be talked through is still a lesson.
        import json

        from agent_newton.llm.base import Completion

        class NeverFinishes:
            label = "fake/model"

            def generate(self, prompt, schema, system):  # noqa: ANN001
                return Completion(
                    text=json.dumps({"text": "and then the concept of "}),
                    model="fake",
                    provider="fake",
                )

        resource = calculus.resources.get("power_rule")
        assert self._tutor(NeverFinishes()).explain(
            resource, TeachingStyle.SOCRATIC
        ) == resource.lesson()

    def test_the_summary_is_authored_and_needs_no_model(self, calculus) -> None:
        # The guarantees belong to the thing the learner keeps. A conversation
        # is re-checked against nothing; the authored account is checked as
        # plain text and checked not to answer any item at any draw.
        resource = calculus.resources.get("power_rule")
        assert resource.lesson()  # composed without a provider at all

    def test_a_styled_lesson_is_used_when_it_comes_back_clean(self, calculus) -> None:
        provider = self.Replies("What happens to the power? It comes down in front.")
        text = self._tutor(provider).explain(
            calculus.resources.get("power_rule"), TeachingStyle.SOCRATIC
        )
        assert text.startswith("What happens to the power?")
        assert provider.calls == 1

    def test_a_backslash_in_the_reply_falls_back_to_the_authored_text(
        self, calculus
    ) -> None:
        """The sitting-2 defect, arriving through a new door.

        A reply comes back as JSON, ``\\f`` parses to a form feed, and a learner
        read ``rac{f(b) - f(a)}{b - a}`` without being able to tell it meant a
        division. Checked against the same pattern the authored content is
        checked against, so a fix to one is a fix to both.
        """
        resource = calculus.resources.get("power_rule")
        provider = self.Replies("the rule is \\frac{n}{x}")
        assert self._tutor(provider).explain(resource, TeachingStyle.SOCRATIC) == (
            resource.lesson()
        )

    def test_a_reply_that_stopped_rephrasing_falls_back(self, calculus) -> None:
        # A re-voicing several times the length of the original has stopped
        # re-voicing and started composing, which is the thing this is built not
        # to do.
        resource = calculus.resources.get("power_rule")
        provider = self.Replies("word " * 4000)
        assert self._tutor(provider).explain(resource, TeachingStyle.SOCRATIC) == (
            resource.lesson()
        )

    def test_a_dead_backend_still_teaches(self, calculus) -> None:
        # The authored lesson is a complete lesson, not a degraded one. Only the
        # style was lost.
        from agent_newton.llm.base import ProviderError

        class Dead:
            label = "fake/dead"

            def generate(self, prompt, schema, system):  # noqa: ANN001
                raise ProviderError("ollama is not running")

        resource = calculus.resources.get("power_rule")
        assert self._tutor(Dead()).explain(resource, TeachingStyle.SOCRATIC) == (
            resource.lesson()
        )


class TestWhatALearnerSeesAndCanAskFor:
    """The demo's half: a lesson has to be visible, and askable for.

    Tested at the observer and the chooser rather than by driving a whole
    sitting, because a sitting needs a model and this is about the wiring.
    """

    def _console(self):
        import io

        from rich.console import Console

        buffer = io.StringIO()
        return Console(file=buffer, width=100, force_terminal=False), buffer

    def test_a_lesson_is_shown_as_its_own_kind_of_panel(self, calculus) -> None:
        # A learner should be able to tell being taught from being corrected.
        # Every other panel on that screen is a response to something they did.
        from agent_newton.demo import DemoObserver

        console, buffer = self._console()
        observer = DemoObserver(console, calculus, _config())
        observer.lesson_offered("power_rule", "a power tells you how fast it grows")
        shown = buffer.getvalue()
        assert "a moment on" in shown
        assert "how fast it grows" in shown
        # And the learner is told how to leave, in the panel that starts it.
        assert ":done" in shown

    def test_the_summary_is_marked_as_the_thing_to_keep(self, calculus) -> None:
        # The conversation above it was written by a model and is checked
        # against nothing; this is the text a person wrote and the validator
        # checked. A learner should be able to tell which one is theirs.
        from agent_newton.demo import DemoObserver

        console, buffer = self._console()
        DemoObserver(console, calculus, _config()).lesson_summary(
            "power_rule", "bring the power down and reduce it by one"
        )
        shown = buffer.getvalue()
        assert "short version" in shown
        assert "yours to keep" in shown

    def test_nothing_is_offered_to_ask_about_during_a_test(self, calculus) -> None:
        """Asking what a concept is mid-test is asking to be told the thing the
        test is measuring.

        The banks are the instrument, and every absolute score before the
        concept came off the question heading is inflated because the display
        named the method. This is the same mistake with a different door.
        """
        from agent_newton.demo import DemoObserver

        console, _ = self._console()
        observer = DemoObserver(console, calculus, _config())
        item = calculus.items.for_concept("power_rule", "practice")[0]
        observer.item_started(item, _board(calculus))
        assert observer.working_concept == "power_rule"
        observer.phase_started("posttest", 5)
        assert observer.working_concept is None

    def test_the_style_chooser_records_what_was_picked(self, calculus) -> None:
        from agent_newton.demo import _ask_how_to_explain

        console, _ = self._console()
        board = _board(calculus)
        _ask_how_to_explain(console, board, lambda prompt, **kw: "2")
        assert board.teaching_style is list(TeachingStyle)[1]

    def test_saying_nothing_leaves_it_to_the_rule(self, calculus) -> None:
        # Not a fallback so much as the better default: a concept explained
        # twice is then put differently the second time, which is what a learner
        # who did not understand the first account needs.
        from agent_newton.demo import _ask_how_to_explain

        console, buffer = self._console()
        board = _board(calculus)
        _ask_how_to_explain(console, board, lambda prompt, **kw: "")
        assert board.teaching_style is None
        assert "differently the second time" in buffer.getvalue()

    def test_nonsense_is_treated_as_no_preference(self, calculus) -> None:
        from agent_newton.demo import _ask_how_to_explain

        console, _ = self._console()
        board = _board(calculus)
        _ask_how_to_explain(console, board, lambda prompt, **kw: "banana")
        assert board.teaching_style is None

    def test_every_style_is_offered_and_described(self, calculus) -> None:
        # A chooser that lists fewer options than exist is how a control quietly
        # stops covering what it claims to.
        from agent_newton.demo import _STYLE_BLURB

        assert set(_STYLE_BLURB) == set(TeachingStyle)


class TestAskingForALesson:
    """``:why`` — the trigger the ideas note lists first.

    Someone saying "I do not know what this is" is better evidence of that than
    three wrong answers are, so it bypasses the difficulty threshold. It does
    not bypass whether the run teaches at all: a run with teaching off has no
    lesson to give, and asking cannot conjure one.
    """

    def test_asking_gets_a_lesson_before_the_threshold(self, calculus) -> None:
        session = build_session("L_ask", 1, calculus, _config(), learner=AlwaysWrong())
        assert not session._offer_lesson("power_rule"), "no errors yet, so nothing owed"
        session.board.request_lesson("power_rule")
        assert session._offer_lesson("power_rule")

    def test_a_request_is_answered_once(self, calculus) -> None:
        # Left standing it would re-teach on every pass, which is the failure
        # the throttle in `should_explain` exists to prevent, arriving through
        # the other door.
        session = build_session("L_ask", 1, calculus, _config(), learner=AlwaysWrong())
        session.board.request_lesson("power_rule")
        assert session._offer_lesson("power_rule")
        assert not session._offer_lesson("power_rule")

    def test_asking_about_one_concept_does_not_teach_another(self, calculus) -> None:
        session = build_session("L_ask", 1, calculus, _config(), learner=AlwaysWrong())
        session.board.request_lesson("power_rule")
        assert not session._offer_lesson("chain_rule")

    def test_asking_cannot_teach_in_a_run_that_does_not_teach(self, calculus) -> None:
        # Structural rather than a matter of nobody calling it: every cohort
        # passes through this branch.
        session = build_session(
            "L_ask", 1, calculus, _config(explain_after=0), learner=AlwaysWrong()
        )
        session.board.request_lesson("power_rule")
        assert not session._offer_lesson("power_rule")

    def test_the_asking_is_on_the_record(self, calculus) -> None:
        session = build_session("L_ask", 1, calculus, _config(), learner=AlwaysWrong())
        session.board.request_lesson("power_rule")
        assert [
            r for r in session.board.audit_log
            if r.evidence.get("asked_for_a_lesson")
        ], "a sitting has to be readable back against what the learner asked for"


class TestTeachingStopsWhenThereIsNothingNewToSay:
    """⚠️ Found by driving a real sitting, not by a unit test.

    Six lessons landed on one concept: three distinct accounts, and then the
    same three again word for word — because ``style_for`` comes back round and
    the response cache returns the identical text for the identical prompt. It
    is §7i's "same hint three times" arriving through a new door, and the honest
    reading is the one recorded there: if three different accounts did not land,
    a fourth identical one will not either.
    """

    def test_the_repertoire_is_the_ceiling(self) -> None:
        assert should_explain(99, 2, after=3, accounts_available=3)
        assert not should_explain(99, 3, after=3, accounts_available=3)

    def test_the_ceiling_is_read_off_the_styles_that_exist(self, calculus) -> None:
        # Rather than configured, so it cannot drift from the number of accounts
        # there actually are. Adding a fourth style raises it on its own.
        session = build_session("L_cap", 1, calculus, _config(), learner=AlwaysWrong())
        session.run()
        by_concept: dict[str, int] = {}
        for lesson in _lessons_in(session.board):
            by_concept[lesson["concept_id"]] = by_concept.get(lesson["concept_id"], 0) + 1
        assert by_concept, "the fixture must provoke lessons"
        assert max(by_concept.values()) <= len(TeachingStyle)

    def test_every_account_is_a_different_one(self, calculus) -> None:
        # The property the ceiling exists to protect: while lessons are given at
        # all, no two on a concept are the same account.
        assert len({style_for(n) for n in range(len(TeachingStyle))}) == len(
            TeachingStyle
        )

    def test_asking_is_still_answered_after_the_ceiling(self, calculus) -> None:
        # Stopping is not withdrawing. An explicit ask is answered whatever the
        # count — a learner who wants to read it again may.
        session = build_session("L_cap", 1, calculus, _config(), learner=AlwaysWrong())
        session.run()
        taught = {lesson["concept_id"] for lesson in _lessons_in(session.board)}
        exhausted = next(iter(taught))
        before = len(_lessons_in(session.board))
        session.board.request_lesson(exhausted)
        assert session._offer_lesson(exhausted)
        # One more *lesson*, however many turns it took.
        assert len(_lessons_in(session.board)) == before + 1


class TestALessonIsAConversation:
    """⚠️ The revision a sitting asked for.

    The first version produced a monologue shaped like a dialogue: the style
    instruction told the model to ask questions and answer each one itself. A
    learner watched it do that and said so — *"I really thought that would be
    more of a dialogue between me and the Tutor."*

    A lesson now opens, listens, and replies, and ends with the authored account
    in writing however it ended.
    """

    def _config_talking(self, turns: int = 3) -> Config:
        config = _config()
        config.teaching.lesson_turns = turns
        return config

    def test_the_learner_is_asked_and_the_tutor_replies(self, calculus) -> None:
        learner = AlwaysWrong(says=["I think it is a ratio", "oh, dividing"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.run()
        assert learner.asked, "the learner was never asked anything"
        # opening + two replies + summary, per lesson
        turns = _turns_in(session.board)
        assert len(turns) > len(_lessons_in(session.board))

    def test_what_the_learner_said_is_kept(self, calculus) -> None:
        learner = AlwaysWrong(says=["I think it is a ratio"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.run()
        lesson_words = [
            u for u in session.board.state.reflections if u.kind == "lesson"
        ]
        assert lesson_words
        assert any("ratio" in u.text for u in lesson_words)

    def test_a_lesson_utterance_belongs_to_no_question(self, calculus) -> None:
        """⚠️ And its ``kind`` is what stops that being read back wrongly.

        `LLMTutor.respond` labels an utterance "on an earlier question, not the
        one above" whenever its item id does not match. A lesson has no question,
        so without a kind of its own what the learner said while being taught
        would come back to them as a remark about something else — the sitting-3
        defect one level finer.
        """
        learner = AlwaysWrong(says=["I do not follow"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.run()
        for utterance in session.board.state.reflections:
            if utterance.kind == "lesson":
                assert utterance.item_id == ""

    def test_the_tutor_is_told_what_was_said(self, calculus) -> None:
        # Handed over like `said_this_item` on `respond`: an agent is told what
        # was said, never given a channel to another agent.
        seen: list[int] = []

        class Listening:
            def respond(self, *a, **kw):  # noqa: ANN001, ANN002, ANN003
                raise AssertionError("not under test")

            def explain(self, resource, style, exchanges=(), closing=False):  # noqa: ANN001
                seen.append(len(exchanges))
                return f"turn {len(exchanges)}"

        learner = AlwaysWrong(says=["a", "b"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.tutor = Listening()
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        # opening sees nothing, then one exchange, then two.
        assert seen[:3] == [0, 1, 2]

    # -- ending it --------------------------------------------------------

    def test_saying_nothing_ends_it(self, calculus) -> None:
        learner = AlwaysWrong(says=[])          # like a simulated learner
        session = build_session(
            "L_quiet", 1, calculus, self._config_talking(), learner=learner
        )
        session.run()
        assert _lessons_in(session.board)
        assert not [
            u for u in session.board.state.reflections if u.kind == "lesson"
        ]

    def test_the_turn_cap_bounds_a_learner_who_keeps_talking(self, calculus) -> None:
        learner = AlwaysWrong(says=["yes"] * 50)
        session = build_session(
            "L_chatty", 1, calculus, self._config_talking(turns=2), learner=learner
        )
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        said = [u for u in session.board.state.reflections if u.kind == "lesson"]
        assert len(said) == 2

    def test_one_turn_is_the_default_everywhere(self) -> None:
        # Today's behaviour, so every existing test and the whole cohort path
        # are untouched by this existing.
        assert Config().teaching.lesson_turns == 0

    # -- the summary ------------------------------------------------------

    def test_every_lesson_leaves_the_authored_account_behind(
        self, calculus
    ) -> None:
        learner = AlwaysWrong(says=["mm"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.run()
        lessons = _lessons_in(session.board)
        summaries = _summaries_in(session.board)
        assert len(summaries) == len(lessons)

    def test_the_summary_is_the_authored_text_verbatim(self, calculus) -> None:
        """The guarantees belong to the thing the learner is left holding.

        The conversation is the model's and is re-checked against nothing. This
        is what a person wrote and what `domain validate` has checked is plain
        text and checked does not answer any item on the concept at any draw.
        """
        learner = AlwaysWrong(says=["mm"])
        session = build_session(
            "L_talk", 1, calculus, self._config_talking(), learner=learner
        )
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        [summary] = [
            t for t in _summaries_in(session.board) if t["concept_id"] == "power_rule"
        ]
        assert summary["text"] == calculus.resources.get("power_rule").lesson()

    def test_declining_to_talk_still_teaches(self, calculus) -> None:
        # A conversational lesson that produces *less* than the one-shot when
        # someone is not in the mood to talk would be worse than the one-shot.
        quiet = AlwaysWrong(says=[])
        session = build_session(
            "L_quiet", 1, calculus, self._config_talking(), learner=quiet
        )
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        assert _summaries_in(session.board)

    # -- the counting hazard ----------------------------------------------

    def test_a_conversation_counts_as_one_lesson(self, calculus) -> None:
        """⚠️ The hazard the revision created, asserted rather than trusted.

        `taught` drives both which account comes next and the ceiling that stops
        teaching once every account has been given. Counting turns instead of
        openings would count every exchange as a fresh lesson: the ceiling would
        trip inside the first conversation and the rotation would skip accounts.
        """
        learner = AlwaysWrong(says=["a", "b", "c"])
        session = build_session(
            "L_count", 1, calculus, self._config_talking(), learner=learner
        )
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        on_power_rule = [
            t for t in _turns_in(session.board) if t["concept_id"] == "power_rule"
        ]
        assert len(on_power_rule) > 1, "the fixture must produce a real conversation"
        assert (
            len([t for t in on_power_rule if t.get("opening")]) == 1
        ), "a conversation must count as one lesson, not one per exchange"

    def test_the_rotation_still_advances_one_account_per_lesson(
        self, calculus
    ) -> None:
        # The observable consequence of the count being right.
        session = build_session(
            "L_rot", 1, calculus, self._config_talking(), learner=AlwaysWrong(says=["a"])
        )
        for _ in range(len(TeachingStyle)):
            session.board.request_lesson("power_rule")
            session._offer_lesson("power_rule")
        levels = [
            t["level"]
            for t in _lessons_in(session.board)
            if t["concept_id"] == "power_rule"
        ]
        assert levels == [s.label for s in TeachingStyle]


class TestTheTutorMaySuggestButNeverDecides:
    """The line a sitting drew.

    *"I don't think that the llm should decide when to quit the dialogue, but it
    could probably recommend the student to continue after it has noticed that
    the student is getting the concept."*

    It is the rule the rest of the tutor already follows: a model may say
    things, it may not decide them. Whether a lesson continues is read off
    nothing the model produces — the loop asks the learner every turn, and the
    learner answers or does not.
    """

    def test_it_is_told_it_may_say_so(self) -> None:
        from agent_newton.core.agents.llm import _STYLE_REPLY

        assert "stop here or keep going" in _STYLE_REPLY

    def test_and_told_the_decision_is_not_its_own(self) -> None:
        from agent_newton.core.agents.llm import _STYLE_REPLY

        assert "never end the conversation yourself" in _STYLE_REPLY

    def test_nothing_reads_the_suggestion_back(self, calculus) -> None:
        """The part that makes it a suggestion rather than a decision.

        A tutor saying "you can stop here" must not stop anything. The learner
        is asked exactly as often either way.
        """

        class Suggests:
            def respond(self, *a, **kw):  # noqa: ANN001, ANN002, ANN003
                raise AssertionError("not under test")

            def explain(self, resource, style, exchanges=(), closing=False):  # noqa: ANN001
                if closing:
                    return "and that is the idea."
                return "You have got it — you can stop here or keep going. Next?"

        learner = AlwaysWrong(says=["a", "b", "c"])
        config = _config()
        config.teaching.lesson_turns = None
        session = build_session("L_sugg", 1, calculus, config, learner=learner)
        session.tutor = Suggests()
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        said = [u for u in session.board.state.reflections if u.kind == "lesson"]
        assert len(said) == 3, "the suggestion must not have ended anything"


class TestACohortCannotBeTalkedTo:
    """The guarantee worth having: an inability, not a setting.

    `explain_after` and `lesson_turns` are both 0 for every experiment config
    and a scan enforces it. But the stronger statement is that a simulated
    learner cannot hold a conversation at all, so a dialogue is unreachable in a
    cohort however the config is set.
    """

    def test_a_simulated_learner_says_nothing(self, calculus) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, calculus.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, calculus, SimulatorConfig())
        assert learner.discuss("power_rule", "what do you think?") is None

    def test_so_a_lesson_collapses_to_one_turn_and_its_summary(
        self, calculus
    ) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        config = _config()
        config.teaching.lesson_turns = 5      # generous, and it will not be used
        profile = sample_profile("L1", 1, calculus.misconceptions, SimulatorConfig())
        session = build_session(
            "L1", 1, calculus, config,
            learner=SimulatedLearner(profile, calculus, SimulatorConfig()),
        )
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        on_power_rule = [
            t for t in _turns_in(session.board) if t["concept_id"] == "power_rule"
        ]
        assert len(on_power_rule) == 2, "an opening and a summary, and nothing else"


class TestReadingTheLearnersOwnWords:
    """⚠️ The trigger a sitting asked for, by writing it down and being ignored.

    Someone wrote *"I factored the denominator with part of the nominator
    (x^2 - 9) = (x+3)(x-3). But I don't understand what a limit is"* in the
    working channel, and then still had to type ``:why`` — for something the
    system was already holding, in the channel that already captures it.
    """

    class Reads:
        """A detector that fires on whatever substring it was given."""

        def __init__(self, needle: str = "don't understand") -> None:
            self._needle = needle
            self.checked: list[str] = []

        def confused(self, concept_id: str, text: str) -> str | None:
            self.checked.append(text)
            return text if self._needle in text else None

    def _config_detecting(self) -> Config:
        config = _config()
        config.teaching.detect_confusion = True
        return config

    def test_saying_so_in_the_working_channel_is_enough(self, calculus) -> None:
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        session._note_if_confused(
            "limits_of_sequences",
            "I factored the denominator. But I don't understand what a limit is.",
        )
        assert session.board.take_lesson_request() == ("limits_of_sequences", True)

    def test_the_words_that_fired_it_reach_the_audit_log(self, calculus) -> None:
        # ⚠️ The reason `confused` returns a quote rather than a bool, and it
        # went unrealised for a while: the session took the string, tested it
        # for None and dropped it, so the log recorded that something fired and
        # never what it read. A trigger whose evidence is a boolean cannot be
        # argued with afterwards, which is the whole claim the string exists to
        # support.
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        said = "I factored the denominator. But I don't understand what a limit is."
        session._note_if_confused("limits_of_sequences", said)

        inferred = [
            r for r in session.board.audit_log
            if r.evidence.get("asked_for_a_lesson") and r.evidence.get("inferred")
        ]
        assert len(inferred) == 1
        quote = inferred[0].evidence.get("quote", "")
        assert quote, "the firing recorded no evidence at all"
        assert quote in said, "the quote must be the learner's own words"

    def test_an_explicit_ask_records_no_quote(self, calculus) -> None:
        # Nothing was read, so there is nothing to quote: the learner typing
        # `:why` *is* the record. An empty string here would be evidence that
        # did not exist.
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.board.request_lesson("power_rule")
        asked = [
            r for r in session.board.audit_log
            if r.evidence.get("asked_for_a_lesson")
        ]
        assert len(asked) == 1
        assert "quote" not in asked[0].evidence

    def test_an_ordinary_wrong_answer_is_not_confusion(self, calculus) -> None:
        # The distinction the whole trigger rests on: someone attempting the
        # work and getting it wrong has met the concept and slipped, and those
        # need different help.
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        session._note_if_confused("power_rule", "I brought the power down but kept it")
        assert session.board.take_lesson_request() is None

    def test_it_leads_to_a_lesson_without_waiting_for_three_errors(
        self, calculus
    ) -> None:
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        assert not session._offer_lesson("power_rule"), "nothing owed yet"
        session._note_if_confused("power_rule", "I don't understand any of this")
        assert session._offer_lesson("power_rule")

    # -- what separates it from asking -------------------------------------

    def test_an_inference_still_stops_at_the_ceiling(self, calculus) -> None:
        """⚠️ And an explicit ask does not, which is the whole distinction.

        A person asking again has decided they want it again. A detector firing
        repeatedly would re-teach the same three accounts round and round, which
        is the repetition the ceiling was added to stop — arriving through a
        door that bypasses it.
        """
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        for _ in range(len(TeachingStyle)):
            session._note_if_confused("power_rule", "I don't understand")
            assert session._offer_lesson("power_rule")
        session._note_if_confused("power_rule", "I don't understand")
        assert not session._offer_lesson("power_rule"), (
            "an inference past the ceiling would cycle the same accounts again"
        )

    def test_but_asking_outright_is_still_answered(self, calculus) -> None:
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.confusion = self.Reads()
        for _ in range(len(TeachingStyle)):
            session._note_if_confused("power_rule", "I don't understand")
            session._offer_lesson("power_rule")
        session.board.request_lesson("power_rule")
        assert session._offer_lesson("power_rule")

    def test_the_two_are_told_apart_on_the_record(self, calculus) -> None:
        # Counted separately, so a detector firing on ordinary mistakes shows up
        # as a rate rather than as mysterious teaching.
        session = build_session(
            "L_lost", 1, calculus, self._config_detecting(), learner=AlwaysWrong()
        )
        session.board.request_lesson("power_rule")
        session.board.request_lesson("chain_rule", inferred=True)
        by_concept = {
            r.evidence["concept_id"]: r.evidence.get("inferred")
            for r in session.board.audit_log
            if r.evidence.get("asked_for_a_lesson")
        }
        assert by_concept == {"power_rule": False, "chain_rule": True}

    def test_the_evidence_is_recorded_not_just_the_verdict(self, calculus) -> None:
        # A trigger whose evidence is a boolean cannot be argued with after the
        # fact, which is why the detector returns the words rather than True.
        detector = self.Reads()
        assert detector.confused("limits", "I don't understand limits") is not None
        assert detector.confused("limits", "the answer is 6") is None

    # -- the guards --------------------------------------------------------

    def test_off_reads_nothing_at_all(self, calculus) -> None:
        # Not merely "detects nothing": the detector is never consulted, so a
        # run that does not want this makes no extra model call.
        detector = self.Reads()
        session = build_session(
            "L_off", 1, calculus, _config(), learner=AlwaysWrong()
        )
        session.confusion = detector
        session._note_if_confused("power_rule", "I don't understand any of this")
        assert detector.checked == []
        assert session.board.take_lesson_request() is None

    def test_the_model_free_detector_says_no_to_everything(self) -> None:
        from agent_newton.core.agents.tutor import NoConfusion

        assert NoConfusion().confused("limits", "I have no idea what this is") is None

    def test_a_model_free_run_is_refused_rather_than_looking_like_it_worked(
        self,
    ) -> None:
        """⚠️ The failure this validator exists to prevent.

        Left unchecked the run would look fine: the detector would be the
        model-free one, it would answer no to everything, nothing would ever
        trigger, and the manifest would record the feature as on. Every number
        would be the kind that looks plausible and means nothing — the same
        shape the human-diagnostic check already guards.
        """
        with pytest.raises(ValueError, match="detect_confusion"):
            Config.model_validate(
                {
                    "teaching": {"detect_confusion": True},
                    "agents": {"tutor": {"impl": "template"}},
                }
            )

    def test_a_cohort_has_nothing_to_read_even_if_it_were_on(self, calculus) -> None:
        # The structural half. A simulated learner writes nothing in either
        # prose channel, so the text this reads is empty by construction rather
        # than by configuration.
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, calculus.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, calculus, SimulatorConfig())
        item = calculus.items.for_concept("power_rule", "practice")[0]
        assert learner.show_working(item, "wrong", required=True) is None
        assert learner.reflect(item, "which part are you unsure of?") is None


class TestTheConfusionDetectorAgreesWithHandLabels:
    """Calibration, on the same terms as the tutor judge's.

    A detector nobody measured is worse than none, because it looks like one.
    The set is balanced — read the count off the fixture, not from here, since it
    has grown once already — so answering "confused" to everything scores half
    rather than well. The `false` half is the hard one: hedging, uncertainty
    about an answer, and "this was confusing" all describe someone who is doing
    the work.

    ⚠️ This is the CI gate. `agent-newton evaluate confusion` is the same
    measurement written down, and the two are not substitutes: a gate that skips
    without a model still passes, and a stored figure is what a claim gets
    quoted from.

    Skipped where no model is reachable — it is a measurement of a model, and a
    version of it that ran without one would be measuring nothing.
    """

    GOLD = Path("tests/fixtures/gold/calculus_confusion_cases.yaml")

    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls):
        import yaml

        return yaml.safe_load(cls.GOLD.read_text())["cases"]

    def test_the_set_is_balanced(self, cases) -> None:
        # Or a detector that always says yes would score well by saying nothing.
        confused = sum(1 for case in cases if case["confused"])
        assert confused == len(cases) - confused

    def test_every_case_says_where_it_came_from(self, cases) -> None:
        # The catalogue's convention: content stays traceable rather than
        # accumulating invented examples.
        for case in cases:
            assert case["source"].strip()

    def test_saying_confused_to_everything_scores_half(self, cases) -> None:
        # The floor the real figure has to beat, stated rather than assumed.
        always = sum(1 for case in cases if case["confused"] is True)
        assert always / len(cases) == 0.5

    def test_the_model_free_detector_scores_the_other_half(self, cases) -> None:
        from agent_newton.core.agents.tutor import NoConfusion

        detector = NoConfusion()
        agreed = sum(
            1
            for case in cases
            if (detector.confused(case["concept_id"], case["text"]) is not None)
            == case["confused"]
        )
        assert agreed / len(cases) == 0.5

    def test_it_is_calibrated_against_the_hand_labels(self, cases) -> None:
        """The real figure. Needs a model, so it skips without one."""
        import os
        import urllib.error
        import urllib.request

        # ⚠️ The same variable the ollama client reads, not a hard-coded
        # localhost. In a container `localhost` is the container, so this
        # skipped even with a reachable server — and a calibration test that
        # silently does not run is worse than one that fails, because its figure
        # is the one every judged rate has to be quoted beside.
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        except (urllib.error.URLError, OSError):
            pytest.skip("no model reachable; this measures one")

        from agent_newton.config import ModelSpec
        from agent_newton.core.agents.llm import LLMConfusionDetector
        from agent_newton.llm.factory import build_provider

        detector = LLMConfusionDetector(
            build_provider(ModelSpec(model="gemma4:12b", think=False), Path(".cache/llm"))
        )
        wrong = [
            case
            for case in cases
            if (detector.confused(case["concept_id"], case["text"]) is not None)
            != case["confused"]
        ]
        assert not wrong, "disagreed on: " + "; ".join(c["text"][:50] for c in wrong)


class TestALessonNeverEndsOnAQuestionNobodyCanAnswer:
    """⚠️ From a sitting, and it is the sharpest thing one has caught yet.

    Every turn of a lesson ends by asking something, so however the conversation
    stops there is one left hanging — and the written summary then answers it.
    The sitting caught it at the worst moment the material allows: the tutor had
    just asked what happens to a secant's gradient as the second point slides
    in, which *is* the limit concept, and the summary appeared instead of a
    reply. *"I was just about to understand something important."*

    Under the Socratic style that is the monologue failure returning by another
    door — the system asks the question and then answers it itself.
    """

    class Recording:
        """A tutor that records how each turn was asked for."""

        def __init__(self) -> None:
            self.closings: list[bool] = []

        def respond(self, *a, **kw):  # noqa: ANN001, ANN002, ANN003
            raise AssertionError("not under test")

        def explain(self, resource, style, exchanges=(), closing=False):  # noqa: ANN001
            self.closings.append(closing)
            return "and so it closes." if closing else "what do you think?"

    def _talking(self, turns: int = 3) -> Config:
        config = _config()
        config.teaching.lesson_turns = turns
        return config

    def _lesson(self, calculus, learner, turns: int = 3):
        session = build_session(
            "L_end", 1, calculus, self._talking(turns), learner=learner
        )
        tutor = self.Recording()
        session.tutor = tutor
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        return session, tutor

    def test_the_last_turn_is_asked_to_close(self, calculus) -> None:
        _, tutor = self._lesson(calculus, AlwaysWrong(says=["a", "b", "c"]))
        assert tutor.closings[-1] is True
        assert tutor.closings.count(True) == 1

    def test_and_it_is_the_last_thing_before_the_summary(self, calculus) -> None:
        session, _ = self._lesson(calculus, AlwaysWrong(says=["a", "b", "c"]))
        turns = [
            t for t in _turns_in(session.board) if t["concept_id"] == "power_rule"
        ]
        assert turns[-1]["level"] == "summary"
        assert turns[-2]["text"] == "and so it closes."

    def test_running_out_of_budget_closes_rather_than_cutting_off(
        self, calculus
    ) -> None:
        # The sitting's case exactly: the learner is still talking and the guard
        # stops the conversation. It must still be tied off.
        _, tutor = self._lesson(calculus, AlwaysWrong(says=["a"] * 50), turns=2)
        assert tutor.closings[-1] is True

    def test_the_learner_stopping_also_closes(self, calculus) -> None:
        # They said their piece and pressed enter. The question the tutor just
        # asked is still hanging, and it is still owed an answer.
        _, tutor = self._lesson(calculus, AlwaysWrong(says=["a"]))
        assert tutor.closings[-1] is True

    def test_declining_at_the_very_first_prompt_does_not(self, calculus) -> None:
        # Nothing was engaged with, so a closing turn would be the system
        # talking to itself. The summary is the right answer there, and it is
        # what the one-shot lesson always did.
        _, tutor = self._lesson(calculus, AlwaysWrong(says=[]))
        assert True not in tutor.closings

    def test_a_cohort_never_reaches_a_closing_turn(self, calculus) -> None:
        # It cannot converse, so nothing is ever left hanging.
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile

        profile = sample_profile("L1", 1, calculus.misconceptions, SimulatorConfig())
        _, tutor = self._lesson(
            calculus, SimulatedLearner(profile, calculus, SimulatorConfig())
        )
        assert True not in tutor.closings

    def test_the_closing_instruction_forbids_asking_anything_new(self) -> None:
        # The rule, stated where the model reads it.
        from agent_newton.core.agents.llm import _STYLE_CLOSING

        assert "do not ask" in _STYLE_CLOSING.lower()
        assert "hanging" in _STYLE_CLOSING.lower()

    def test_and_it_is_checked_rather_than_hoped_for(self) -> None:
        """⚠️ Because asking did not work.

        Told to stop asking, the model asked anyway — the system prompt said
        "say a little and then ask" on *every* turn, so the closing instruction
        was unfollowable. That is the fourth time a global rule here has
        outlived the case it was written for: `_TUTOR_SYSTEM` once demanded two
        sentences while `WORKED_STEP` asked for the step to be worked through,
        and `HintReply`'s field description carried the same demand one layer
        down.

        The conflict is fixed, and this is what makes the rule checkable: a
        constraint a model can talk itself out of is not one.
        """
        import pydantic

        from agent_newton.core.agents.schemas import ClosingReply

        with pytest.raises(pydantic.ValidationError):
            ClosingReply(text="So what happens as the point slides in?")

    def test_a_closing_turn_may_still_contain_a_question(self) -> None:
        # Only a *trailing* one is refused. Taking up what the learner asked is
        # exactly what a closing turn is for.
        from agent_newton.core.agents.schemas import ClosingReply

        text = (
            "You asked what happens as it slides in: the gradients settle on a "
            "single value."
        )
        assert ClosingReply(text=text).text == text

    def test_the_system_prompt_no_longer_demands_a_question(self) -> None:
        # The cause, not the symptom. Whether a turn asks something belongs to
        # the turn, not to a rule applied to every turn.
        from agent_newton.core.agents.llm import _EXPLAIN_SYSTEM

        assert "say a little and then ask" not in _EXPLAIN_SYSTEM.lower()


class TestTheGuardIsNotALength:
    """A bound a learner can reach while still engaged is an interruption.

    It was 3, then 12, and a sitting reached both — the second time
    mid-derivation, with the tutor having just asked them to expand
    ``(x + h)^2``. The bound is gone for a person now, and the reason it can be
    is that it was never the thing doing the bounding: a turn requires a reply
    and a reply requires someone to type one.
    """

    def test_a_person_is_not_bounded_at_all(self) -> None:
        demo = Config.from_yaml("experiments/configs/demo.yaml")
        assert demo.teaching.lesson_turns is None

    def test_an_unbounded_lesson_still_ends_when_the_learner_does(
        self, calculus
    ) -> None:
        """Which is why unbounded is safe rather than reckless.

        The loop cannot advance without a reply. A learner who stops talking
        stops the lesson, and a simulated one stops it at the first turn — so
        "no bound" is bounded by the only thing that was ever bounding it.
        """
        learner = AlwaysWrong(says=["a", "b", "c"])
        config = _config()
        config.teaching.lesson_turns = None
        session = build_session("L_free", 1, calculus, config, learner=learner)
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        said = [u for u in session.board.state.reflections if u.kind == "lesson"]
        assert len(said) == 3, "it ran exactly as long as the learner talked"

    def test_a_number_is_still_honoured_for_anything_that_wants_one(
        self, calculus
    ) -> None:
        learner = AlwaysWrong(says=["a"] * 50)
        config = _config()
        config.teaching.lesson_turns = 2
        session = build_session("L_capped", 1, calculus, config, learner=learner)
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        said = [u for u in session.board.state.reflections if u.kind == "lesson"]
        assert len(said) == 2

    def test_a_cohort_is_still_pinned_to_one_turn(self) -> None:
        # `None` must not leak into an experiment config: unbounded there would
        # be a different run from every measured one. The scan checks `== 0`,
        # which None fails.
        for name in ("calculus", "smoke"):
            assert (
                Config.from_yaml(f"experiments/configs/{name}.yaml").teaching.lesson_turns
                == 0
            )

    def test_the_learner_can_always_end_it_sooner(self, calculus) -> None:
        learner = AlwaysWrong(says=["one thing"])
        config = _config()
        config.teaching.lesson_turns = 12
        session = build_session("L_short", 1, calculus, config, learner=learner)
        session.board.request_lesson("power_rule")
        session._offer_lesson("power_rule")
        said = [u for u in session.board.state.reflections if u.kind == "lesson"]
        assert len(said) == 1, "one reply, then they stopped, and it ended there"
