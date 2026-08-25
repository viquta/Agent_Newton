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

    def __init__(self) -> None:
        self.hints: list[str | None] = []

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

    def receive_hint(self, targeted_misconception: str | None) -> bool:
        self.hints.append(targeted_misconception)
        return False

    def remediation_ratio(self) -> float | None:
        return None


def _lessons_in(board) -> list[dict]:
    return [
        record.evidence
        for record in board.audit_log
        if record.cause == "tutor"
        and record.evidence.get("move") == TutorMove.EXPLAIN.value
    ]


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
        for lesson in _lessons_in(session.board):
            assert lesson["move"] == "explain"
            # The style stands where a hint records its support level. A lesson
            # has no support level — it is not a quantity of the answer — and
            # recording one would invite it to be read as a rung on the ladder,
            # which is exactly what it is not.
            assert lesson["level"] in {s.label for s in TeachingStyle}


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

    def test_plain_calls_no_model_at_all(self, calculus) -> None:
        # So a model-free run still teaches, and the first lesson on every
        # concept is exactly what was authored.
        provider = self.Replies("should never be used")
        resource = calculus.resources.get("power_rule")
        assert (
            self._tutor(provider).explain(resource, TeachingStyle.PLAIN)
            == resource.lesson()
        )
        assert provider.calls == 0

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
