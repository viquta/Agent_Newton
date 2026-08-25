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
from agent_newton.core.pedagogy import TutorMove, check_move, should_explain
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
            assert lesson["level"] == "lesson"


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
