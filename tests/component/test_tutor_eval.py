"""The tutor evaluation.

Every check here decides something about the text a learner reads, and the
instructional rules say nothing about that text — they choose the move and the
support level, and a reply can satisfy both while doing the opposite of what the
level means. So each check is exercised on a reply that breaks it as well as on
one that does not: a check that cannot fail proves nothing, and these are the
only guards standing between a stated teaching constraint and prose that ignores
it.

Driven by a scripted provider throughout. Whether a real model writes good hints
is a measurement, not a test; that is what `agent-newton evaluate tutor` is for.
"""

from __future__ import annotations

import json

from pathlib import Path
from agent_newton.core.state.schema import AuditRecord
import pytest

from agent_newton.config import ZPDConfig
from agent_newton.core.agents.llm import FALLBACK_HINT, LLMTutor
from agent_newton.core.evaluation import tutor as evaluation
from agent_newton.core.evaluation.tutor import (
    ANSWER_LEAKED,
    LATEX_IN_REPLY,
    OVER_LENGTH,
    REFLECT_TELLS,
    Turn,
    TurnCase,
)
from agent_newton.core.pedagogy import HintLevel, TutorMove
from agent_newton.domains import registry
from agent_newton.domains.base import DomainError
from agent_newton.llm.base import Completion

BAND = ZPDConfig()

GOLD = "tests/fixtures/gold/calculus_tutor_cases.yaml"


class Scripted:
    """Replies from a list; repeats the last one once exhausted."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or ["{}"]
        self.calls: list[str] = []

    @property
    def label(self) -> str:
        return "fake/judge-1"

    def generate(self, prompt: str, schema, system):  # noqa: ANN001
        self.calls.append(prompt)
        return Completion(
            text=self._replies[min(len(self.calls) - 1, len(self._replies) - 1)],
            model="judge-1",
            provider="fake",
        )


def hint(*texts: str) -> Scripted:
    return Scripted(*(json.dumps({"text": t}) for t in texts))


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


def turn_for(
    calculus,
    text: str,
    *,
    item_id: str = "ca_pow_p1",
    move: TutorMove = TutorMove.HINT,
    level: HintLevel = HintLevel.NUDGE,
    working: str = "",
) -> Turn:
    item = calculus.items.get(item_id)
    case = TurnCase(
        id=f"{item_id}|test",
        item_id=item_id,
        concept_id=item.concept_id,
        misconception_id=item.probes[0] if item.probes else "",
        wrong_answer="5*x**5",
        mastery=0.5,
        prior_failures=0,
        diagnosed=move is not TutorMove.HINT,
        working=working,
    )
    return Turn(
        case=case,
        move=move.value,
        level=level.label,
        targets=None,
        text=text,
        seconds=0.0,
    )


class TestLeakingTheAnswer:
    """The check the instructional rules cannot make.

    ``hint_level`` decides how much support to give; nothing stops the prose
    carrying more. A nudge that states the answer satisfies every predicate in
    ``core/pedagogy`` and defeats the rule it was issued under.
    """

    def test_a_verbatim_answer_is_caught(self, calculus) -> None:
        item = calculus.items.get("ca_pow_p1")  # answer 5*x**4
        assert evaluation.leaks_answer("The derivative is 5*x**4.", item, calculus)

    def test_an_equivalent_form_is_caught_too(self, calculus) -> None:
        # The whole reason the domain's verifier decides this rather than a
        # string comparison: the same thing has been given away.
        item = calculus.items.get("ca_pow_p1")
        assert evaluation.leaks_answer("You should reach 5x^4 here.", item, calculus)

    @pytest.mark.parametrize(
        ("item_id", "text"),
        [
            # Most answers in this bank have a space around their operator, and
            # splitting the reply on whitespace took them apart into fragments
            # none of which verifies. A whole run of 329 turns reported no leak
            # at all, partly because of this.
            ("ca_anti_p1", "The expression should be x^3 + C."),
            ("ca_poly_p1", "Differentiating term by term gives 3x^2 - 6x here."),
            ("ca_stat_p1", "The two roots are 0, 2."),
        ],
    )
    def test_an_answer_written_with_spaces_is_caught(
        self, calculus, item_id: str, text: str
    ) -> None:
        item = calculus.items.get(item_id)
        assert evaluation.leaks_answer(text, item, calculus)

    def test_a_genuine_nudge_does_not_trip_it(self, calculus) -> None:
        item = calculus.items.get("ca_pow_p1")
        assert not evaluation.leaks_answer(
            "Compare the exponent you wrote with the one you started from.",
            item,
            calculus,
        )

    def test_quoting_the_question_is_not_giving_anything_away(self, calculus) -> None:
        # ca_lim_p1 asks about (x^2 - 4)/(x - 2) and its answer is 4. A hint
        # repeating the question would otherwise read as a leak on every item
        # whose answer happens to appear in its own prompt.
        item = calculus.items.get("ca_lim_p1")
        assert not evaluation.leaks_answer(
            "Look at the factor x^2 - 4 in the numerator.", item, calculus
        )

    def test_several_values_are_caught_when_listed_together(self, calculus) -> None:
        # ca_stat_p1's answer is "0, 2". Either root alone is not the answer;
        # both of them are.
        item = calculus.items.get("ca_stat_p1")
        assert not evaluation.leaks_answer("One root is 2.", item, calculus)
        assert evaluation.leaks_answer("The roots are 0, 2 here.", item, calculus)

    def test_it_is_a_violation_below_worked_step(self, calculus) -> None:
        for level in (HintLevel.NUDGE, HintLevel.TARGETED):
            turn = turn_for(calculus, "The answer is 5*x**4.", level=level)
            item = calculus.items.get(turn.case.item_id)
            rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
            assert ANSWER_LEAKED in rules

    def test_a_worked_step_is_not_allowed_to_show_it_either(self, calculus) -> None:
        """⚠️ This test asserted the exemption, and the exemption was wrong.

        The reasoning was that the level's own instruction is to work the step
        through, so the answer appearing in it is the rule being followed. Then
        a person read one: the reply assembled the whole answer and the only
        thing left to do was type it back, which the learner model records as
        knowing it. *"The worked step should not actually show the final
        answer."*

        The instruction now says to stop one line short, and this is what makes
        that a rule rather than a suggestion.
        """
        turn = turn_for(
            calculus, "Bring the 5 down and reduce the power: 5*x**4.",
            level=HintLevel.WORKED_STEP,
        )
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert ANSWER_LEAKED in rules

    def test_working_the_step_without_finishing_it_is_clean(self, calculus) -> None:
        # The other half, or the check above would be satisfied by a tutor that
        # says nothing useful. Everything a worked step is for — naming the
        # rule, substituting the student's own expression, showing the line —
        # is still permitted.
        turn = turn_for(
            calculus,
            "The power rule brings the exponent down and reduces it by one. "
            "Here that is 5 times x to the power 5 minus 1.",
            level=HintLevel.WORKED_STEP,
        )
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert ANSWER_LEAKED not in rules


class TestLatexNeverReachesTheLearner:
    """Guarded at the reply rather than at the prompt.

    The system prompt has banned backslash commands since the second sitting,
    and the ban was asserted only on the prompt — so a model ignoring it would
    have gone unnoticed exactly as it did the first time.
    """

    def test_a_backslash_command_is_caught(self, calculus) -> None:
        turn = turn_for(calculus, r"Use \frac{f(b) - f(a)}{b - a} here.")
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert LATEX_IN_REPLY in rules

    def test_the_form_it_takes_after_json_unescaping_is_caught(self, calculus) -> None:
        # What the learner actually saw: "\frac" arrives as a form feed followed
        # by "rac", and the division the hint was explaining disappears. Checking
        # only for a literal backslash would miss the case that was reported.
        turn = turn_for(calculus, "Use \x0crac{f(b) - f(a)}{b - a} here.")
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert LATEX_IN_REPLY in rules

    def test_plain_text_mathematics_passes(self, calculus) -> None:
        turn = turn_for(calculus, "Use (f(b) - f(a)) / (b - a) here.")
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert LATEX_IN_REPLY not in rules


class TestAReflectiveTurnMustNotCorrect:
    """The error-first rule inserts a step; a turn that corrects has skipped it.

    ``check_move`` proves remediation was preceded by a reflective *move*. It
    cannot see whether the words issued under that move handed over the answer.
    """

    def test_a_prompt_that_gives_the_answer_is_caught(self, calculus) -> None:
        turn = turn_for(
            calculus,
            "Does 5*x**4 look like what you wrote?",
            move=TutorMove.REFLECT,
        )
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert REFLECT_TELLS in rules

    def test_a_real_reflective_prompt_passes(self, calculus) -> None:
        turn = turn_for(
            calculus,
            "Which part of that step are you least sure about?",
            move=TutorMove.REFLECT,
        )
        item = calculus.items.get(turn.case.item_id)
        assert evaluation.check_turn(turn, item, calculus) == ()

    def test_an_imperative_request_passes_too(self, calculus) -> None:
        # The first reply a real model produced under this move, and an earlier
        # version of the check called it a violation for having no question
        # mark. A reflective prompt need not be punctuated as a question, and a
        # check that says otherwise measures punctuation.
        turn = turn_for(
            calculus,
            "Please take another look at your expression and tell me which part "
            "you feel least confident about.",
            move=TutorMove.REFLECT,
        )
        item = calculus.items.get(turn.case.item_id)
        assert evaluation.check_turn(turn, item, calculus) == ()


class TestLength:
    """The budget belongs to the level, not to every reply alike.

    It was one number, and it was two — which would have marked a proper worked
    step as a violation for obeying the instruction telling it to show the
    working a line at a time.
    """

    def test_a_third_sentence_is_caught_at_nudge(self, calculus) -> None:
        turn = turn_for(calculus, "One thing. Then another. And a third.")
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert OVER_LENGTH in rules

    def test_two_sentences_pass(self, calculus) -> None:
        turn = turn_for(calculus, "One thing. Then another.")
        item = calculus.items.get(turn.case.item_id)
        assert evaluation.check_turn(turn, item, calculus) == ()

    def test_a_worked_step_may_run_longer(self, calculus) -> None:
        # Six short lines is what the level asks for; two would make the
        # instruction impossible to follow.
        turn = turn_for(
            calculus,
            "Use the product rule. Take the first factor. Differentiate it. "
            "Leave the second alone. Now swap the roles. Add the two pieces.",
            level=HintLevel.WORKED_STEP,
        )
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert OVER_LENGTH not in rules

    def test_but_not_indefinitely(self, calculus) -> None:
        turn = turn_for(
            calculus, "One. Two. Three. Four. Five. Six. Seven.",
            level=HintLevel.WORKED_STEP,
        )
        item = calculus.items.get(turn.case.item_id)
        rules = [v.rule for v in evaluation.check_turn(turn, item, calculus)]
        assert OVER_LENGTH in rules


class TestAFallbackIsNotABadHint:
    """A failure to produce a turn is not a wrongly-pitched one.

    The same distinction the diagnostic agent draws between a failed call and a
    wrong label: nothing was written, so scoring it as prose would be scoring an
    absence.
    """

    def test_it_is_recognised(self, calculus) -> None:
        assert turn_for(calculus, FALLBACK_HINT).is_fallback
        assert not turn_for(calculus, "Look at the exponent.").is_fallback

    def test_it_is_counted_apart(self, calculus) -> None:
        report = evaluation.score(
            calculus,
            [turn_for(calculus, FALLBACK_HINT), turn_for(calculus, "Look again.")],
        )
        assert report.fallbacks == 1
        assert report.total == 2

    def test_it_is_still_checked(self, calculus) -> None:
        # It has never broken a rule. Exempting it would make the exemption the
        # thing being measured.
        turn = turn_for(calculus, FALLBACK_HINT)
        item = calculus.items.get(turn.case.item_id)
        assert evaluation.check_turn(turn, item, calculus) == ()


class TestTheCasesComeFromTheRules:
    def test_a_case_names_no_move_and_no_level(self, calculus) -> None:
        # They are the rules' decisions. A case that asserted them would let
        # this drift from the policy the session actually runs.
        case = evaluation.cases(calculus, BAND)[0]
        assert not hasattr(case, "move")
        assert not hasattr(case, "level")

    def test_the_mastery_values_follow_the_band(self, calculus) -> None:
        # A threshold sweep moves the band, and the case set has to move with it
        # rather than keep measuring the old one.
        wide = ZPDConfig(theta_lower=0.30, theta_upper=0.95)
        narrow = ZPDConfig(theta_lower=0.80, theta_upper=0.90)
        assert {c.mastery for c in evaluation.cases(calculus, wide)} != {
            c.mastery for c in evaluation.cases(calculus, narrow)
        }

    @pytest.mark.parametrize("policy", ["banded", "banded_plus"])
    @pytest.mark.parametrize("level", list(HintLevel))
    def test_each_intended_level_is_what_the_rules_assign(self, level, policy) -> None:
        """A case's mastery must land on the level it was generated for.

        Only for levels the policy can assign. ``banded`` never returns ``none``
        — above ``theta_lower`` it gives a nudge and stops — so requiring it
        here would be asserting a level into existence rather than measuring
        one. ``_assignable`` decides, by asking the rule, and the next test
        pins that the two policies really do differ.
        """
        from agent_newton.core.pedagogy import hint_level

        if level not in evaluation._assignable(BAND, policy):
            pytest.skip(f"{policy} cannot assign {level.label}")
        mastery = evaluation._mastery_for(level, BAND)
        assert hint_level(mastery, prior_failures=0, band=BAND, policy=policy) is level

    def test_only_the_richer_ladder_offers_a_case_at_none(self) -> None:
        # Without this the skip above could hide a policy that assigns nothing:
        # every case would skip and the test would pass having checked nothing.
        assert HintLevel.NONE not in evaluation._assignable(BAND, "banded")
        assert HintLevel.NONE in evaluation._assignable(BAND, "banded_plus")

    def test_every_case_resolves_to_a_real_item(self, calculus) -> None:
        for case in evaluation.cases(calculus, BAND):
            item = calculus.items.get(case.item_id)
            assert case.concept_id == item.concept_id
            assert case.misconception_id in item.probes

    def test_the_view_carries_the_working_when_there_is_any(self, calculus) -> None:
        case = evaluation.cases(calculus, BAND)[0]
        bare = evaluation.view_for(case, BAND)
        assert bare.reflections == ()

        from dataclasses import replace

        shown = evaluation.view_for(replace(case, working="I guessed"), BAND)
        assert shown.said_about(case.concept_id)[0].text == "I guessed"


class TestAskingTheTutor:
    """The real agent, driven by a scripted model."""

    def test_the_rules_assign_the_move_and_the_level(self, calculus) -> None:
        tutor = LLMTutor(hint("Look at the exponent."), BAND)
        situations = [
            c for c in evaluation.cases(calculus, BAND)
            if c.item_id == "ca_pow_p1" and c.misconception_id
        ]
        turns = list(evaluation.ask(calculus, tutor, BAND, situations))
        assert {t.move for t in turns} == {"hint", "reflect", "remediate"}
        assert {t.level for t in turns} == {"nudge", "targeted", "worked_step"}

    def test_only_remediation_targets_anything(self, calculus) -> None:
        tutor = LLMTutor(hint("Look at the exponent."), BAND)
        situations = [c for c in evaluation.cases(calculus, BAND)[:14]]
        for turn in evaluation.ask(calculus, tutor, BAND, situations):
            if turn.move != "remediate":
                assert turn.targets is None
            else:
                assert turn.targets == turn.case.misconception_id

    def test_a_report_aggregates_by_level_move_and_bank(self, calculus) -> None:
        tutor = LLMTutor(hint("Look at the exponent."), BAND)
        situations = evaluation.cases(calculus, BAND)[:14]
        report = evaluation.score(
            calculus, evaluation.ask(calculus, tutor, BAND, situations)
        )
        assert report.total == 14
        assert set(report.by_move()) <= {"hint", "reflect", "remediate"}
        assert all(0.0 <= row["clean_rate"] <= 1.0 for row in report.by_level().values())
        # Only practice turns reach a learner: the test banks are administered
        # without hints, so that row is what a running system exposes anyone to.
        assert set(report.by_bank(calculus)) <= {"practice", "pretest", "posttest"}

    def test_a_leaking_model_is_caught_end_to_end(self, calculus) -> None:
        # The path that matters: a model ignoring its instructions, through the
        # real tutor, into the report.
        tutor = LLMTutor(hint("The answer is 5x^4."), BAND)
        situations = [
            c for c in evaluation.cases(calculus, BAND)
            if c.item_id == "ca_pow_p1" and c.id.endswith("hint|nudge")
        ]
        report = evaluation.score(
            calculus, evaluation.ask(calculus, tutor, BAND, situations)
        )
        assert report.by_rule().get(ANSWER_LEAKED)


class TestTheGoldSet:
    def test_it_loads_and_every_item_exists(self, calculus) -> None:
        cases = evaluation.load_gold(GOLD, calculus)
        assert cases
        for case in cases:
            calculus.items.get(case.item_id)  # raises on an unknown id

    def test_it_is_balanced(self, calculus) -> None:
        # A judge that always answers "grounded" would score well on a set where
        # most cases are, and the agreement figure would hide it.
        cases = evaluation.load_gold(GOLD, calculus)
        grounded = sum(1 for c in cases if c.grounded)
        assert grounded == len(cases) - grounded

    def test_the_sitting_that_produced_it_is_recorded(self, calculus) -> None:
        cases = evaluation.load_gold(GOLD, calculus)
        assert any(c.source for c in cases), "no case carries its provenance"

    def test_an_unknown_item_is_refused(self, calculus, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "cases:\n"
            "  - id: x\n"
            "    item_id: no_such_item\n"
            "    response: '1'\n"
            "    hint: 'look again'\n"
            "    grounded: true\n"
        )
        with pytest.raises(DomainError):
            evaluation.load_gold(path, calculus)

    def test_an_empty_file_is_refused(self, calculus, tmp_path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("cases: []\n")
        with pytest.raises(DomainError, match="no cases"):
            evaluation.load_gold(path, calculus)


class TestTheJudgeIsItselfMeasured:
    """Its verdicts are only as good as its agreement with the hand labels."""

    def _judged(self, calculus, *replies: str):
        provider = Scripted(*replies)
        return evaluation.calibrate(
            provider, calculus, evaluation.load_gold(GOLD, calculus)
        )

    def test_a_perfect_judge_agrees_completely(self, calculus) -> None:
        cases = evaluation.load_gold(GOLD, calculus)
        provider = Scripted(
            *(json.dumps({"grounded": c.grounded, "reason": ""}) for c in cases)
        )
        report = evaluation.calibrate(provider, calculus, cases)
        assert report.agreement == 1.0
        assert report.disagreements() == []

    def test_a_judge_that_always_says_yes_is_caught(self, calculus) -> None:
        # The failure the balanced set exists to expose.
        report = self._judged(calculus, json.dumps({"grounded": True, "reason": ""}))
        assert report.agreement == 0.5
        assert report.disagreements()

    def test_a_failed_call_counts_for_neither_side(self, calculus) -> None:
        # Nothing was judged. Folding it into either side would move the rate it
        # is supposed to measure.
        report = self._judged(calculus, "not json at all")
        assert report.scored == []
        assert report.agreement == 0.0
        assert report.unobtainable == len(evaluation.load_gold(GOLD, calculus))

    def test_reflective_turns_are_not_judged_for_groundedness(self, calculus) -> None:
        # They make no claim about the step by design, so including them would
        # dilute the rate with cases that cannot fail.
        provider = Scripted(json.dumps({"grounded": True, "reason": ""}))
        turns = [
            turn_for(calculus, "What are you least sure of?", move=TutorMove.REFLECT),
            turn_for(calculus, "The coefficient is right."),
        ]
        report = evaluation.judge_turns(provider, calculus, turns)
        assert len(report.verdicts) == 1

    def test_a_fallback_is_not_judged_either(self, calculus) -> None:
        provider = Scripted(json.dumps({"grounded": True, "reason": ""}))
        report = evaluation.judge_turns(
            provider, calculus, [turn_for(calculus, FALLBACK_HINT)]
        )
        assert report.verdicts == []


class TestLevelsAreVisibleInTheText:
    """The fading property as a learner would experience it.

    ``check_fading`` proves ``hint_level`` never rises with mastery. That is a
    property of the function; whether the difference reaches the page is not
    something the function can answer.
    """

    def _turns(self, calculus):
        return [
            turn_for(calculus, "Look at the exponent.", level=HintLevel.NUDGE),
            turn_for(calculus, "Bring the 5 down: 5*x**4.", level=HintLevel.WORKED_STEP),
        ]

    def test_the_pair_is_ordered_by_level_not_by_position(self, calculus) -> None:
        provider = Scripted(json.dumps({"choice": "second"}))
        report = evaluation.rank_levels(provider, calculus, self._turns(calculus))
        assert report.pairs
        _, low, high, ok = report.pairs[0]
        assert (low, high) == ("nudge", "worked_step")
        assert ok is True

    def test_a_judge_reading_it_backwards_is_recorded_as_disagreeing(
        self, calculus
    ) -> None:
        provider = Scripted(json.dumps({"choice": "first"}))
        report = evaluation.rank_levels(provider, calculus, self._turns(calculus))
        assert report.agreement == 0.0

    def test_turns_at_one_level_produce_no_pair(self, calculus) -> None:
        provider = Scripted(json.dumps({"choice": "second"}))
        same = [
            turn_for(calculus, "a", level=HintLevel.NUDGE),
            turn_for(calculus, "b", level=HintLevel.NUDGE),
        ]
        assert evaluation.rank_levels(provider, calculus, same).pairs == []


class TestGroundednessOverLessonTurns:
    """The same question `evaluate tutor` asks of hints, over lessons.

    A sitting is why it exists. A learner wrote ``x2 + h - 3^2 / x + h - x`` and
    the tutor replied *"You've set up the calculation perfectly!"* — which is the
    sitting-2 failure exactly ("your calculation is correct" to someone who had
    multiplied), in the one part of the system that check had never been pointed
    at.
    """

    GOLD = Path("tests/fixtures/gold/calculus_lesson_grounding_cases.yaml")

    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls):
        import yaml

        return yaml.safe_load(cls.GOLD.read_text())["cases"]

    def test_the_set_is_balanced(self, cases) -> None:
        # Or a judge that answers "grounded" to everything scores well by
        # saying nothing — which is what one local model did on the hint set.
        grounded = sum(1 for case in cases if case["grounded"])
        assert grounded == len(cases) - grounded

    def test_the_sitting_that_asked_for_this_is_in_it(self, cases) -> None:
        [case] = [c for c in cases if c["id"] == "gl_praised_a_malformed_expression"]
        assert case["grounded"] is False
        assert "perfectly" in case["reply"]

    def test_every_case_says_why_it_is_labelled_that_way(self, cases) -> None:
        # Carried rather than left as a YAML comment, so a disagreement can be
        # read against the reasoning that produced the label.
        for case in cases:
            assert case["note"].strip()

    # -- reconstructing a conversation -------------------------------------

    def _log(self, *records):
        return list(records)

    def _tutor(self, text: str, concept: str = "power_rule", level: str = "plain"):
        return AuditRecord(
            version=1,
            cause="tutor",
            summary="explain",
            evidence={
                "move": "explain",
                "concept_id": concept,
                "level": level,
                "text": text,
            },
        )

    def _said(self, text: str, concept: str = "power_rule"):
        return AuditRecord(
            version=1,
            cause="annotation",
            summary="said",
            evidence={"reflection": text, "kind": "lesson", "concept_id": concept},
        )

    def test_each_turn_is_paired_with_what_came_before_it(self) -> None:
        from agent_newton.core.evaluation.tutor import lesson_exchanges

        exchanges = lesson_exchanges(
            self._log(
                self._tutor("what do you think?"),
                self._said("a ratio"),
                self._tutor("exactly, a ratio."),
            )
        )
        assert [(e.said, e.reply) for e in exchanges] == [
            ("", "what do you think?"),
            ("a ratio", "exactly, a ratio."),
        ]

    def test_the_pairing_comes_from_the_log_rather_than_two_tables(self) -> None:
        """Because the pairing *is* an ordering.

        Who spoke when is a property of one sequence. Reading turns and
        utterances out of their separate tables and zipping them would be
        inventing that ordering rather than recovering it.
        """
        from agent_newton.core.evaluation.tutor import lesson_exchanges

        exchanges = lesson_exchanges(
            self._log(
                self._said("said before anything was asked"),
                self._tutor("first"),
            )
        )
        assert exchanges[0].said == "said before anything was asked"

    def test_the_written_summary_is_not_judged(self) -> None:
        # It is authored content, not a claim about anyone. Judging it would
        # score a person's prose as if a model had asserted it.
        from agent_newton.core.evaluation.tutor import lesson_exchanges

        exchanges = lesson_exchanges(
            self._log(self._tutor("the authored account", level="summary"))
        )
        assert exchanges == []

    def test_a_reply_is_not_carried_over_to_the_next_turn(self) -> None:
        # Cleared once used, or a turn would be judged against something the
        # learner said two turns ago and be marked ungrounded for it.
        from agent_newton.core.evaluation.tutor import lesson_exchanges

        exchanges = lesson_exchanges(
            self._log(
                self._tutor("q1"),
                self._said("an answer"),
                self._tutor("q2"),
                self._tutor("q3"),
            )
        )
        assert [e.said for e in exchanges] == ["", "an answer", ""]

    def test_openings_are_excluded_from_judging(self) -> None:
        """The same exclusion reflective prompts get, for the same reason.

        An opening makes no claim about the learner's work — there is none yet —
        so groundedness has nothing to bite on, and including it would dilute
        the rate with cases that cannot fail.
        """
        from agent_newton.core.evaluation.tutor import (
            JudgeReport,
            judge_lesson_turns,
            lesson_exchanges,
        )

        class NeverAsked:
            label = "fake/judge"

            def generate(self, prompt, schema, system):  # noqa: ANN001
                raise AssertionError("an opening must not be judged")

        exchanges = lesson_exchanges(self._log(self._tutor("what do you think?")))
        report = judge_lesson_turns(NeverAsked(), exchanges, JudgeReport())
        assert report.verdicts == []

    # -- the question itself ------------------------------------------------

    def test_the_question_is_asked_the_way_round_the_field_reads(self) -> None:
        """⚠️ The defect that made the hint judge look worse than chance.

        Asked the other way — "does the reply claim anything the work does not
        show" — a model answering correctly fills `grounded` with the opposite
        of what it means. It read as a judge performing below chance rather than
        as a prompt that inverts it: 30% against 90% once fixed. Nothing but the
        hand labels would have told them apart.
        """
        import inspect

        from agent_newton.core.evaluation.tutor import judge_lesson_grounded

        source = inspect.getsource(judge_lesson_grounded)
        assert "Is every claim" in source
        assert "supported by" in source
