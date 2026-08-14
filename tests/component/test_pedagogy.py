"""Instructional rules.

These are the predicates the tutor and the planner's guardrail layer must
satisfy. Each is tested as the property it claims to be, not at a sample point.
"""

from __future__ import annotations

import itertools

import pytest

from agent_newton.config import Config, ZPDConfig
from agent_newton.core.agents.base import Diagnosis
from agent_newton.core.agents.tutor import TemplateTutor
from agent_newton.core.pedagogy import (
    BAND_MEMBERSHIP,
    ERROR_FIRST,
    HintLevel,
    TutorMove,
    check_fading,
    check_move,
    hint_level,
    may_select,
    next_required_move,
)
from agent_newton.core.state.store import new_blackboard
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains import registry

BAND = ZPDConfig(theta_lower=0.70, theta_upper=0.90)


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


class TestBandMembership:
    def test_permits_a_concept_in_the_frontier(self) -> None:
        assert may_select("distribute", Frontier(frozenset({"distribute"}))) is None

    def test_refuses_a_concept_outside_it(self) -> None:
        violation = may_select("solve_linear", Frontier(frozenset({"distribute"})))
        assert violation is not None
        assert violation.rule == BAND_MEMBERSHIP

    def test_refuses_everything_when_the_frontier_is_empty(self) -> None:
        # An empty frontier means the learner is done; nothing is selectable.
        violation = may_select("distribute", Frontier(frozenset()))
        assert violation is not None
        assert "empty" in violation.message

    def test_the_message_names_what_was_available(self) -> None:
        violation = may_select("x", Frontier(frozenset({"a", "b"})))
        assert violation is not None
        assert "a" in violation.message and "b" in violation.message


class TestScaffolding:
    @pytest.mark.parametrize(
        ("mastery", "expected"),
        [
            (0.05, HintLevel.WORKED_STEP),  # near the bottom: show the step
            (0.50, HintLevel.TARGETED),  # mid band: name the misconception
            (0.85, HintLevel.NUDGE),  # near mastery: a nudge suffices
        ],
    )
    def test_support_follows_the_mastery_estimate(
        self, mastery: float, expected: HintLevel
    ) -> None:
        assert hint_level(mastery, prior_failures=0, band=BAND) is expected

    def test_repeated_failure_escalates_support(self) -> None:
        # A learner who is stuck must not be nudged over and over.
        levels = [hint_level(0.85, prior_failures=n, band=BAND) for n in range(3)]
        assert levels == [HintLevel.NUDGE, HintLevel.TARGETED, HintLevel.WORKED_STEP]

    def test_escalation_is_bounded(self) -> None:
        assert hint_level(0.05, prior_failures=99, band=BAND) is HintLevel.WORKED_STEP


class TestFading:
    def test_support_never_rises_with_mastery(self) -> None:
        assert check_fading(BAND) is None

    @pytest.mark.parametrize("prior_failures", [0, 1, 2, 5])
    def test_holds_at_every_escalation_level(self, prior_failures: int) -> None:
        # "All else equal" means at fixed failure count. Escalation may raise
        # support, but never as a function of rising mastery.
        assert check_fading(BAND, prior_failures=prior_failures) is None

    @pytest.mark.parametrize(
        "band",
        [
            ZPDConfig(theta_lower=0.50, theta_upper=0.95),
            ZPDConfig(theta_lower=0.80, theta_upper=0.90),
            ZPDConfig(theta_lower=0.10, theta_upper=0.20),
        ],
    )
    def test_holds_for_any_band(self, band: ZPDConfig) -> None:
        # The threshold sweep moves the band, so the property must not depend
        # on the default values.
        assert check_fading(band) is None

    def test_pairwise_across_the_whole_range(self) -> None:
        levels = [hint_level(i / 40, 0, BAND) for i in range(41)]
        assert all(a >= b for a, b in itertools.pairwise(levels))

    def test_the_check_would_catch_an_inversion(self) -> None:
        """A property test that cannot fail proves nothing."""
        from agent_newton.core.pedagogy import policy

        original = policy.hint_level
        try:
            # Support rising with mastery — exactly what fading forbids.
            policy.hint_level = lambda mastery, prior_failures, band: (  # type: ignore[assignment]
                HintLevel.WORKED_STEP if mastery > 0.5 else HintLevel.NUDGE
            )
            violation = policy.check_fading(BAND)
            assert violation is not None
            assert violation.rule == "fading"
        finally:
            policy.hint_level = original  # type: ignore[assignment]


class TestErrorFirst:
    def test_remediation_after_reflection_is_permitted(self) -> None:
        assert (
            check_move(
                TutorMove.REMEDIATE,
                [TutorMove.HINT, TutorMove.REFLECT],
                misconception_confirmed=True,
            )
            is None
        )

    def test_remediation_without_reflection_is_refused(self) -> None:
        violation = check_move(
            TutorMove.REMEDIATE, [TutorMove.HINT], misconception_confirmed=True
        )
        assert violation is not None
        assert violation.rule == ERROR_FIRST

    def test_the_rule_only_applies_after_a_confirmed_misconception(self) -> None:
        # No confirmed misconception means nothing to reflect on, so plain
        # remediation is fine.
        assert check_move(TutorMove.REMEDIATE, [], misconception_confirmed=False) is None

    @pytest.mark.parametrize("move", [TutorMove.HINT, TutorMove.REFLECT])
    def test_other_moves_are_unconstrained(self, move: TutorMove) -> None:
        assert check_move(move, [], misconception_confirmed=True) is None

    def test_the_required_move_is_reflection(self) -> None:
        assert next_required_move([], misconception_confirmed=True) is TutorMove.REFLECT

    def test_nothing_is_required_once_reflection_has_happened(self) -> None:
        assert next_required_move([TutorMove.REFLECT], misconception_confirmed=True) is None

    def test_nothing_is_required_without_a_confirmation(self) -> None:
        assert next_required_move([], misconception_confirmed=False) is None

    def test_driving_from_the_requirement_satisfies_the_check(self) -> None:
        # A tutor that asks what is required and does it must never violate the
        # rule it is being checked against.
        history: list[TutorMove] = []
        required = next_required_move(history, misconception_confirmed=True)
        assert required is TutorMove.REFLECT
        history.append(required)
        assert check_move(TutorMove.REMEDIATE, history, misconception_confirmed=True) is None


class TestViolationsAreAuditable:
    def test_carry_a_rule_and_a_readable_message(self) -> None:
        violation = may_select("x", Frontier(frozenset({"a"})))
        assert violation is not None
        assert violation.rule and violation.message
        assert str(violation).startswith(f"[{BAND_MEMBERSHIP}]")


class TestTemplateTutorIgnoresTheStep:
    """The step the learner wrote reaches the tutor, but not this one's text.

    The model-backed tutor needs it — without the step it invents one to match
    the misconception's description. The template tutor must not use it: the
    cohorts run ``impl: template``, so a hint whose wording varied with the
    response would make every measured number depend on something that was not
    part of the manipulation.
    """

    def _hint(self, toy, response: str):
        board = new_blackboard("L1", 1, toy.concepts, Config(domain="toy_algebra"))
        return TemplateTutor(BAND).respond(
            toy.items.get("ta_dist_p1"),
            Diagnosis("distribute_first_term_only", 0.9),
            board.view(),
            toy,
            response=response,
            mastery=0.0,
            prior_failures=1,
            moves_this_item=[TutorMove.REFLECT],
        )

    def test_the_text_does_not_vary_with_the_response(self, toy) -> None:
        first = self._hint(toy, "3*x + 4")
        second = self._hint(toy, "something else entirely")
        assert first.text == second.text
        assert first.move is second.move and first.level is second.level

    def test_the_response_never_appears_in_the_hint(self, toy) -> None:
        # The stronger form: not merely stable, but not present at all.
        hint = self._hint(toy, "ZZQQ-marker")
        assert "ZZQQ-marker" not in hint.text
