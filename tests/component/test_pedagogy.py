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
    Support,
    TutorMove,
    check_fading,
    check_move,
    check_support_fading,
    hint_level,
    may_select,
    move_for,
    next_required_move,
    support_at_presentation,
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
            policy.hint_level = lambda mastery, prior_failures, band, **_: (  # type: ignore[assignment]
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


class TestTheBandedPlusLadder:
    """Every cut point read off the band, and one region the band excludes.

    The original ladder used ``theta_lower / 2`` as its lower boundary — half of
    a band edge, which is not a quantity the zone is defined in terms of. This
    one is: above the band nothing is disclosed, inside it a nudge, below it the
    two levels that name and show. ``theta_lower / 2`` survives only as the
    boundary between naming and showing, which is the one thing it was deciding.
    """

    @pytest.mark.parametrize(
        "mastery,expected",
        [
            (0.95, HintLevel.NONE),  # above the band: nothing is disclosed
            (0.90, HintLevel.NONE),  # theta_upper itself is above it
            (0.85, HintLevel.NUDGE),  # inside the band
            (0.50, HintLevel.TARGETED),  # below it: name the error
            (0.05, HintLevel.WORKED_STEP),  # far below: show the step
        ],
    )
    def test_the_regions(self, mastery: float, expected: HintLevel) -> None:
        assert hint_level(mastery, 0, BAND, policy="banded_plus") is expected

    def test_inside_the_band_escalation_stops_short_of_the_step(self) -> None:
        """The substantive change, and the one a sitting will judge.

        Failing repeatedly on a concept the model believes is nearly mastered
        used to hand over the worked step. The belief and the failures disagree
        there, and this resolves it by trusting the belief — the learner is told
        what went wrong and left to finish it.
        """
        levels = [hint_level(0.85, n, BAND, policy="banded_plus") for n in range(5)]
        assert levels == [
            HintLevel.NUDGE,
            HintLevel.TARGETED,
            HintLevel.TARGETED,
            HintLevel.TARGETED,
            HintLevel.TARGETED,
        ]

    def test_below_the_band_escalation_still_reaches_the_step(self) -> None:
        # The cap is about the top of the band, not about escalation in general.
        # A learner the model does not think is close still gets everything.
        levels = [hint_level(0.50, n, BAND, policy="banded_plus") for n in range(3)]
        assert levels == [
            HintLevel.TARGETED,
            HintLevel.WORKED_STEP,
            HintLevel.WORKED_STEP,
        ]

    def test_nothing_escalates_out_of_the_silent_region(self) -> None:
        # Above `theta_upper` no number of failures buys a hint. Failing is what
        # should move the *estimate*; when it has moved, the concept comes back
        # round lower down and the ladder gives something.
        assert all(
            hint_level(0.95, n, BAND, policy="banded_plus") is HintLevel.NONE
            for n in range(10)
        )

    @pytest.mark.parametrize("mastery", [0.0, 0.05, 0.34, 0.36, 0.5, 0.71, 0.85, 0.89])
    @pytest.mark.parametrize("failures", [0, 1, 2, 3])
    def test_the_original_ladder_is_untouched(
        self, mastery: float, failures: int
    ) -> None:
        """``banded`` must be exactly what it was, everywhere below the band.

        Every measured result was produced under it, and the renumbering of
        ``HintLevel`` to make room for ``NONE`` shifted both the base and the
        ceiling — so this is not a formality. It pins the arithmetic.
        """
        level = hint_level(mastery, failures, BAND)
        if mastery > BAND.theta_lower:
            base = HintLevel.NUDGE
        elif mastery > BAND.theta_lower / 2:
            base = HintLevel.TARGETED
        else:
            base = HintLevel.WORKED_STEP
        assert level is HintLevel(min(int(base) + failures, int(HintLevel.WORKED_STEP)))

    def test_the_original_ladder_never_falls_silent(self) -> None:
        # `NONE` is the new level and it must not leak into the old policy: a
        # cohort under `banded` that suddenly stopped hinting near mastery would
        # be a different run from the one every number came from.
        assert all(
            hint_level(m / 100, n, BAND) is not HintLevel.NONE
            for m in range(101)
            for n in range(4)
        )

    @pytest.mark.parametrize("failures", [0, 1, 2, 5])
    @pytest.mark.parametrize(
        "band",
        [
            ZPDConfig(theta_lower=0.70, theta_upper=0.90),
            ZPDConfig(theta_lower=0.30, theta_upper=0.95),
            ZPDConfig(theta_lower=0.80, theta_upper=0.85),
        ],
    )
    def test_fading_holds_under_the_new_ladder_too(
        self, band: ZPDConfig, failures: int
    ) -> None:
        assert check_fading(band, failures, policy="banded_plus") is None


class TestSupportAtPresentation:
    """The second axis: what is shown before the learner has done anything."""

    @pytest.mark.parametrize(
        "mastery,expected",
        [
            (0.95, Support.NONE),
            (0.85, Support.NONE),  # inside the band, the question stands alone
            (0.70, Support.FORMULA),  # theta_lower itself is below the line
            (0.50, Support.FORMULA),
            (0.35, Support.FORMULA_AND_EXAMPLE),
            (0.0, Support.FORMULA_AND_EXAMPLE),
        ],
    )
    def test_the_regions(self, mastery: float, expected: Support) -> None:
        assert support_at_presentation(mastery, BAND) is expected

    def test_its_boundaries_are_the_hint_ladder_s(self) -> None:
        """A learner must not sit in one tier for the question and another for
        the reply. Checked across the range rather than at the two edges, since
        the claim is that the boundaries coincide everywhere."""
        for index in range(201):
            mastery = index / 200
            shows_example = support_at_presentation(mastery, BAND).shows_example
            worked = (
                hint_level(mastery, 0, BAND, policy="banded_plus")
                is HintLevel.WORKED_STEP
            )
            assert shows_example is worked

    @pytest.mark.parametrize(
        "band",
        [
            ZPDConfig(theta_lower=0.70, theta_upper=0.90),
            ZPDConfig(theta_lower=0.30, theta_upper=0.95),
            ZPDConfig(theta_lower=0.05, theta_upper=0.99),
        ],
    )
    def test_support_never_rises_with_mastery(self, band: ZPDConfig) -> None:
        assert check_support_fading(band) is None

    def test_the_check_would_catch_an_inversion(self) -> None:
        """A property test that cannot fail proves nothing."""
        from agent_newton.core.pedagogy import policy

        original = policy.support_at_presentation
        try:
            policy.support_at_presentation = lambda mastery, band: (  # type: ignore[assignment]
                Support.FORMULA_AND_EXAMPLE if mastery > 0.5 else Support.NONE
            )
            violation = policy.check_support_fading(BAND)
            assert violation is not None
            assert violation.rule == "support_fading"
        finally:
            policy.support_at_presentation = original  # type: ignore[assignment]


class TestTheMoveFollowsTheLevel:
    """``move_for`` is one rule where the two tutors had a copy each."""

    def test_silence_forces_a_reflective_turn(self) -> None:
        # The whole of `HintLevel.NONE`: remediation is the only move that
        # teaches, and it is withheld where the model already believes the
        # learner has the concept.
        assert move_for(HintLevel.NONE, (), True) is TutorMove.REFLECT
        assert move_for(HintLevel.NONE, (TutorMove.REFLECT,), True) is TutorMove.REFLECT
        assert move_for(HintLevel.NONE, (), True, True) is TutorMove.REFLECT

    def test_below_it_the_error_first_ordering_is_unchanged(self) -> None:
        assert move_for(HintLevel.TARGETED, (), True) is TutorMove.REFLECT
        assert (
            move_for(HintLevel.TARGETED, (TutorMove.REFLECT,), True)
            is TutorMove.REMEDIATE
        )
        assert move_for(HintLevel.TARGETED, (), False) is TutorMove.HINT

    def test_an_explained_step_still_satisfies_the_rule(self) -> None:
        assert move_for(HintLevel.NUDGE, (), True, True) is TutorMove.REMEDIATE

    def test_a_silent_turn_targets_nothing(self, toy) -> None:
        """Nothing to copy, and nothing credited as remediation.

        Read off the tutor rather than off ``move_for``, because ``targets`` is
        the tutor's to set and it is the field the learner model reads.
        """
        tutor = TemplateTutor(BAND, "banded_plus")
        # An item on a concept the catalogue covers: two of toy_algebra's
        # concepts carry no misconception, and `domain validate` warns about
        # exactly that.
        labelled = {m.concept_id for m in toy.misconceptions.all()}
        item = next(
            i for i in toy.items.bank("practice") if i.concept_id in labelled
        )
        board = new_blackboard("L1", 1, toy.concepts, Config(domain="toy_algebra"))
        hint = tutor.respond(
            item,
            Diagnosis(toy.misconceptions.for_concept(item.concept_id)[0].id),
            board.view(),
            toy,
            response="whatever",
            mastery=0.95,
            prior_failures=2,
            moves_this_item=(TutorMove.REFLECT,),
        )
        assert hint.level is HintLevel.NONE
        assert hint.move is TutorMove.REFLECT
        assert hint.targets is None
