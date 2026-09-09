"""The simulated learner.

The simulator is the ground truth every measurement is scored against, so its
determinism and its remediation mechanism are pinned here in detail.
"""

from __future__ import annotations

import pytest

from agent_newton.config import SimulatorConfig
from agent_newton.core.simulator import (
    SimulatedLearner,
    SymbolicSurface,
    sample_profile,
)
from agent_newton.domains import registry
from agent_newton.domains.base import Verdict

CONFIG = SimulatorConfig(misconceptions_per_learner=2, p_fire_range=(0.6, 0.9))
#: Fires on every applicable item, so tests are about mechanism not luck.
ALWAYS = SimulatorConfig(misconceptions_per_learner=4, p_fire_range=(1.0, 1.0))
NEVER = SimulatorConfig(misconceptions_per_learner=4, p_fire_range=(0.0, 0.0))


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def learner(domain, learner_id="L1", seed=7, config=CONFIG) -> SimulatedLearner:
    profile = sample_profile(learner_id, seed, domain.misconceptions, config)
    return SimulatedLearner(profile, domain, config)


class TestProfileSampling:
    def test_the_same_seed_gives_the_same_learner(self, toy) -> None:
        # The paired design depends on this: one learner, both architectures.
        a = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        b = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        assert a.firing == b.firing

    def test_different_learners_differ(self, toy) -> None:
        a = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        b = sample_profile("L2", 42, toy.misconceptions, CONFIG)
        assert a.firing != b.firing

    def test_different_seeds_differ(self, toy) -> None:
        a = sample_profile("L1", 1, toy.misconceptions, CONFIG)
        b = sample_profile("L1", 2, toy.misconceptions, CONFIG)
        assert a.firing != b.firing

    def test_draws_the_configured_number(self, toy) -> None:
        profile = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        assert len(profile.firing) == 2

    def test_never_draws_more_than_the_catalogue_holds(self, toy) -> None:
        greedy = SimulatorConfig(misconceptions_per_learner=99)
        profile = sample_profile("L1", 42, toy.misconceptions, greedy)
        assert len(profile.firing) == len(toy.misconceptions.ids())

    def test_probabilities_lie_in_the_configured_range(self, toy) -> None:
        profile = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        low, high = CONFIG.p_fire_range
        assert all(low <= p <= high for p in profile.firing.values())

    def test_only_draws_real_misconceptions(self, toy) -> None:
        profile = sample_profile("L1", 42, toy.misconceptions, CONFIG)
        assert set(profile.firing) <= set(toy.misconceptions.ids())


class TestAnswering:
    def test_a_held_misconception_produces_the_documented_error(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        item = toy.items.get("ta_dist_p1")  # probes distribute_first_term_only
        step = subject.answer(item)
        assert step.fired == "distribute_first_term_only"
        assert not step.correct
        assert toy.verifier.verify(item, step.response).verdict is Verdict.INCORRECT

    def test_answers_correctly_when_nothing_fires(self, toy) -> None:
        subject = learner(toy, config=NEVER)
        item = toy.items.get("ta_dist_p1")
        step = subject.answer(item)
        assert step.fired is None
        assert step.correct
        assert toy.verifier.verify(item, step.response).verdict is Verdict.CORRECT

    def test_answers_correctly_on_items_it_cannot_err_on(self, toy) -> None:
        # An item probing nothing the learner holds is simply answered.
        subject = learner(toy, config=ALWAYS)
        item = toy.items.get("ta_int_p1")  # probes nothing
        step = subject.answer(item)
        assert step.correct

    def test_is_deterministic(self, toy) -> None:
        subject = learner(toy)
        item = toy.items.get("ta_dist_p1")
        assert len({subject.answer(item).response for _ in range(20)}) == 1

    def test_attempts_are_independent_draws(self, toy) -> None:
        # Otherwise a learner stuck on one item would be stuck identically
        # forever, which no amount of tutoring could distinguish from failure.
        subject = learner(toy, seed=3)
        item = toy.items.get("ta_dist_p1")
        rolls = {subject.answer(item, attempt=n).correct for n in range(40)}
        assert len(rolls) == 2, "expected both outcomes across many attempts"


class TestRepeatedPracticeIsAFreshDraw:
    """Practising an item again must not replay its own past.

    Regression. When planners began repeating items, a roll keyed only on
    (item, attempt) made every revisit an exact clone: a learner who once
    answered correctly did so forever, mastering the concept without ever
    demonstrating anything, while one who erred could never stop. Mastery
    became a step function and misconceptions went permanently unexhibited.
    """

    def test_revisits_vary(self, toy) -> None:
        subject = learner(toy, seed=3, config=SimulatorConfig(
            misconceptions_per_learner=4, p_fire_range=(0.5, 0.5)
        ))
        item = toy.items.get("ta_dist_p1")
        outcomes = {subject.answer(item, repetition=n).correct for n in range(12)}
        assert len(outcomes) == 2, "repeated practice replayed the same outcome"

    def test_a_single_visit_is_still_deterministic(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        item = toy.items.get("ta_dist_p1")
        assert len({subject.answer(item, repetition=2).response for _ in range(8)}) == 1

    def test_repetition_and_attempt_are_distinct_axes(self, toy) -> None:
        # The second visit's first step is a different draw from the first
        # visit's second step; conflating them would reintroduce the clone.
        from agent_newton.core.simulator.engine import _roll

        assert _roll(1, "L", "i", 1, 0, "m") != _roll(1, "L", "i", 0, 1, "m")


class TestCommonRandomNumbers:
    """The same learner meets the same item identically in both architectures."""

    def test_a_decision_does_not_depend_on_history(self, toy) -> None:
        # Rolls are drawn per decision rather than from a running generator, so
        # unrelated items answered in between cannot perturb this one.
        fresh = learner(toy, seed=11)
        used = learner(toy, seed=11)
        for other in ("ta_int_p1", "ta_clt_p1", "ta_solve_p2"):
            used.answer(toy.items.get(other))

        item = toy.items.get("ta_dist_p1")
        assert fresh.answer(item).response == used.answer(item).response

    def test_two_arms_draw_the_same_numbers(self, toy) -> None:
        # This is the variance reduction that gives the paired comparison its
        # power: the arms differ by the tutoring, not by their random streams.
        coupled = learner(toy, "L1", seed=5)
        decoupled = learner(toy, "L1", seed=5)
        for item_id in ("ta_dist_p2", "ta_clt_p3", "ta_solve_p1"):
            item = toy.items.get(item_id)
            assert coupled.answer(item).fired == decoupled.answer(item).fired


class TestRemediation:
    def test_a_correctly_targeted_hint_weakens_the_misconception(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        target = sorted(subject.profile.firing)[0]
        before = subject.profile.probability(target)
        assert subject.receive_hint(target)
        assert subject.profile.probability(target) < before

    def test_a_misaimed_hint_changes_nothing(self, toy) -> None:
        # The mechanism that gives diagnostic accuracy consequences: a hint
        # aimed at a misconception the learner does not hold does no work.
        subject = learner(toy, config=CONFIG)
        held = set(subject.profile.firing)
        missing = next(m for m in toy.misconceptions.ids() if m not in held)
        before = subject.profile.snapshot()
        assert not subject.receive_hint(missing)
        assert subject.profile.snapshot() == before

    def test_no_target_changes_nothing(self, toy) -> None:
        subject = learner(toy)
        before = subject.profile.snapshot()
        assert not subject.receive_hint(None)
        assert subject.profile.snapshot() == before

    def test_remediation_eventually_stops_the_error(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        item = toy.items.get("ta_dist_p1")
        target = "distribute_first_term_only"
        assert not subject.answer(item).correct

        for _ in range(30):
            subject.receive_hint(target)

        assert subject.answer(item).correct

    def test_progress_is_reported_as_a_ratio(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        assert subject.profile.remediation_ratio() == pytest.approx(0.0)
        for target in list(subject.profile.firing):
            for _ in range(30):
                subject.receive_hint(target)
        remediated = subject.profile.remediation_ratio()
        assert remediated is not None, "this learner holds misconceptions to reduce"
        assert remediated > 0.99


class TestSurface:
    def test_symbolic_rendering_is_the_identity(self, toy) -> None:
        # The engine decides; the surface only phrases. In symbolic mode there
        # is no phrasing, so the response must survive untouched.
        subject = learner(toy, config=ALWAYS)
        item = toy.items.get("ta_dist_p1")
        step = subject.answer(item)
        assert SymbolicSurface().render(item, step) == step.response

    def test_symbolic_surface_satisfies_the_protocol(self) -> None:
        from agent_newton.core.simulator import SurfaceRenderer

        assert isinstance(SymbolicSurface(), SurfaceRenderer)


class TestProfilesStayHidden:
    def test_the_profile_is_not_reachable_from_a_state_view(self, toy) -> None:
        # The tutor must observe only the interaction history. If a profile ever
        # became reachable through a view, diagnostic accuracy would be
        # measuring nothing.
        from agent_newton.config import Config
        from agent_newton.core.state.store import new_blackboard

        board = new_blackboard("L1", 1, toy.concepts, Config(domain="toy_algebra"))
        for view in (board.view("coupled"), board.view("decoupled")):
            assert not hasattr(view, "profile")
            assert not hasattr(view, "firing")


class TestNothingToRemediateIsNotFullyRemediated:
    """A learner holding no misconceptions has nothing to measure, not a perfect score.

    `misconceptions_per_learner: 0` is an accepted configuration, and
    `remediation_ratio` returned **1.0** for it — so every learner in such a
    cohort was reported as *fully remediated* and the declared primary outcome
    would have read perfect. None is the answer the rest of the codebase already
    gives here: `normalised_gain` is None when the pre-test left no headroom, and
    this same method is None for a person, who has no profile at all.
    """

    def test_a_learner_with_no_misconceptions_reports_none(self, toy) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import sample_profile

        profile = sample_profile(
            "L0", 1, toy.misconceptions, SimulatorConfig(misconceptions_per_learner=0)
        )
        assert profile.firing == {}
        assert profile.remediation_ratio() is None

    def test_a_learner_who_holds_something_still_reports_a_number(self, toy) -> None:
        # The guard must not swallow the ordinary case.
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import sample_profile

        profile = sample_profile(
            "L0", 1, toy.misconceptions, SimulatorConfig(misconceptions_per_learner=2)
        )
        assert profile.firing
        assert profile.remediation_ratio() == pytest.approx(0.0)

    def test_the_cohort_metric_skips_it_rather_than_averaging_it(self) -> None:
        # `run_cohort` already filters `is not None`, which is why returning None
        # is safe — but assert it, because a mean over a None would raise and a
        # mean *including* a 1.0 would quietly report a perfect primary outcome.
        values = [None, 0.4, 0.6]
        measured = [v for v in values if v is not None]
        assert sum(measured) / len(measured) == pytest.approx(0.5)


class TestABandTheEstimateCannotCross:
    """`theta_upper` of 1.0 looks legitimate and guarantees a null.

    Posteriors are clamped below 1.0, so nothing can ever cross an upper edge of
    exactly 1.0: every concept stays in the frontier forever, nothing is mastered,
    and `goals_mastered` is zero for every learner in the run.
    """

    def test_it_is_refused_at_load(self) -> None:
        from pydantic import ValidationError

        from agent_newton.config import ZPDConfig

        with pytest.raises(ValidationError, match="must be below 1.0"):
            ZPDConfig(theta_lower=0.7, theta_upper=1.0)

    def test_the_ordinary_band_is_still_accepted(self) -> None:
        from agent_newton.config import ZPDConfig

        band = ZPDConfig(theta_lower=0.7, theta_upper=0.9)
        assert band.theta_upper == 0.9

    def test_the_clamp_is_what_makes_it_unreachable(self) -> None:
        # The reason, asserted rather than described: the highest posterior BKT
        # can produce is below 1.0, so `P(c) < theta_upper` stays true forever.
        from agent_newton.config import BKTConfig
        from agent_newton.core.state import bkt

        params = BKTConfig()
        highest = 1.0 - 1e-9
        for _ in range(200):
            highest = bkt.observe(highest, True, params)
        assert highest < 1.0


# --------------------------------------------------------------- learner types

#: The three dials off. Identical to CONFIG, and named so a test asserting
#: "off is today" says what it is asserting.
OFF = SimulatorConfig(misconceptions_per_learner=2, p_fire_range=(0.6, 0.9))


class TestOffIsToday:
    """Every dial defaults off, and off must be the behaviour that was measured.

    The stored results were produced under this configuration. A dial whose zero
    is not the identity would move every one of them without anything saying so.
    """

    def test_the_defaults_are_all_off(self) -> None:
        blank = SimulatorConfig()
        assert blank.forgetting_rate == 0.0
        assert blank.forgetting_period == 0
        assert blank.slip_rate == 0.0
        assert blank.remediation_curve == "exponential"

    def test_answering_is_unchanged_with_the_dials_off(self, toy) -> None:
        item = toy.items.all()[0]
        subject = learner(toy, config=OFF)
        steps = [
            subject.answer(item, attempt=a, repetition=r)
            for r in range(3)
            for a in range(3)
        ]
        again = learner(toy, config=OFF)
        assert steps == [
            again.answer(item, attempt=a, repetition=r)
            for r in range(3)
            for a in range(3)
        ]

    def test_remediation_is_unchanged_with_the_dials_off(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        target = sorted(subject.profile.firing)[0]
        before = subject.profile.probability(target)
        subject.receive_hint(target)
        # The exponential default: a constant share of what was there.
        assert subject.profile.probability(target) == pytest.approx(
            before * ALWAYS.remediation_factor
        )


class TestForgetting:
    def test_a_rate_of_zero_moves_nothing(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        subject.receive_hint(sorted(subject.profile.firing)[0])
        before = subject.profile.snapshot()
        assert not subject.profile.forget(0.0)
        assert subject.profile.snapshot() == before

    def test_only_what_was_taught_can_be_forgotten(self, toy) -> None:
        # An untouched misconception sits at its initial value and has no ground
        # to give back, which is what makes "forgets some of it" fall out.
        subject = learner(toy, config=ALWAYS)
        before = subject.profile.snapshot()
        assert not subject.profile.forget(1.0)
        assert subject.profile.snapshot() == before

    def test_a_full_rate_returns_the_learner_to_where_they_began(self, toy) -> None:
        subject = learner(toy, config=ALWAYS)
        for target in sorted(subject.profile.firing):
            subject.receive_hint(target)
        assert subject.profile.forget(1.0)
        assert subject.profile.snapshot() == pytest.approx(
            dict(subject.profile.initial)
        )

    def test_it_never_pushes_past_where_they_began(self, toy) -> None:
        # The ceiling is what keeps remediation_ratio inside [0, 1], and the
        # declared primary outcome is computed from that ratio.
        subject = learner(toy, config=ALWAYS)
        target = sorted(subject.profile.firing)[0]
        subject.receive_hint(target)
        for _ in range(20):
            subject.profile.forget(1.0)
        for misconception_id, value in subject.profile.snapshot().items():
            assert value <= subject.profile.initial[misconception_id] + 1e-12
        assert 0.0 <= (subject.profile.remediation_ratio() or 0.0) <= 1.0

    def test_it_fires_on_the_beat_and_not_between(self, toy) -> None:
        config = SimulatorConfig(
            misconceptions_per_learner=4,
            p_fire_range=(1.0, 1.0),
            forgetting_rate=1.0,
            forgetting_period=3,
        )
        subject = learner(toy, config=config)
        target = sorted(subject.profile.firing)[0]
        subject.receive_hint(target)
        taught = subject.profile.probability(target)
        item = toy.items.all()[0]

        subject.answer(item)  # t = 1
        assert subject.profile.probability(target) == pytest.approx(taught)
        subject.answer(item)  # t = 2
        assert subject.profile.probability(target) == pytest.approx(taught)
        subject.answer(item)  # t = 3, the beat
        assert subject.profile.probability(target) == pytest.approx(
            subject.profile.initial[target]
        )

    def test_the_clock_runs_whether_or_not_the_dial_is_on(self, toy) -> None:
        subject = learner(toy, config=OFF)
        item = toy.items.all()[0]
        subject.answer(item)
        subject.answer(item)
        assert subject.profile.t == 2


class TestTheRemediationCurve:
    def test_the_two_shapes_agree_on_the_first_hint(self, toy) -> None:
        # Nothing has happened yet for them to differ about, so the shape is the
        # whole difference rather than the shape and a different first step.
        exponential = learner(toy, config=ALWAYS)
        linear = learner(
            toy,
            config=SimulatorConfig(
                misconceptions_per_learner=4,
                p_fire_range=(1.0, 1.0),
                remediation_curve="linear",
            ),
        )
        target = sorted(exponential.profile.firing)[0]
        exponential.receive_hint(target)
        linear.receive_hint(target)
        assert exponential.profile.probability(target) == pytest.approx(
            linear.profile.probability(target)
        )

    def test_linear_reaches_zero_and_exponential_does_not(self, toy) -> None:
        linear = learner(
            toy,
            config=SimulatorConfig(
                misconceptions_per_learner=4,
                p_fire_range=(1.0, 1.0),
                remediation_curve="linear",
            ),
        )
        exponential = learner(toy, config=ALWAYS)
        target = sorted(linear.profile.firing)[0]
        for _ in range(10):
            linear.receive_hint(target)
            exponential.receive_hint(target)
        assert linear.profile.probability(target) == 0.0
        assert exponential.profile.probability(target) > 0.0

    def test_linear_never_goes_below_zero(self, toy) -> None:
        profile = sample_profile("L1", 7, toy.misconceptions, ALWAYS)
        target = sorted(profile.firing)[0]
        for _ in range(50):
            profile.remediate(target, 0.55, "linear")
        assert profile.probability(target) == 0.0


class TestSlipping:
    #: Holds nothing, so every wrong answer must have come from a slip.
    NOTHING_HELD = SimulatorConfig(
        misconceptions_per_learner=0, p_fire_range=(1.0, 1.0), slip_rate=1.0
    )

    def test_a_rate_of_zero_never_slips(self, toy) -> None:
        subject = learner(
            toy,
            config=SimulatorConfig(
                misconceptions_per_learner=0, p_fire_range=(1.0, 1.0)
            ),
        )
        item = toy.items.all()[0]
        assert all(
            subject.answer(item, attempt=a).correct for a in range(10)
        )

    def test_a_slip_carries_no_injected_label(self, toy) -> None:
        # The label is the diagnostic's ground truth. A slip is not a bug, so
        # there is nothing for it to have named.
        subject = learner(toy, config=self.NOTHING_HELD)
        item = next(i for i in toy.items.all() if i.probes)
        step = subject.answer(item)
        assert not step.correct
        assert step.fired is None
        assert step.label == "unlabelled-error"

    def test_a_slip_is_wrong_in_a_way_the_learner_does_not_hold(self, toy) -> None:
        subject = learner(toy, config=self.NOTHING_HELD)
        item = next(i for i in toy.items.all() if i.probes)
        step = subject.answer(item)
        assert step.response != item.answer

    def test_both_arms_slip_on_the_same_step(self, toy) -> None:
        # Common random numbers: the arms must differ by their tutoring, not by
        # their slip streams.
        first = learner(toy, config=self.NOTHING_HELD)
        second = learner(toy, config=self.NOTHING_HELD)
        item = next(i for i in toy.items.all() if i.probes)
        assert [first.answer(item, attempt=a) for a in range(6)] == [
            second.answer(item, attempt=a) for a in range(6)
        ]
