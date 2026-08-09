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
        assert subject.profile.remediation_ratio() > 0.99


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
