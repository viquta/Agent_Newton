"""Scoring a planner against a policy that knows the learner's profile.

The reference is the only agent in the system handed the true profile, so the
first thing tested here is that nothing else can reach one. The rest checks that
the measures mean what they are reported to mean — regret in particular, since
it is a difference between two value functions and would look plausible if
either side were computed wrongly.
"""

from __future__ import annotations

import pytest

from agent_newton.config import BKTConfig, Config, ZPDConfig
from agent_newton.core.agents.llm import LLMPlanner
from agent_newton.core.agents.planner import (
    FixedOrderPlanner,
    GoalDirectedPlanner,
    OraclePlanner,
)
from agent_newton.core.evaluation.planning import Decision, PlanningReport, evaluate
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.state import bkt
from agent_newton.core.state.schema import Emphasis, Plan
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry

BAND = ZPDConfig()
PRIOR = bkt.initial(BKTConfig())


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def config_for(arm: str = "coupled", **planner) -> Config:
    return Config.model_validate(
        {
            "domain": "toy_algebra",
            "arm": arm,
            "cohort": {"n_learners": 3, "max_items": 12},
            "simulator": {"surface": "symbolic"},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed", **planner},
            },
        }
    )


class TestOnlyTheReferenceSeesTheProfile:
    """Ground truth is a capability here, as it is for the diagnostic.

    The oracle planner is handed the profile because a config named it. Every
    other planner must have no route to one — otherwise the regret it is being
    measured on would be computed against information it secretly shared.
    """

    @pytest.mark.parametrize(
        "planner",
        [
            GoalDirectedPlanner(BAND, PRIOR, Emphasis.CONSOLIDATE),
            FixedOrderPlanner(2),
        ],
    )
    def test_a_measured_planner_holds_no_profile(self, planner) -> None:
        reachable = [getattr(planner, name) for name in vars(planner)]
        assert not any(isinstance(value, dict) and value for value in reachable), (
            "a planner under test is holding a mapping that could be a profile"
        )

    def test_the_model_backed_planner_takes_no_profile(self) -> None:
        import inspect

        parameters = set(inspect.signature(LLMPlanner.__init__).parameters)
        assert not parameters & {"profile", "firing", "learner"}

    def test_the_oracle_reads_the_profile_live(self, toy) -> None:
        # Held by reference, so remediation during a session is visible without
        # the loop having to report it.
        firing = {"combine_unlike_terms": 0.8}
        oracle = OraclePlanner(firing, BAND, PRIOR)
        item = next(
            i for i in toy.items.all() if "combine_unlike_terms" in i.probes
        )
        assert oracle.value(item) == pytest.approx(0.8)
        firing["combine_unlike_terms"] = 0.1
        assert oracle.value(item) == pytest.approx(0.1)


class TestTheReferencePolicy:
    def test_prefers_the_item_carrying_the_most_held_mass(self, toy) -> None:
        board = new_blackboard("L1", 1, toy.concepts, config_for())
        board.record_plan(Plan(goal="solve_linear"))
        view = board.view()

        held = {m: 0.9 for m in toy.misconceptions.ids()}
        oracle = OraclePlanner(held, BAND, PRIOR)
        chosen = oracle.select(view, toy, {})
        assert chosen is not None

        options = oracle.candidate_items(view, toy, {})  # pyright: ignore[reportArgumentType]
        assert oracle.value(chosen) == max(oracle.value(i) for i in options)

    def test_ignores_misconceptions_the_learner_does_not_hold(self, toy) -> None:
        board = new_blackboard("L1", 1, toy.concepts, config_for())
        board.record_plan(Plan(goal="solve_linear"))
        oracle = OraclePlanner({}, BAND, PRIOR)
        for item in oracle.candidate_items(board.view(), toy, {}):  # pyright: ignore[reportArgumentType]
            assert oracle.value(item) == 0.0

    def test_is_deterministic(self, toy) -> None:
        board = new_blackboard("L1", 1, toy.concepts, config_for())
        board.record_plan(Plan(goal="solve_linear"))
        held = {m: 0.5 for m in toy.misconceptions.ids()}
        chosen = {
            OraclePlanner(held, BAND, PRIOR).select(board.view(), toy, {}).id  # type: ignore[union-attr]
            for _ in range(5)
        }
        assert len(chosen) == 1


class TestRegret:
    def _decision(self, chosen: float, reference: float) -> Decision:
        return Decision(
            learner_id="L",
            goal="g",
            chosen_item="a",
            chosen_concept="c",
            reference_item="b",
            reference_concept="d",
            chosen_value=chosen,
            reference_value=reference,
            in_frontier=True,
            options=2,
        )

    def test_is_zero_when_the_choice_was_equally_good(self) -> None:
        # Disagreement is not by itself a cost, which is why regret rather than
        # agreement is the figure to read.
        assert self._decision(0.5, 0.5).regret == 0.0

    def test_is_never_negative(self) -> None:
        # The reference maximises over the same options, so it cannot be beaten.
        assert self._decision(0.9, 0.5).regret == 0.0

    def test_measures_what_was_given_up(self) -> None:
        assert self._decision(0.2, 0.7).regret == pytest.approx(0.5)

    def test_the_share_normalises_by_what_was_available(self) -> None:
        # The arms drive different trajectories, so the reference faces
        # different states; raw regret is not comparable between them.
        report = PlanningReport([self._decision(0.2, 0.8)])
        assert report.regret_share == pytest.approx(0.75)


class TestNoSelectionIsNotABadChoice:
    def test_it_is_excluded_from_the_frontier_rate(self) -> None:
        # A planner with nothing left to give did not choose badly; counting it
        # as outside the band reported the decoupled arm straying on every
        # learner, which it never does.
        made = Decision("L", "g", "a", "c", "a", "c", 0.0, 0.0, True, 1)
        none = Decision("L", "g", None, None, None, None, 0.0, 0.0, False, 0)
        report = PlanningReport([made, none])
        assert report.in_frontier_rate == 1.0
        assert report.no_selection == 1


class TestAgainstARealSession:
    """The decisions come through the loop, not a re-implementation of it."""

    def _report(self, arm: str) -> PlanningReport:
        config = config_for(arm)
        domain = registry.load_domain(config.domain)
        return evaluate(["L0000", "L0001", "L0002"], domain, config, build_session)

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_decisions_are_recorded(self, arm: str) -> None:
        assert self._report(arm).total > 0

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_every_selection_lies_in_the_band(self, arm: str) -> None:
        assert self._report(arm).in_frontier_rate == 1.0

    def test_shadowing_does_not_change_the_session(self) -> None:
        # The reference watches; the planner under test still drives. If this
        # fails, every outcome measured through the harness is of a different
        # session than the one being reported on.
        config = config_for("coupled")
        domain = registry.load_domain(config.domain)
        plain = build_session("L0000", config.seed, domain, config).run()

        from agent_newton.core.evaluation.planning import ShadowedPlanner

        session = build_session("L0000", config.seed, domain, config)
        session.planner = ShadowedPlanner(
            session.planner,
            OraclePlanner(session.learner.profile.firing, config.zpd, PRIOR),
            [],
            learner_id="L0000",
            frontier=lambda: session.board.frontier,
            reference_view=lambda: session.board.view(arm="coupled"),
        )
        shadowed = session.run()

        assert shadowed.items_attempted == plain.items_attempted
        assert shadowed.remediation_ratio == pytest.approx(plain.remediation_ratio)
        assert shadowed.posttest.score == pytest.approx(plain.posttest.score)

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_the_reference_is_never_beaten(self, arm: str) -> None:
        # The property regret rests on, and the one that would break silently.
        # If a planner could choose something the reference did not consider,
        # the difference could go negative and regret would be measuring two
        # different option sets rather than one decision.
        for decision in self._report(arm).decisions:
            assert decision.reference_value >= decision.chosen_value - 1e-9, (
                f"{decision.chosen_concept} scored above the reference's "
                f"{decision.reference_concept}"
            )

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    def test_a_selection_always_came_from_the_offered_options(self, arm: str) -> None:
        for decision in self._report(arm).decisions:
            if decision.chosen_item is not None:
                assert decision.options >= 1

    # Which arm tracks the reference more closely is a *result*, not an
    # invariant, so it is not asserted here: on toy_algebra the reference often
    # has a single option and agreement is high for both. Measured on calculus,
    # where the graph is large enough for the choice to matter, it is 74.0%
    # against 23.9% — recorded with the run artifacts rather than as a test that
    # would fail for a legitimate change in behaviour.


class TestTheOrderingProbes:
    """Planners built to violate prerequisite order, for a validity check.

    They exist to answer whether the graph does anything beyond constraining
    selection. That only works if they cover the same material as the planner
    they are compared against — a probe that wandered off and never met the
    learner's misconceptions would lose for reasons unrelated to ordering.
    """

    def _view(self, toy, mastery: dict[str, float]):
        from agent_newton.core.state.views import FullStateView
        from agent_newton.core.state.zpd import compute

        return FullStateView(
            mastery=mastery,
            error_trace=(),
            frontier=compute(mastery, toy.concepts, BAND, PRIOR),
            outcomes=(),
            version=1,
            plan=Plan(goal="solve_linear"),
        )

    def _probes(self):
        from agent_newton.core.agents.planner import ReverseOrderPlanner, ShuffledPlanner

        return ReverseOrderPlanner(BAND, PRIOR), ShuffledPlanner(BAND, PRIOR, 7)

    def test_reverse_really_violates_the_order(self, toy) -> None:
        # The point of the probe. On a fresh learner the goal-directed planner
        # takes a root; reverse takes the goal itself, whose prerequisites are
        # untouched.
        view = self._view(toy, {})
        reverse, _ = self._probes()
        directed = GoalDirectedPlanner(BAND, PRIOR, Emphasis.CONSOLIDATE)

        taken = reverse.select(view, toy, {})
        respected = directed.select(view, toy, {})
        assert taken is not None and respected is not None
        assert taken.concept_id != respected.concept_id

        unmet = [
            p
            for p in toy.concepts.all_prerequisites(taken.concept_id)
            if view.mastery.get(p, PRIOR) < BAND.theta_upper
        ]
        assert unmet, "reverse selected a concept with nothing outstanding behind it"

    def test_the_goal_directed_planner_never_does(self, toy) -> None:
        # The contrast that makes the probe a probe.
        view = self._view(toy, {})
        directed = GoalDirectedPlanner(BAND, PRIOR, Emphasis.CONSOLIDATE)
        chosen = directed.select(view, toy, {})
        assert chosen is not None
        assert chosen.concept_id in view.frontier

    def test_both_probes_stay_on_the_way_to_the_goal(self, toy) -> None:
        # Coverage has to be comparable by construction, not by luck.
        from agent_newton.core.state import route

        view = self._view(toy, {})
        relevant = route.relevant("solve_linear", toy.concepts)
        for probe in self._probes():
            chosen = probe.select(view, toy, {})
            assert chosen is not None and chosen.concept_id in relevant

    def test_the_shuffled_probe_is_reproducible(self, toy) -> None:
        from agent_newton.core.agents.planner import ShuffledPlanner

        view = self._view(toy, {})
        first = ShuffledPlanner(BAND, PRIOR, 7).select(view, toy, {})
        second = ShuffledPlanner(BAND, PRIOR, 7).select(view, toy, {})
        assert first is not None and second is not None
        assert first.id == second.id

    def test_a_different_seed_can_choose_differently(self, toy) -> None:
        # Otherwise the seed is decorative and the probe is just another
        # fixed-order planner.
        from agent_newton.core.agents.planner import ShuffledPlanner

        view = self._view(toy, {})
        chosen = {
            ShuffledPlanner(BAND, PRIOR, seed).select(view, toy, {}).concept_id  # type: ignore[union-attr]
            for seed in range(40)
        }
        assert len(chosen) > 1

    def test_they_stop_when_the_goal_is_done(self, toy) -> None:
        mastered = {c: BAND.theta_upper + 0.05 for c in toy.concepts.ids()}
        view = self._view(toy, mastered)
        for probe in self._probes():
            assert probe.select(view, toy, {}) is None
