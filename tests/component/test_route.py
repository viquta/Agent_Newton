"""Routing toward a goal, derived from the blackboard.

The goal is durable state; the route is not. Everything here is a pure function
of the posteriors, the error trace and the graph, so it is tested directly
rather than through a session.

The property that matters most is the last one: acting on the learner's stated
intent requires the learner model, so a planner without one behaves the same
whichever intent was asked for.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_newton.config import BKTConfig, ZPDConfig
from agent_newton.core.state import bkt, route
from agent_newton.core.state.schema import Emphasis, ErrorEvent
from agent_newton.core.state.zpd import Frontier, compute
from agent_newton.domains import registry
from agent_newton.domains.base import Concept
from agent_newton.domains.content import YamlConceptGraph

BAND = ZPDConfig()
PRIOR = bkt.initial(BKTConfig())
MASTERED = BAND.theta_upper + 0.05

#   a -> b -> d
#   a -> c -> d       e is off the path to d entirely
CHAIN = YamlConceptGraph(
    [
        Concept("a", "a"),
        Concept("b", "b", ("a",)),
        Concept("c", "c", ("a",)),
        Concept("d", "d", ("b", "c")),
        Concept("e", "e", ("a",)),
    ],
    goals=("d",),
)


def frontier_for(mastery: dict[str, float], graph=CHAIN) -> Frontier:
    return compute(mastery, graph, BAND, PRIOR)


class TestRelevance:
    def test_is_the_closure_plus_the_goal(self) -> None:
        assert route.relevant("d", CHAIN) == {"a", "b", "c", "d"}

    def test_excludes_what_the_goal_does_not_need(self) -> None:
        # The whole point of a goal: it takes concepts out of scope.
        assert "e" not in route.relevant("d", CHAIN)

    def test_a_root_needs_only_itself(self) -> None:
        assert route.relevant("a", CHAIN) == {"a"}


class TestCandidates:
    def test_are_reachable_and_on_the_way(self) -> None:
        mastery = {"a": MASTERED}
        available = route.candidates("d", frontier_for(mastery), CHAIN)
        # b and c are now open; e is open too but is not on the way to d.
        assert set(available) == {"b", "c"}

    def test_exclude_what_is_already_mastered(self) -> None:
        mastery = {"a": MASTERED, "b": MASTERED}
        available = route.candidates("d", frontier_for(mastery), CHAIN)
        assert "a" not in available and "b" not in available

    def test_are_empty_once_everything_relevant_is_mastered(self) -> None:
        mastery = {c: MASTERED for c in ("a", "b", "c", "d")}
        assert route.candidates("d", frontier_for(mastery), CHAIN) == ()

    @settings(max_examples=50, deadline=None)
    @given(
        mastery=st.dictionaries(
            st.sampled_from(["a", "b", "c", "d", "e"]),
            st.floats(min_value=0.0, max_value=1.0),
            max_size=5,
        )
    )
    def test_never_offers_a_concept_the_goal_does_not_need(
        self, mastery: dict[str, float]
    ) -> None:
        needed = route.relevant("d", CHAIN)
        for concept in route.candidates("d", frontier_for(mastery), CHAIN):
            assert concept in needed

    @settings(max_examples=50, deadline=None)
    @given(
        mastery=st.dictionaries(
            st.sampled_from(["a", "b", "c", "d", "e"]),
            st.floats(min_value=0.0, max_value=1.0),
            max_size=5,
        )
    )
    def test_something_is_always_available_while_work_remains(
        self, mastery: dict[str, float]
    ) -> None:
        # Topological order puts every prerequisite first, so the earliest
        # unmastered relevant concept always has its prerequisites met. If this
        # ever fails, the frontier and the graph disagree.
        outstanding = route.remaining("d", mastery, CHAIN, BAND, PRIOR)
        available = route.candidates("d", frontier_for(mastery), CHAIN)
        assert bool(outstanding) == bool(available)


class TestGoalSequence:
    def test_takes_the_first_goal_not_yet_reached(self) -> None:
        graph = YamlConceptGraph(list(CHAIN.concepts()), goals=("b", "d"))
        assert route.next_goal(graph.goals(), {}, BAND, PRIOR) == "b"

    def test_moves_on_once_a_goal_is_reached(self) -> None:
        graph = YamlConceptGraph(list(CHAIN.concepts()), goals=("b", "d"))
        assert route.next_goal(graph.goals(), {"b": MASTERED}, BAND, PRIOR) == "d"

    def test_none_when_every_goal_is_reached(self) -> None:
        graph = YamlConceptGraph(list(CHAIN.concepts()), goals=("b", "d"))
        reached = {"b": MASTERED, "d": MASTERED}
        assert route.next_goal(graph.goals(), reached, BAND, PRIOR) is None


class TestRemaining:
    def test_counts_only_what_is_unmastered_and_relevant(self) -> None:
        mastery = {"a": MASTERED, "e": 0.0}
        assert route.remaining("d", mastery, CHAIN, BAND, PRIOR) == ("b", "c", "d")

    def test_is_empty_at_the_goal(self) -> None:
        mastery = {c: MASTERED for c in ("a", "b", "c", "d")}
        assert route.remaining("d", mastery, CHAIN, BAND, PRIOR) == ()


class TestEmphasis:
    """The learner's intent, and the evidence each reading needs."""

    def _step(self, emphasis: Emphasis, mastery, trace=()):
        return route.next_step(
            goal="d",
            emphasis=emphasis,
            mastery=mastery,
            error_trace=trace,
            frontier=frontier_for(mastery),
            graph=CHAIN,
            band=BAND,
            prior=PRIOR,
        )

    def test_consolidate_goes_where_the_errors_are(self) -> None:
        # b and c are both open and equally unmastered. The error trace is the
        # only thing separating them, and it is the thing the decoupled arm
        # does not have.
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.5}
        trace = (ErrorEvent(t=1, item_id="i", concept_id="c"),)
        step = self._step(Emphasis.CONSOLIDATE, mastery, trace)
        assert step is not None and step.concept_id == "c"

    def test_consolidate_prefers_the_furthest_from_mastery_among_equals(self) -> None:
        # b and c sit at the same depth, so the posterior decides.
        mastery = {"a": MASTERED, "b": 0.8, "c": 0.2}
        step = self._step(Emphasis.CONSOLIDATE, mastery)
        assert step is not None and step.concept_id == "c"

    def test_consolidate_finishes_what_was_started_before_opening_a_sibling(self) -> None:
        # b and c sit at the same depth with no errors against either. b has
        # been worked to 0.5; c has never been attempted, so it sits at the
        # prior — *below* b. Ranking on the posterior alone would send a
        # consolidating learner to the untouched one, which is the same
        # absence-is-not-evidence error the depth key guards across levels.
        mastery = {"a": MASTERED, "b": 0.5}
        assert PRIOR < 0.5, "the premise of this test"
        step = self._step(Emphasis.CONSOLIDATE, mastery)
        assert step is not None and step.concept_id == "b"

    def test_the_posterior_still_decides_between_measured_concepts(self) -> None:
        # The tiebreak is not disabled, only made to come after measurement.
        mastery = {"a": MASTERED, "b": 0.8, "c": 0.2}
        step = self._step(Emphasis.CONSOLIDATE, mastery)
        assert step is not None and step.concept_id == "c"

    def test_advance_is_unaffected_by_what_has_been_measured(self) -> None:
        # It ranks on depth alone, so it has no posterior tiebreak to get wrong.
        mastery = {"a": MASTERED, "b": 0.5}
        step = self._step(Emphasis.ADVANCE, mastery)
        assert step is not None and step.concept_id in {"b", "c"}

    def test_consolidate_does_not_open_new_material_first(self) -> None:
        # An unstarted concept sits at the prior, below anything the learner has
        # worked on. Ranking by posterior alone would send a consolidating
        # learner forward into it, which is the opposite of consolidating.
        mastery = {"a": MASTERED, "b": 0.75, "c": 0.75}
        step = self._step(Emphasis.CONSOLIDATE, mastery)
        assert step is not None
        assert step.concept_id in {"b", "c"}, "should finish b/c before opening d"

    def test_advance_takes_the_deepest_reachable(self) -> None:
        # With b mastered but c not, d is still shut. Once both are past
        # theta_lower, advance prefers d over finishing either.
        mastery = {"a": MASTERED, "b": BAND.theta_lower + 0.05, "c": BAND.theta_lower + 0.05}
        step = self._step(Emphasis.ADVANCE, mastery)
        assert step is not None and step.concept_id == "d"

    def test_consolidate_stays_behind_where_advance_moves_on(self) -> None:
        # The same state, read two ways. This is the difference the band's width
        # makes: a concept only has to clear theta_lower to open the next one,
        # while staying selectable itself until theta_upper.
        mastery = {"a": MASTERED, "b": BAND.theta_lower + 0.05, "c": BAND.theta_lower + 0.05}
        advancing = self._step(Emphasis.ADVANCE, mastery)
        consolidating = self._step(Emphasis.CONSOLIDATE, mastery)
        assert advancing is not None and consolidating is not None
        assert advancing.concept_id != consolidating.concept_id

    def test_neither_leaves_the_route(self) -> None:
        mastery = {"a": MASTERED}
        for emphasis in Emphasis:
            step = self._step(emphasis, mastery)
            assert step is not None and step.concept_id != "e"

    def test_ranking_does_not_depend_on_declaration_order(self) -> None:
        # Sorting by position in the topological order would let the order
        # concepts happen to be written in the YAML decide what a learner works.
        forward = YamlConceptGraph(list(CHAIN.concepts()), goals=("d",))
        reverse = YamlConceptGraph(list(reversed(list(CHAIN.concepts()))), goals=("d",))
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.5}
        both = {
            graph: route.next_step(
                goal="d",
                emphasis=Emphasis.ADVANCE,
                mastery=mastery,
                error_trace=(),
                frontier=compute(mastery, graph, BAND, PRIOR),
                graph=graph,
                band=BAND,
                prior=PRIOR,
            )
            for graph in (forward, reverse)
        }
        chosen = {step.concept_id for step in both.values() if step is not None}
        assert len(chosen) == 1


class TestNothingLeft:
    def test_returns_none_when_the_goal_is_done(self) -> None:
        mastery = {c: MASTERED for c in ("a", "b", "c", "d")}
        step = route.next_step(
            goal="d",
            emphasis=Emphasis.CONSOLIDATE,
            mastery=mastery,
            error_trace=(),
            frontier=frontier_for(mastery),
            graph=CHAIN,
            band=BAND,
            prior=PRIOR,
        )
        assert step is None

    def test_recovers_loudly_if_the_frontier_and_graph_disagree(self) -> None:
        # Should be unreachable for a consistent graph, so it is forced here.
        # Stalling a cohort would be the worse failure, but a silent recovery
        # would hide a real inconsistency.
        mastery = {"a": 0.0}
        step = route.next_step(
            goal="d",
            emphasis=Emphasis.CONSOLIDATE,
            mastery=mastery,
            error_trace=(),
            frontier=Frontier(frozenset()),  # empty, while work plainly remains
            graph=CHAIN,
            band=BAND,
            prior=PRIOR,
        )
        assert step is not None
        assert step.fallback
        assert "should be impossible" in step.reason


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


class TestAgainstTheRealDomain:
    def test_the_first_goal_narrows_the_graph(self, calculus) -> None:
        goal = calculus.concepts.goals()[0]
        relevant = route.relevant(goal, calculus.concepts)
        assert len(relevant) < len(calculus.concepts.ids())

    def test_a_fresh_learner_has_somewhere_to_start(self, calculus) -> None:
        goal = calculus.concepts.goals()[0]
        frontier = compute({}, calculus.concepts, BAND, PRIOR)
        assert route.candidates(goal, frontier, calculus.concepts)
