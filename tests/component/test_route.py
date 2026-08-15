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


class TestSettingAConceptAside:
    """The dwelling cap, which is off unless a config asks for it.

    ``consolidate`` ranks by recent errors, so a concept a learner keeps failing
    attracts further selection. That is what consolidation means and it has no
    floor: a learner who stays stuck can spend a whole budget in one place.
    """

    def _step(self, mastery, trace=(), deprioritised=frozenset()):
        return route.next_step(
            goal="d",
            emphasis=Emphasis.CONSOLIDATE,
            mastery=mastery,
            error_trace=trace,
            frontier=frontier_for(mastery),
            graph=CHAIN,
            band=BAND,
            prior=PRIOR,
            deprioritised=deprioritised,
        )

    def test_without_a_cap_failure_keeps_attracting_selection(self) -> None:
        # Today's behaviour, and what every measured result was produced under.
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.5}
        trace = tuple(ErrorEvent(t=n, item_id="i", concept_id="c") for n in range(5))
        step = self._step(mastery, trace)
        assert step is not None and step.concept_id == "c"

    def test_a_concept_set_aside_yields_to_anything_else(self) -> None:
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.5}
        trace = tuple(ErrorEvent(t=n, item_id="i", concept_id="c") for n in range(5))
        step = self._step(mastery, trace, deprioritised=frozenset({"c"}))
        assert step is not None and step.concept_id == "b"

    def test_it_is_set_aside_rather_than_withdrawn(self) -> None:
        # A learner whose only remaining work is the thing they are stuck on
        # must still be given work. Withdrawing it would end the session.
        mastery = {"a": MASTERED, "b": MASTERED, "c": 0.5}
        step = self._step(mastery, deprioritised=frozenset({"c"}))
        assert step is not None and step.concept_id == "c"

    def test_the_reason_says_it_was_set_aside(self) -> None:
        mastery = {"a": MASTERED, "b": MASTERED, "c": 0.5}
        step = self._step(mastery, deprioritised=frozenset({"c"}))
        assert step is not None and "worked enough for now" in step.reason

    def test_an_empty_set_changes_no_ordering(self) -> None:
        # The default. It must be exactly the old code path, or every measured
        # result would have moved.
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.2}
        trace = (ErrorEvent(t=1, item_id="i", concept_id="b"),)
        assert route.rank(
            ("b", "c"), Emphasis.CONSOLIDATE, mastery, trace, CHAIN, PRIOR
        ) == route.rank(
            ("b", "c"), Emphasis.CONSOLIDATE, mastery, trace, CHAIN, PRIOR, frozenset()
        )

    def test_the_ordering_within_each_group_is_unchanged(self) -> None:
        # Deprioritising moves a concept to the back; it does not reshuffle what
        # is in front of it, or what is behind.
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.2, "e": 0.4}
        plain = route.rank(
            ("b", "c", "e"), Emphasis.CONSOLIDATE, mastery, (), CHAIN, PRIOR
        )
        held = route.rank(
            ("b", "c", "e"), Emphasis.CONSOLIDATE, mastery, (), CHAIN, PRIOR,
            frozenset({plain[0]}),
        )
        assert held == plain[1:] + (plain[0],)

    def test_advance_honours_it_too(self) -> None:
        # Both emphases route toward a goal, and a learner set aside from a
        # concept should not be handed it back by asking to move on.
        mastery = {"a": MASTERED, "b": 0.5, "c": 0.5}
        step = route.next_step(
            goal="d", emphasis=Emphasis.ADVANCE, mastery=mastery, error_trace=(),
            frontier=frontier_for(mastery), graph=CHAIN, band=BAND, prior=PRIOR,
            deprioritised=frozenset({"b"}),
        )
        assert step is not None and step.concept_id == "c"


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


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


class TestAgainstTheRealDomain:
    def test_the_first_goal_narrows_the_graph(self, calculus) -> None:
        goal = calculus.concepts.goals()[0]
        relevant = route.relevant(goal, calculus.concepts)
        assert len(relevant) < len(calculus.concepts.ids())

    def test_a_fresh_learner_has_somewhere_to_start(self, calculus) -> None:
        goal = calculus.concepts.goals()[0]
        frontier = compute({}, calculus.concepts, BAND, PRIOR)
        assert route.candidates(goal, frontier, calculus.concepts)


class TestNoisedOracleRateIsRealised:
    """The noised condition is defined by its rate, so the rate has to hold.

    Corruption is deterministic in the situation, which is what keeps both arms
    misdiagnosed identically. Keyed on the situation *alone* it would also make a
    repeated situation reproduce its own past — and a session revisits a handful
    of situations many times, so a single unlucky draw would decide most of the
    run rather than the configured rate.
    """

    def _agent(self, rate: float):
        from agent_newton.core.agents.diagnostic import NoisedOracleDiagnostic

        return NoisedOracleDiagnostic(rate, seed=20260807)

    def test_a_repeated_situation_draws_again(self, calculus) -> None:
        # The defect this guards: at rate 0.5, twenty encounters with one
        # situation must not all come back the same way.
        agent = self._agent(0.5)
        item = calculus.items.get("ca_chain_p1")
        label = "chain_rule_omits_inner"
        agent.observe_ground_truth(label)
        verdicts = {
            agent.diagnose(item, "cos(x**2)", calculus).misconception_id
            for _ in range(20)
        }
        assert len(verdicts) > 1, "a repeated situation replayed its own past"

    @pytest.mark.parametrize("rate", [0.1, 0.25, 0.5])
    def test_the_rate_holds_over_repeats_of_one_situation(self, calculus, rate) -> None:
        agent = self._agent(rate)
        item = calculus.items.get("ca_chain_p1")
        label = "chain_rule_omits_inner"
        agent.observe_ground_truth(label)
        n = 400
        wrong = sum(
            agent.diagnose(item, "cos(x**2)", calculus).misconception_id != label
            for _ in range(n)
        )
        # Generous band: this is one hash stream, not a statistical claim.
        assert abs(wrong / n - rate) < 0.08, f"realised {wrong / n:.3f} for nominal {rate}"

    def test_both_arms_meeting_the_same_situation_agree(self, calculus) -> None:
        # The property the determinism exists for: two independently constructed
        # agents on the same seed misdiagnose the nth encounter identically.
        item = calculus.items.get("ca_chain_p1")
        left, right = self._agent(0.5), self._agent(0.5)
        for agent in (left, right):
            agent.observe_ground_truth("chain_rule_omits_inner")
        assert [left.diagnose(item, "cos(x**2)", calculus).misconception_id for _ in range(10)] == [
            right.diagnose(item, "cos(x**2)", calculus).misconception_id for _ in range(10)
        ]


class TestALearnerCanAskForSomething:
    """Learner input, not learner model — and only one arm can act on it.

    Asking to work on something is asking to be *routed* toward it, so the
    request moves which goal comes next rather than which concept does.
    Re-ranking inside the frontier could not honour it at all: a concept off the
    way to the current goal is not a candidate in the first place, so the
    request would look accepted and change nothing.
    """

    def test_no_request_takes_the_declared_order(self, toy) -> None:
        # Every cohort. The default must be byte-identical to what it was.
        assert route.next_goal(
            toy.concepts.goals(), {}, BAND, PRIOR
        ) == route.next_goal(
            toy.concepts.goals(), {}, BAND, PRIOR, frozenset(), toy.concepts
        )

    def test_a_request_moves_the_goal_to_one_that_reaches_it(self, calculus) -> None:
        goals = calculus.concepts.goals()
        plain = route.next_goal(goals, {}, BAND, PRIOR)
        asked = route.next_goal(
            goals, {}, BAND, PRIOR, frozenset({"chain_rule"}), calculus.concepts
        )
        assert plain != asked
        assert asked is not None
        assert "chain_rule" in route.relevant(asked, calculus.concepts)

    def test_the_prerequisites_still_come_first(self, calculus) -> None:
        # The guard against reading this as a skip: the goal moves, the route to
        # it does not, so everything on the way is still outstanding.
        goal = route.next_goal(
            calculus.concepts.goals(), {}, BAND, PRIOR,
            frozenset({"chain_rule"}), calculus.concepts,
        )
        assert goal is not None
        outstanding = route.remaining(goal, {}, calculus.concepts, BAND, PRIOR)
        assert outstanding[0] != "chain_rule"
        assert set(calculus.concepts.all_prerequisites("chain_rule")) <= set(outstanding)

    def test_a_request_for_something_already_reached_is_ignored(self, calculus) -> None:
        # A goal that is done is done. Honouring the request there would mean a
        # request can waive the band, which would make mastery mean one thing
        # for the model and another for the person who asked.
        mastery = {c: 0.99 for c in calculus.concepts.ids()}
        assert (
            route.next_goal(
                calculus.concepts.goals(), mastery, BAND, PRIOR,
                frozenset({"chain_rule"}), calculus.concepts,
            )
            is None
        )

    def test_a_request_nothing_serves_falls_back_to_the_order(self, toy) -> None:
        assert route.next_goal(
            toy.concepts.goals(), {}, BAND, PRIOR,
            frozenset({"not_a_concept"}), toy.concepts,
        ) == route.next_goal(toy.concepts.goals(), {}, BAND, PRIOR)

    def test_without_a_graph_it_cannot_be_resolved(self, toy) -> None:
        # Inert rather than wrong: resolving a request needs the closure, and a
        # caller that has no graph has not asked for the behaviour.
        assert route.next_goal(
            toy.concepts.goals(), {}, BAND, PRIOR, frozenset({"distribute"})
        ) == route.next_goal(toy.concepts.goals(), {}, BAND, PRIOR)


class TestARequestIsHonouredWhereItCanHelp:
    """⚠️ A sitting honoured a request in the one way that could not help.

    The learner asked for two concepts. The pre-test then put the first at 0.98,
    the goal moved to serve *that* one — the thing he had just demonstrated —
    and the second sat at 0.32 in the frontier for the whole sitting, never
    selected, because it was not on the way to the goal his own request had
    chosen. *"Even the things I chose to work on, were not entirely there in the
    session."*

    Two fixes, and both are needed: the goal is chosen to serve a request the
    learner cannot yet do, and a requested concept among the candidates is
    worked first.
    """

    def test_a_mastered_request_does_not_move_the_goal(self, calculus) -> None:
        mastery = {"polynomial_differentiation": 0.98}
        assert route.next_goal(
            calculus.concepts.goals(), mastery, BAND, PRIOR,
            frozenset({"polynomial_differentiation"}), calculus.concepts,
        ) == route.next_goal(calculus.concepts.goals(), mastery, BAND, PRIOR)

    def test_an_unmastered_one_still_does(self, calculus) -> None:
        # The guard can fail: the same request, below the band, must move it.
        goals = calculus.concepts.goals()
        mastery = {"polynomial_differentiation": 0.30}
        moved = route.next_goal(
            goals, mastery, BAND, PRIOR,
            frozenset({"polynomial_differentiation"}), calculus.concepts,
        )
        assert moved != route.next_goal(goals, mastery, BAND, PRIOR)
        assert moved is not None
        assert "polynomial_differentiation" in route.relevant(moved, calculus.concepts)

    def test_the_live_half_of_a_mixed_request_wins(self, calculus) -> None:
        # Exactly the sitting: one demonstrated, one not. The goal must serve
        # the one still to be learned.
        mastery = {"polynomial_differentiation": 0.98, "antiderivative": 0.32}
        goal = route.next_goal(
            calculus.concepts.goals(), mastery, BAND, PRIOR,
            frozenset({"polynomial_differentiation", "antiderivative"}),
            calculus.concepts,
        )
        assert goal is not None
        assert "antiderivative" in route.relevant(goal, calculus.concepts)

    def test_a_requested_candidate_is_worked_first(self, toy) -> None:
        available = ["integer_arithmetic", "combine_like_terms"]
        plain = route.rank(available, Emphasis.CONSOLIDATE, {}, (), toy.concepts, PRIOR)
        asked = route.rank(
            available, Emphasis.CONSOLIDATE, {}, (), toy.concepts, PRIOR,
            preferred=frozenset({"combine_like_terms"}),
        )
        assert asked[0] == "combine_like_terms"
        assert set(asked) == set(plain), "a request must reorder, never remove"

    def test_an_empty_request_changes_no_order(self, toy) -> None:
        available = ["integer_arithmetic", "combine_like_terms", "distribute"]
        assert route.rank(
            available, Emphasis.CONSOLIDATE, {}, (), toy.concepts, PRIOR,
            preferred=frozenset(),
        ) == route.rank(available, Emphasis.CONSOLIDATE, {}, (), toy.concepts, PRIOR)

    def test_the_step_says_it_was_asked_for(self, toy) -> None:
        # A planning decision that cannot be explained afterwards is not
        # auditable, and "why this concept" is the question a learner asks.
        frontier = compute({}, toy.concepts, BAND, PRIOR)
        step = route.next_step(
            goal="solve_linear", emphasis=Emphasis.CONSOLIDATE, mastery={},
            error_trace=(), frontier=frontier, graph=toy.concepts, band=BAND,
            prior=PRIOR, preferred=frozenset(frontier),
        )
        assert step is not None and "asked for" in step.reason
