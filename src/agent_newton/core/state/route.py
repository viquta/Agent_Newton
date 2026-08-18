"""How to get to the goal, derived from the blackboard.

The goal is durable and lives in :class:`~agent_newton.core.state.schema.Plan`.
The route is not: it is recomputed from the current posteriors, the error trace
and the graph every time it is asked for, the same way the frontier is. Nothing
is committed to, so evidence arriving mid-session changes the next step rather
than having to invalidate a stored sequence.

Three things are derived here:

``relevant``
    The goal's prerequisite closure, plus the goal. Everything outside it is out
    of scope until the goal is reached — this is what makes planning *directed*
    rather than a walk over whatever happens to be reachable.

``candidates``
    ``frontier ∩ relevant``. Reachable *and* on the way to the goal.

``next_step``
    One candidate, chosen by the learner's :class:`Emphasis`.

Computing any of them needs per-concept posteriors and the graph, so a view
carrying only item correctness cannot produce them. That is the same structural
limit the frontier has, and it extends to the learner's stated intent: acting on
``CONSOLIDATE`` needs the error trace and acting on ``ADVANCE`` needs the
posteriors, so a planner without them cannot honour either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from agent_newton.config import ZPDConfig
from agent_newton.core.state.schema import Emphasis, ErrorEvent
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains.base import ConceptGraph


@dataclass(frozen=True, slots=True)
class Step:
    """The concept to work next, and why it was chosen.

    ``reason`` goes to the audit log. A planning decision that cannot be
    explained after the run is not auditable, and the audit trail is the point.
    """

    concept_id: str
    reason: str = ""
    #: True when the ranking found nothing and a recovery choice was made.
    fallback: bool = False


def relevant(goal: str, graph: ConceptGraph) -> frozenset[str]:
    """Concepts that matter for reaching ``goal``: its closure, plus itself."""
    return graph.all_prerequisites(goal) | {goal}


def remaining(
    goal: str,
    mastery: Mapping[str, float],
    graph: ConceptGraph,
    band: ZPDConfig,
    prior: float,
) -> tuple[str, ...]:
    """Relevant concepts not yet mastered, in topological order.

    Reported as the distance still to travel, and shown in the demo. Not a
    commitment: the planner does not walk this list, it recomputes the next step
    each time.
    """
    needed = relevant(goal, graph)
    return tuple(
        c
        for c in graph.topological_order()
        if c in needed and mastery.get(c, prior) < band.theta_upper
    )


def reached(
    goal: str, mastery: Mapping[str, float], band: ZPDConfig, prior: float
) -> bool:
    return mastery.get(goal, prior) >= band.theta_upper


def next_goal(
    goals: Sequence[str],
    mastery: Mapping[str, float],
    band: ZPDConfig,
    prior: float,
    requested: frozenset[str] = frozenset(),
    graph: ConceptGraph | None = None,
    reviewing: frozenset[str] = frozenset(),
) -> str | None:
    """The first declared goal not yet reached, or None when all are.

    Order is the domain's curriculum decision, so this takes the goals as given
    rather than choosing among them.

    ``requested`` is what the learner said they wanted to work on, and it moves
    which goal comes next rather than which concept does. That is the honest
    reading of the request: asking to work on something is asking to be routed
    toward it, and the route is what a goal decides. Re-ranking within a
    frontier could not honour it at all — a concept off the way to the current
    goal is not a candidate in the first place, so the request would look
    accepted and change nothing.

    The order still decides between goals that would serve the request, so a
    learner cannot skip a prerequisite by naming something further on: the goal
    moves, the route to it does not, and everything on the way is still worked
    first. Empty for every cohort, and inert without a graph to resolve it.

    ⚠️ **Requests the learner cannot yet do come first, and that ordering is
    load-bearing.** A person asked for two concepts; the pre-test then put one
    of them at 0.98, and the goal moved to serve that one — the thing he had
    just demonstrated — while the other sat at 0.32 in the frontier and was
    never reached, because it was not on the way to the goal his own request had
    chosen. Serving the mastered half of a request first is serving it in the
    only way that cannot help.

    ``reviewing`` is the *fallback* to that, never a competitor: when nothing
    requested is still outstanding, aim at a goal whose route passes through
    what was asked for anyway. Without this the frontier reopening the concept
    achieves nothing — a reopened concept that is not on the way to the current
    goal is not a candidate, and ``implicit_differentiation`` is a sibling of
    ``integration_by_substitution`` rather than a prerequisite, so the goal that
    follows it does not pass through it.
    """
    outstanding = [goal for goal in goals if not reached(goal, mastery, band, prior)]
    if requested and graph is not None:
        # A reviewed goal is servable although it is reached — that is what
        # being under review means, and `outstanding` excludes it by definition.
        # Without this a request for a *goal* could not be honoured at all:
        # `relevant` includes the goal itself, so nothing else names it either.
        servable = [
            goal
            for goal in goals
            if goal in reviewing or not reached(goal, mastery, band, prior)
        ]
        live = {c for c in requested if not reached(c, mastery, band, prior)}
        for goal in servable:
            if relevant(goal, graph) & live:
                return goal
        for goal in servable:
            if relevant(goal, graph) & reviewing:
                return goal
    return outstanding[0] if outstanding else None


def candidates(goal: str, frontier: Frontier, graph: ConceptGraph) -> tuple[str, ...]:
    """Reachable concepts that are on the way to the goal, in topological order."""
    needed = relevant(goal, graph)
    return tuple(c for c in graph.topological_order() if c in needed and c in frontier)


def _difficulty(concept_id: str, error_trace: Sequence[ErrorEvent]) -> int:
    """Errors recorded against this concept in the rolling window.

    The trace is already bounded to a recent window by the arbitration config,
    so a plain count is a recency-weighted measure without needing to weight
    anything.
    """
    return sum(1 for event in error_trace if event.concept_id == concept_id)


def rank(
    available: Sequence[str],
    emphasis: Emphasis,
    mastery: Mapping[str, float],
    error_trace: Sequence[ErrorEvent],
    graph: ConceptGraph,
    prior: float,
    deprioritised: frozenset[str] = frozenset(),
    preferred: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Order candidates by what the learner is here for.

    Both orders break ties on depth then id, so neither depends on the order
    concepts happen to be declared in the domain's YAML.

    ``preferred`` is what the learner asked to work on, and it sorts first among
    the candidates. Choosing the goal is not enough on its own: a request can be
    reachable *and* on the route and still never come up, because the emphasis
    ranks by difficulty or depth and something else wins every time. A person
    watched a concept he had asked for sit in the frontier for a whole sitting.
    Empty for every cohort.

    ``deprioritised`` concepts sort last under either emphasis. They are not
    removed: a learner who has been stuck on something should be moved along,
    not have the material withdrawn, and if nothing else on the way to the goal
    is available then the alternative to offering it again is offering nothing.
    Empty by default, which is every run that does not set
    ``cohort.max_visits_per_concept``.

    Once *everything* available has been set aside the ordering within that
    group decides again, so a learner who is stuck on all of it settles on
    whichever concept the emphasis ranks first. That is the intended end state
    rather than a gap: every concept is on the record as outstanding, and the
    remaining budget goes to the one the evidence says is hardest.
    """
    if preferred:
        # Applied outside the dwelling split, so a request outranks having been
        # set aside: the learner asking for something is a better reason to
        # offer it than the session's own bookkeeping is to hold it back.
        asked = tuple(c for c in available if c in preferred)
        others = tuple(c for c in available if c not in preferred)
        if asked and others:
            return rank(
                asked, emphasis, mastery, error_trace, graph, prior, deprioritised
            ) + rank(
                others, emphasis, mastery, error_trace, graph, prior, deprioritised
            )

    if deprioritised:
        held = tuple(c for c in available if c in deprioritised)
        rest = tuple(c for c in available if c not in deprioritised)
        if rest:
            return rank(rest, emphasis, mastery, error_trace, graph, prior) + rank(
                held, emphasis, mastery, error_trace, graph, prior
            )
        available = held

    if emphasis is Emphasis.ADVANCE:
        # Deepest first: as soon as a further concept becomes reachable, work
        # it, leaving shallower ones at whatever they have reached. The band's
        # width is what makes this possible — a prerequisite only has to clear
        # theta_lower to open the next concept, while staying selectable itself
        # until theta_upper.
        return tuple(sorted(available, key=lambda c: (-graph.depth(c), c)))

    # CONSOLIDATE: where the evidence says the learner is struggling, then the
    # earliest unfinished thing, then what they have started but not finished,
    # then what they are furthest from mastering.
    #
    # Two keys guard the same mistake, which is worth stating because it is easy
    # to make twice: **a concept with no observations is not evidence of
    # anything.** It sits at the prior, below every posterior the learner has
    # actually moved, so ranking by posterior alone sends a consolidating
    # learner *forward* into untouched material — the opposite of consolidating.
    #
    # Depth handles it across levels. `c not in mastery` handles it within a
    # level: False sorts before True, so a concept worked to 0.5 is preferred
    # over one never attempted, and the posterior only decides between concepts
    # that have both been measured.
    #
    # The same distinction the verifier draws with UNPARSEABLE: failing to
    # measure is not a finding about the learner.
    #
    # Measured on calculus: the touched-first key changes the winner at 20 of
    # 587 ranking calls across a 20-learner cohort, and the cohort outcomes are
    # identical with and without it. It is here because it is what
    # ``consolidate`` means, not because it moves a number — depth ordering is
    # what drives the difference between the two emphases.
    return tuple(
        sorted(
            available,
            key=lambda c: (
                -_difficulty(c, error_trace),
                graph.depth(c),
                c not in mastery,
                mastery.get(c, prior),
                c,
            ),
        )
    )


def next_step(
    goal: str,
    emphasis: Emphasis,
    mastery: Mapping[str, float],
    error_trace: Sequence[ErrorEvent],
    frontier: Frontier,
    graph: ConceptGraph,
    band: ZPDConfig,
    prior: float,
    deprioritised: frozenset[str] = frozenset(),
    preferred: frozenset[str] = frozenset(),
) -> Step | None:
    """The concept to work next on the way to ``goal``.

    None means every concept on the way to the goal has been mastered, so the
    caller should move to the next goal.
    """
    available = candidates(goal, frontier, graph)

    if available:
        chosen = rank(
            available,
            emphasis,
            mastery,
            error_trace,
            graph,
            prior,
            deprioritised,
            preferred,
        )[0]
        held = " (worked enough for now)" if chosen in deprioritised else ""
        asked = " (asked for)" if chosen in preferred else ""
        return Step(
            concept_id=chosen,
            reason=(
                f"{emphasis.value} toward {goal}: chose {chosen} from "
                f"{len(available)} candidate(s){held}{asked}"
            ),
        )

    outstanding = remaining(goal, mastery, graph, band, prior)
    if not outstanding:
        return None

    # Should be unreachable. The earliest unmastered relevant concept has only
    # relevant concepts as prerequisites, and all of those are earlier and
    # therefore mastered — so it clears theta_lower and is in the frontier.
    # Reaching this means the frontier and the graph disagree; recover so a
    # cohort does not stall, but make it visible.
    return Step(
        concept_id=outstanding[0],
        fallback=True,
        reason=(
            f"no concept on the way to {goal} was reachable, which should be "
            f"impossible for a consistent graph; fell back to {outstanding[0]!r} "
            f"with {len(outstanding)} concept(s) outstanding"
        ),
    )
