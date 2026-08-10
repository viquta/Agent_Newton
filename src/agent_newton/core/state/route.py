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
) -> str | None:
    """The first declared goal not yet reached, or None when all are.

    Order is the domain's curriculum decision, so this takes the goals as given
    rather than choosing among them.
    """
    for goal in goals:
        if not reached(goal, mastery, band, prior):
            return goal
    return None


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
) -> tuple[str, ...]:
    """Order candidates by what the learner is here for.

    Both orders break ties on depth then id, so neither depends on the order
    concepts happen to be declared in the domain's YAML.
    """
    if emphasis is Emphasis.ADVANCE:
        # Deepest first: as soon as a further concept becomes reachable, work
        # it, leaving shallower ones at whatever they have reached. The band's
        # width is what makes this possible — a prerequisite only has to clear
        # theta_lower to open the next concept, while staying selectable itself
        # until theta_upper.
        return tuple(sorted(available, key=lambda c: (-graph.depth(c), c)))

    # CONSOLIDATE: where the evidence says the learner is struggling, then the
    # earliest unfinished thing, then what they are furthest from mastering.
    #
    # Depth comes before the posterior deliberately. An unstarted concept sits
    # at the prior, which is lower than anything the learner has worked on, so
    # ranking by posterior alone would send a consolidating learner *forward*
    # into new material — the opposite of consolidating. Shallowest-first keeps
    # them finishing what they have already begun.
    return tuple(
        sorted(
            available,
            key=lambda c: (
                -_difficulty(c, error_trace),
                graph.depth(c),
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
) -> Step | None:
    """The concept to work next on the way to ``goal``.

    None means every concept on the way to the goal has been mastered, so the
    caller should move to the next goal.
    """
    available = candidates(goal, frontier, graph)

    if available:
        chosen = rank(available, emphasis, mastery, error_trace, graph, prior)[0]
        return Step(
            concept_id=chosen,
            reason=(
                f"{emphasis.value} toward {goal}: chose {chosen} from "
                f"{len(available)} candidate(s)"
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
