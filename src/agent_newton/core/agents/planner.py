"""Planners — where the ablation becomes behaviour.

Both planners know the **syllabus**: the item bank and the prerequisite graph
are static curriculum, available to either arm. What differs is what they know
about *this learner*. Keeping curriculum knowledge common is what makes the
comparison fair — otherwise the decoupled arm would be handicapped by ignorance
of the subject rather than of the learner.

:class:`FrontierPlanner` reads the frontier and selects from it.
:class:`FixedOrderPlanner` cannot: its view has neither the posteriors nor any
per-concept estimate, so it walks the syllabus in topological order and advances
on consecutive correct answers. That is not a worse heuristic over the same
information — it is the best available given strictly less.

Both may **repeat** an item, preferring the least-used one for a concept.
Mastery takes several correct answers to establish, and a bank holding fewer
items than that per concept would otherwise make the concept permanently
unmasterable: the planner would run out of unseen work and stop while the
learner still had everything to learn.
"""

from __future__ import annotations

from typing import Mapping

from agent_newton.core.agents.base import StateView
from agent_newton.core.pedagogy import may_select
from agent_newton.core.state.views import FullStateView
from agent_newton.domains.base import Domain, Item


def _least_used(domain: Domain, concept_id: str, given: Mapping[str, int]) -> Item | None:
    """The least-practised item for a concept, ties broken by id for stability."""
    items = domain.items.for_concept(concept_id, "practice")
    if not items:
        return None
    return min(items, key=lambda item: (given.get(item.id, 0), item.id))


class FrontierPlanner:
    """Selects from the frontier, nearest the syllabus start first.

    Ties are broken by topological position, so among reachable concepts it
    prefers the one whose prerequisites were satisfied earliest, and the walk is
    reproducible.

    Returns None only when the frontier is empty — which means every concept is
    mastered, not that the planner ran out of material.
    """

    def select(self, view: StateView, domain: Domain, given: Mapping[str, int]) -> Item | None:
        if not isinstance(view, FullStateView):
            raise TypeError(
                "FrontierPlanner requires the full state view; the decoupled arm "
                "must use FixedOrderPlanner"
            )

        order = list(domain.concepts.topological_order())
        for concept_id in sorted(view.frontier, key=order.index):
            # The band-membership rule is the planner's own guardrail, so a
            # selection violating it never reaches the learner.
            if may_select(concept_id, view.frontier) is not None:
                continue
            item = _least_used(domain, concept_id, given)
            if item is not None:
                return item

        return None


class FixedOrderPlanner:
    """Walks the syllabus in order, advancing on consecutive correct answers.

    The decoupled arm. Its view carries a right/wrong stream and nothing else,
    so it cannot ask which concepts are reachable — only whether the learner has
    been getting things right lately.
    """

    def __init__(self, advance_after: int = 2) -> None:
        self._advance_after = advance_after
        self._position = 0
        #: Length of the outcome stream when the position last moved. Without
        #: it, a streak that persists would advance the position on *every*
        #: call rather than once, racing through the syllabus an item at a time.
        self._advanced_at = 0

    def select(self, view: StateView, domain: Domain, given: Mapping[str, int]) -> Item | None:
        order = list(domain.concepts.topological_order())
        answered = len(view.outcomes)

        if (
            view.consecutive_correct() >= self._advance_after
            and answered - self._advanced_at >= self._advance_after
        ):
            self._position += 1
            self._advanced_at = answered

        while self._position < len(order):
            item = _least_used(domain, order[self._position], given)
            if item is not None:
                return item
            # A concept with no practice items at all cannot be taught; step
            # past it rather than stalling.
            self._position += 1

        return None
