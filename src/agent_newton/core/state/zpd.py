"""The mastery frontier.

A concept is in the frontier when it is not yet independently mastered but every
prerequisite is:

    frontier = { c : P(c) < theta_upper  and  for all p in prereqs(c): P(p) > theta_lower }

Beyond current capability, reachable with support. Computing it needs
per-concept posteriors **and** the prerequisite graph, which is why only
``FullStateView`` can produce one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agent_newton.config import ZPDConfig
from agent_newton.domains.base import ConceptGraph


@dataclass(frozen=True, slots=True)
class Frontier:
    """The selectable set, plus why it looks the way it does.

    ``reason`` and ``fallback`` exist so a run can report *why* the planner had
    the options it had, rather than only what it chose.
    """

    concepts: frozenset[str]
    #: True when the predicate yielded nothing and a recovery choice was made.
    fallback: bool = False
    reason: str = ""

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self.concepts

    def __bool__(self) -> bool:
        return bool(self.concepts)

    def __iter__(self):
        return iter(sorted(self.concepts))

    def __len__(self) -> int:
        return len(self.concepts)


def compute(
    mastery: Mapping[str, float],
    graph: ConceptGraph,
    band: ZPDConfig,
    prior: float,
) -> Frontier:
    """Derive the frontier from mastery estimates and the prerequisite graph.

    Pure: same inputs, same output. The blackboard caches the result per state
    version so every agent sees one consistent zone within a step.
    """

    def p(concept_id: str) -> float:
        return mastery.get(concept_id, prior)

    unmastered = [c for c in graph.ids() if p(c) < band.theta_upper]

    if not unmastered:
        # Nothing left to teach. An empty frontier here means the learner is
        # done, which is categorically different from being unable to find
        # anything reachable — hence not a fallback.
        return Frontier(frozenset(), reason="all concepts mastered")

    in_zone = frozenset(
        c for c in unmastered if all(p(q) > band.theta_lower for q in graph.prerequisites(c))
    )
    if in_zone:
        return Frontier(in_zone)

    # Should be unreachable. Topological order places every prerequisite before
    # its dependants, so the *first* unmastered concept in that order cannot
    # have an unmastered prerequisite — an unmastered prerequisite would itself
    # have come first — and is therefore always in the zone. Reaching this means
    # the graph and the mastery map disagree, so recover, but say so loudly.
    order = list(graph.topological_order())
    earliest = min(unmastered, key=order.index)
    return Frontier(
        frozenset({earliest}),
        fallback=True,
        reason=(
            f"no concept had its prerequisites met, which should be impossible "
            f"for a consistent graph; fell back to the earliest unmastered "
            f"concept {earliest!r}"
        ),
    )


def is_unconstrained(frontier: Frontier, graph: ConceptGraph) -> bool:
    """Whether the band admits essentially the whole graph.

    Not an error — a learner at the very start legitimately has many reachable
    concepts — but if it holds throughout a run the band is not doing any
    selecting, and any result about frontier-constrained planning is hollow.
    Reported as a run diagnostic and read directly by the threshold sweep.
    """
    return len(frontier) >= max(1, len(graph.ids()) - 1)
