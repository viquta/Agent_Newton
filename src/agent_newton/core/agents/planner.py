"""Planners — where the ablation becomes behaviour.

Both planners know the **syllabus**: the item bank, the prerequisite graph and
the declared goals are static curriculum, available to either arm. What differs
is what they know about *this learner*. Keeping curriculum knowledge common is
what makes the comparison fair — otherwise the decoupled arm would be
handicapped by ignorance of the subject rather than of the learner.

Both restrict their work to the goal's prerequisite closure, because that too is
curriculum: which concepts lie on the way to a target is a property of the
graph. So the arms are not separated by *scope*.

They are separated by **routing**. :class:`GoalDirectedPlanner` reads the
posteriors and the error trace, so it can skip what is already mastered and
order what remains by the learner's stated intent.
:class:`FixedOrderPlanner` has neither, so it walks the closure in topological
order and advances on consecutive correct answers. That is not a worse heuristic
over the same information — it is the best available given strictly less.

:class:`FrontierPlanner` is kept as the undirected baseline: it selects from the
frontier with no goal at all, which is what the system did before planning had a
target.

All three may **repeat** an item, preferring the least-used one for a concept.
Mastery takes several correct answers to establish, and a bank holding fewer
items than that per concept would otherwise make the concept permanently
unmasterable: the planner would run out of unseen work and stop while the
learner still had everything to learn.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from agent_newton.config import ZPDConfig
from agent_newton.core.agents.base import StateView
from agent_newton.core.pedagogy import may_select
from agent_newton.core.state import route
from agent_newton.core.state.schema import Emphasis, Plan
from agent_newton.core.state.views import FullStateView
from agent_newton.domains.base import Domain, Item


def _least_used(domain: Domain, concept_id: str, given: Mapping[str, int]) -> Item | None:
    """The least-practised item for a concept, ties broken by id for stability."""
    items = domain.items.for_concept(concept_id, "practice")
    if not items:
        return None
    return min(items, key=lambda item: (given.get(item.id, 0), item.id))


class GoalDirectedPlanner:
    """Aims at a declared goal and routes toward it from the learner model.

    The goal is chosen from the domain's ordered list — the first not yet
    mastered. The next concept is derived from the blackboard every time it is
    asked for rather than fixed when the goal is set, so evidence arriving
    mid-session changes where the learner is taken next.
    """

    def __init__(self, band: ZPDConfig, prior: float, emphasis: Emphasis) -> None:
        self._band = band
        self._prior = prior
        self._emphasis = emphasis

    def _require_full(self, view: StateView) -> FullStateView:
        if not isinstance(view, FullStateView):
            raise TypeError(
                "GoalDirectedPlanner requires the full state view; the decoupled "
                "arm must use FixedOrderPlanner"
            )
        return view

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        full = self._require_full(view)
        goal = route.next_goal(
            domain.concepts.goals(), full.mastery, self._band, self._prior
        )
        if goal is None:
            return None
        outstanding = route.remaining(
            goal, full.mastery, domain.concepts, self._band, self._prior
        )
        return Plan(
            goal=goal,
            emphasis=self._emphasis,
            reason=f"{len(outstanding)} concept(s) still needed for {goal}",
        )

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None:
        full = self._require_full(view)
        if full.plan is None:
            return None

        step = route.next_step(
            goal=full.plan.goal,
            emphasis=full.plan.emphasis,
            mastery=full.mastery,
            error_trace=full.error_trace,
            frontier=full.frontier,
            graph=domain.concepts,
            band=self._band,
            prior=self._prior,
            # Read from the view rather than tracked here. Empty unless the
            # dwelling cap is configured, and it is off for every cohort.
            deprioritised=full.weaknesses,
        )
        if step is None:
            return None

        # The band-membership rule is the planner's own guardrail, so a
        # selection violating it never reaches the learner. The fallback branch
        # is exempt: it fires only when the frontier and the graph disagree, and
        # stalling the session would be the worse failure.
        if not step.fallback and may_select(step.concept_id, full.frontier) is not None:
            return None
        return _least_used(domain, step.concept_id, given)


class _ClosureWalker:
    """Selects from the goal's unmastered closure, ignoring the band.

    The base for the ordering probes. It takes exactly the material
    :class:`GoalDirectedPlanner` would eventually cover —
    ``route.remaining`` is the same set — and differs only in which of it to
    take next. Holding the material constant is what makes a null result
    meaningful: a planner that wandered off and never met the learner's
    misconceptions would do worse for reasons that have nothing to do with
    ordering.

    The band-membership guardrail is deliberately skipped. Violating it is the
    point, and a subclass that respected it could not violate the prerequisite
    order it exists to violate.
    """

    def __init__(self, band: ZPDConfig, prior: float) -> None:
        self._band = band
        self._prior = prior

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        if not isinstance(view, FullStateView):
            raise TypeError(f"{type(self).__name__} requires the full state view")
        goal = route.next_goal(
            domain.concepts.goals(), view.mastery, self._band, self._prior
        )
        return None if goal is None else Plan(goal=goal, reason="ordering probe")

    def _order(self, remaining: tuple[str, ...], view: FullStateView) -> str:
        raise NotImplementedError

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None:
        if not isinstance(view, FullStateView):
            raise TypeError(f"{type(self).__name__} requires the full state view")
        if view.plan is None:
            return None
        remaining = route.remaining(
            view.plan.goal, view.mastery, domain.concepts, self._band, self._prior
        )
        if not remaining:
            return None
        return _least_used(domain, self._order(remaining, view), given)


class ReverseOrderPlanner(_ClosureWalker):
    """Works the goal's closure backwards: every dependant before its prerequisite.

    The worst sequencing the prerequisite graph allows, and therefore the sharp
    test of whether the graph has any bearing on what a simulated learner
    learns. If outcomes match a planner that respects the order, the structure
    is constraining selection and nothing else.
    """

    def _order(self, remaining: tuple[str, ...], view: FullStateView) -> str:
        return remaining[-1]


class ShuffledPlanner(_ClosureWalker):
    """Takes an arbitrary concept from the closure, ignoring structure entirely.

    Guards against reverse order being special in some way not foreseen. The
    choice is a hash of the seed and the concepts still outstanding rather than
    a running generator, so it is reproducible and does not depend on how many
    other selections happened first — the same convention the simulator's rule
    engine uses.
    """

    def __init__(self, band: ZPDConfig, prior: float, seed: int = 0) -> None:
        super().__init__(band, prior)
        self._seed = seed

    def _order(self, remaining: tuple[str, ...], view: FullStateView) -> str:
        digest = hashlib.sha256(f"{self._seed}|{'|'.join(remaining)}".encode()).digest()
        return remaining[int.from_bytes(digest[:8], "big") % len(remaining)]


class OraclePlanner:
    """The reference policy: selects what would elicit the most, knowing the truth.

    Given the learner's live misconception profile, it takes the selectable item
    whose probes carry the greatest remaining firing probability — the most
    misconception mass a single item could bring to the surface, where it can be
    diagnosed, hinted at and remediated.

    **A reference, not an optimum.** A truly optimal policy would look ahead over
    the mastery dynamics and the remediation schedule; this is greedy on one
    step. It is an upper bound on what a planner could do with the information it
    has, which is what regret needs, and it is deterministic, which is what
    comparing against it needs.

    It stays inside the frontier and on the way to the goal, so it is a fair
    reference for a planner bound by the same rules: the comparison is about
    which of the permitted choices was taken, not about who was allowed more.

    The profile is a **live** mapping, so remediation applied during the session
    is reflected here without the session having to tell it anything. No
    model-backed planner is given one — there is a test on that.
    """

    def __init__(
        self,
        firing: Mapping[str, float],
        band: ZPDConfig,
        prior: float,
    ) -> None:
        self._firing = firing
        self._band = band
        self._prior = prior

    def value(self, item: Item) -> float:
        """Remaining firing probability this item could bring to the surface."""
        return sum(self._firing.get(m, 0.0) for m in item.probes)

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        if not isinstance(view, FullStateView):
            raise TypeError("OraclePlanner requires the full state view")
        goal = route.next_goal(
            domain.concepts.goals(), view.mastery, self._band, self._prior
        )
        return None if goal is None else Plan(goal=goal, reason="oracle reference policy")

    def candidate_items(
        self, view: FullStateView, domain: Domain, given: Mapping[str, int]
    ) -> list[Item]:
        """One item per permitted concept — what any planner here could choose."""
        concepts = (
            route.candidates(view.plan.goal, view.frontier, domain.concepts)
            if view.plan is not None
            else tuple(sorted(view.frontier))
        )
        found = [_least_used(domain, c, given) for c in concepts]
        return [item for item in found if item is not None]

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None:
        if not isinstance(view, FullStateView):
            raise TypeError("OraclePlanner requires the full state view")
        options = self.candidate_items(view, domain, given)
        if not options:
            return None
        # Ties broken by id so the reference is reproducible.
        return max(options, key=lambda item: (self.value(item), item.id))


class FrontierPlanner:
    """Selects from the frontier, nearest the syllabus start first.

    The undirected baseline: no goal, so nothing constrains selection to what is
    on the way anywhere. Ties are broken by topological position, so among
    reachable concepts it prefers the one whose prerequisites were satisfied
    earliest, and the walk is reproducible.

    Returns None only when the frontier is empty — which means every concept is
    mastered, not that the planner ran out of material.
    """

    def plan(self, view: StateView, domain: Domain) -> Plan | None:  # noqa: ARG002
        """No target. Selection is not directed anywhere."""
        return None

    def select(self, view: StateView, domain: Domain, given: Mapping[str, int]) -> Item | None:
        if not isinstance(view, FullStateView):
            raise TypeError(
                "FrontierPlanner requires the full state view; the decoupled arm "
                "must use FixedOrderPlanner"
            )

        order = list(domain.concepts.topological_order())
        for concept_id in sorted(view.frontier, key=order.index):
            if may_select(concept_id, view.frontier) is not None:
                continue
            item = _least_used(domain, concept_id, given)
            if item is not None:
                return item

        return None


class FixedOrderPlanner:
    """Walks the goal's prerequisite closure in order, advancing on correctness.

    The decoupled arm. Its view carries a right/wrong stream and nothing else,
    so it cannot ask which concepts are reachable or which are already known —
    only whether the learner has been getting things right lately.

    It knows the goal, and it knows from the graph which concepts lie on the way
    to it. Both are curriculum. What it cannot do is skip the ones this learner
    has already mastered, or work the one they are struggling with first.
    """

    def __init__(self, advance_after: int = 2, on_exhaustion: str = "stop") -> None:
        self._advance_after = advance_after
        #: ``stop`` ends the session at the end of the walk; ``cycle`` begins it
        #: again. Stopping is not something the missing learner model forces —
        #: it simply leaves this arm attempting fewer items than the other,
        #: which confounds any outcome that grows with practice.
        self._on_exhaustion = on_exhaustion
        self._position = 0
        #: Length of the outcome stream when the position last moved. Without
        #: it, a streak that persists would advance the position on *every*
        #: call rather than once, racing through the syllabus an item at a time.
        self._advanced_at = 0

    def snapshot(self) -> dict[str, Any]:
        """Where this walk had got to. Satisfies ``Resumable``.

        The position advances on a streak of correct answers *at the moment
        select is called*, so it cannot be recovered from the outcome stream
        alone — the stream does not record when the planner was asked. It has to
        be carried.

        That this planner needs carrying and the coupled one does not is the
        architecture showing through: a planner routing from the blackboard
        keeps nothing, because the blackboard already persists.
        """
        return {"position": self._position, "advanced_at": self._advanced_at}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self._position = int(snapshot.get("position", 0))
        self._advanced_at = int(snapshot.get("advanced_at", 0))

    def _walk(self, domain: Domain) -> list[str]:
        """The concepts this planner will walk, in order.

        The union of every declared goal's closure — everything on the way to
        anything the domain aims at. Curriculum, so restricting to it is fair.

        Deliberately *not* narrowed to the current goal. Narrowing would mean
        stepping past a concept that no current goal needs but a later one does,
        and the walk only moves forward, so that concept would never be
        revisited. The coupled planner can narrow safely because it knows which
        goals are already reached; this one cannot, and pretending otherwise
        would hand the comparison an advantage that has nothing to do with the
        learner model.
        """
        needed: set[str] = set()
        for goal in domain.concepts.goals():
            needed |= route.relevant(goal, domain.concepts)
        order = list(domain.concepts.topological_order())
        return [c for c in order if c in needed] if needed else order

    def plan(self, view: StateView, domain: Domain) -> Plan | None:  # noqa: ARG002
        """The first declared goal this walk has not yet passed.

        Whether a goal has been *reached* is a fact about the learner, and this
        planner cannot see it. Its position in its own walk is the only progress
        signal it has, so that is what the goal advances on.
        """
        walk = self._walk(domain)
        for goal in domain.concepts.goals():
            if goal in walk and walk.index(goal) >= min(self._position, len(walk)):
                return Plan(
                    goal=goal,
                    reason=f"walking the syllabus; position {self._position}",
                )
        return None

    def select(self, view: StateView, domain: Domain, given: Mapping[str, int]) -> Item | None:
        walk = self._walk(domain)
        answered = len(view.outcomes)

        if (
            view.consecutive_correct() >= self._advance_after
            and answered - self._advanced_at >= self._advance_after
        ):
            self._position += 1
            self._advanced_at = answered

        if self._position >= len(walk) and self._on_exhaustion == "cycle":
            # Round again. `_least_used` keeps the revision spread over the
            # bank rather than replaying one item, and the session's own item
            # budget is what ends the run.
            self._position = 0
            self._advanced_at = answered

        while self._position < len(walk):
            item = _least_used(domain, walk[self._position], given)
            if item is not None:
                return item
            # A concept with no practice items at all cannot be taught; step
            # past it rather than stalling.
            self._position += 1

        return None
