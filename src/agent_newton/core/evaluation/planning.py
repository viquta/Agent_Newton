"""Planner accuracy, measured against a reference policy.

The planner is the one component whose output is a *choice* rather than a
judgement, so it cannot be scored against a labelled set the way the verifier
and the diagnostic are. It is scored against what a policy holding the learner's
true profile would have chosen from the same options.

**Decisions are collected through the real session.** :class:`ShadowedPlanner`
satisfies the ``Planner`` protocol, delegates every call to the planner under
test, and asks the reference the same question on the way past. The session runs
unchanged and the planner under test still drives it — the reference only
watches. Re-implementing the loop to harvest decisions would risk measuring a
loop that had drifted from the real one.

Three measures, and they answer different questions:

``agreement``
    How often the same item was chosen. Blunt — a planner can disagree with the
    reference and lose nothing, if the two options were worth the same.

``regret``
    What the disagreement cost, in remaining misconception probability the
    chosen item could not bring to the surface and the reference's could. Zero
    when a different choice was equally good. This is the one to read.

``in_frontier``
    Whether the choice was inside the mastery band at all. For a planner reading
    the frontier this is 1.0 by construction and measures nothing; for one that
    cannot see it, it is the rate at which work lands outside the zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from agent_newton.core.agents.base import Planner, StateView
from agent_newton.core.agents.planner import OraclePlanner
from agent_newton.core.state.schema import Plan
from agent_newton.core.state.views import FullStateView
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains.base import Domain, Item


@dataclass(frozen=True, slots=True)
class Decision:
    """One selection, beside what the reference would have made."""

    learner_id: str
    goal: str | None
    chosen_item: str | None
    chosen_concept: str | None
    reference_item: str | None
    reference_concept: str | None
    chosen_value: float
    reference_value: float
    in_frontier: bool
    options: int

    @property
    def agrees(self) -> bool:
        return self.chosen_item == self.reference_item

    @property
    def regret(self) -> float:
        """Elicitable misconception probability left on the table.

        Non-negative: the reference maximises this quantity over the same
        options, so it cannot be beaten on it.
        """
        return max(0.0, self.reference_value - self.chosen_value)


@dataclass
class PlanningReport:
    decisions: list[Decision] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.decisions)

    @property
    def agreement(self) -> float:
        if not self.decisions:
            return 0.0
        return sum(1 for d in self.decisions if d.agrees) / self.total

    @property
    def concept_agreement(self) -> float:
        """Softer, and often the more informative one.

        Two items on the same concept are close to interchangeable — the item is
        the least-practised one, not a separate judgement — so agreeing on the
        concept and differing on the item is barely a disagreement.
        """
        if not self.decisions:
            return 0.0
        same = sum(1 for d in self.decisions if d.chosen_concept == d.reference_concept)
        return same / self.total

    @property
    def mean_regret(self) -> float:
        if not self.decisions:
            return 0.0
        return sum(d.regret for d in self.decisions) / self.total

    @property
    def costly_disagreements(self) -> int:
        """Disagreements that actually gave something up."""
        return sum(1 for d in self.decisions if d.regret > 1e-9)

    @property
    def no_selection(self) -> int:
        """Decisions where the planner had nothing to give.

        The session ends here. Counted apart from everything else: a planner
        that selected nothing did not make a bad choice, it made none.
        """
        return sum(1 for d in self.decisions if d.chosen_item is None)

    @property
    def in_frontier_rate(self) -> float:
        """Of the selections actually made, how many were inside the band.

        The denominator excludes decisions with no selection. Counting those as
        "outside the frontier" would report a planner as choosing badly at the
        exact moment it correctly declined to choose — and since the decoupled
        arm ends every session that way, it would read as straying outside the
        zone on every learner.
        """
        made = [d for d in self.decisions if d.chosen_item is not None]
        if not made:
            return 0.0
        return sum(1 for d in made if d.in_frontier) / len(made)

    @property
    def reference_value(self) -> float:
        """What the reference had available, as the scale regret sits on."""
        if not self.decisions:
            return 0.0
        return sum(d.reference_value for d in self.decisions) / self.total

    @property
    def regret_share(self) -> float:
        """Regret as a fraction of what was available to take.

        **This is the figure to compare between arms, not raw regret.** The two
        arms drive the learner along different trajectories, so the reference
        faces different states and has different amounts of misconception mass
        in front of it. A planner that leaves more behind in absolute terms may
        simply have been offered more.
        """
        available = self.reference_value
        return self.mean_regret / available if available > 0 else 0.0

    def worst(self, limit: int = 5) -> list[Decision]:
        return sorted(self.decisions, key=lambda d: -d.regret)[:limit]


class ShadowedPlanner:
    """Runs a planner and a reference over the same state, recording both.

    Returns the planner's own choice, so the session it sits in behaves exactly
    as it would without it.
    """

    def __init__(
        self,
        planner: Planner,
        reference: OraclePlanner,
        record: list[Decision],
        *,
        learner_id: str,
        frontier: Callable[[], Frontier],
        reference_view: Callable[[], StateView],
    ) -> None:
        self._planner = planner
        self._reference = reference
        self._record = record
        self._learner_id = learner_id
        self._frontier = frontier
        # The reference always needs the full view, even when the planner under
        # test is holding an item-correctness one. Reading it from the board
        # rather than the arm's view is what lets a decoupled planner be scored
        # against a policy that can see everything.
        self._reference_view = reference_view

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        return self._planner.plan(view, domain)

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None:
        chosen = self._planner.select(view, domain, given)

        full = self._reference_view()
        assert isinstance(full, FullStateView)  # the board always affords one
        reference = self._reference.select(full, domain, given)
        options = self._reference.candidate_items(full, domain, given)

        self._record.append(
            Decision(
                learner_id=self._learner_id,
                goal=full.plan.goal if full.plan else None,
                chosen_item=chosen.id if chosen else None,
                chosen_concept=chosen.concept_id if chosen else None,
                reference_item=reference.id if reference else None,
                reference_concept=reference.concept_id if reference else None,
                chosen_value=self._reference.value(chosen) if chosen else 0.0,
                reference_value=self._reference.value(reference) if reference else 0.0,
                in_frontier=bool(chosen and chosen.concept_id in self._frontier()),
                options=len(options),
            )
        )
        return chosen


def evaluate(
    learner_ids: Sequence[str],
    domain: Domain,
    config,  # agent_newton.config.Config — annotated loosely to avoid a cycle
    build,  # build_session, injected so core/ does not import orchestration
) -> PlanningReport:
    """Run each learner with the planner shadowed by the reference."""
    from agent_newton.core.state import bkt

    report = PlanningReport()
    for learner_id in learner_ids:
        session = build(learner_id, config.seed, domain, config)
        profile = getattr(session.learner, "profile", None)
        if profile is None:
            raise TypeError(
                "the planner evaluation needs the learner's true profile, so it "
                "runs against a simulated learner only"
            )
        reference = OraclePlanner(
            profile.firing, config.zpd, bkt.initial(config.bkt)
        )
        session.planner = ShadowedPlanner(
            session.planner,
            reference,
            report.decisions,
            learner_id=learner_id,
            frontier=lambda s=session: s.board.frontier,
            reference_view=lambda s=session: s.board.view(arm="coupled"),
        )
        session.run()
    return report
