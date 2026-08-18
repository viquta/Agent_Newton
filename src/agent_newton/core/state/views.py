"""What each agent is allowed to see.

**The ablation lives here.** Both arms run the same tutor, diagnostic, verifier,
items and learner seeds. The only thing that differs is which view the planner
receives, and the two views are windows onto the *same* state object rather than
two implementations — so the arms cannot drift apart in any other respect.

``FullStateView`` exposes per-concept posteriors, the error trace and the
frontier. ``ItemCorrectnessView`` exposes a stream of right/wrong outcomes and
nothing else. The difference is not that the second is coarser: it has neither
the posteriors nor the graph, so **it cannot compute a frontier at all**. There
is no method on it that would return one, which is the point — a planner given
this view is structurally incapable of frontier-based selection rather than
merely discouraged from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agent_newton.core.state.schema import ErrorEvent, Plan, Utterance
from agent_newton.core.state.zpd import Frontier


class PlannerView(Protocol):
    """What every planner can rely on, whichever arm is running."""

    @property
    def outcomes(self) -> Sequence[bool]:
        """Item-level correctness, oldest first."""
        ...

    @property
    def plan(self) -> Plan | None:
        """The goal being worked toward. Curriculum, so both arms see it."""
        ...

    def consecutive_correct(self) -> int: ...


@dataclass(frozen=True)
class FullStateView:
    """The coupled arm's view: everything the shared layer carries."""

    mastery: dict[str, float]
    error_trace: tuple[ErrorEvent, ...]
    frontier: Frontier
    outcomes: tuple[bool, ...]
    version: int
    plan: Plan | None = None
    #: What the learner said in words, most recent last. Only the coupled view
    #: carries it: it is something the learner told us about themselves, which
    #: is exactly what the decoupled arm does without.
    reflections: tuple[Utterance, ...] = ()
    #: Concepts worked often enough in this sitting to be set aside for now.
    #: Empty unless ``cohort.max_visits_per_concept`` is set, which is every run
    #: that has produced a measured number.
    #:
    #: On the view rather than inside the planner. A planner that kept its own
    #: record of where it had dwelt would be holding private state, which is the
    #: thing the blackboard design exists to avoid — and the asymmetry between
    #: the two planners' memory is an argument the project makes, so it must not
    #: be quietly spent here.
    weaknesses: frozenset[str] = frozenset()
    #: Concepts the learner asked to work on, said before the sitting started.
    #:
    #: Learner *input*, not learner *model*: it is a thing a person stated about
    #: what they want, not an inference about what they know. Both arms could be
    #: handed it fairly — and only one can act on it, because acting means
    #: choosing a goal whose route reaches it, which needs the posteriors and
    #: the graph. That is the same asymmetry ``Emphasis`` already has, arriving
    #: through a different door.
    #:
    #: Empty for every cohort. Nothing but the demo sets it.
    requested: frozenset[str] = frozenset()
    #: Requested concepts the band would otherwise have closed, reopened for
    #: review. A subset of ``requested``, and empty unless
    #: ``cohort.review_on_request`` is on.
    #:
    #: Here rather than recomputed by each reader because the board is what
    #: knows the configuration, and two readers deciding it separately could
    #: disagree about which zone the session is actually in.
    reviewing: frozenset[str] = frozenset()

    def consecutive_correct(self) -> int:
        count = 0
        for correct in reversed(self.outcomes):
            if not correct:
                break
            count += 1
        return count

    def probability(self, concept_id: str, default: float = 0.0) -> float:
        return self.mastery.get(concept_id, default)

    def said_about(self, concept_id: str, window: int = 2) -> tuple[Utterance, ...]:
        """The learner's most recent words **on this concept**.

        Filtered rather than taken from the end of the list. Unfiltered, a
        reflection carried over from an earlier concept and the tutor asked
        someone differentiating ``2/x^2`` to revisit their explanation of
        limits.
        """
        on_topic = [u for u in self.reflections if u.concept_id == concept_id]
        return tuple(on_topic[-window:])

    def recent_misconceptions(self, window: int | None = None) -> list[str]:
        events = self.error_trace if window is None else self.error_trace[-window:]
        return [e.misconception_label for e in events if e.misconception_label]


@dataclass(frozen=True)
class ItemCorrectnessView:
    """The decoupled arm's view: a right/wrong stream, and the goal.

    Deliberately has no ``mastery``, no ``frontier``, no ``error_trace``. A
    planner holding this cannot ask which concepts are reachable, because the
    information required to answer is absent rather than withheld.

    It knows *where* the learner is going and cannot work out how to get there.
    That extends to the learner's stated intent: consolidating on a difficulty
    needs the error trace and advancing past what is already known needs the
    posteriors, so a planner on this view behaves identically whichever the
    learner asked for.
    """

    outcomes: tuple[bool, ...]
    version: int
    plan: Plan | None = None

    def consecutive_correct(self) -> int:
        count = 0
        for correct in reversed(self.outcomes):
            if not correct:
                break
            count += 1
        return count
