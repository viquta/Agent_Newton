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

from agent_newton.core.state.schema import ErrorEvent
from agent_newton.core.state.zpd import Frontier


class PlannerView(Protocol):
    """What every planner can rely on, whichever arm is running."""

    @property
    def outcomes(self) -> Sequence[bool]:
        """Item-level correctness, oldest first."""
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

    def consecutive_correct(self) -> int:
        count = 0
        for correct in reversed(self.outcomes):
            if not correct:
                break
            count += 1
        return count

    def probability(self, concept_id: str, default: float = 0.0) -> float:
        return self.mastery.get(concept_id, default)

    def recent_misconceptions(self, window: int | None = None) -> list[str]:
        events = self.error_trace if window is None else self.error_trace[-window:]
        return [e.misconception_label for e in events if e.misconception_label]


@dataclass(frozen=True)
class ItemCorrectnessView:
    """The decoupled arm's view: a right/wrong stream, and nothing more.

    Deliberately has no ``mastery``, no ``frontier``, no ``error_trace``. A
    planner holding this cannot ask which concepts are reachable, because the
    information required to answer is absent rather than withheld.
    """

    outcomes: tuple[bool, ...]
    version: int

    def consecutive_correct(self) -> int:
        count = 0
        for correct in reversed(self.outcomes):
            if not correct:
                break
            count += 1
        return count
