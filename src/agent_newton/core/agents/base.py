"""Agent interfaces.

Every role has a model-backed implementation and at least one model-free
counterpart. Those counterparts are run conditions, not test doubles: the oracle
and noised-oracle diagnostics are the comparison conditions the error-propagation
analysis needs, and a fully model-free configuration runs the whole pipeline
without inference.

Agents never call one another. Each receives a view of the shared state and
returns a decision; the session writes the consequences back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from agent_newton.core.pedagogy import HintLevel, TutorMove
from agent_newton.core.state.views import FullStateView, ItemCorrectnessView
from agent_newton.domains.base import Domain, Item

StateView = FullStateView | ItemCorrectnessView


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What the diagnostic agent concluded about an incorrect step."""

    misconception_id: str | None
    confidence: float = 0.0

    @property
    def named(self) -> bool:
        return self.misconception_id is not None


@dataclass(frozen=True, slots=True)
class Hint:
    """A tutor turn.

    ``targets`` is what the hint actually addresses. It is the sole route by
    which a learner improves, so a hint naming the wrong misconception does no
    work — which is how diagnostic error reaches learning outcomes.
    """

    text: str
    move: TutorMove
    level: HintLevel
    targets: str | None = None


@runtime_checkable
class Tutor(Protocol):
    def respond(
        self,
        item: Item,
        diagnosis: Diagnosis,
        view: StateView,
        domain: Domain,
        *,
        failed_attempts: int,
        moves_this_item: Sequence[TutorMove],
    ) -> Hint: ...


@runtime_checkable
class Diagnostic(Protocol):
    """Classifies an incorrect step into the domain's misconception catalogue."""

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis: ...


@runtime_checkable
class OracleAccess(Protocol):
    """Marks an implementation that is *given* the injected label.

    Kept as a separate protocol so that reading ground truth is an explicit
    capability rather than an argument every implementation happens to receive.
    A model-backed diagnostic must never satisfy this — if it did, the label it
    is supposed to infer would be sitting in its inputs, and its measured
    accuracy would mean nothing.
    """

    def observe_ground_truth(self, label: str | None) -> None: ...


@runtime_checkable
class Planner(Protocol):
    """Selects the next item.

    Both arms' planners know the syllabus — the item bank and the prerequisite
    graph are static curriculum, not learner state. They differ only in what
    they know about *this learner*, which is the single variable under test.
    """

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None: ...
