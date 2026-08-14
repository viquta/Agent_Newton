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
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from agent_newton.core.pedagogy import HintLevel, TutorMove
from agent_newton.core.state.schema import Plan
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
    """Writes the turn the learner reads.

    ``response`` is the step being responded to. It is here because without it a
    model-backed tutor has only the misconception's description to work from and
    reconstructs a plausible step rather than addressing the actual one — which
    is how a human session was told its calculation was correct when it was not.

    ``said_this_item`` is what this tutor has already said on this item, oldest
    first. It is here because a tutor with no memory of its own turn repeats it:
    three empty answers to one question produced three identical replies, since
    the prompt was unchanged and the response cache is keyed on the prompt.
    Carried through the session like ``moves_this_item`` rather than read from
    anywhere — an agent is told what it said, not given a channel to another
    agent.
    """

    def respond(
        self,
        item: Item,
        diagnosis: Diagnosis,
        view: StateView,
        domain: Domain,
        *,
        response: str,
        unresolved_steps: int,
        moves_this_item: Sequence[TutorMove],
        said_this_item: Sequence[str] = (),
    ) -> Hint: ...


@runtime_checkable
class Diagnostic(Protocol):
    """Classifies an incorrect step into the domain's misconception catalogue."""

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis: ...


@runtime_checkable
class Resumable(Protocol):
    """An agent whose own bookkeeping has to survive between sessions.

    Kept as a capability, like :class:`OracleAccess`, so that carrying agent
    state across a gap is something an implementation declares rather than
    something the runner assumes it may do.

    Only the decoupled planner needs this, and that is the architectural point
    rather than an accident: it walks a syllabus and its position in that walk
    is the only progress signal it has, so the position lives inside it. The
    coupled planner holds nothing — everything it needs is on the blackboard,
    which already persists. Without this, a returning learner would restart the
    decoupled walk at the first concept every session, and the coupled arm would
    win a comparison about persistence rather than about routing.

    The snapshot never reaches a view, so it is not a channel between agents: it
    goes from an agent to the store and back to the same agent.
    """

    def snapshot(self) -> dict[str, Any]:
        """Serialisable internal state."""
        ...

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Take back a snapshot produced by an earlier session."""
        ...


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
    """Chooses what to work toward, and what to work on next.

    Both arms' planners know the syllabus — the item bank, the prerequisite
    graph and the declared goals are static curriculum, not learner state. They
    differ only in what they know about *this learner*, which is the single
    variable under test.

    Two decisions at two timescales, and the session is what moves information
    between them. :meth:`plan` names the target; :meth:`select` chooses the next
    item on the way to it. Both read the same view, so a planner that cannot see
    the learner model is limited in the same way at both.
    """

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        """The goal to work toward, or None when every goal is reached."""
        ...

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None: ...
