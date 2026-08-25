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

from agent_newton.core.pedagogy import HintLevel, TeachingStyle, TutorMove
from agent_newton.core.state.schema import Plan
from agent_newton.core.state.views import FullStateView, ItemCorrectnessView
from agent_newton.domains.base import ConceptResource, Domain, Item

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

    ``mastery`` and ``prior_failures`` are the scaffolding rule's two inputs,
    and the session supplies both rather than leaving the tutor to read them.
    Not a loss of autonomy — the level was never the tutor's to choose — but a
    matter of *when* each is read, which the tutor cannot know: ``mastery`` is
    the posterior as it stood when the question was posed, before this answer
    moved it, and ``prior_failures`` excludes the step being responded to. The
    tutor read the view instead, so both inputs already carried the current
    failure and every turn in two human sittings came out at ``worked_step``.

    The session derives ``mastery`` from **its own arm's view**, so a tutor in
    the decoupled arm still gets the 0.0 its view would have yielded. The
    ablation is unaffected: nothing here is a channel to state the arm
    withholds.

    ``explained`` says the learner set out their reasoning on this step when
    asked for it. The error-first rule is satisfied by that, so the tutor may
    remediate straight away rather than asking a second time what they have just
    said — see ``core/pedagogy``.
    """

    def respond(
        self,
        item: Item,
        diagnosis: Diagnosis,
        view: StateView,
        domain: Domain,
        *,
        response: str,
        mastery: float,
        prior_failures: int,
        moves_this_item: Sequence[TutorMove],
        said_this_item: Sequence[str] = (),
        explained: bool = False,
    ) -> Hint: ...

    def explain(
        self,
        resource: ConceptResource,
        style: TeachingStyle,
        exchanges: Sequence[tuple[str, str]] = (),
    ) -> str:
        """The lesson a learner reads, in the account the rules chose.

        Separate from :meth:`respond` because it answers a different question.
        Every hint comments on an attempt and presupposes the learner has the
        concept; this presupposes they may not, and there is nothing for it to
        respond to — it is called between items, with no step in front of it.

        ``resource`` carries the authored content, already checked as plain text
        and already checked not to answer any item on the concept at any
        template draw. **A tutor may re-voice it; it does not write it.** That
        keeps the mathematics something a person wrote and validated, and leaves
        the model the part it is good at.

        ``style`` comes from :func:`~agent_newton.core.pedagogy.policy.style_for`
        rather than from a prompt, like the support level and the move. A
        model-free tutor is free to ignore it, and the one the cohorts run does.

        ``exchanges`` is the conversation so far, oldest first, as
        ``(what the tutor said, what the learner said back)``. Empty on the
        opening turn. Handed over the same way ``said_this_item`` is on
        :meth:`respond`, and for the same reason: an agent is *told* what was
        said, never given a channel to another agent — the learner's words reach
        here through the session and the board, like everything else.
        """
        ...


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
