"""Agent interfaces.

Every role has a model-backed implementation and at least one model-free
counterpart. Those counterparts are run conditions, not test doubles: the oracle
and noised-oracle diagnostics are the comparison conditions the error-propagation
analysis needs, and a fully model-free configuration runs the whole pipeline
without inference.

Agents never call one another. Each receives a view of the shared state and
returns a decision; the session writes the consequences back.

vh comment: note to self: base.py never executes, it just has type declarations 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from agent_newton.core.pedagogy import HintLevel, TeachingStyle, TutorMove
from agent_newton.core.state.schema import Plan
from agent_newton.core.state.views import FullStateView, ItemCorrectnessView
from agent_newton.domains.base import ConceptResource, Domain, Item

StateView = FullStateView | ItemCorrectnessView

#used in session where the student attempts the problem
@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What the diagnostic agent concluded about an incorrect step."""

    misconception_id: str | None
    confidence: float = 0.0

    @property
    def named(self) -> bool:
        return self.misconception_id is not None


@dataclass(frozen=True, slots=True)
class Hint: #vh comment: i would have called it TutorTurn or TutorAction or something similar rather than Hint, cause the word Hint brings confusion. Naja, doesn't matter.
    # answer: fair, and the store already agrees with you — the `turn` table
    # calls it a turn and records move/level/targets/text. The rename is
    # mechanical (five files and their tests) and nothing measured depends on
    # the class name. It has not been done because `HintLevel`, `FALLBACK_HINT`
    # and `receive_hint` would all want renaming with it, and `receive_hint` is
    # on the Learner protocol. Worth doing as one commit, not piecemeal.
    """A tutor turn.

    ``targets`` is what the hint actually addresses. It is the the main route by
    which a learner improves, so a hint naming the wrong misconception does no
    work — which is how diagnostic error reaches learning outcomes.
    vh updated comment: cause now there is also "explain" which also results in the learner
    learning.
    """

    text: str
    move: TutorMove
    level: HintLevel
    targets: str | None = None


@runtime_checkable
class Tutor(Protocol):
    """
    vh: responds (as a protocol) and explains (dialogue) (is this right?)
    """

    def respond(
        self,
        item: Item,
        diagnosis: Diagnosis, #output from the diagnostic agent --> a diagnosis obj
        view: StateView,
        domain: Domain,
        *,
        response: str,
        mastery: float,
        prior_failures: int,
        moves_this_item: Sequence[TutorMove], #in pedagogy.py, such as hint, reflect, remediate, after teacher-layer branch -->present and explain.
        said_this_item: Sequence[str] = (),
        explained: bool = False,
    ) -> Hint: 
        """
        for def respond:
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
                failure and every turn in two human sittings came out at ``worked_step``. vh comment:This is a clear example of how this architecture is great for this ITSM.
        
                notes for the simulated students: 
                    The session derives ``mastery`` from **its own arm's view**, so a tutor in
                    the decoupled arm still gets the 0.0 its view would have yielded. The
                    ablation is unaffected: nothing here is a channel to state the arm
                    withholds.
        
                ``explained`` says the learner set out their reasoning on this step when
                asked for it. The error-first rule is satisfied by that, so the tutor may
                remediate straight away rather than asking a second time what they have just
                said — see ``core/pedagogy``. vh comment: but is this input also used also in the explain method? It would be valuable.

                answer: no, and deliberately. ``explain`` takes only
                ``resource, style, exchanges, closing``. ``explained`` exists to satisfy
                the error-first rule, which orders *replies to a step* — reflect before
                remediate. A lesson is not a reply to a step, and ``check_move`` returns
                early on ``EXPLAIN`` for the same reason. There is no ordering for it to
                satisfy, so the parameter would sit unused.
           
        """
        ... #vh comment: it's confusing that the output obj is called "hint", cause the class is called Hint, but the tutormove could be reflect, remediate ... ect
        # answer: agreed — see the note on the class. And note this body is `...`:
        # a Protocol method is a *declaration*, never executed. The code that runs
        # is `LLMTutor.respond` and `TemplateTutor.respond`.

    def explain(
        self,
        resource: ConceptResource, # vh comment: i need to check what this is, and how it is used. it could be part of the content that the tutor uses to explain the concept to the student.
        # answer: exactly that. One entry per concept from
        # `domains/<domain>/resources.yaml`, behind the `ConceptResources`
        # Protocol in `domains/base.py`. It carries the rule shown beside a
        # question, a worked example, and two optional lesson fields;
        # `resource.lesson()` composes the authored lesson. `domain validate`
        # checks it is plain text and that the example answers no item on the
        # concept at any template draw.
        style: TeachingStyle, #vh comment: i should check this also, cause style 1 and 2 are both pretty similar. I could just put the Socratic as the default.
        # answer: they differ more than they look. PLAIN states the concept in
        # two or three sentences and then asks; SOCRATIC is told *not* to explain
        # yet and to ask one question first. But PLAIN cannot be replaced as the
        # default: it **is** the authored text, so it needs no model —
        # `TemplateTutor.explain` returns `resource.lesson()` and every
        # model-free run still teaches. Defaulting to SOCRATIC would leave a
        # model-free run with no lesson at all. `style_for` also rotates, so a
        # second lesson on a concept differs from the first — which is the point
        # of the account ceiling.
        exchanges: Sequence[tuple[str, str]] = (), # vh comment: does the llm hold the entire dialogue in its cache or is it only seeing the most recent reply and its most recent response?
        # answer: the whole dialogue, every turn, and the model holds nothing
        # between calls. Each call is stateless; `explain` formats every pair
        # into the prompt as "You: … The student: …" and re-sends it. That is
        # also why the response cache behaves: a longer conversation is a
        # different prompt, so it misses rather than returning the earlier turn.
        closing: bool = False,
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
        the model the part it is good at. vh comment: i need to check this, and see what it means.

        answer: the split is authored *mathematics* against generated *wording*.
        The lesson a person wrote is what the learner keeps; the model is asked
        to say the same thing in the chosen style. Two guards send its reply
        back to the authored text — a plain-text check and a length cap — so a
        failure costs the styling and never the content.

        ``style`` comes from :func:`~agent_newton.core.pedagogy.policy.style_for`
        rather than from a prompt, like the support level and the move. A
        model-free tutor is free to ignore it, and the one the cohorts run does.

        ``closing`` says this is the last turn: answer what is hanging and ask
        nothing new. ⚠️ Without it a lesson always ended on a question the
        learner had no way to reply to, and the written summary answered it for
        them — at the exact moment a sitting reported being *"just about to
        understand something important"*.

        ``exchanges`` is the conversation so far, oldest first, as
        ``(what the tutor said, what the learner said back)``. Empty on the
        opening turn. Handed over the same way ``said_this_item`` is on
        :meth:`respond`, and for the same reason: an agent is *told* what was
        said, never given a channel to another agent — the learner's words reach
        here through the session and the board, like everything else.
        """
        ...



@runtime_checkable
class Diagnostic(Protocol): #vh comment: see in my private notes for revision of diagnostic agent (maybe I'll post them in docs later)

    """Classifies an incorrect step into the domain's misconception catalogue."""
    # returns a Diagnosis obj
    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis: ...

@runtime_checkable
class ConfusionDetector(Protocol): #vh comment: i need to check where this is used... is this for the tutor?
    # answer: not the tutor — the session. `session.py :: _note_if_confused`
    # is the only caller, reading the working and reflection channels. It is
    # the third of the three things that can buy a lesson (see diagram 16);
    # the other two are an explicit `:why` and the error trace.
    # Implementations: `LLMConfusionDetector` (llm.py) and `NoConfusion`
    # (tutor.py), the null object every cohort runs.
    """Reads the learner's own words for "I do not know what this is".

    Not an agent in the blackboard sense, and worth being clear about why. It
    plans nothing, teaches nothing and holds no view of the learner model — it
    classifies one string, the way the verifier classifies one answer. What it
    produces is written to the board and acted on by the session, like every
    other observation.

    ⚠️ It is the one place a model is permitted to decide something, and the
    reason is that it is **detection rather than instructional policy**. Whether
    a lesson is given, which account it takes and when it stops are all still
    rules; this only answers whether the learner said a thing. And it is counted,
    so a detector that fires on ordinary mistakes shows up as a rate rather than
    as mysterious teaching.

    The model-free implementation says "no" to everything, which is what every
    cohort runs.
    """

    def confused(self, concept_id: str, text: str) -> str | None:
        """The words saying so, or ``None``.

        Returning the quote rather than a bare ``True`` so the audit log can
        record *what* was read that way. A trigger whose evidence is a boolean
        cannot be argued with after the fact.
        """
        ...


@runtime_checkable
class Resumable(Protocol): #vh comment: i need to check where this is used... 
    # answer: `FixedOrderPlanner` alone. `demo.py` snapshots it at the end of a
    # sitting and `session.py` restores it at the start. The ablation shows up
    # even here: the decoupled planner's position in the syllabus walk is its
    # only progress signal, so it must carry it; the coupled planner declares
    # nothing, because everything it routes from is already on the board.
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
