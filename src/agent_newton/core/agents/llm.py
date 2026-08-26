"""Model-backed agents.

Each mirrors a model-free counterpart, so a run can swap one in by config and
change nothing else. Prompts are built from the loaded domain — none contains
subject-specific text — and every reply comes back through a schema closed over
that domain's own ids.

Two properties this module exists to hold:

* **The diagnostic never sees the injected label.** It does not implement
  ``OracleAccess`` and is given no channel to one. Its whole purpose is to infer
  what the oracle is handed, so an accuracy figure measured with the answer in
  scope would be worthless.
* **The planner's latitude is bounded by rules it cannot argue with.** It
  proposes; a deterministic guardrail accepts or replaces. A model that suggests
  an out-of-band concept has that suggestion overridden and the override
  recorded, so the rate is measurable rather than invisible.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from agent_newton.config import LabelSpace, ScaffoldingPolicy, ZPDConfig
from pydantic import BaseModel

from agent_newton.core.agents.base import Diagnosis, Hint, StateView
from agent_newton.core.agents.planner import GoalDirectedPlanner, _least_used
from agent_newton.core.state import route
from agent_newton.core.state.schema import Emphasis, Plan
from agent_newton.core.agents.schemas import UNKNOWN, diagnosis_schema, plan_schema
from agent_newton.core.agents.schemas import (
    ClosingReply,
    ConfusionReply,
    HintReply,
    LessonReply,
)
from agent_newton.core.pedagogy import (
    HintLevel,
    TeachingStyle,
    TutorMove,
    hint_level,
    may_select,
    move_for,
)
from agent_newton.core.state.views import FullStateView
from agent_newton.llm.base import (
    LLMProvider,
    MalformedResponse,
    ProviderError,
    complete,
)
from agent_newton.domains.base import (
    PLAIN_TEXT_ONLY,
    ConceptResource,
    Domain,
    Item,
    Misconception,
)

log = logging.getLogger(__name__)

_DIAGNOSTIC_SYSTEM = (
    "You identify the single misconception behind a student's incorrect step in "
    "a mathematics exercise. Work in three stages: list the catalogue entries "
    "consistent with what the student wrote; distinguish between them using the "
    "specific error made; then commit to one. Choose 'unknown' if no entry fits "
    "— a forced guess is worse than an admission."
)

_TUTOR_SYSTEM = (
    "You are a mathematics tutor. Reply addressed to the student, within the "
    "length the instruction gives you. Never state the final answer: the "
    "student is here to reach it. "
    "Write mathematics in plain text — (f(b) - f(a)) / (b - a), x^2, sqrt(x). "
    "Never use LaTeX or backslash commands. "
    "Describe only what the student's written step actually shows. Do not tell "
    "them which operation they performed, which part they got right, or where "
    "they went wrong, unless their step shows it — if it does not, say what the "
    "step should have been instead of narrating what they did."
)
# The LaTeX ban is not a style preference. Replies arrive as JSON, and a
# backslash command inside a JSON string is eaten by escape processing:
# "\frac{a}{b}" parses to a form-feed character followed by "rac{a}{b}", which
# a learner sees as "$rac{a}{b}". Observed in a human session, where it hid the
# division the hint was trying to explain.
#
# The ban on narrating the student's working has the same origin. The tutor was
# not given the response at all until this existed, so it had nothing but the
# misconception's description to work from and invented a step to match: a
# learner who had multiplied was told their division was correct. The response
# is now in the prompt, and the instruction keeps the model from embroidering
# past it.

#: Said when the model produces nothing usable. A constant because the tutor
#: evaluation counts these apart from bad hints — a failure to produce a turn is
#: not a wrongly-pitched one — and a literal in two places would drift.
FALLBACK_HINT = "That step is not right yet — take another look at it."

#: How much longer than the authored lesson a re-voiced one may run.
#:
#: Generous, because a Socratic account genuinely needs more words than a
#: declarative one. It is here to catch a model that has stopped rephrasing and
#: started composing, not to police style.
_STYLED_LESSON_LIMIT = 3.0

_PLANNER_SYSTEM = (
    "You choose what a student should work on next, given what they have shown "
    "they can and cannot do. Choose exactly one concept from those offered."
)


def _offered(domain: Domain, item: Item, label_space: LabelSpace) -> tuple[Misconception, ...]:
    """The misconceptions this diagnosis may choose between.

    Under ``concept``, only those belonging to the item's own concept. The wide
    space is not neutral: with the whole catalogue on offer the agent will name
    a misconception from an unrelated concept rather than abstain, which a
    learner then reads in the panel as an explanation of their error. It is also
    the harder task, so accuracy under the two is not the same measurement —
    which is why the space is configured and recorded rather than assumed.

    Falls back to the whole catalogue when a concept has no entry at all.
    Offering nothing would leave only ``unknown``, making abstention the sole
    legal reply and the measurement meaningless.
    """
    if label_space == "concept":
        own = tuple(domain.misconceptions.for_concept(item.concept_id))
        if own:
            return own
    return tuple(domain.misconceptions.all())


def _describe(misconceptions: Sequence[Misconception]) -> str:
    return "\n".join(
        f"- {m.id}: {' '.join(m.description.split())}" for m in misconceptions
    )


class LLMDiagnostic:
    """Classifies an incorrect step into the domain's catalogue.

    Deliberately does **not** implement ``OracleAccess``. There is no method by
    which the injected label could reach it.
    """

    def __init__(
        self, provider: LLMProvider, label_space: LabelSpace = "concept"
    ) -> None:
        self._provider = provider
        self._label_space: LabelSpace = label_space
        #: Replies that could not be obtained at all, reported per run. These
        #: are not wrong answers and must not be scored as such.
        self.failures = 0

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis:
        offered = _offered(domain, item, self._label_space)
        schema = diagnosis_schema(domain.name, tuple(m.id for m in offered))
        prompt = (
            f"Exercise: {item.prompt}\n"
            f"Correct answer: {item.answer}\n"
            f"The student wrote: {response}\n\n"
            f"Catalogue of misconceptions:\n{_describe(offered)}\n\n"
            f"Which one explains this step?"
        )
        try:
            reply = complete(self._provider, prompt, schema, system=_DIAGNOSTIC_SYSTEM)
        except ProviderError as exc:
            # Distinct from an incorrect label: nothing was inferred. Recorded
            # so the rate is visible, and returned as no-diagnosis so it cannot
            # be counted as a wrong prediction.
            #
            # ⚠️ `ProviderError` rather than `MalformedResponse`, which it
            # subclasses. A provider that is *unreachable* was not caught here, so
            # a dead or timed-out backend propagated out of the session — and the
            # demo stored nothing, losing the sitting. From this loop's point of
            # view "the model said nonsense" and "the model did not answer" are
            # the same event: nothing was inferred. Which of the two it was goes
            # to the log.
            self.failures += 1
            log.warning(
                "diagnostic produced nothing usable for %s (%s)",
                item.id,
                type(exc).__name__,
                extra={
                    "event": "diagnostic.failed",
                    "item_id": item.id,
                    "reason": type(exc).__name__,
                },
            )
            return Diagnosis(None)

        label = getattr(reply, "misconception_id", UNKNOWN)
        confidence = float(getattr(reply, "confidence", 0.0))
        if label == UNKNOWN:
            return Diagnosis(None, confidence=confidence)
        return Diagnosis(label, confidence=confidence)


class LLMTutor:
    """Writes the hint. Support level and move come from the rules, not the model.

    The model chooses words; it does not choose how much to give away or whether
    remediation may happen yet. Leaving those to a prompt would make the
    instructional constraints advisory, and a constraint a model can talk itself
    out of is not one.
    """

    def __init__(
        self,
        provider: LLMProvider,
        band: ZPDConfig,
        policy: ScaffoldingPolicy = "banded",
    ) -> None:
        self._provider = provider
        self._band = band
        self._policy: ScaffoldingPolicy = policy

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
    ) -> Hint:
        # Both inputs come from the session. Read from the view here, they each
        # carried the failure being responded to — see the Tutor protocol.
        level = hint_level(mastery, prior_failures, self._band, policy=self._policy)
        move = move_for(
            level,
            moves_this_item,
            misconception_confirmed=diagnosis.named,
            already_explained=explained,
        )

        # The length budget belongs to the level, not to the system prompt. It
        # used to say "at most two sentences" globally, which made a worked step
        # impossible: the level asks the tutor to work the step through and the
        # prompt forbade the room to do it, so the model restated the rule in
        # the abstract instead. A learner put a chatbot's answer beside it and
        # said the difference plainly — the other one decomposed the problem and
        # substituted their own expression, and this one did not.
        #
        # ⚠️ No level states the answer, worked step included. It used to be
        # permitted there — "you may state the result" — on the reasoning that
        # working the step through is what the level is for. What that produced
        # was the answer assembled in the reply: `(2x)(x^6 + 2) + (x^2 + 1)(6x^5)`
        # for a question asking exactly that. The learner's verdict was that a
        # worked step should stop short of it, and the mastery estimate agrees —
        # an answer read off a hint and typed back is recorded as knowing it.
        #
        # The answer is still given, at the point where it costs nothing: when
        # the item is over, and the next question on that concept carries
        # different numbers. That reveal was asked for and is worth keeping.
        instruction = {
            # Reached only above `theta_upper`, where the move is always
            # `REFLECT` and this is overwritten below. Present so the mapping is
            # total: a level with no entry would raise at the keyboard, and a
            # `.get` with a default would quietly pitch an unknown level at
            # whatever the default happened to be.
            HintLevel.NONE: (
                "Ask the student to look again at their own step. Do not tell "
                "them anything about it. At most two sentences."
            ),
            HintLevel.NUDGE: (
                "Point at the part of the step that is wrong, without naming it. "
                "At most two sentences."
            ),
            HintLevel.TARGETED: (
                "Name the specific error, but do not give the answer. "
                "At most two sentences."
            ),
            HintLevel.WORKED_STEP: (
                "Work the step through on this student's own expression: name "
                "the rule and prepare each piece it needs, one line at a time. "
                "Then stop. Do not combine the pieces and do not write the "
                "result — putting them together is the step the student has "
                "left to take. Up to six short lines."
            ),
        }[level]
        if move is TutorMove.REFLECT:
            instruction = (
                "Ask the student to look again at their own reasoning and say which "
                "part they are least sure of. Do not correct them yet."
            )

        context = ""
        if diagnosis.named and move is not TutorMove.REFLECT:
            assert diagnosis.misconception_id is not None
            described = domain.misconceptions.get(diagnosis.misconception_id).description
            context = f"\nThe error is: {' '.join(described.split())}"

        # Only what was said about *this* concept. Taking the last two
        # regardless of subject once had the tutor ask a learner differentiating
        # 2/x^2 to revisit their explanation of limits.
        #
        # And each utterance says which question it was about, because filtering
        # by concept alone is not enough: working shown on the previous question
        # arrived as unlabelled context for the next one, and the tutor told a
        # learner to review their derivative of u^4 on a question containing no
        # u^4. Words from an earlier question are still worth having — they are
        # how the tutor knows what this learner keeps finding hard — but they
        # have to be marked as being about something else.
        said = ""
        if isinstance(view, FullStateView):
            for utterance in view.said_about(item.concept_id):
                if utterance.kind == "lesson":
                    # ⚠️ Its own branch, and the reason `Utterance.kind` gained a
                    # third value. A lesson is about a *concept* and often has no
                    # question in front of it, so its utterances carry an empty
                    # item id — which the test below would read as "some other
                    # question" and hand back to the learner as a remark about
                    # work they were not doing. It is the sitting-3 defect one
                    # level finer, and this is where it would have surfaced.
                    label = (
                        "The student said this while this concept was being "
                        "explained to them, not about the question above"
                    )
                else:
                    when = (
                        "on this question"
                        if utterance.item_id == item.id
                        else "on an earlier question, not the one above"
                    )
                    label = (
                        f"The student showed this working {when}"
                        if utterance.kind == "working"
                        else f"The student said {when}, when asked what they were unsure of"
                    )
                said += f"\n{label}: {utterance.text}"

        # What this tutor has already said on this question, and an instruction
        # not to say it again. Two things needed it.
        #
        # The support level escalates on a step that did not resolve the item,
        # so the prompt usually changes on its own — but the level has a
        # ceiling, and at `worked_step` nothing else in the prompt moves when
        # the student's response does not either. Three empty answers to one
        # question therefore produced three byte-identical replies: same item,
        # same response, same level, same move, so the response cache returned
        # its stored text. Correct caching, useless teaching.
        #
        # Kept as an instruction rather than as a retry, because the honest
        # reading of a student who has not got there twice is that the reply did
        # not land — not that it needs to be phrased better in the abstract.
        again = ""
        if said_this_item:
            already = "\n".join(f"- {text}" for text in said_this_item)
            again = (
                f"\n\nYou have already replied on this question, and it did not "
                f"get them there:\n{already}\n"
                f"Do not repeat any of that. Take a different line — be more "
                f"concrete than you were, or start from something earlier."
            )
        prompt = (
            f"Exercise: {item.prompt}\n"
            f"The student wrote: {response}\n"
            f"Correct answer: {item.answer}{context}{said}\n\n"
            f"{instruction}{again}"
        )
        try:
            reply = complete(self._provider, prompt, HintReply, system=_TUTOR_SYSTEM)
            text = reply.text
        except ProviderError:
            # A hint is prose; failing to produce it should not end a session.
            # Falling back keeps the turn's *targeting* intact, which is the
            # part that affects the learner. `ProviderError` covers an
            # unreachable backend as well as a malformed reply — a sitting must
            # survive ollama dying, and the fallback is what lets it.
            text = FALLBACK_HINT

        return Hint(
            text=text,
            move=move,
            level=level,
            # A reflective prompt teaches nothing by design, so it targets
            # nothing; only remediation carries a target.
            targets=diagnosis.misconception_id if move is TutorMove.REMEDIATE else None,
        )

    def explain(
        self,
        resource: ConceptResource,
        style: TeachingStyle,
        exchanges: Sequence[tuple[str, str]] = (),
        closing: bool = False,
    ) -> str:
        """Re-voice the authored lesson in the style the rules chose.

        **The model does not write the lesson.** It is handed text a person
        wrote, that ``domain validate`` has already checked is plain text and
        does not answer any item on the concept at any template draw, and it is
        asked to say the same thing differently. That is deliberate: the
        mathematics a learner is taught should not be generated fresh at a
        keyboard, and the guarantees the content carries are guarantees about
        *that* text.

        A lesson is a conversation and this writes one side of it. With no
        ``exchanges`` it opens — a little, and then a question the learner can
        answer. With exchanges it replies to what they said and puts the next
        piece to them. Neither turn delivers the whole account: the learner is
        given that in writing when the conversation ends, and it is the authored
        text rather than anything generated here.

        Two guards on what comes back, and a fallback to the authored text if
        either trips:

        * **Plain text.** The LaTeX ban is the sitting-2 defect — a reply
          arrives as JSON, ``\\f`` parses to a form feed, and a learner read
          ``rac{f(b) - f(a)}{b - a}`` without being able to tell it meant a
          division. Checked against the same pattern the content is checked
          against, so a fix to one is a fix to both.
        * **Length.** A re-voicing that runs to several times the original has
          stopped re-voicing and started writing, which is the thing this is
          built not to do.

        ⚠️ What is *not* guaranteed: that a generated turn avoids answering an
        item. The authored example is checked against every item and every draw;
        a model talking around it could in principle arrive at an item's
        numbers. Re-checking every turn against every form of every item on the
        concept is affordable and is not done here. The honest statement is that
        the **summary** carries the guarantee, because the summary is the
        authored text, and the conversation inherits it only as far as "do not
        add mathematics that is not in it" is obeyed.
        """
        authored = resource.lesson()
        if exchanges:
            said = "\n".join(
                f"You: {mine}\nThe student: {theirs}" for mine, theirs in exchanges
            )
            context = f"\n\nThe conversation so far:\n{said}"
        else:
            context = ""
        schema: type[BaseModel] = LessonReply
        if closing:
            instruction = _STYLE_CLOSING
            # A rule a model can talk itself out of is not one. This turn is
            # refused if it ends on a question, and the repair loop shows the
            # model its own reply and asks again — the same machinery that
            # catches a turn stopping mid-sentence.
            schema = ClosingReply
        elif exchanges:
            instruction = _STYLE_REPLY
        else:
            instruction = _STYLE_OPENING[style]

        prompt = f"{instruction}\n\nThe explanation to work from:\n{authored}{context}"
        try:
            reply = complete(
                self._provider, prompt, schema, system=_EXPLAIN_SYSTEM
            )
            text = reply.text
        except ProviderError:
            # A sitting must survive a dead backend, and the authored lesson is
            # a complete lesson rather than a degraded one — the style was the
            # only thing lost.
            return authored

        if PLAIN_TEXT_ONLY.search(text):
            log.warning(
                "styled lesson for %s came back with a backslash command; "
                "using the authored text",
                resource.concept_id,
                extra={"event": "tutor.lesson_rejected", "reason": "not_plain_text"},
            )
            return authored
        # Measured against the authored account, which a single conversational
        # turn should come in well under rather than near. It is here to catch a
        # model that has abandoned the conversation and delivered the lecture,
        # which is the failure this design exists to avoid.
        if len(text) > _STYLED_LESSON_LIMIT * len(authored):
            log.warning(
                "styled lesson for %s ran to %d characters against an authored "
                "%d; using the authored text",
                resource.concept_id,
                len(text),
                len(authored),
                extra={"event": "tutor.lesson_rejected", "reason": "over_length"},
            )
            return authored
        return text


#: How each style **opens** a lesson.
#:
#: ⚠️ These used to say how to *voice* a finished account, and the Socratic one
#: said to answer each question "yourself in a line before asking the next". The
#: model did exactly that and produced a monologue shaped like a dialogue — a
#: learner watched it ask and answer its own questions and said so: *"I really
#: thought that would be more of a dialogue between me and the Tutor."* The
#: clause that caused it is gone, and every style now ends by putting something
#: to the learner that they can actually reply to.
_STYLE_OPENING = {
    TeachingStyle.PLAIN: (
        "Say plainly what this concept is, in two or three sentences. Then ask "
        "the student one short question to find out where they are with it."
    ),
    TeachingStyle.SOCRATIC: (
        "Do not explain it yet. Ask the student one short question that starts "
        "them towards the idea — something they can have a go at from what they "
        "already know. One question only, and then stop."
    ),
    TeachingStyle.REAL_WORLD: (
        "Open with one concrete situation where this idea is actually used, in "
        "two or three sentences. Then ask the student one short question "
        "connecting that situation to the mathematics."
    ),
}

#: How it **ends**.
#:
#: ⚠️ Every other turn ends by asking something, so without this a lesson always
#: stopped on a question nobody answered — and the written summary then answered
#: it for the learner. A sitting caught it at the worst possible moment: the
#: tutor had just asked *"what do you think happens to the gradient of that
#: secant as it moves closer to the first one?"*, which is the limit concept
#: itself, and the summary appeared instead of a reply. Their words: *"I was
#: just about to understand something important."*
#:
#: Under the Socratic style that is the monologue failure returning by another
#: door — the system asks the question and then answers it.
_STYLE_CLOSING = (
    "This is the last thing you will say. Answer the question you left hanging, "
    "briefly, taking up whatever the student worked out. Then stop. Do not ask "
    "anything new — they have no way to reply to it."
)

#: How it **continues**, once the student has said something back.
#:
#: The last line is the only place the tutor may say anything about *ending*,
#: and it may only say it — the learner ends a lesson and nothing here does. A
#: sitting drew that line: "I don't think that the llm should decide when to
#: quit the dialogue, but it could probably recommend the student to continue
#: after it has noticed that the student is getting the concept."
#:
#: It fits the rule the rest of the tutor already follows. A model may say
#: things; it may not decide them. Whether a lesson continues is not read off
#: this text by anything — the loop asks the learner, every turn, and the
#: learner answers or does not.
_STYLE_REPLY = (
    "Reply to what the student just said. Take up what they got right, and put "
    "the next small piece to them as a question they can answer. Two or three "
    "sentences, then the question. Do not deliver the whole explanation — they "
    "will be given it in writing when you are done.\n"
    "If their answers show they have got the idea, say so plainly and add that "
    "they can stop here or keep going, as they prefer. Say it and then carry on "
    "as normal — it is their decision, not yours, and you never end the "
    "conversation yourself."
)

_EXPLAIN_SYSTEM = (
    "You are a mathematics tutor talking a student through a concept they may "
    "never have met. You are given an explanation that has already been written "
    "and checked; use it as the ground you are working from. Do not add "
    "mathematics that is not in it, do not correct it, and do not extend it.\n"
    "This is a conversation: the student does some of the thinking, so never "
    "deliver the whole explanation at once and never answer your own question "
    "in the same breath as asking it.\n"
    "The instruction you are given says what this particular turn is for. "
    "Follow it.\n"
    "Write mathematics in plain text — (f(b) - f(a)) / (b - a), x^2, sqrt(x). "
    "Never use LaTeX or backslash commands."
)
# ⚠️ "Say a little and then ask" used to live in the line above, and it made the
# closing instruction unfollowable: asked to stop asking, the model asked anyway,
# because the system prompt told it to on every turn. Whether a turn asks
# something belongs to the turn.
#
# That is the fourth time one instruction has contradicted another here.
# `_TUTOR_SYSTEM` once demanded two sentences while `WORKED_STEP` asked for the
# step to be worked through; `HintReply`'s field description carried the same
# demand one layer further down. The pattern is always a global rule outliving
# the case it was written for.


_CONFUSION_SYSTEM = (
    "You are reading one thing a mathematics student wrote, to answer a single "
    "question about it: do they say they do not know what the concept IS?\n\n"
    "True means they are telling you the idea itself is missing — they do not "
    "know what the thing is or what it means, so there is nothing for them to "
    "apply.\n"
    "False means anything else, and this is the harder half:\n"
    "  - attempting the work and getting it wrong -> false\n"
    "  - using a wrong method confidently -> false\n"
    "  - not being sure their answer is right -> false\n"
    "  - hedging: 'I think', 'not totally sure', 'maybe' -> false\n"
    "  - saying a step was hard, or that they found it confusing -> false\n\n"
    "Someone who describes a method, even a wrong one, has met the concept. "
    "Being unsure of an answer is not the same as not knowing what the question "
    "is about, and a student who says both is doing the work.\n"
    "If it is true, copy the words that say so, exactly."
)


class LLMConfusionDetector:
    """Asks a model whether the learner said they do not understand the concept.

    Narrow on purpose. It decides nothing instructional: whether a lesson
    happens, which account it takes and when it stops are all still rules. What
    it produces is one fact about one string, written to the board like any
    other observation.

    ``detections`` and ``checks`` are reported per run. That is not
    bookkeeping — a detector that fires on ordinary mistakes would teach a
    learner who was doing fine, and the only thing that would show it is the
    rate. It is the same reading `UNPARSEABLE` gets: a rising number is a fact
    about the instrument, not about the learner.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.checks = 0
        self.detections = 0

    @property
    def rate(self) -> float:
        return self.detections / self.checks if self.checks else 0.0

    def confused(self, concept_id: str, text: str) -> str | None:
        if not text.strip():
            return None
        self.checks += 1
        prompt = (
            f"The student is working on: {concept_id}\n"
            f"They wrote:\n{text}\n\n"
            f"Do they say they do not know what this concept is?"
        )
        try:
            reply = complete(
                self._provider, prompt, ConfusionReply, system=_CONFUSION_SYSTEM
            )
        except ProviderError:
            # Not knowing is not the same as "no", but it has to become one
            # somewhere: a sitting must survive a dead backend, and the fallback
            # here costs a lesson that would have been offered rather than
            # giving one that should not have been. The failure is logged so the
            # rate stays honest.
            log.warning(
                "confusion detector produced nothing usable for %s",
                concept_id,
                extra={"event": "confusion.failed", "concept_id": concept_id},
            )
            return None
        if not reply.confused:
            return None
        self.detections += 1
        # The quote, when there is one, so the audit log records *what* was read
        # that way. Falling back to the text itself rather than to True keeps
        # the evidence readable either way.
        return reply.quote.strip() or text.strip()


class LLMPlanner:
    """Proposes the next concept; a guardrail decides whether it stands.

    The model sees only what its arm's view carries. The guardrail is the
    deterministic goal-directed planner: if the proposal is outside the band, or
    off the way to the goal, or names a concept with no material left, the
    guardrail's own choice is used instead and the override is counted.

    **The model does not choose the goal.** Which target comes next is settled
    by the domain's declared order and the posteriors, so letting a model decide
    it would add an uncontrolled variable to the one decision the whole
    comparison is framed around. The model's latitude is where to go *next*,
    within the route.
    """

    def __init__(
        self,
        provider: LLMProvider,
        band: ZPDConfig,
        prior: float,
        emphasis: Emphasis = Emphasis.CONSOLIDATE,
    ) -> None:
        self._provider = provider
        self._band = band
        self._prior = prior
        self._fallback = GoalDirectedPlanner(band, prior, emphasis)
        #: Proposals replaced by the guardrail. Reported per run: a high rate
        #: means the model is not usefully planning, whatever the outcome says.
        self.overrides = 0
        self.proposals = 0

    @property
    def override_rate(self) -> float:
        return self.overrides / self.proposals if self.proposals else 0.0

    def plan(self, view: StateView, domain: Domain) -> Plan | None:
        return self._fallback.plan(view, domain)

    def select(
        self, view: StateView, domain: Domain, given: Mapping[str, int]
    ) -> Item | None:
        if not isinstance(view, FullStateView):
            raise TypeError("LLMPlanner requires the full state view")

        guarded = self._fallback.select(view, domain, given)
        if guarded is None:
            return None  # nothing selectable at all; not the model's call to make

        # Only concepts on the way to the goal are offered. A proposal outside
        # the route is not a judgement call the model gets to make.
        offered = (
            list(route.candidates(view.plan.goal, view.frontier, domain.concepts))
            if view.plan is not None
            else sorted(view.frontier)
        )
        if not offered:
            return guarded

        self.proposals += 1
        schema = plan_schema(domain.name, tuple(domain.concepts.ids()))
        goal_line = f"Working toward: {view.plan.goal}\n" if view.plan else ""
        prompt = (
            f"{goal_line}"
            f"Concepts the student is ready for: {', '.join(offered)}\n"
            f"Mastery estimates: "
            f"{', '.join(f'{c}={view.probability(c, 0.0):.2f}' for c in offered)}\n"
            f"Recent misconceptions: "
            f"{', '.join(view.recent_misconceptions(5)) or 'none'}\n\n"
            f"Which of the offered concepts should they work on next?"
        )
        try:
            reply = complete(self._provider, prompt, schema, system=_PLANNER_SYSTEM)
            proposed = getattr(reply, "concept_id")
        except ProviderError:
            self.overrides += 1
            return guarded

        if proposed not in offered or may_select(proposed, view.frontier) is not None:
            self.overrides += 1
            log.info(
                "planner proposed concept %s, which is not on offer", proposed,
                extra={"event": "planner.override", "proposed": proposed},
            )
            return guarded

        item = _least_used(domain, proposed, given)
        if item is None:
            self.overrides += 1
            return guarded
        return item
