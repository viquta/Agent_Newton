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

from agent_newton.config import LabelSpace, ZPDConfig
from agent_newton.core.agents.base import Diagnosis, Hint, StateView
from agent_newton.core.agents.planner import GoalDirectedPlanner, _least_used
from agent_newton.core.state import route
from agent_newton.core.state.schema import Emphasis, Plan
from agent_newton.core.agents.schemas import UNKNOWN, diagnosis_schema, plan_schema
from agent_newton.core.agents.schemas import HintReply
from agent_newton.core.pedagogy import (
    HintLevel,
    TutorMove,
    hint_level,
    may_select,
    next_required_move,
)
from agent_newton.core.state.views import FullStateView
from agent_newton.llm.base import LLMProvider, MalformedResponse, complete
from agent_newton.domains.base import Domain, Item, Misconception

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
        except MalformedResponse:
            # Distinct from an incorrect label: nothing was inferred. Recorded
            # so the rate is visible, and returned as no-diagnosis so it cannot
            # be counted as a wrong prediction.
            self.failures += 1
            log.warning(
                "diagnostic produced nothing usable for %s", item.id,
                extra={"event": "diagnostic.failed", "item_id": item.id},
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

    def __init__(self, provider: LLMProvider, band: ZPDConfig) -> None:
        self._provider = provider
        self._band = band

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
        level = hint_level(mastery, prior_failures, self._band)
        required = next_required_move(
            moves_this_item,
            misconception_confirmed=diagnosis.named,
            already_explained=explained,
        )
        move = required or (TutorMove.REMEDIATE if diagnosis.named else TutorMove.HINT)

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
        except MalformedResponse:
            # A hint is prose; failing to produce it should not end a session.
            # Falling back keeps the turn's *targeting* intact, which is the
            # part that affects the learner.
            text = FALLBACK_HINT

        return Hint(
            text=text,
            move=move,
            level=level,
            # A reflective prompt teaches nothing by design, so it targets
            # nothing; only remediation carries a target.
            targets=diagnosis.misconception_id if move is TutorMove.REMEDIATE else None,
        )


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
        except MalformedResponse:
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
