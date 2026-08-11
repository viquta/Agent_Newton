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

from agent_newton.config import ZPDConfig
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
from agent_newton.domains.base import Domain, Item

log = logging.getLogger(__name__)

_DIAGNOSTIC_SYSTEM = (
    "You identify the single misconception behind a student's incorrect step in "
    "a mathematics exercise. Work in three stages: list the catalogue entries "
    "consistent with what the student wrote; distinguish between them using the "
    "specific error made; then commit to one. Choose 'unknown' if no entry fits "
    "— a forced guess is worse than an admission."
)

_TUTOR_SYSTEM = (
    "You are a mathematics tutor. Reply with at most two sentences, addressed to "
    "the student. Never state the final answer unless explicitly asked to. "
    "Write mathematics in plain text — (f(b) - f(a)) / (b - a), x^2, sqrt(x). "
    "Never use LaTeX or backslash commands."
)
# The LaTeX ban is not a style preference. Replies arrive as JSON, and a
# backslash command inside a JSON string is eaten by escape processing:
# "\frac{a}{b}" parses to a form-feed character followed by "rac{a}{b}", which
# a learner sees as "$rac{a}{b}". Observed in a human session, where it hid the
# division the hint was trying to explain.

_PLANNER_SYSTEM = (
    "You choose what a student should work on next, given what they have shown "
    "they can and cannot do. Choose exactly one concept from those offered."
)


def _catalogue(domain: Domain) -> str:
    return "\n".join(
        f"- {m.id}: {' '.join(m.description.split())}" for m in domain.misconceptions.all()
    )


class LLMDiagnostic:
    """Classifies an incorrect step into the domain's catalogue.

    Deliberately does **not** implement ``OracleAccess``. There is no method by
    which the injected label could reach it.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        #: Replies that could not be obtained at all, reported per run. These
        #: are not wrong answers and must not be scored as such.
        self.failures = 0

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis:
        schema = diagnosis_schema(domain.name, tuple(domain.misconceptions.ids()))
        prompt = (
            f"Exercise: {item.prompt}\n"
            f"Correct answer: {item.answer}\n"
            f"The student wrote: {response}\n\n"
            f"Catalogue of misconceptions:\n{_catalogue(domain)}\n\n"
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
        failed_attempts: int,
        moves_this_item: Sequence[TutorMove],
    ) -> Hint:
        mastery = (
            view.probability(item.concept_id, 0.0)
            if isinstance(view, FullStateView)
            else 0.0
        )
        level = hint_level(mastery, failed_attempts, self._band)
        required = next_required_move(moves_this_item, misconception_confirmed=diagnosis.named)
        move = required or (TutorMove.REMEDIATE if diagnosis.named else TutorMove.HINT)

        instruction = {
            HintLevel.NUDGE: "Point at the part of the step that is wrong, without naming it.",
            HintLevel.TARGETED: "Name the specific error, but do not give the answer.",
            HintLevel.WORKED_STEP: "Explain the error and work the step through.",
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

        said = (
            "\nThe student said, when asked what they were unsure of: "
            + "; ".join(view.reflections[-2:])
            if isinstance(view, FullStateView) and view.reflections
            else ""
        )
        prompt = (
            f"Exercise: {item.prompt}\n"
            f"Correct answer: {item.answer}{context}{said}\n\n"
            f"{instruction}"
        )
        try:
            reply = complete(self._provider, prompt, HintReply, system=_TUTOR_SYSTEM)
            text = reply.text
        except MalformedResponse:
            # A hint is prose; failing to produce it should not end a session.
            # Falling back keeps the turn's *targeting* intact, which is the
            # part that affects the learner.
            text = "That step is not right yet — take another look at it."

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
