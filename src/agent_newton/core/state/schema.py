"""The shared learner state.

Concept and misconception identifiers are opaque strings throughout. Nothing
here knows what a derivative is, which is what lets the same state serve any
domain.

The state is versioned and every mutation is recorded. Two things depend on
that: the frontier is cached per version so all agents see one consistent zone
within a step, and every planning decision can be traced back to the evidence
that caused it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

#: What produced a state change. Recorded on every audit entry.
#:
#: ``plan`` is separate from ``replan``: setting a goal is not a replanning
#: trigger, and folding it in would put it among the trigger counts that the
#: threshold analysis reads.
#:
#: ``seed`` is separate from ``observation`` for the same reason. Both move a
#: posterior, but a seeded update comes from a held-out test rather than from
#: practice, so anything counting what the learner did during the session must
#: be able to leave it out.
#: ``decay`` is separate again, for the same reason: it moves posteriors without
#: the learner having done anything at all. Anything counting evidence must be
#: able to leave it out, and anything asking why an estimate fell must be able to
#: tell "they got it wrong" from "time passed".
#:
#: ``tutor`` is what the system said back. It moves no estimate, so it could have
#: been an annotation; it is its own cause because a reader looking for what the
#: tutor did should not have to sift it out of everything else that changes
#: nothing, and because a transcript without it cannot show what the learner was
#: actually taught.
Cause = Literal[
    "observation",
    "seed",
    "decay",
    "plan",
    "replan",
    "reset",
    "annotation",
    "tutor",
    # An inference *about* the learner drawn from the graph rather than from
    # something they did. Its own cause so anything counting what the learner
    # did can leave it out — it is not an observation, and reporting it as one
    # would credit them with an attempt nobody made.
    "doubt",
]


class Emphasis(str, Enum):
    """How a learner wants to get to their goal.

    Both orderings are legitimate, and the difference between them is real
    because the mastery band has width. A concept opens once its prerequisites
    clear ``theta_lower`` but stays selectable until it clears ``theta_upper``,
    so there is a range in which a learner may either move on or stay.

    ``CONSOLIDATE`` stays, working wherever the error trace shows difficulty.
    ``ADVANCE`` moves on to the deepest concept now reachable, leaving
    partly-mastered ones behind in favour of progress toward the goal.
    """

    CONSOLIDATE = "consolidate"
    ADVANCE = "advance"


class ErrorEvent(BaseModel):
    """One incorrect step, as it enters the rolling trace."""

    t: int
    item_id: str
    concept_id: str
    #: None when the diagnostic could not name a misconception. Distinct from a
    #: label of "unknown": absence means nothing was inferred.
    misconception_label: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: The verifier's verdict, carried so analysis can separate genuine errors
    #: from responses that merely failed to parse.
    verifier_label: str = "incorrect"


class Utterance(BaseModel):
    """Something the learner said in words, and what it was about.

    Two kinds, both prose and neither evidence: a ``reflection`` answers a
    question the tutor asked, and ``working`` is the steps the learner took,
    volunteered unprompted. Kept apart because the tutor should address them
    differently — one is a reply, the other is a record of reasoning.

    ``concept_id`` is here because it was once dropped. Only the text reached
    the state, so the tutor was handed whatever had been said most recently
    regardless of subject, and asked a learner working on ``y = 2/x^2`` to
    reconsider their explanation of limits.
    """

    text: str
    item_id: str
    concept_id: str
    kind: Literal["reflection", "working"] = "reflection"


class AuditRecord(BaseModel):
    """One entry in the append-only log.

    ``evidence`` holds whatever justified the change. For a replan that is the
    trigger and its inputs, which is what makes a decision reconstructible after
    the run rather than merely observable during it.
    """

    version: int
    cause: Cause
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """What the learner is working toward, and how.

    Deliberately small. It holds the *target* and nothing about the route,
    because the route is derived from the rest of the state every time it is
    needed — see :mod:`agent_newton.core.state.route`. Freezing a sequence of
    concepts here would make the path curriculum-driven again, which is the
    thing the derived route exists to avoid.
    """

    #: The terminal concept currently aimed at.
    goal: str
    emphasis: Emphasis = Emphasis.CONSOLIDATE
    #: State version at which this plan was set, so the audit log can be read
    #: back against the estimates that justified it.
    set_at_version: int = 0
    reason: str = ""


class LearnerState(BaseModel):
    """Per-learner state: mastery estimates, recent errors, provenance.

    Mutated only through :class:`~agent_newton.core.state.store.Blackboard`, so
    that no path exists which changes state without bumping the version and
    writing an audit entry.
    """

    learner_id: str
    seed: int

    #: concept_id -> P(mastery). Absent concepts have not been observed and are
    #: treated as sitting at the BKT prior.
    mastery: dict[str, float] = Field(default_factory=dict)

    #: Most recent errors, oldest first, bounded by the configured length.
    error_trace: list[ErrorEvent] = Field(default_factory=list)

    #: item_id -> times this learner has been given it, over their whole
    #: history. Session-local until sessions could span sittings, and moving it
    #: here is not bookkeeping tidiness: the simulated learner's answer is a
    #: function of the repetition index, and the item templates draw their
    #: variant from it. A count that restarted each session would replay session
    #: one's answers *and* its questions — the same defect the rule engine fixed
    #: at P6, returning by a different door.
    items_given: dict[str, int] = Field(default_factory=dict)

    #: Item-level correctness, oldest first. This is the whole of what the
    #: decoupled view carries, so it lives on the state rather than on the
    #: blackboard: a learner resuming a sequence must bring it with them, or the
    #: decoupled arm would restart its run of consecutive answers every session
    #: while the coupled arm resumed with everything. That difference would not
    #: be the manipulation — it would be one arm being handed persistence.
    outcomes: list[bool] = Field(default_factory=list)

    #: What the learner said, in words, most recent last. Prose, not evidence:
    #: it updates no estimate and enters no error trace, because it is not a
    #: graded step. It is here so the tutor can read back what the learner said
    #: they were unsure of, and the working they showed.
    reflections: list[Utterance] = Field(default_factory=list)

    #: What this learner is working toward. None before the first plan is set.
    #: This is the shared representation of goals: it lives in the state every
    #: agent reads, not in the loop that happens to be running.
    plan: Plan | None = None

    #: Monotonic. Bumped by every mutation; the frontier cache keys on it.
    version: int = 0

    #: Steps taken, used as the timestamp on error events.
    t: int = 0

    def probability(self, concept_id: str, prior: float) -> float:
        """Mastery estimate, falling back to the prior for unseen concepts."""
        return self.mastery.get(concept_id, prior)

    def recent_misconceptions(self, window: int | None = None) -> list[str]:
        """Labels from the trace, most recent last, skipping unlabelled events."""
        events = self.error_trace if window is None else self.error_trace[-window:]
        return [e.misconception_label for e in events if e.misconception_label]

    def misconception_count(self, misconception_id: str, window: int | None = None) -> int:
        """How often one misconception appears in the window.

        The arbitration policy's ``k_repeats`` trigger reads this.
        """
        return self.recent_misconceptions(window).count(misconception_id)
