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

from typing import Any, Literal

from pydantic import BaseModel, Field

#: What produced a state change. Recorded on every audit entry.
Cause = Literal["observation", "replan", "reset", "annotation"]


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
