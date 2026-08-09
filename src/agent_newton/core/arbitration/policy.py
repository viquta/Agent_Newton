"""When new evidence may revise the plan.

Without this, the planner is consulted after every item and the plan changes
continuously — which makes the coupling trivially total and leaves nothing to
measure. The policy decides *when* the plan is reopened, so the threshold
governs how readily real-time evidence reaches long-horizon planning.

Three triggers, each a different kind of evidence:

* ``frontier_crossed`` — the concept being worked has left the zone. The
  strongest signal: there is nothing further to do here.
* ``mastery_delta`` — the estimate has moved by more than ``theta`` since the
  plan was set. This is the threshold the sensitivity analysis sweeps.
* ``misconception_repeat`` — the same misconception has recurred ``k_repeats``
  times. Evidence that the current work is not landing.

Two guardrails, which suppress a trigger rather than create one:

* **Rate limit.** At least ``min_items_between_replans`` items must have been
  worked. Without it, a threshold set low enough to be sensitive also makes the
  planner thrash between concepts on single observations.
* **Verifier confirmation.** Only verifier-confirmed errors count toward the
  repeat trigger. A diagnostic agent's label is an opinion about an error; the
  verifier is what establishes there was one. Letting a model's say-so alone
  move the plan would let diagnostic error propagate straight into planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_newton.config import ArbitrationConfig
from agent_newton.core.state.schema import ErrorEvent
from agent_newton.core.state.zpd import Frontier

NO_PLAN = "no_plan"
FRONTIER_CROSSED = "frontier_crossed"
MASTERY_DELTA = "mastery_delta"
MISCONCEPTION_REPEAT = "misconception_repeat"
RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether to replan, and the evidence either way.

    Carried into the audit log whole. A replan that cannot be explained after
    the fact is not auditable, and the audit trail is the point.
    """

    replan: bool
    trigger: str | None = None
    suppressed_by: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.replan:
            return f"replan triggered by {self.trigger}"
        if self.suppressed_by:
            return f"{self.trigger} suppressed by {self.suppressed_by}"
        return "no trigger"


class ArbitrationPolicy:
    """Holds the baseline the mastery delta is measured against."""

    def __init__(self, config: ArbitrationConfig) -> None:
        self._config = config
        self._baseline: dict[str, float] = {}
        self._items_since_replan = 0
        self.replans = 0
        #: Triggers that fired but were suppressed. A high count means the rate
        #: limit is doing the deciding rather than the threshold, which matters
        #: for interpreting a threshold sweep.
        self.suppressed = 0

    def note_item(self) -> None:
        self._items_since_replan += 1

    def accept(self, mastery: dict[str, float]) -> None:
        """Record that a replan happened; reset the baseline and the rate limit."""
        self._baseline = dict(mastery)
        self._items_since_replan = 0
        self.replans += 1

    def evaluate(
        self,
        *,
        current_concept: str | None,
        mastery: dict[str, float],
        frontier: Frontier,
        error_trace: list[ErrorEvent],
        prior: float,
    ) -> Decision:
        """Decide whether the plan may be reopened."""
        if current_concept is None:
            return Decision(True, trigger=NO_PLAN)

        # Nothing left to work on here: not suppressible, because continuing
        # would mean giving items for a concept that has left the zone.
        if current_concept not in frontier:
            return Decision(
                True,
                trigger=FRONTIER_CROSSED,
                evidence={
                    "concept": current_concept,
                    "mastery": mastery.get(current_concept, prior),
                },
            )

        trigger, evidence = self._find_trigger(current_concept, mastery, error_trace, prior)
        if trigger is None:
            return Decision(False)

        if self._items_since_replan < self._config.min_items_between_replans:
            self.suppressed += 1
            return Decision(
                False,
                trigger=trigger,
                suppressed_by=RATE_LIMITED,
                evidence={
                    **evidence,
                    "items_since_replan": self._items_since_replan,
                    "required": self._config.min_items_between_replans,
                },
            )

        return Decision(True, trigger=trigger, evidence=evidence)

    def _find_trigger(
        self,
        concept: str,
        mastery: dict[str, float],
        error_trace: list[ErrorEvent],
        prior: float,
    ) -> tuple[str | None, dict[str, Any]]:
        before = self._baseline.get(concept, prior)
        after = mastery.get(concept, prior)
        delta = abs(after - before)
        if delta > self._config.theta:
            return MASTERY_DELTA, {
                "concept": concept,
                "before": before,
                "after": after,
                "delta": delta,
                "theta": self._config.theta,
            }

        window = error_trace[-self._config.error_trace_length :]
        # Only verifier-confirmed errors count. A diagnostic label is an opinion
        # about an error; the verifier is what establishes there was one.
        confirmed = [e for e in window if e.verifier_label == "incorrect"]
        counts: dict[str, int] = {}
        for event in confirmed:
            if event.misconception_label:
                counts[event.misconception_label] = counts.get(event.misconception_label, 0) + 1

        for label, count in sorted(counts.items()):
            if count >= self._config.k_repeats:
                return MISCONCEPTION_REPEAT, {
                    "misconception": label,
                    "count": count,
                    "k_repeats": self._config.k_repeats,
                }

        return None, {}
