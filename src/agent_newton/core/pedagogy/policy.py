"""Instructional rules, expressed as checkable predicates.

Teaching rules stated as prose cannot be enforced or tested. Stated as
predicates over the shared state they can be: every one below is a function the
tutor and the arbitration layer must satisfy, each returns a
:class:`Violation` rather than merely failing, and each violation carries enough
detail to be written to the audit log.

Four rules:

* **Band membership** — an item may be selected only if its concept is in the
  current frontier.
* **Scaffolding** — how much support a hint gives is chosen from the mastery
  estimate and how many steps on this item have already failed to resolve it.
* **Fading** — support is non-increasing in mastery, all else equal. This is a
  monotonicity property, so it can be checked across a grid rather than
  spot-checked.
* **Error first** — after a misconception is confirmed, the learner is prompted
  to reflect before being given the remediation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Sequence

from agent_newton.config import ZPDConfig
from agent_newton.core.state.zpd import Frontier


class HintLevel(IntEnum):
    """How much support a hint carries. Higher means more.

    Ordered so that "support falls as mastery rises" is an ordinary comparison
    on the value, which is what makes the fading property checkable.
    """

    NUDGE = 0  # points at the region of the error without naming it
    TARGETED = 1  # names the misconception
    WORKED_STEP = 2  # shows the step

    @property
    def label(self) -> str:
        return self.name.lower()


class TutorMove(str, Enum):
    """What the tutor does on a turn."""

    HINT = "hint"
    REFLECT = "reflect"
    REMEDIATE = "remediate"


@dataclass(frozen=True, slots=True)
class Violation:
    """A rule that was broken, and the detail needed to audit it."""

    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


BAND_MEMBERSHIP = "band_membership"
ERROR_FIRST = "error_first"


def may_select(concept_id: str, frontier: Frontier) -> Violation | None:
    """Whether an item on this concept may be given to the learner.

    The planner's guardrail layer calls this on every proposal. A concept
    outside the frontier is either already mastered or has an unmet
    prerequisite; in both cases the item is not the one to give next.
    """
    if concept_id in frontier:
        return None
    return Violation(
        BAND_MEMBERSHIP,
        f"concept {concept_id!r} is outside the current frontier "
        f"({', '.join(frontier) or 'empty'})",
    )


def hint_level(
    mastery: float,
    unresolved_steps: int,
    band: ZPDConfig,
) -> HintLevel:
    """How much support to give.

    Two inputs. The mastery estimate sets the baseline: a learner near the top
    of the band needs a nudge, one near the bottom needs the step worked. Steps
    on the *current* item that did not resolve it escalate from there, so a
    learner who is stuck is not nudged repeatedly.

    ``unresolved_steps`` counts steps rather than attempts, and the two are no
    longer the same thing: a response the verifier could not read costs no
    attempt — it is a failure to measure, not a wrong answer — but it still
    leaves the learner not having got there. Support escalates on it. Saying
    the same thing again to someone who has now not answered twice is the one
    reply that is certainly not working.

    Escalation is why fading is stated "all else equal" — it varies support at
    fixed mastery, and would otherwise look like a violation.
    """
    if mastery > band.theta_lower:
        base = HintLevel.NUDGE
    elif mastery > band.theta_lower / 2:
        base = HintLevel.TARGETED
    else:
        base = HintLevel.WORKED_STEP

    escalated = int(base) + max(0, unresolved_steps)
    return HintLevel(min(escalated, int(HintLevel.WORKED_STEP)))


def check_fading(
    band: ZPDConfig, unresolved_steps: int = 0, steps: int = 50
) -> Violation | None:
    """Verify support is non-increasing in mastery across the whole range.

    Checked as a property rather than at sample points, because the rule is a
    claim about the shape of the function and a spot check would not catch a
    single inverted step.
    """
    previous = HintLevel.WORKED_STEP
    for index in range(steps + 1):
        mastery = index / steps
        level = hint_level(mastery, unresolved_steps, band)
        if level > previous:
            return Violation(
                "fading",
                f"support rose from {previous.label} to {level.label} as mastery "
                f"reached {mastery:.2f}; it must never increase with mastery",
            )
        previous = level
    return None


def check_move(
    move: TutorMove,
    moves_since_confirmation: Sequence[TutorMove],
    misconception_confirmed: bool,
) -> Violation | None:
    """Whether this tutor move is permitted now.

    Once a misconception is confirmed, remediation must be preceded by a
    reflective prompt: the learner is asked to look at their own reasoning
    before being handed the correction. Without the ordering constraint a tutor
    can skip straight to the answer, which is the behaviour this rule exists to
    prevent.

    ``moves_since_confirmation`` is the tutor's turns on this item since the
    misconception was confirmed, oldest first.
    """
    if move is not TutorMove.REMEDIATE or not misconception_confirmed:
        return None
    if TutorMove.REFLECT in moves_since_confirmation:
        return None
    return Violation(
        ERROR_FIRST,
        "remediation was offered before any reflective prompt, after a "
        "confirmed misconception",
    )


def next_required_move(
    moves_since_confirmation: Sequence[TutorMove],
    misconception_confirmed: bool,
) -> TutorMove | None:
    """The move the rules require next, if any.

    Lets the tutor be driven by the constraint rather than checked against it
    afterwards.
    """
    if not misconception_confirmed:
        return None
    if TutorMove.REFLECT not in moves_since_confirmation:
        return TutorMove.REFLECT
    return None
