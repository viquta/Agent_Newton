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
  estimate as it stood when the question was posed, and how many readable
  attempts at this item have already failed.
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
    prior_failures: int,
    band: ZPDConfig,
) -> HintLevel:
    """How much support to give.

    Two inputs, and they must stay independent or the same failure raises the
    level twice.

    ``mastery`` is what was believed **when the question was posed** — not what
    is believed after the answer being responded to has been folded in. One
    wrong answer at 0.40 moves the posterior to 0.26, which is below
    ``theta_lower / 2`` and therefore a full worked step on its own; adding an
    escalation on top of that counts the same failure in both inputs. Two human
    sittings received a worked step on every single turn that way.

    ``prior_failures`` counts readable answers on this item that were already
    wrong **before** this one. Excluding the current step is what makes the
    mastery baseline reachable: the tutor is only ever called after a failure,
    so a count that included it meant the baseline was the one level a learner
    could never be given. ``nudge`` was unreachable outright.

    Readable only. A response the verifier could not read is a failure to
    measure, not a wrong answer, and support is earned on work that could be
    read: an unreadable step that escalated bought a full worked step — answer
    included — for a blank submission, at no cost, which is a way to be handed
    the answer rather than a way to be taught.

    Escalation is why fading is stated "all else equal" — it varies support at
    fixed mastery, and would otherwise look like a violation.
    """
    if mastery > band.theta_lower:
        base = HintLevel.NUDGE
    elif mastery > band.theta_lower / 2:
        base = HintLevel.TARGETED
    else:
        base = HintLevel.WORKED_STEP

    escalated = int(base) + max(0, prior_failures)
    return HintLevel(min(escalated, int(HintLevel.WORKED_STEP)))


def check_fading(
    band: ZPDConfig, prior_failures: int = 0, steps: int = 50
) -> Violation | None:
    """Verify support is non-increasing in mastery across the whole range.

    Checked as a property rather than at sample points, because the rule is a
    claim about the shape of the function and a spot check would not catch a
    single inverted step.
    """
    previous = HintLevel.WORKED_STEP
    for index in range(steps + 1):
        mastery = index / steps
        level = hint_level(mastery, prior_failures, band)
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
