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
* **Support at presentation** — what is shown *beside* the question, before any
  attempt, chosen from the same estimate. The reactive rule above answers a
  failure; this one answers the position in the band that made the failure
  likely.
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

from agent_newton.config import ScaffoldingPolicy, ZPDConfig
from agent_newton.core.state.zpd import Frontier


class HintLevel(IntEnum):
    """How much support a hint carries. Higher means more.

    Ordered so that "support falls as mastery rises" is an ordinary comparison
    on the value, which is what makes the fading property checkable.
    """

    #: Nothing is disclosed. The learner is asked to look again, and told
    #: nothing they could copy. Reachable only where the model already believes
    #: they can do it — see :func:`hint_level`.
    NONE = 0
    NUDGE = 1  # points at the region of the error without naming it
    TARGETED = 2  # names the misconception
    WORKED_STEP = 3  # shows the step

    @property
    def label(self) -> str:
        return self.name.lower()


class Support(IntEnum):
    """What is shown *beside* the question, before any attempt has been made.

    A second axis, and deliberately not more levels on the first one. The
    existing ladder answers a failure: it is chosen after a step, it escalates
    with further steps, and every level of it is a reply. This one answers a
    position — the learner has not done anything yet, and the estimate already
    says the question is a long way past what they can do unaided.

    Ordered like :class:`HintLevel` so "support falls as mastery rises" stays an
    ordinary comparison, and so the same property check applies to both.
    """

    NONE = 0  # the question, as it is written
    FORMULA = 1  # the rule the question is about, stated
    FORMULA_AND_EXAMPLE = 2  # and a solved instance, on other numbers

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def shows_example(self) -> bool:
        return self is Support.FORMULA_AND_EXAMPLE


class TutorMove(str, Enum):
    """What the tutor does on a turn."""

    HINT = "hint"
    REFLECT = "reflect"
    REMEDIATE = "remediate"
    #: Support given with the question, before any attempt. The only move that
    #: is not a reply to something the learner did — which is why it is a move
    #: at all rather than a property of the item: it is an instructional
    #: decision, it is chosen by a rule, and it has to appear in the record of
    #: what a learner was taught.
    PRESENT = "present"
    #: The concept explained: what it is, and why it behaves as it does.
    #:
    #: The second move that is not a reply, and it differs from every other one
    #: in *kind* rather than in degree. ``HINT`` and ``REMEDIATE`` comment on an
    #: attempt and presuppose the learner has the concept; ``PRESENT`` states
    #: the rule beside the question. None of them ever says what a derivative
    #: *is*. So when a learner fails the same concept repeatedly, the honest
    #: reading may not be that they hold a misconception — it may be that they
    #: were never taught it, and no amount of hinting on a failed attempt is
    #: teaching.
    #:
    #: ⚠️ Deliberately **not** a fourth ``HintLevel``. That ladder is a scale of
    #: how much of the answer is revealed, and mastery decides where a learner
    #: starts on it — but the decoupled view carries no posteriors, so its
    #: tutor reads 0.0 and already sits at the top. A level above
    #: ``WORKED_STEP`` would be reached after one failure in that arm and three
    #: in the other, handing the arm defined by having *less* information
    #: substantially *more* teaching. It would not look like a bug; it would
    #: look like the coupling advantage disappearing.
    EXPLAIN = "explain"


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
    *,
    policy: ScaffoldingPolicy = "banded",
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

    ``policy`` selects the ladder. ``banded`` is the original one and is what
    every measured result was produced under.

    ``banded_plus`` reads every cut point off the band instead of off a constant
    beside it, and differs in two places:

    * **Above ``theta_upper`` nothing is disclosed.** A concept the model
      believes is mastered is not one to be taught, so the reply says to look
      again and gives nothing that could be copied. Note this region is not
      reachable through selection — an item may be given only from the frontier,
      and the frontier stops at ``theta_upper`` — so it is a boundary the rule
      states rather than a case a learner meets. There is a test asserting a
      session never enters it.
    * **Inside the band, escalation stops at ``targeted``.** Failing twice on a
      concept the model believes is nearly mastered used to hand over the worked
      step. The belief and the failures then disagree, and the ladder resolved it
      by trusting the failures; this resolves it by trusting the belief, which
      is what the band is for. Naming the error is still permitted — the learner
      is told what went wrong, not shown how to finish.

    Below ``theta_lower`` the two policies are identical. ``theta_lower / 2``
    survives there as the boundary between naming the error and working the
    step, which is the one place it was ever deciding anything.
    """
    if policy == "banded_plus":
        if mastery >= band.theta_upper:
            return HintLevel.NONE
        if mastery > band.theta_lower:
            base, ceiling = HintLevel.NUDGE, HintLevel.TARGETED
        elif mastery > band.theta_lower / 2:
            base, ceiling = HintLevel.TARGETED, HintLevel.WORKED_STEP
        else:
            base, ceiling = HintLevel.WORKED_STEP, HintLevel.WORKED_STEP
    else:
        ceiling = HintLevel.WORKED_STEP
        if mastery > band.theta_lower:
            base = HintLevel.NUDGE
        elif mastery > band.theta_lower / 2:
            base = HintLevel.TARGETED
        else:
            base = HintLevel.WORKED_STEP

    escalated = int(base) + max(0, prior_failures)
    return HintLevel(min(escalated, int(ceiling)))


def should_explain(
    errors_on_concept: int,
    lessons_already_given: int,
    *,
    after: int,
) -> bool:
    """Whether the learner is owed an explanation of this concept.

    ``after`` is how many recorded errors on one concept buy a lesson; ``0``
    disables it, which is every cohort.

    **Both inputs are counted from things that mean the same in both arms**, and
    that is the whole design rather than a detail. ``errors_on_concept`` comes
    from the error trace, which lives on the shared state and is written
    identically whichever planner is running — only the *view* differs, and the
    session reads the board. ``lessons_already_given`` comes from the audit log.
    Neither is mastery, neither is the frontier, and neither is a hint level; a
    trigger derived from any of those would fire at different rates in the two
    arms and would be measuring the manipulation instead of the learner.

    ⚠️ ``UNPARSEABLE`` responses never enter the error trace, so they cannot buy
    a lesson. That is correct and worth saying out loud: "the learner keeps
    getting this wrong" and "the verifier keeps failing to read them" are
    different events, and a lesson triggered by the second would be teaching
    someone who may have been right all along.

    Repeats are throttled by requiring the threshold again for each one, so a
    learner who keeps struggling is taught again rather than either once or
    every time. The second lesson is where explaining it *differently* starts to
    matter — see :func:`style_for`.
    """
    if after <= 0:
        return False
    return errors_on_concept >= after * (lessons_already_given + 1)


def check_fading(
    band: ZPDConfig,
    prior_failures: int = 0,
    steps: int = 50,
    *,
    policy: ScaffoldingPolicy = "banded",
) -> Violation | None:
    """Verify support is non-increasing in mastery across the whole range.

    Checked as a property rather than at sample points, because the rule is a
    claim about the shape of the function and a spot check would not catch a
    single inverted step.

    Holds under both ladders, and more strongly under ``banded_plus``: the top
    of the range returns ``none`` where the original returned ``nudge``, so
    support reaches zero rather than bottoming out at a token hint.
    """
    previous = HintLevel.WORKED_STEP
    for index in range(steps + 1):
        mastery = index / steps
        level = hint_level(mastery, prior_failures, band, policy=policy)
        if level > previous:
            return Violation(
                "fading",
                f"support rose from {previous.label} to {level.label} as mastery "
                f"reached {mastery:.2f}; it must never increase with mastery",
            )
        previous = level
    return None


def support_at_presentation(mastery: float, band: ZPDConfig) -> Support:
    """What to show with the question, from where the learner sits in the band.

    A pure function of the estimate and the band — whether it is *acted on* is a
    configuration decision (``scaffolding.offer_at_presentation``), kept out of
    here so this stays a statement about the pedagogy rather than about a run.

    Nothing above ``theta_lower``. A learner in the upper part of the band is
    close enough to unaided that handing them the rule pre-empts the recall the
    question is asking for; if they turn out to need it, the reactive ladder is
    still there and costs them one attempt.

    The rule below ``theta_lower``, and a solved instance as well below
    ``theta_lower / 2``. Both boundaries are the ones :func:`hint_level` already
    uses, so a learner does not sit in one tier for the question and another for
    the reply.

    ⚠️ The example must be on **other numbers than the question's**. An example
    that solves the item is the answer with extra steps, which is the failure a
    sitting described exactly — "the system's hint cheated for me" — arriving
    through a new door. The content side enforces it: ``domain validate``
    refuses a resource whose worked answer verifies as any item's on that
    concept.
    """
    if mastery > band.theta_lower:
        return Support.NONE
    if mastery > band.theta_lower / 2:
        return Support.FORMULA
    return Support.FORMULA_AND_EXAMPLE


def check_support_fading(band: ZPDConfig, steps: int = 50) -> Violation | None:
    """Verify presentation support is non-increasing in mastery.

    The same property as :func:`check_fading`, over the other axis, and checked
    the same way and for the same reason: it is a claim about the shape of the
    function, so a spot check would pass over a single inverted step.

    There is no "all else equal" qualifier here, because there is nothing else.
    This support is chosen before the learner has done anything, so no
    escalation varies it at fixed mastery — which makes the monotonicity plain
    rather than conditional.
    """
    previous = Support.FORMULA_AND_EXAMPLE
    for index in range(steps + 1):
        mastery = index / steps
        support = support_at_presentation(mastery, band)
        if support > previous:
            return Violation(
                "support_fading",
                f"support at presentation rose from {previous.label} to "
                f"{support.label} as mastery reached {mastery:.2f}; it must "
                f"never increase with mastery",
            )
        previous = support
    return None


def move_for(
    level: HintLevel,
    moves_since_confirmation: Sequence[TutorMove],
    misconception_confirmed: bool,
    already_explained: bool = False,
) -> TutorMove:
    """The move to make, once the support level is known.

    Both tutors decided this for themselves and decided it the same way, which
    is two copies of a rule that has to stay one. It is here so a tutor is
    *driven* by the instructional layer rather than agreeing with it.

    ``HintLevel.NONE`` forces a reflective turn, and that is the substantive
    part. Remediation is the only move that teaches, so offering it to a learner
    the model already believes has the concept spends instruction where the
    evidence says none is needed — and hands over a correction to someone whose
    own next attempt was the better source of it. They are asked to look again
    instead. If they keep failing, the estimate falls, and the concept comes
    back round at a level that gives them something.
    """
    if level is HintLevel.NONE:
        return TutorMove.REFLECT
    required = next_required_move(
        moves_since_confirmation,
        misconception_confirmed=misconception_confirmed,
        already_explained=already_explained,
    )
    if required is not None:
        return required
    return TutorMove.REMEDIATE if misconception_confirmed else TutorMove.HINT


def check_move(
    move: TutorMove,
    moves_since_confirmation: Sequence[TutorMove],
    misconception_confirmed: bool,
    already_explained: bool = False,
) -> Violation | None:
    """Whether this tutor move is permitted now.

    Once a misconception is confirmed, remediation must be preceded by a
    reflective prompt: the learner is asked to look at their own reasoning
    before being handed the correction. Without the ordering constraint a tutor
    can skip straight to the answer, which is the behaviour this rule exists to
    prevent.

    ``moves_since_confirmation`` is the tutor's turns on this item since the
    misconception was confirmed, oldest first.

    ``already_explained`` says the learner has *just* set out their reasoning on
    this step, unprompted by any reflective turn. The rule is satisfied: what it
    requires is that the learner looks at their own thinking before being handed
    the correction, not that a particular question was asked. Asking anyway is
    how a sitting came to demand the same thing twice in a row — "how did you
    get there?", answered, and then "which part are you least sure of?" — which
    is a tax rather than a step, and was asked for the other way round: *"I like
    the hint better first, and then a reflection."*
    """
    # ⚠️ Stated rather than reached by falling through the test below, which is
    # what a new move would otherwise do. A lesson is permitted at any point,
    # including after a misconception is confirmed and before any reflective
    # turn, and the reason is that the error-first rule is about *correction*:
    # it puts the learner's own reasoning between their error and the answer to
    # it. A lesson is not the answer to their error. It is the account of the
    # concept they may never have had, and withholding it until they have
    # explained reasoning they do not have is the wrong way round.
    #
    # Written as its own branch because the alternative — letting it pass
    # through `move is not REMEDIATE` — is the disjunction shape that has
    # already hidden one defect here: a check that admits a case by accident
    # reads exactly like one that admits it on purpose.
    if move is TutorMove.EXPLAIN:
        return None
    if move is not TutorMove.REMEDIATE or not misconception_confirmed:
        return None
    if already_explained or TutorMove.REFLECT in moves_since_confirmation:
        return None
    return Violation(
        ERROR_FIRST,
        "remediation was offered before any reflective prompt, after a "
        "confirmed misconception",
    )


def next_required_move(
    moves_since_confirmation: Sequence[TutorMove],
    misconception_confirmed: bool,
    already_explained: bool = False,
) -> TutorMove | None:
    """The move the rules require next, if any.

    Lets the tutor be driven by the constraint rather than checked against it
    afterwards.

    Nothing is required of a learner who has already explained the step in their
    own words — see :func:`check_move`. The reflective turn exists to put their
    reasoning between the error and the correction, and their reasoning is
    already there.
    """
    if not misconception_confirmed or already_explained:
        return None
    if TutorMove.REFLECT not in moves_since_confirmation:
        return TutorMove.REFLECT
    return None
