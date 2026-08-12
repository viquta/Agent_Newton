"""Outcome measurement.

Tests are administered **without hints and without updating the learner model**:
they measure what the learner can do unaided, so they must not themselves teach
or disturb the estimates being reported.

**An outcome compared between arms must be derived from the shared state, never
from an agent's own bookkeeping.** The agents are what differ, so a counter one
of them keeps means something different in each arm — and it will not look
wrong, because a plausible number is exactly what it produces. ``goal_changes``
below is the worked example: the decoupled planner cannot see mastery, so it
retargets on its own position in the syllabus and reports goals "reached" while
still far from them. ``goals_mastered`` measures the same idea from the state
and is comparable.

Audited against that rule: ``triggers`` comes from the audit log, and
``diagnoses`` is recorded by the session, so both are state-derived.
``suppressed`` is arbitration bookkeeping, but the arbitration policy is
identical in both arms and reads the board rather than the arm's view, so it is
comparable — stated here rather than left to be assumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from agent_newton.core.simulator import SurfaceRenderer
from agent_newton.core.simulator.engine import Learner
from agent_newton.domains.base import Domain, Item, Verdict

if TYPE_CHECKING:  # the audit log's shape, without importing the state layer
    from agent_newton.core.state.schema import AuditRecord


@dataclass(frozen=True, slots=True)
class ItemResult:
    """How one held-out item was answered.

    Carries the verdict rather than a bare boolean so an unreadable answer stays
    distinguishable from a wrong one — the same distinction the verifier draws
    and the learner model honours.
    """

    item_id: str
    concept_id: str
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class TestResult:
    """One administration of a held-out bank."""

    correct: int
    total: int
    unmeasurable: int = 0
    #: Misconceptions that fired during the test — what the learner still shows.
    exhibited: frozenset[str] = field(default_factory=frozenset)
    #: Every item, in the order administered. Aggregates alone cannot say *what*
    #: was missed, which is what a learner is owed after sitting a test and what
    #: seeding the learner model from a pre-test would need.
    per_item: tuple[ItemResult, ...] = ()

    @property
    def score(self) -> float:
        """Correct out of everything administered.

        Unreadable answers sit in this denominator, so it is the raw test score
        and not a statement about the learner. Use :attr:`measured_score` for
        that, and see it for why the two are kept apart.
        """
        return self.correct / self.total if self.total else 0.0

    @property
    def measured_score(self) -> float:
        """Correct out of what the verifier could actually read.

        The verifier failing to parse an answer is a failure to measure, not a
        finding about the learner — the same distinction ``is_evidence`` draws
        for the learner model, and ``concepts_missed`` draws below. It was not
        drawn here, so a bank with ten unreadable answers and five wrong ones
        reported zero, while the panel showing it said unreadable answers were
        not counted.

        Zero when nothing was measurable: no answer was read, so the learner
        demonstrated nothing either way, and there is no score to state.
        """
        measured = self.total - self.unmeasurable
        return self.correct / measured if measured > 0 else 0.0

    @property
    def measured(self) -> int:
        """Answers the verifier could read. The denominator that means something."""
        return self.total - self.unmeasurable

    @property
    def concepts_missed(self) -> tuple[str, ...]:
        """Concepts answered incorrectly, in order, without repeats.

        Unreadable answers are excluded: the verifier could not measure, which
        is not a finding about the learner and must not be shown to one as if
        it were.
        """
        missed: list[str] = []
        for result in self.per_item:
            if result.verdict is Verdict.INCORRECT and result.concept_id not in missed:
                missed.append(result.concept_id)
        return tuple(missed)

    @property
    def administered(self) -> bool:
        """Whether the bank was actually run.

        A skipped test and a test scored zero both have ``score == 0.0``, and
        they mean opposite things. Check this before reporting a score.
        """
        return self.total > 0


@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """Everything one learner's session produced."""

    learner_id: str
    arm: str
    pretest: TestResult
    posttest: TestResult
    items_attempted: int
    #: Practice items consumed when the session ended because there was nothing
    #: left to teach. None when the learner did not get there within budget.
    items_to_exhaustion: int | None
    #: None when there is no ground-truth profile to measure against — a
    #: person. Unavailable, not zero: zero would read as 'nothing remediated'.
    remediation_ratio: float | None
    unmeasurable_steps: int
    #: (injected label, inferred label) per diagnosed step, for scoring the
    #: diagnostic agent against the ground truth it never saw.
    diagnoses: tuple[tuple[str | None, str | None], ...] = ()

    #: Replans by trigger. Reported separately because the triggers compete: a
    #: threshold that suppresses one pathway lets another take up the slack, so
    #: a total replan count can stay flat while the threshold is doing plenty.
    #: A sweep reading only totals would conclude the threshold does nothing.
    triggers: dict[str, int] = field(default_factory=dict)
    #: Triggers that fired and were held back by the rate limit.
    suppressed: int = 0

    #: The goal being worked toward when the session ended. None for a planner
    #: with no target.
    goal: str | None = None
    #: Times the planner moved to a new target.
    #:
    #: **Not comparable between arms.** The coupled planner retargets when a
    #: goal's mastery clears the band; the decoupled one cannot see mastery and
    #: retargets on its own position in the syllabus, so it can be working
    #: toward a goal it is nowhere near. Use ``goals_mastered`` to compare.
    goal_changes: int = 0
    #: Declared goals whose mastery cleared the band by the end. Derived from
    #: the state rather than from planner bookkeeping, so it means the same
    #: thing in both arms.
    goals_mastered: int = 0
    #: Diagnoses naming a misconception that belongs to a different concept
    #: than the item worked. The catalogue is offered whole, so the agent may
    #: reach outside the concept at hand — an incoherence nothing checked until
    #: a human demo produced one. Counted rather than prevented: narrowing the
    #: label space would also make the offline accuracy figure easier and no
    #: longer comparable to what is already measured.
    cross_concept_diagnoses: int = 0
    #: Concepts still needed for the first goal not yet mastered. Derived from
    #: mastery, not from the planner's current target — those differ, and using
    #: the target would measure the two arms against different goals. Zero when
    #: every declared goal is mastered; None when the domain declares none.
    distance_to_goal: int | None = None
    #: Why training stopped: ``budget_spent``, ``every_goal_reached`` or
    #: ``nothing_left_to_select``. Distinct from ``items_to_exhaustion``, which
    #: says *when* the material ran out and is None when it did not. A session
    #: that spent its budget and one that taught everything it had both end, and
    #: comparing arms without separating them would read a cap as a result.
    stop_reason: str = "budget_spent"

    @property
    def gain(self) -> float:
        """Pre to post improvement.

        Meaningful only when both banks were administered — see
        ``TestResult.administered``. Zero otherwise, and zero would read as "no
        improvement" rather than "not measured".

        Taken over what the verifier could read. An answer it could not parse
        says nothing about whether the learner improved, and leaving it in the
        denominator charges the verifier's failure to the teaching. For a
        simulated cohort the two are identical — ``domain validate`` requires
        every stated answer to verify correct and every buggy rule's output to
        verify incorrect, so nothing unreadable can be produced — and there is a
        test on that, which is what makes this inert for every measured result.
        """
        return self.posttest.measured_score - self.pretest.measured_score

    @property
    def normalised_gain(self) -> float | None:
        """Gain as a share of the improvement that was available.

        ``(post - pre) / (1 - pre)``. Raw gain is bounded by how much a learner
        did not already know, and that bound is not the same for everyone: a
        cohort whose mean pre-test is 0.91 has nine points of room, so an
        improvement of five reads as small while being most of what could be
        had.

        None when the pre-test was perfect. Nothing was available to gain, so
        the ratio is undefined — unavailable rather than zero, on the same
        grounds as ``remediation_ratio`` for a person. A learner at ceiling can
        only lose, and averaging them in as a zero would report that loss as an
        absence of teaching.
        """
        if not (self.pretest.administered and self.posttest.administered):
            return None
        headroom = 1.0 - self.pretest.measured_score
        if headroom <= 0.0:
            return None
        return self.gain / headroom


@dataclass(frozen=True, slots=True)
class ConceptChange:
    """What happened to one concept between the two held-out banks.

    An aggregate cannot answer "did this teach me anything". A sitting can end
    with a lower post-test than pre-test and still have taught everything it
    reached, or improve without having touched a single thing the learner did
    not already know — both have happened, and both were only visible once the
    banks were compared concept by concept.
    """

    concept_id: str
    before: Verdict | None
    after: Verdict | None

    @property
    def state(self) -> str:
        """``fixed``, ``lost``, ``still_wrong``, ``still_right`` or ``unmeasured``.

        ``unmeasured`` covers a concept the verifier could not read at either
        end. It is not a fifth kind of learning outcome; it is the absence of a
        measurement, and folding it into ``still_wrong`` would report a failure
        to parse as a failure to learn.
        """
        if self.before not in (Verdict.CORRECT, Verdict.INCORRECT):
            return "unmeasured"
        if self.after not in (Verdict.CORRECT, Verdict.INCORRECT):
            return "unmeasured"
        if self.before is Verdict.INCORRECT:
            return "fixed" if self.after is Verdict.CORRECT else "still_wrong"
        return "still_right" if self.after is Verdict.CORRECT else "lost"


def per_concept_change(
    pretest: TestResult, posttest: TestResult
) -> tuple[ConceptChange, ...]:
    """Concept by concept, what moved between the two banks.

    Both banks carry one item per concept, so a concept's verdict is that item's.
    Where a bank holds several, the last is taken: they are administered in
    order and the later one is the more recent measurement.
    """
    before = {result.concept_id: result.verdict for result in pretest.per_item}
    after = {result.concept_id: result.verdict for result in posttest.per_item}
    return tuple(
        ConceptChange(concept_id, before.get(concept_id), after.get(concept_id))
        for concept_id in sorted(set(before) | set(after))
    )


def dose_by_concept(audit_log: Sequence["AuditRecord"]) -> dict[str, int]:
    """Training steps spent on each concept, from the audit log.

    Read from ``observation`` entries only. Seeding a learner model from a
    held-out test moves posteriors without the learner having practised
    anything, and decay moves them without the learner having done anything at
    all — counting either as instruction would report the model changing its
    mind as time spent teaching.

    State-derived, so it means the same thing whichever planner ran, and it
    needs no ground-truth profile: it is the one measure of what the tutoring
    *did* that is available for a person.
    """
    spent: Counter[str] = Counter()
    for record in audit_log:
        if record.cause == "observation":
            concept_id = record.evidence.get("concept_id")
            if concept_id:
                spent[str(concept_id)] += 1
    return dict(spent)


def dose_on_gap(
    audit_log: Sequence["AuditRecord"], pretest: TestResult
) -> float | None:
    """The share of training that went to something the pre-test showed wrong.

    None when the pre-test was not administered, or when it found no gaps —
    there was nothing to aim at, so a share of zero would read as a failure to
    aim rather than as an absence of targets.

    This is what a sitting needs and could not previously produce. A session
    once spent 21 of 24 steps re-proving concepts the pre-test had already shown
    the learner knew, and it took a hand count of the transcript to see it.
    """
    if not pretest.administered:
        return None
    gaps = set(pretest.concepts_missed)
    if not gaps:
        return None
    spent = dose_by_concept(audit_log)
    total = sum(spent.values())
    if total == 0:
        return None
    return sum(count for c, count in spent.items() if c in gaps) / total


def administer(
    items: Sequence[Item],
    learner: Learner,
    domain: Domain,
    surface: SurfaceRenderer,
    on_answer: Callable[[int, int, Item], None] | None = None,
) -> TestResult:
    """Run a held-out bank. No hints, no state update, no remediation.

    ``on_answer`` reports progress and nothing else — it receives the index and
    the item, never the verdict. A front end must not be able to leak the
    result back to the learner, because feedback during a test would make it
    measure something other than unaided ability.
    """
    correct = 0
    unmeasurable = 0
    exhibited: set[str] = set()
    per_item: list[ItemResult] = []

    for index, item in enumerate(items):
        step = learner.answer(item, attempt=0)
        if on_answer is not None:
            on_answer(index, len(items), item)
        response = surface.render(item, step)
        result = domain.verifier.verify(item, response)
        per_item.append(ItemResult(item.id, item.concept_id, result.verdict))

        if not result.is_evidence:
            unmeasurable += 1
            continue
        if result.verdict is Verdict.CORRECT:
            correct += 1
        elif step.fired:
            exhibited.add(step.fired)

    return TestResult(
        correct=correct,
        total=len(items),
        unmeasurable=unmeasurable,
        exhibited=frozenset(exhibited),
        per_item=tuple(per_item),
    )
