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

from dataclasses import dataclass, field
from typing import Callable, Sequence

from agent_newton.core.simulator import SurfaceRenderer
from agent_newton.core.simulator.engine import Learner
from agent_newton.domains.base import Domain, Item, Verdict


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
        return self.correct / self.total if self.total else 0.0

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

    @property
    def gain(self) -> float:
        """Pre to post improvement.

        Meaningful only when both banks were administered — see
        ``TestResult.administered``. Zero otherwise, and zero would read as "no
        improvement" rather than "not measured".
        """
        return self.posttest.score - self.pretest.score


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
