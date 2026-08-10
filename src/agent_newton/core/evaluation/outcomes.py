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
from typing import Sequence

from agent_newton.core.simulator import SurfaceRenderer
from agent_newton.core.simulator.engine import Learner
from agent_newton.domains.base import Domain, Item, Verdict


@dataclass(frozen=True, slots=True)
class TestResult:
    """One administration of a held-out bank."""

    correct: int
    total: int
    unmeasurable: int = 0
    #: Misconceptions that fired during the test — what the learner still shows.
    exhibited: frozenset[str] = field(default_factory=frozenset)

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else 0.0

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
) -> TestResult:
    """Run a held-out bank. No hints, no state update, no remediation."""
    correct = 0
    unmeasurable = 0
    exhibited: set[str] = set()

    for item in items:
        step = learner.answer(item, attempt=0)
        response = surface.render(item, step)
        result = domain.verifier.verify(item, response)

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
    )
