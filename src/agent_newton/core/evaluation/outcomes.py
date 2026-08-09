"""Outcome measurement.

Tests are administered **without hints and without updating the learner model**:
they measure what the learner can do unaided, so they must not themselves teach
or disturb the estimates being reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from agent_newton.core.simulator import SimulatedLearner, SurfaceRenderer
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
    remediation_ratio: float
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

    @property
    def gain(self) -> float:
        return self.posttest.score - self.pretest.score


def administer(
    items: Sequence[Item],
    learner: SimulatedLearner,
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
