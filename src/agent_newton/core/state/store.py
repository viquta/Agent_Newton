"""The blackboard.

Every mutation goes through here, so no path exists that changes state without
bumping the version and appending to the audit log. Agents hold a view, never
the state itself.

Two invariants this file exists to hold:

* **Only evidence updates the learner model.** A verdict whose ``is_evidence``
  is False — unreadable input, or an answer the verifier could not decide — is a
  failure to measure, not information about the learner. It is recorded so the
  rate stays visible, and it moves nothing.
* **The frontier is consistent within a step.** It is cached against the state
  version, so two agents reading it between mutations cannot see different
  zones.
"""

from __future__ import annotations

from typing import Any

from agent_newton.config import Config
from agent_newton.core.state import bkt, zpd
from agent_newton.core.state.schema import AuditRecord, Cause, ErrorEvent, LearnerState
from agent_newton.core.state.views import FullStateView, ItemCorrectnessView
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains.base import ConceptGraph, VerificationResult, Verdict


class Blackboard:
    """Shared learner state, its audit log, and the views onto it."""

    def __init__(self, state: LearnerState, graph: ConceptGraph, config: Config) -> None:
        self._state = state
        self._graph = graph
        self._config = config
        self._audit: list[AuditRecord] = []
        self._outcomes: list[bool] = []
        self._frontier_cache: tuple[int, Frontier] | None = None
        #: Observations discarded because they were not evidence. Reported per
        #: run: a high rate means the verifier is failing, not the learner.
        self.unmeasurable = 0

    # -- reading ----------------------------------------------------------

    @property
    def state(self) -> LearnerState:
        """Read-only access. Mutate through the recording methods."""
        return self._state

    @property
    def version(self) -> int:
        return self._state.version

    @property
    def audit_log(self) -> tuple[AuditRecord, ...]:
        """Append-only. Returned as a tuple so callers cannot rewrite history."""
        return tuple(self._audit)

    @property
    def frontier(self) -> Frontier:
        """The current zone, recomputed only when the state has moved."""
        if self._frontier_cache is None or self._frontier_cache[0] != self._state.version:
            frontier = zpd.compute(
                self._state.mastery,
                self._graph,
                self._config.zpd,
                prior=bkt.initial(self._config.bkt),
            )
            self._frontier_cache = (self._state.version, frontier)
        return self._frontier_cache[1]

    def probability(self, concept_id: str) -> float:
        return self._state.probability(concept_id, bkt.initial(self._config.bkt))

    def view(self, arm: str | None = None) -> FullStateView | ItemCorrectnessView:
        """The view this arm's planner receives.

        This single choice is the independent variable of the whole experiment.
        """
        arm = arm or self._config.arm
        if arm == "coupled":
            return FullStateView(
                mastery=dict(self._state.mastery),
                error_trace=tuple(self._state.error_trace),
                frontier=self.frontier,
                outcomes=tuple(self._outcomes),
                version=self._state.version,
            )
        return ItemCorrectnessView(
            outcomes=tuple(self._outcomes),
            version=self._state.version,
        )

    # -- writing ----------------------------------------------------------

    def _bump(self, cause: Cause, summary: str, **evidence: Any) -> None:
        self._state.version += 1
        self._audit.append(
            AuditRecord(
                version=self._state.version,
                cause=cause,
                summary=summary,
                evidence=evidence,
            )
        )

    def record_observation(
        self,
        *,
        item_id: str,
        concept_id: str,
        result: VerificationResult,
        misconception_label: str | None = None,
        confidence: float = 0.0,
    ) -> bool:
        """Record one graded step. Returns whether it counted as evidence.

        An unmeasurable result still bumps the version and lands in the audit
        log — the attempt happened, and hiding it would make the record
        incomplete — but leaves mastery, the outcome stream and the error trace
        untouched.
        """
        self._state.t += 1

        if not result.is_evidence:
            self.unmeasurable += 1
            self._bump(
                "observation",
                f"unmeasurable response on {item_id}; learner model unchanged",
                item_id=item_id,
                concept_id=concept_id,
                verdict=result.verdict.value,
                detail=result.detail,
            )
            return False

        correct = result.verdict is Verdict.CORRECT
        before = self.probability(concept_id)
        after = bkt.observe(before, correct, self._config.bkt)

        self._state.mastery[concept_id] = after
        self._outcomes.append(correct)

        if not correct:
            self._state.error_trace.append(
                ErrorEvent(
                    t=self._state.t,
                    item_id=item_id,
                    concept_id=concept_id,
                    misconception_label=misconception_label,
                    confidence=confidence,
                    verifier_label=result.verdict.value,
                )
            )
            # Bounded: the trace is a rolling window, and the arbitration policy
            # counts repeats within it.
            excess = len(self._state.error_trace) - self._config.arbitration.error_trace_length
            if excess > 0:
                del self._state.error_trace[:excess]

        self._bump(
            "observation",
            f"{result.verdict.value} on {item_id}; "
            f"P({concept_id}) {before:.3f} -> {after:.3f}",
            item_id=item_id,
            concept_id=concept_id,
            verdict=result.verdict.value,
            mastery_before=before,
            mastery_after=after,
            delta=after - before,
            misconception_label=misconception_label,
        )
        return True

    def record_replan(self, summary: str, **evidence: Any) -> None:
        """Record a planning decision and what triggered it.

        The evidence is what makes a replan reconstructible after the run rather
        than only observable during it.
        """
        self._bump("replan", summary, **evidence)

    def annotate(self, summary: str, **evidence: Any) -> None:
        """Record something worth auditing that changes no estimate."""
        self._bump("annotation", summary, **evidence)


def new_blackboard(
    learner_id: str,
    seed: int,
    graph: ConceptGraph,
    config: Config,
) -> Blackboard:
    """A blackboard for a learner who has done nothing yet."""
    board = Blackboard(LearnerState(learner_id=learner_id, seed=seed), graph, config)
    board.annotate(
        f"initialised learner {learner_id!r}",
        seed=seed,
        concepts=len(graph.ids()),
        arm=config.arm,
    )
    return board
