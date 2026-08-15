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

from typing import Any, Iterable, Literal

from agent_newton.config import Config
from agent_newton.core.state import bkt, decay, zpd
from agent_newton.core.state.schema import (
    AuditRecord,
    Cause,
    ErrorEvent,
    LearnerState,
    Plan,
    Utterance,
)
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
        self._frontier_cache: tuple[int, Frontier] | None = None
        #: Observations discarded because they were not evidence. Reported per
        #: run: a high rate means the verifier is failing, not the learner.
        self.unmeasurable = 0
        #: Times each concept has been worked in *this* sitting, and the ones
        #: worked often enough to be set aside. Session-scoped rather than on
        #: the state, so neither is persisted: being stuck on something in
        #: March is not a reason to skip it in May, and the decay model is what
        #: decides whether a concept comes round again.
        self._visits: dict[str, int] = {}
        self._weak: set[str] = set()
        #: What the learner asked to work on this sitting. Session-scoped like
        #: the visit counts: what someone wanted to practise in March is not a
        #: standing instruction, and a request that outlived the sitting it was
        #: made in would steer routing nobody had asked for.
        self._requested: frozenset[str] = frozenset()

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
                # Always empty unless a dwelling cap is configured. Marking a
                # weakness bumps the version, so the cache invalidates at
                # exactly the step the zone changes.
                waived=self.weaknesses,
            )
            self._frontier_cache = (self._state.version, frontier)
        return self._frontier_cache[1]

    def probability(self, concept_id: str) -> float:
        return self._state.probability(concept_id, bkt.initial(self._config.bkt))

    @property
    def plan(self) -> Plan | None:
        """What this learner is working toward. None before the first plan."""
        return self._state.plan

    @property
    def weaknesses(self) -> frozenset[str]:
        """Concepts worked enough in this sitting to be set aside for now.

        Always empty unless ``cohort.max_visits_per_concept`` is set, which is
        every run that has produced a measured number.
        """
        return frozenset(self._weak)

    def visits(self, concept_id: str) -> int:
        return self._visits.get(concept_id, 0)

    @property
    def requested(self) -> frozenset[str]:
        """Concepts the learner asked for. Empty unless a front end asked."""
        return self._requested

    def view(self, arm: str | None = None) -> FullStateView | ItemCorrectnessView:
        """The view this arm's planner receives.

        This single choice is the independent variable of the whole experiment.

        The plan is on **both** views. A goal is curriculum, like the item bank
        and the graph, so withholding it would make the comparison measure
        ignorance of the target rather than inability to route toward it. What
        the decoupled view still lacks is everything needed to compute a route:
        the posteriors, the error trace and the frontier.
        """
        arm = arm or self._config.arm
        if arm == "coupled":
            return FullStateView(
                mastery=dict(self._state.mastery),
                error_trace=tuple(self._state.error_trace),
                frontier=self.frontier,
                outcomes=tuple(self._state.outcomes),
                version=self._state.version,
                plan=self._state.plan,
                reflections=tuple(self._state.reflections),
                weaknesses=self.weaknesses,
                requested=self._requested,
            )
        return ItemCorrectnessView(
            outcomes=tuple(self._state.outcomes),
            version=self._state.version,
            plan=self._state.plan,
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
        attempt: int = 0,
        response: str = "",
    ) -> bool:
        """Record one graded step. Returns whether it counted as evidence.

        An unmeasurable result still bumps the version and lands in the audit
        log — the attempt happened, and hiding it would make the record
        incomplete — but leaves mastery, the outcome stream and the error trace
        untouched.

        ``attempt`` is here only so the first attempt at an item increments the
        lifetime count of how often it has been given. Doing it on this path
        rather than a separate one keeps the invariant at the top of this
        module: nothing changes state without bumping the version and writing an
        audit entry.

        ``response`` is what the learner actually wrote. It was not kept here,
        so the audit log recorded that an answer was wrong without recording
        what it was — and a sitting could only be read back by joining the log
        against a separate transcript file. The verdict is the evidence; the
        answer is what makes the verdict readable.
        """
        self._state.t += 1
        if attempt == 0:
            self._state.items_given[item_id] = self._state.items_given.get(item_id, 0) + 1

        if not result.is_evidence:
            self.unmeasurable += 1
            self._bump(
                "observation",
                f"unmeasurable response on {item_id}; learner model unchanged",
                item_id=item_id,
                concept_id=concept_id,
                verdict=result.verdict.value,
                detail=result.detail,
                response=response,
            )
            return False

        correct = result.verdict is Verdict.CORRECT
        before = self.probability(concept_id)
        after = bkt.observe(before, correct, self._config.bkt)

        self._state.mastery[concept_id] = after
        self._state.outcomes.append(correct)

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
            response=response,
        )
        return True

    def seed_from_test(
        self,
        results: Iterable[tuple[str, Verdict]],
        weight: int = 1,
        floor: float = 0.0,
    ) -> int:
        """Fold a held-out test's results into the learner model. Returns the count.

        One BKT update per measurable item, before any practice happens. Close
        to what ``record_observation`` performs, and deliberately not the same
        method: four things must differ.

        * **No learning transition.** ``bkt.revise``, not ``bkt.observe`` — a
          held-out item is answered unaided and without feedback, so there was
          no opportunity to learn from it. This one is not a nicety: the
          transition is large enough that ``observe`` would *raise* the estimate
          on a wrong answer, seeding the model backwards.
        * **A separate audit cause.** ``seed`` rather than ``observation``, so
          anything counting what the learner did *during* the session can leave
          these out. They came from a test, not from practice.
        * **No error-trace events.** A test is administered without diagnosis,
          so a wrong answer here carries no misconception label. The arbitration
          policy counts label repeats in the trace, and unlabelled entries would
          trip replans on evidence that does not exist.
        * **No outcome stream entry.** The stream is the practice record the
          decoupled view is built from, and a test the learner sat unaided is
          not part of it.

        Unreadable answers are skipped, for the reason they are skipped
        everywhere: the verifier failed to measure, which says nothing about
        mastery.

        ``weight`` treats one answer as that many independent observations. At
        1 a correct answer lands well below ``theta_upper``, so a concept the
        learner has just demonstrated is still selectable and costs more items
        to re-prove; a human sitting spent 21 of 24 training steps that way.
        Applied in both directions: it says how much one held-out item is worth,
        and weighting only the direction that suits the budget would be a thumb
        on the scale rather than a claim about evidence.

        ``floor`` is the lowest a seeded estimate may land. It binds only on the
        way down — a correct answer lands far above any legal floor — and it
        exists because weighting the evidence up also drove a missed concept to
        about 0.0003, where the scaffolding rule reads bottom-of-band and gives
        maximum support on the first attempt. The learner then received a worked
        step every time, on every concept the pre-test had flagged, and never a
        nudge. The floor keeps the concept a priority without claiming the
        learner is at the very bottom of it; see ``CohortConfig.seed_floor`` for
        why it must stay under ``theta_lower``.
        """
        seeded = 0
        for concept_id, verdict in results:
            if verdict not in (Verdict.CORRECT, Verdict.INCORRECT):
                continue
            before = self.probability(concept_id)
            after = before
            for _ in range(weight):
                after = bkt.revise(after, verdict is Verdict.CORRECT, self._config.bkt)
            after = max(after, floor)
            self._state.mastery[concept_id] = after
            seeded += 1
            self._bump(
                "seed",
                f"{verdict.value} on a held-out item; "
                f"P({concept_id}) {before:.3f} -> {after:.3f}",
                concept_id=concept_id,
                verdict=verdict.value,
                weight=weight,
                mastery_before=before,
                mastery_after=after,
            )
        return seeded

    def apply_decay(self, elapsed_days: float) -> int:
        """Let the model go stale over a gap between sessions. Returns the count.

        Every *observed* concept relaxes toward the prior. Unobserved concepts
        are skipped because they are already at the prior — decaying them would
        write an estimate that says nothing, and would make an untouched concept
        indistinguishable in the audit log from one that was worked and
        forgotten.

        The error trace and the outcome stream are untouched. What the learner
        did is a fact about the past and does not become less true; it is the
        *inference* from it that goes stale.
        """
        if not self._config.decay.enabled or elapsed_days <= 0.0:
            return 0
        half_life = self._config.decay.half_life_days
        assert half_life is not None  # enabled implies it is set
        prior = bkt.initial(self._config.bkt)

        moved = 0
        for concept_id, before in list(self._state.mastery.items()):
            after = decay.relax(before, prior, elapsed_days, half_life)
            if after == before:
                continue
            self._state.mastery[concept_id] = after
            moved += 1
            self._bump(
                "decay",
                f"{elapsed_days:g} day(s) elapsed; "
                f"P({concept_id}) {before:.3f} -> {after:.3f}",
                concept_id=concept_id,
                elapsed_days=elapsed_days,
                half_life_days=half_life,
                mastery_before=before,
                mastery_after=after,
            )
        return moved

    def record_replan(self, summary: str, **evidence: Any) -> None:
        """Record a planning decision and what triggered it.

        The evidence is what makes a replan reconstructible after the run rather
        than only observable during it.
        """
        self._bump("replan", summary, **evidence)

    def record_reflection(
        self,
        text: str,
        item_id: str,
        concept_id: str,
        kind: Literal["reflection", "working"] = "reflection",
    ) -> None:
        """Record what the learner said, in words.

        Prose, not evidence. It updates no estimate and enters no error trace —
        the learner was asked a question in words and answered it, or showed
        their working, and neither is a graded attempt at anything. It is kept
        because the tutor can use it, and because a session where someone said
        what confused them and the system discarded it is not a shared learner
        state worth the name.

        ``concept_id`` is stored rather than only audited. It was audited only,
        once, and the tutor consequently read back whatever had been said most
        recently regardless of subject.
        """
        self._state.reflections.append(
            Utterance(text=text, item_id=item_id, concept_id=concept_id, kind=kind)
        )
        self._bump(
            "annotation",
            f"{kind} on {item_id}",
            item_id=item_id,
            concept_id=concept_id,
            reflection=text,
            kind=kind,
        )

    def record_request(self, concepts: Iterable[str]) -> frozenset[str]:
        """What the learner said they want to work on. Returns what was taken.

        Recorded through the same path as everything else, so a sitting that
        routed somewhere unexpected can be read back against what the learner
        asked for rather than against an assumption about it. It changes which
        goal is aimed at, never which prerequisites may be skipped — see
        ``route.next_goal``.
        """
        self._requested = frozenset(concepts)
        self._bump(
            "annotation",
            "the learner asked to work on "
            + (", ".join(sorted(self._requested)) or "nothing in particular"),
            requested=sorted(self._requested),
        )
        return self._requested

    def note_visit(self, concept_id: str, cap: int | None) -> bool:
        """Count one working of this concept. Returns whether it just became a weakness.

        ``cap`` of None is unlimited, and nothing is recorded beyond the count —
        which is what every run that produced a measured number does.

        Above the cap the concept is set aside rather than withdrawn: it sorts
        last, so it is still offered when nothing else on the way to the goal
        is available. Withdrawing it would mean a learner whose only remaining
        work is the thing they are stuck on gets no work at all.
        """
        self._visits[concept_id] = self._visits.get(concept_id, 0) + 1
        if cap is None or self._visits[concept_id] < cap or concept_id in self._weak:
            return False
        self._weak.add(concept_id)
        self._bump(
            "annotation",
            f"{concept_id} worked {self._visits[concept_id]} time(s); setting it "
            f"aside for now and noting it as something to come back to",
            concept_id=concept_id,
            visits=self._visits[concept_id],
            max_visits_per_concept=cap,
            weakness=True,
        )
        return True

    def record_turn(
        self,
        *,
        item_id: str,
        concept_id: str,
        move: str,
        level: str,
        targets: str | None,
        text: str,
        mastery: float = 0.0,
        prior_failures: int = 0,
    ) -> None:
        """Record what the tutor said, and under which rules it said it.

        Changes no estimate: a hint is instruction, and whether it worked shows
        up in the next graded step rather than here. It is recorded because it
        was not, and a sitting's transcript consequently held every answer the
        learner gave and nothing the system replied — leaving no way to read back
        what a person was actually taught.

        ``move`` and ``level`` are the rules' decisions rather than the model's,
        so storing them beside the text is what makes a turn checkable after the
        fact: a reply can be read against the support level it was supposed to
        carry.

        ``mastery`` and ``prior_failures`` are the two inputs the level was
        chosen from, and they are here because the level alone could not be
        argued with. Two sittings ran entirely at ``worked_step`` and the
        transcripts could say only that; whether the cause was the belief, the
        escalation or both took a wrapper around the tutor to find out. Stored,
        the question is answerable from the record.
        """
        self._bump(
            "tutor",
            f"{move} at {level} on {item_id}",
            item_id=item_id,
            concept_id=concept_id,
            move=move,
            level=level,
            targets=targets,
            mastery=mastery,
            prior_failures=prior_failures,
            text=text,
        )

    def record_plan(self, plan: Plan, **evidence: Any) -> Plan:
        """Set what the learner is working toward.

        Goes through the same path as every other mutation, so a change of goal
        bumps the version and lands in the audit log. Without that, the state
        could carry a target nobody could account for afterwards.
        """
        stamped = plan.model_copy(update={"set_at_version": self._state.version + 1})
        self._state.plan = stamped
        self._bump(
            "plan",
            f"goal set to {stamped.goal!r} ({stamped.emphasis.value})",
            goal=stamped.goal,
            emphasis=stamped.emphasis.value,
            reason=stamped.reason,
            **evidence,
        )
        return stamped

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


def resumed_blackboard(
    state: LearnerState,
    graph: ConceptGraph,
    config: Config,
) -> Blackboard:
    """A blackboard for a learner picking up where they left off.

    The state carries: mastery, the error trace, the plan, what the learner
    said, the outcome stream and ``t``. The audit log does not — it is the
    record of *one* session, written to the store when that session ended, and
    starting a new one with the last one's log would make every run's log grow
    without bound and stop being readable as the account of a sitting.

    The version counter continues rather than restarting, so an audit entry's
    version still orders events within a learner's whole history.
    """
    board = Blackboard(state, graph, config)
    board.annotate(
        f"resumed learner {state.learner_id!r}",
        concepts=len(graph.ids()),
        arm=config.arm,
        known_concepts=len(state.mastery),
        prior_steps=state.t,
    )
    return board
