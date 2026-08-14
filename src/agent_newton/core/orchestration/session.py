"""One learner's tutoring session.

The loop that closes the feedback path: the planner selects, the learner
answers, the verifier grades, the diagnostic labels, the state updates, and the
tutor responds. Agents never call one another — the session is the only thing
that moves information between them, and it moves all of it through the
blackboard.

Assembly lives in :func:`build_session`, which turns a config into the set of
agents it names. That is where the arm takes effect: it decides which view the
planner receives and therefore which planner can be used at all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from agent_newton.config import Config
from agent_newton.core.agents.base import (
    Diagnosis,
    Diagnostic,
    Hint,
    OracleAccess,
    Planner,
    Resumable,
    Tutor,
)
from agent_newton.core.agents.diagnostic import NoisedOracleDiagnostic, OracleDiagnostic
from agent_newton.core.agents.llm import LLMDiagnostic, LLMPlanner, LLMTutor
from agent_newton.core.agents.planner import (
    FixedOrderPlanner,
    FrontierPlanner,
    GoalDirectedPlanner,
    OraclePlanner,
    ReverseOrderPlanner,
    ShuffledPlanner,
)
from agent_newton.core.agents.tutor import TemplateTutor
from agent_newton.core.arbitration.policy import ArbitrationPolicy
from agent_newton.core.state import bkt, route
from agent_newton.llm.factory import build_provider
from agent_newton.core.evaluation.outcomes import SessionOutcome, TestResult, administer
from agent_newton.core.pedagogy import TutorMove, check_move
from agent_newton.core.simulator import (
    SimulatedLearner,
    SurfaceRenderer,
    SymbolicSurface,
    sample_profile,
)
from agent_newton.core.simulator.engine import Learner
from agent_newton.core.simulator.profile import MisconceptionProfile
from agent_newton.core.state.schema import LearnerState
from agent_newton.core.state.store import Blackboard, new_blackboard, resumed_blackboard
from agent_newton.core.state.views import FullStateView
from agent_newton.domains.base import Domain, Item, VerificationResult, Verdict


class NotImplementedForModels(NotImplementedError):
    """Raised when a config names a model-backed agent before one exists."""


@runtime_checkable
class SessionObserver(Protocol):
    """Watches a session without taking part in it.

    Exists so a front end can render what is happening without the loop
    knowing there is one, and without a second loop being written that would
    drift from this one. A session with no observer behaves exactly as one
    written before observers existed.

    Implement it structurally, or subclass :class:`Watching` to take no-ops for
    the hooks you do not want. The session calls every hook unconditionally, so
    a partial structural implementation fails at the moment the session first
    reaches the missing one — halfway through somebody's sitting, not at
    startup.
    """

    def item_started(self, item: Item, board: Blackboard) -> None: ...

    def step_graded(
        self, item: Item, response: str, result: VerificationResult, diagnosis: Diagnosis
    ) -> None: ...

    def item_finished(
        self, item: Item, solved: bool, reason: str = "attempts_spent"
    ) -> None:
        """The session is done with this item, whether or not it was answered.

        Exists because running out of attempts previously looked identical to
        nothing happening: the next question simply appeared, and a person had
        no way to tell whether they had got the last one right.

        ``reason`` refines the unsolved case, which now has two forms that must
        not be reported as one: ``attempts_spent`` means the tries ran out, and
        ``unreadable`` means nothing the learner sent could be read, so no
        attempt was spent and nothing about them was measured. Telling someone
        "that is all the attempts" when they still have all of them is the same
        category error the attempt budget itself carried.
        """
        ...

    def tutor_replied(self, item: Item, hint: Hint) -> None: ...

    def reflection_recorded(self, item: Item, text: str) -> None: ...

    def working_recorded(self, item: Item, text: str) -> None: ...

    def session_resumed(self, elapsed_days: float, concepts_decayed: int) -> None:
        """A gap since the last sitting, and what it did to the model.

        Fires only when something actually moved, so a first session and a
        no-gap sequence stay silent.
        """
        ...

    def training_finished(self, reason: str, items: int) -> None:
        """Training is over, and why.

        The three reasons are different events — the budget ran out, every goal
        was reached, or nothing was left to select — and without saying which,
        the post-test appearing straight after a failed item reads as the system
        giving up on the learner.
        """
        ...

    def phase_started(self, phase: str, total: int) -> None: ...

    def phase_answer(self, phase: str, index: int, total: int, item: Item) -> None: ...

    def phase_finished(self, phase: str, result: "TestResult") -> None: ...


class Watching:
    """A :class:`SessionObserver` that does nothing. Subclass and override.

    Here so a front end can take the hooks it wants without having to track
    every one the session grows later. The alternative is that adding a hook
    breaks existing observers the first time a session reaches it, which is
    mid-sitting rather than at startup.
    """

    def item_started(self, item: Item, board: Blackboard) -> None:
        return None

    def step_graded(
        self, item: Item, response: str, result: VerificationResult, diagnosis: Diagnosis
    ) -> None:
        return None

    def item_finished(
        self, item: Item, solved: bool, reason: str = "attempts_spent"
    ) -> None:
        return None

    def tutor_replied(self, item: Item, hint: Hint) -> None:
        return None

    def reflection_recorded(self, item: Item, text: str) -> None:
        return None

    def working_recorded(self, item: Item, text: str) -> None:
        return None

    def session_resumed(self, elapsed_days: float, concepts_decayed: int) -> None:
        return None

    def training_finished(self, reason: str, items: int) -> None:
        return None

    def phase_started(self, phase: str, total: int) -> None:
        return None

    def phase_answer(self, phase: str, index: int, total: int, item: Item) -> None:
        return None

    def phase_finished(self, phase: str, result: "TestResult") -> None:
        return None


@dataclass
class Session:
    """A single learner working through practice items."""

    learner: Learner
    board: Blackboard
    planner: Planner
    tutor: Tutor
    diagnostic: Diagnostic
    surface: SurfaceRenderer
    domain: Domain
    config: Config
    arbitration: ArbitrationPolicy
    observer: SessionObserver | None = None
    #: Days since this learner's previous session. Drives decay; zero for a
    #: first sitting and for every single-session run, which is what keeps those
    #: identical to what they were before sequences existed.
    elapsed_days: float = 0.0
    #: Whether this session picked up stored state. Only ``cohort.pretest_scope``
    #: reads it, and only to decide whether there is a history to narrow the
    #: held-out banks against — a first sitting has none and needs the baseline.
    resumed: bool = False

    def run(self) -> SessionOutcome:
        # Before anything is measured. The pre-test then measures what the
        # learner can currently do, and seeding folds *that* in — decaying
        # afterwards would throw away the evidence just collected.
        decayed = self.board.apply_decay(self.elapsed_days)
        if decayed and self.observer is not None:
            self.observer.session_resumed(self.elapsed_days, decayed)

        # Fixed once, before the pre-test, and used again for the post-test.
        # Recomputing it after training would ask the two banks about different
        # concepts and call the difference a gain. Computed after decay, so a
        # concept that went stale over the gap is back on the route and is
        # re-checked — which is the measurement a returning learner is here for.
        measured = self._bank_scope()
        pretest = self._administer("pretest", measured)
        if self.config.cohort.seed_from_pretest:
            # Off for every cohort — see CohortConfig. It moves the starting
            # frontier, and only one arm can route from a frontier.
            self.board.seed_from_test(
                ((result.concept_id, result.verdict) for result in pretest.per_item),
                weight=self.config.cohort.pretest_weight,
                floor=self.config.cohort.seed_floor,
            )

        # Two counts, because they answer different questions.
        #
        # `lifetime` is how often this learner has ever been given each item. It
        # drives the repetition index — which decides the simulated learner's
        # answer and which variant of the question is asked — and it drives
        # least-used selection, so someone returning gets fresh numbers rather
        # than session one again. It lives on the state and is incremented by
        # `record_observation`.
        #
        # `given` counts only this sitting, and is what `items_attempted`
        # reports. Conflating them would make a second session look like it
        # attempted everything the first one did.
        lifetime = self.board.state.items_given
        given: Counter[str] = Counter()
        diagnoses: list[tuple[str | None, str | None]] = []
        exhausted: int | None = None
        #: Why training stopped. Overwritten by any of the early exits; if none
        #: fires, the budget is what ended it. Recorded because the three are
        #: not the same event and a reader — or a learner watching — cannot tell
        #: them apart from the fact that training simply stopped.
        stop_reason = "budget_spent"

        #: The concept currently being worked. The *goal* lives on the board;
        #: this is only where the learner is along the way to it.
        working: str | None = None
        goal_changes = 0
        #: Set when the last item exhausted the dwelling cap. Selection has to be
        #: reopened for the concept to actually be left: without it the weakness
        #: is recorded, the frontier widens, and the loop carries on asking for
        #: the next item on the concept it just set aside — because nothing else
        #: reconsiders. Always False when no cap is configured, which is every
        #: cohort.
        set_aside = False

        for _ in range(self.config.cohort.max_items):
            decision = self.arbitration.evaluate(
                current_concept=working,
                mastery=dict(self.board.state.mastery),
                frontier=self.board.frontier,
                error_trace=list(self.board.state.error_trace),
                prior=bkt.initial(self.config.bkt),
            )

            if decision.replan or set_aside:
                # Macro first: is the target still the right one? Then micro:
                # what to work on the way to it. Both come from the planner, and
                # the goal is written to the board before the item is chosen, so
                # the selection is made against the plan that will be recorded.
                if self._retarget():
                    goal_changes += 1
                if self.board.plan is None and self._wants_a_goal():
                    exhausted = sum(given.values())
                    stop_reason = "every_goal_reached"
                    self.board.annotate(
                        "every goal reached", items_given=sum(given.values())
                    )
                    break

                item = self.planner.select(self.board.view(), self.domain, lifetime)
                if item is None:
                    # Nothing left to teach: the frontier emptied, or the
                    # syllabus ran out. When that happened is an outcome.
                    exhausted = sum(given.values())
                    stop_reason = "nothing_left_to_select"
                    self.board.annotate(
                        "nothing left to select", items_given=sum(given.values())
                    )
                    break
                working = item.concept_id
                if decision.replan:
                    self.board.record_replan(decision.summary, **decision.evidence)
                else:
                    # Its own trigger name, so the threshold sweep reads it apart
                    # from the ones the arbitration policy owns. It cannot fire
                    # without a dwelling cap, and no experiment config sets one.
                    self.board.record_replan(
                        "replan triggered by concept_set_aside", set_aside=True
                    )
                self.arbitration.accept(dict(self.board.state.mastery))
                set_aside = False
            else:
                if decision.suppressed_by:
                    # A trigger that fired and was held back is worth recording:
                    # it is the difference between the threshold deciding and
                    # the rate limit deciding.
                    self.board.annotate(decision.summary, **decision.evidence)
                item = self._next_item_for(working, lifetime)
                if item is None:
                    exhausted = sum(given.values())
                    stop_reason = "nothing_left_to_select"
                    self.board.annotate(
                        "no item left on the current concept",
                        items_given=sum(given.values()),
                        concept_id=working,
                    )
                    break

            # Before the item is worked, so it is the count of previous
            # givings; `record_observation` does the incrementing.
            repetition = lifetime.get(item.id, 0)
            given[item.id] += 1
            self.arbitration.note_item()
            # Counted always, acted on only when a cap is configured — which is
            # no cohort. `consolidate` ranks by recent errors, so failing a
            # concept attracts more of the same, and with the pre-test skipping
            # what a learner has demonstrated there are fewer places left to be
            # moved along to.
            set_aside = self.board.note_visit(
                item.concept_id, self.config.cohort.max_visits_per_concept
            )
            self._work_item(item, diagnoses, repetition=repetition)

        if stop_reason == "budget_spent":
            # The one exit that recorded nothing. To a person it looked like the
            # session breaking off after a failed item — the post-test simply
            # appeared — because nothing said that the budget was what ended it.
            self.board.annotate(
                "item budget spent",
                items_given=sum(given.values()),
                max_items=self.config.cohort.max_items,
            )
        if self.observer is not None:
            self.observer.training_finished(stop_reason, sum(given.values()))

        posttest = self._administer("posttest", measured)

        return SessionOutcome(
            learner_id=self.learner.learner_id,
            arm=self.config.arm,
            pretest=pretest,
            posttest=posttest,
            items_attempted=sum(given.values()),
            items_to_exhaustion=exhausted,
            remediation_ratio=self.learner.remediation_ratio(),
            unmeasurable_steps=self.board.unmeasurable,
            diagnoses=tuple(diagnoses),
            triggers=self._trigger_counts(),
            suppressed=self.arbitration.suppressed,
            goal=self.board.plan.goal if self.board.plan else None,
            goal_changes=goal_changes,
            goals_mastered=self._goals_mastered(),
            stop_reason=stop_reason,
            cross_concept_diagnoses=self._cross_concept(),
            distance_to_goal=self._distance_to_goal(),
        )

    def _bank_scope(self) -> frozenset[str] | None:
        """Which concepts this sitting's held-out banks measure. None is all.

        ``full`` is None here, which is every cohort and every first sitting.

        ``route`` narrows a returning learner's banks to what is still on the
        way to their next goal — the concepts a sitting could plausibly move,
        which is what makes the re-check short. Deliberately wider than the
        frontier: training opens concepts behind it as prerequisites are met,
        and a concept taught but not measured at both ends is a step nothing
        can account for.

        None again when there is no goal left to reach. There is no route to
        narrow to, and the whole bank is the right instrument at that point
        anyway — what a learner with everything mastered comes back to find out
        is what has gone stale.
        """
        if self.config.cohort.pretest_scope != "route" or not self.resumed:
            return None

        mastery = dict(self.board.state.mastery)
        prior = bkt.initial(self.config.bkt)
        goal = route.next_goal(
            self.domain.concepts.goals(), mastery, self.config.zpd, prior
        )
        if goal is None:
            return None
        on_route = route.remaining(
            goal, mastery, self.domain.concepts, self.config.zpd, prior
        )
        return frozenset(on_route) or None

    def _administer(self, bank, concepts: frozenset[str] | None = None) -> TestResult:
        """Run a held-out bank, unless the config says not to.

        A skipped bank returns an empty result rather than a zero score, so
        `TestResult.administered` can tell the two apart.

        ``concepts`` restricts it to the items on those concepts. A restriction
        that would leave nothing is ignored: an empty bank scores zero out of
        zero, which reads as "not administered" and would silently drop the
        measurement rather than shorten it.
        """
        if not self.config.cohort.administer_tests:
            return TestResult(correct=0, total=0)

        items = self.domain.items.bank(bank)
        if concepts is not None:
            narrowed = [item for item in items if item.concept_id in concepts]
            items = narrowed or items
        if self.observer is None:
            return administer(items, self.learner, self.domain, self.surface)

        self.observer.phase_started(bank, len(items))
        result = administer(
            items,
            self.learner,
            self.domain,
            self.surface,
            on_answer=lambda i, n, item: self.observer.phase_answer(bank, i, n, item)
            if self.observer
            else None,
        )
        self.observer.phase_finished(bank, result)
        return result

    def _cross_concept(self) -> int:
        """Diagnoses naming a misconception from another concept.

        Read from the error trace rather than counted as it happens, so it
        describes the state rather than the agent's bookkeeping.
        """
        catalogue = self.domain.misconceptions
        return sum(
            1
            for event in self.board.state.error_trace
            if event.misconception_label
            and catalogue.get(event.misconception_label).concept_id != event.concept_id
        )

    def _goals_mastered(self) -> int:
        """Declared goals whose mastery cleared the band.

        Read from the state rather than counted as the planner retargets, so it
        means the same thing whichever planner ran.
        """
        mastery = dict(self.board.state.mastery)
        prior = bkt.initial(self.config.bkt)
        return sum(
            1
            for goal in self.domain.concepts.goals()
            if route.reached(goal, mastery, self.config.zpd, prior)
        )

    def _retarget(self) -> bool:
        """Ask the planner what to aim at; record it if it changed.

        Returns whether a goal was *completed* — that is, whether a goal was
        already set and has now been replaced. A planner with no notion of goals
        returns None here and nothing is recorded, which is how the undirected
        baseline coexists with the directed one.
        """
        proposed = self.planner.plan(self.board.view(), self.domain)
        existing = self.board.plan

        if proposed is None:
            return existing is not None
        if existing is not None and existing.goal == proposed.goal:
            return False

        self.board.record_plan(proposed)
        return existing is not None

    def _wants_a_goal(self) -> bool:
        """Whether this planner plans toward goals at all.

        Distinguishes 'every goal reached' from 'this planner has no goals',
        which look identical from a None return.
        """
        return bool(self.domain.concepts.goals()) and not isinstance(
            self.planner, FrontierPlanner
        )

    def _distance_to_goal(self) -> int | None:
        """Concepts still needed for the first goal not yet mastered.

        Measured against a goal derived from **mastery**, not against whatever
        the planner happens to be aiming at. Those are not the same thing, and
        using the planner's target makes this incomparable between arms: a
        planner that advances its target on its own progress signal can be
        reported as close to an easy goal while another is reported as close to
        a hard one, and the two numbers get compared as though they meant the
        same. Measured that way the two arms once read 0.50 and 0.53 — against
        the last goal and the first.

        None only when the domain declares no goals at all; zero when every
        declared goal has been mastered.
        """
        goals = self.domain.concepts.goals()
        if not goals:
            return None
        mastery = dict(self.board.state.mastery)
        prior = bkt.initial(self.config.bkt)
        goal = route.next_goal(goals, mastery, self.config.zpd, prior)
        if goal is None:
            return 0
        return len(
            route.remaining(goal, mastery, self.domain.concepts, self.config.zpd, prior)
        )

    def _trigger_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in self.board.audit_log:
            if record.cause == "replan":
                counts[record.summary.replace("replan triggered by ", "")] += 1
        return dict(counts)

    def _next_item_for(self, concept_id: str | None, given: Mapping[str, int]):
        """Continue with the current plan: the least-practised item on it.

        ``given`` is the lifetime count, so a learner returning to a concept
        gets the item they have seen least often across their whole history
        rather than the one they saw least this sitting.
        """
        if concept_id is None:
            return None
        items = self.domain.items.for_concept(concept_id, "practice")
        if not items:
            return None
        return min(items, key=lambda item: (given.get(item.id, 0), item.id))

    def _work_item(
        self,
        item,
        diagnoses: list[tuple[str | None, str | None]],
        *,
        repetition: int = 0,
    ) -> None:
        moves: list[TutorMove] = []
        #: What the tutor has already said on this item. Handed back to it so it
        #: does not repeat itself; see the Tutor protocol.
        replies: list[str] = []
        confirmed = False

        # Three counters over the same steps, because they answer different
        # questions — and conflating two of them is what made an answer nobody
        # could read cost the learner one of their tries.
        #
        # `attempts` is what the budget bounds: steps the verifier could read.
        # `unreadable` is steps it could not, bounded separately. Those update
        # no estimate and enter no error trace, so charging one against the
        # attempt budget would spend the learner's turns on the verifier's
        # failure — but without a bound of their own an empty answer would be
        # asked for forever, so both bind.
        # `step_index` is every step in order, whatever came of it. It tells
        # `record_observation` which step was the first, so the lifetime count
        # of how often this item has been given still moves exactly once.
        attempts = 0
        unreadable = 0
        step_index = 0

        # What this arm's view believed when the question was posed, and the
        # baseline the scaffolding rule reads. Taken once, here, for two
        # reasons.
        #
        # It must be read *before* any of this item's answers are recorded. The
        # tutor read it from the live view instead, so the wrong answer it was
        # responding to had already lowered the posterior — one failure at 0.40
        # lands at 0.26, under `theta_lower / 2`, which is a full worked step on
        # its own before any escalation is added.
        #
        # And it is read from `self.board.view()`, the arm's own window, so a
        # decoupled tutor still gets the 0.0 its view would have yielded. Taking
        # it from the state directly would hand that arm a posterior the
        # manipulation withholds.
        seen = self.board.view()
        mastery = (
            seen.probability(item.concept_id, 0.0)
            if isinstance(seen, FullStateView)
            else 0.0
        )

        # A concept is worked until its posterior clears the band, and most
        # concepts carry one item — so without this the same question is asked
        # verbatim each time. The variant keeps the item's id, because it is the
        # same item asked again and the session counts repetitions by id.
        item = self.domain.variant(item, repetition)

        if self.observer is not None:
            self.observer.item_started(item, self.board)

        while attempts < self.config.cohort.max_steps_per_item:
            # The learner is told which attempt this is, not which step: an
            # unreadable one did not use anything up, and the heading a person
            # reads must not say it did.
            step = self.learner.answer(item, attempt=attempts, repetition=repetition)
            response = self.surface.render(item, step)
            result = self.domain.verifier.verify(item, response)

            diagnosis = Diagnosis(None)
            if result.verdict is Verdict.INCORRECT:
                # Only oracles are handed the injected label, and only through
                # the capability that says so.
                if isinstance(self.diagnostic, OracleAccess):
                    self.diagnostic.observe_ground_truth(step.fired)
                diagnosis = self.diagnostic.diagnose(item, response, self.domain)
                diagnoses.append((step.fired, diagnosis.misconception_id))
                confirmed = confirmed or diagnosis.named

            self.board.record_observation(
                item_id=item.id,
                concept_id=item.concept_id,
                result=result,
                misconception_label=diagnosis.misconception_id,
                confidence=diagnosis.confidence,
                attempt=step_index,
            )
            step_index += 1

            if self.observer is not None:
                self.observer.step_graded(item, response, result, diagnosis)

            # Asked after the answer is recorded, so the working cannot become a
            # hint the learner writes for themselves before committing. Prose,
            # on the same terms as a reflection: no estimate moves.
            shown = self.learner.show_working(item, response)
            if shown:
                self.board.record_reflection(
                    shown, item.id, item.concept_id, kind="working"
                )
                if self.observer is not None:
                    self.observer.working_recorded(item, shown)

            if result.verdict is Verdict.CORRECT:
                if self.observer is not None:
                    self.observer.item_finished(item, solved=True, reason="solved")
                return

            # Readable failures *before* this one. Read before the counter
            # moves, because the tutor is only ever called after a failure — a
            # count including the current step made the mastery baseline the one
            # level a learner could never be given, and `nudge` unreachable.
            prior_failures = attempts

            if result.is_evidence:
                attempts += 1
            else:
                unreadable += 1
                # And an unreadable step leaves `prior_failures` where it was.
                # Support is earned on work the verifier could read: escalating
                # on a blank submission bought a full worked step, answer
                # included, for typing nothing — at no cost, since the step
                # spends no attempt either. A person found that within a minute
                # and described it exactly: "the system's hint cheated for me."
                # What stops the reply repeating is `said_this_item`, which
                # costs nothing that was not earned.

            hint = self.tutor.respond(
                item,
                diagnosis,
                self.board.view(),
                self.domain,
                response=response,
                mastery=mastery,
                prior_failures=prior_failures,
                moves_this_item=moves,
                said_this_item=replies,
            )

            violation = check_move(hint.move, moves, misconception_confirmed=confirmed)
            if violation is not None:
                # The tutor is driven by the rules, so this should not fire.
                # Recorded rather than raised: one bad turn should not abort a
                # cohort, but it must not pass unnoticed either.
                self.board.annotate(f"pedagogy violation: {violation}", rule=violation.rule)

            # Written before the observer sees it, so a violation and the turn
            # that caused it are adjacent in the log. Nothing downstream reads
            # this; it exists so a sitting can be read back afterwards, and so
            # the tutor's output can be scored against the rules that chose its
            # move and its support level.
            self.board.record_turn(
                item_id=item.id,
                concept_id=item.concept_id,
                move=hint.move.value,
                level=hint.level.label,
                targets=hint.targets,
                text=hint.text,
            )

            if self.observer is not None:
                self.observer.tutor_replied(item, hint)

            if hint.move is TutorMove.REFLECT:
                # The tutor asked a question in words. The answer is prose, so
                # it goes nowhere near the verifier: it is not an attempt at the
                # exercise, it costs no attempt, and it is not an unmeasurable
                # step. Sending it to be graded — which is what happened before
                # this existed — told a learner their reflection was unreadable.
                said = self.learner.reflect(item, hint.text)
                if said:
                    self.board.record_reflection(said, item.id, item.concept_id)
                    if self.observer is not None:
                        self.observer.reflection_recorded(item, said)

            moves.append(hint.move)
            replies.append(hint.text)

            # Only remediation teaches. A reflective prompt costs a turn and
            # changes nothing, which is what gives the error-first rule a price.
            if hint.move is TutorMove.REMEDIATE:
                self.learner.receive_hint(hint.targets)

            if unreadable >= self.config.cohort.max_unreadable_per_item:
                # Checked after the reply, so the last thing that happens is the
                # tutor saying something rather than the question vanishing. The
                # item is over without a single attempt having been spent, which
                # is the honest bookkeeping: nothing about this learner was
                # measured here.
                self.board.annotate(
                    f"{unreadable} unreadable response(s) on {item.id}; moving on "
                    f"without having measured anything",
                    item_id=item.id,
                    concept_id=item.concept_id,
                    unreadable=unreadable,
                    max_unreadable_per_item=(
                        self.config.cohort.max_unreadable_per_item
                    ),
                )
                if self.observer is not None:
                    self.observer.item_finished(
                        item, solved=False, reason="unreadable"
                    )
                return

        # The attempts ran out. Said explicitly, because it previously looked
        # exactly like nothing happening — the next question simply appeared,
        # and a person had no way to tell whether they had got the last one
        # right. Nothing about the loop changes here; it is a report.
        if self.observer is not None:
            self.observer.item_finished(item, solved=False, reason="attempts_spent")


def build_session(
    learner_id: str,
    seed: int,
    domain: Domain,
    config: Config,
    learner: Learner | None = None,
    observer: SessionObserver | None = None,
    state: LearnerState | None = None,
    profile: MisconceptionProfile | None = None,
    planner_state: Mapping[str, Any] | None = None,
    elapsed_days: float = 0.0,
) -> Session:
    """Assemble a session from a config.

    Each role is built from the implementation its config names. Providers are
    constructed only for roles that actually use one, so an oracle diagnostic
    never opens a connection.

    ``learner`` and ``observer`` let a front end put a person in this loop and
    watch it, rather than writing a second loop that would drift from the one
    the cohorts run.

    ``state``, ``profile`` and ``elapsed_days`` are how a learner continues a
    sequence. Both must be passed together for a simulated learner: the state is
    what the system believes and the profile is what is true, and resuming one
    without the other would put a model that remembers alongside a learner who
    starts over — or the reverse. The caller supplies them because it holds the
    store; nothing here reads a database.
    """
    agents = config.agents
    cache_dir = config.paths.cache_dir

    if agents.tutor.impl == "template":
        tutor: Tutor = TemplateTutor(config.zpd)
    else:
        tutor = LLMTutor(build_provider(agents.tutor, cache_dir), config.zpd)

    if agents.diagnostic.impl == "oracle":
        diagnostic: Diagnostic = OracleDiagnostic()
    elif agents.diagnostic.impl == "noised_oracle":
        diagnostic = NoisedOracleDiagnostic(agents.diagnostic.noise_rate, seed=seed)
    else:
        diagnostic = LLMDiagnostic(
            build_provider(agents.diagnostic, cache_dir),
            label_space=agents.diagnostic.label_space,
        )

    if config.simulator.surface != "symbolic":
        raise NotImplementedForModels(
            "the model-backed surface renderer is not built yet; "
            "use simulator.surface: symbolic"
        )

    # The arm decides the view, and the view decides which planners are even
    # usable: no planner that routes from the learner model can run on a view
    # that carries none.
    prior = bkt.initial(config.bkt)
    # A resumed profile carries the remediation and forgetting of every prior
    # session; a fresh one is drawn from the seed. Same draw either way for a
    # first session, which is what keeps single-session runs unchanged.
    profile = profile or sample_profile(
        learner_id, seed, domain.misconceptions, config.simulator
    )

    if config.arm == "decoupled":
        planner: Planner = FixedOrderPlanner(
            agents.planner.advance_after, agents.planner.on_exhaustion
        )
    elif agents.planner.impl == "llm":
        planner = LLMPlanner(
            build_provider(agents.planner, cache_dir),
            config.zpd,
            prior,
            agents.planner.emphasis,
        )
    elif agents.planner.impl == "greedy":
        planner = FrontierPlanner()
    elif agents.planner.impl == "reverse":
        planner = ReverseOrderPlanner(config.zpd, prior)
    elif agents.planner.impl == "shuffled":
        planner = ShuffledPlanner(config.zpd, prior, seed)
    elif agents.planner.impl == "oracle":
        # The only agent handed the learner's true profile, and only because the
        # config names it. `profile.firing` is passed live, so remediation during
        # the session is visible without the loop reporting anything.
        planner = OraclePlanner(profile.firing, config.zpd, prior)
    else:
        planner = GoalDirectedPlanner(config.zpd, prior, agents.planner.emphasis)
    # Only a planner that declares `Resumable` has anything to take back,
    # and only the decoupled one does — see the protocol for why that is the
    # architecture showing through rather than an oversight.
    if planner_state is not None and isinstance(planner, Resumable):
        planner.restore(planner_state)

    board = (
        resumed_blackboard(state, domain.concepts, config)
        if state is not None
        else new_blackboard(learner_id, seed, domain.concepts, config)
    )
    return Session(
        learner=learner or SimulatedLearner(profile, domain, config.simulator),
        board=board,
        planner=planner,
        tutor=tutor,
        diagnostic=diagnostic,
        surface=SymbolicSurface(),
        domain=domain,
        config=config,
        arbitration=ArbitrationPolicy(config.arbitration),
        observer=observer,
        elapsed_days=elapsed_days,
        resumed=state is not None,
    )
