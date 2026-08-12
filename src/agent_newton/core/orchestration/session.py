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
from typing import Protocol, runtime_checkable

from agent_newton.config import Config
from agent_newton.core.agents.base import (
    Diagnosis,
    Diagnostic,
    Hint,
    OracleAccess,
    Planner,
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
from agent_newton.core.state.store import Blackboard, new_blackboard
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

    def item_finished(self, item: Item, solved: bool) -> None:
        """The session is done with this item, whether or not it was answered.

        Exists because running out of attempts previously looked identical to
        nothing happening: the next question simply appeared, and a person had
        no way to tell whether they had got the last one right.
        """
        ...

    def tutor_replied(self, item: Item, hint: Hint) -> None: ...

    def reflection_recorded(self, item: Item, text: str) -> None: ...

    def working_recorded(self, item: Item, text: str) -> None: ...

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

    def item_finished(self, item: Item, solved: bool) -> None:
        return None

    def tutor_replied(self, item: Item, hint: Hint) -> None:
        return None

    def reflection_recorded(self, item: Item, text: str) -> None:
        return None

    def working_recorded(self, item: Item, text: str) -> None:
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

    def run(self) -> SessionOutcome:
        pretest = self._administer("pretest")
        if self.config.cohort.seed_from_pretest:
            # Off for every cohort — see CohortConfig. It moves the starting
            # frontier, and only one arm can route from a frontier.
            self.board.seed_from_test(
                (result.concept_id, result.verdict) for result in pretest.per_item
            )

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

        for _ in range(self.config.cohort.max_items):
            decision = self.arbitration.evaluate(
                current_concept=working,
                mastery=dict(self.board.state.mastery),
                frontier=self.board.frontier,
                error_trace=list(self.board.state.error_trace),
                prior=bkt.initial(self.config.bkt),
            )

            if decision.replan:
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

                item = self.planner.select(self.board.view(), self.domain, given)
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
                self.board.record_replan(decision.summary, **decision.evidence)
                self.arbitration.accept(dict(self.board.state.mastery))
            else:
                if decision.suppressed_by:
                    # A trigger that fired and was held back is worth recording:
                    # it is the difference between the threshold deciding and
                    # the rate limit deciding.
                    self.board.annotate(decision.summary, **decision.evidence)
                item = self._next_item_for(working, given)
                if item is None:
                    exhausted = sum(given.values())
                    stop_reason = "nothing_left_to_select"
                    self.board.annotate(
                        "no item left on the current concept",
                        items_given=sum(given.values()),
                        concept_id=working,
                    )
                    break

            given[item.id] += 1
            self.arbitration.note_item()
            self._work_item(item, diagnoses, repetition=given[item.id] - 1)

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

        posttest = self._administer("posttest")

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

    def _administer(self, bank) -> TestResult:
        """Run a held-out bank, unless the config says not to.

        A skipped bank returns an empty result rather than a zero score, so
        `TestResult.administered` can tell the two apart.
        """
        if not self.config.cohort.administer_tests:
            return TestResult(correct=0, total=0)

        items = self.domain.items.bank(bank)
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

    def _next_item_for(self, concept_id: str | None, given: Counter[str]):
        """Continue with the current plan: the least-practised item on it."""
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
        confirmed = False
        failed = 0

        # A concept is worked until its posterior clears the band, and most
        # concepts carry one item — so without this the same question is asked
        # verbatim each time. The variant keeps the item's id, because it is the
        # same item asked again and the session counts repetitions by id.
        item = self.domain.variant(item, repetition)

        if self.observer is not None:
            self.observer.item_started(item, self.board)

        for attempt in range(self.config.cohort.max_steps_per_item):
            step = self.learner.answer(item, attempt=attempt, repetition=repetition)
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
            )

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
                    self.observer.item_finished(item, solved=True)
                return
            failed += 1

            hint = self.tutor.respond(
                item,
                diagnosis,
                self.board.view(),
                self.domain,
                response=response,
                failed_attempts=failed,
                moves_this_item=moves,
            )

            violation = check_move(hint.move, moves, misconception_confirmed=confirmed)
            if violation is not None:
                # The tutor is driven by the rules, so this should not fire.
                # Recorded rather than raised: one bad turn should not abort a
                # cohort, but it must not pass unnoticed either.
                self.board.annotate(f"pedagogy violation: {violation}", rule=violation.rule)

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

            # Only remediation teaches. A reflective prompt costs a turn and
            # changes nothing, which is what gives the error-first rule a price.
            if hint.move is TutorMove.REMEDIATE:
                self.learner.receive_hint(hint.targets)

        # The attempts ran out. Said explicitly, because it previously looked
        # exactly like nothing happening — the next question simply appeared,
        # and a person had no way to tell whether they had got the last one
        # right. Nothing about the loop changes here; it is a report.
        if self.observer is not None:
            self.observer.item_finished(item, solved=False)


def build_session(
    learner_id: str,
    seed: int,
    domain: Domain,
    config: Config,
    learner: Learner | None = None,
    observer: SessionObserver | None = None,
) -> Session:
    """Assemble a session from a config.

    Each role is built from the implementation its config names. Providers are
    constructed only for roles that actually use one, so an oracle diagnostic
    never opens a connection.

    ``learner`` and ``observer`` let a front end put a person in this loop and
    watch it, rather than writing a second loop that would drift from the one
    the cohorts run.
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
    profile = sample_profile(learner_id, seed, domain.misconceptions, config.simulator)

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
    return Session(
        learner=learner or SimulatedLearner(profile, domain, config.simulator),
        board=new_blackboard(learner_id, seed, domain.concepts, config),
        planner=planner,
        tutor=tutor,
        diagnostic=diagnostic,
        surface=SymbolicSurface(),
        domain=domain,
        config=config,
        arbitration=ArbitrationPolicy(config.arbitration),
        observer=observer,
    )
