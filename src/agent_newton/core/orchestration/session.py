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

from agent_newton.config import Config
from agent_newton.core.agents.base import (
    Diagnosis,
    Diagnostic,
    OracleAccess,
    Planner,
    Tutor,
)
from agent_newton.core.agents.diagnostic import NoisedOracleDiagnostic, OracleDiagnostic
from agent_newton.core.agents.planner import FixedOrderPlanner, FrontierPlanner
from agent_newton.core.agents.tutor import TemplateTutor
from agent_newton.core.evaluation.outcomes import SessionOutcome, administer
from agent_newton.core.pedagogy import TutorMove, check_move
from agent_newton.core.simulator import (
    SimulatedLearner,
    SurfaceRenderer,
    SymbolicSurface,
    sample_profile,
)
from agent_newton.core.state.store import Blackboard, new_blackboard
from agent_newton.domains.base import Domain, Verdict


class NotImplementedForModels(NotImplementedError):
    """Raised when a config names a model-backed agent before one exists."""


@dataclass
class Session:
    """A single learner working through practice items."""

    learner: SimulatedLearner
    board: Blackboard
    planner: Planner
    tutor: Tutor
    diagnostic: Diagnostic
    surface: SurfaceRenderer
    domain: Domain
    config: Config

    def run(self) -> SessionOutcome:
        pretest = administer(
            self.domain.items.bank("pretest"), self.learner, self.domain, self.surface
        )

        given: Counter[str] = Counter()
        diagnoses: list[tuple[str | None, str | None]] = []
        exhausted: int | None = None

        for _ in range(self.config.cohort.max_items):
            item = self.planner.select(self.board.view(), self.domain, given)
            if item is None:
                # Nothing left to teach: the frontier emptied, or the syllabus
                # ran out. When that happened is an outcome in its own right.
                exhausted = sum(given.values())
                self.board.annotate(
                    "nothing left to select", items_given=sum(given.values())
                )
                break

            given[item.id] += 1
            self._work_item(item, diagnoses)

        posttest = administer(
            self.domain.items.bank("posttest"), self.learner, self.domain, self.surface
        )

        return SessionOutcome(
            learner_id=self.learner.profile.learner_id,
            arm=self.config.arm,
            pretest=pretest,
            posttest=posttest,
            items_attempted=sum(given.values()),
            items_to_exhaustion=exhausted,
            remediation_ratio=self.learner.profile.remediation_ratio(),
            unmeasurable_steps=self.board.unmeasurable,
            diagnoses=tuple(diagnoses),
        )

    def _work_item(
        self, item, diagnoses: list[tuple[str | None, str | None]]
    ) -> None:
        moves: list[TutorMove] = []
        confirmed = False
        failed = 0

        for attempt in range(self.config.cohort.max_steps_per_item):
            step = self.learner.answer(item, attempt=attempt)
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

            if result.verdict is Verdict.CORRECT:
                return
            failed += 1

            hint = self.tutor.respond(
                item,
                diagnosis,
                self.board.view(),
                self.domain,
                failed_attempts=failed,
                moves_this_item=moves,
            )

            violation = check_move(hint.move, moves, misconception_confirmed=confirmed)
            if violation is not None:
                # The tutor is driven by the rules, so this should not fire.
                # Recorded rather than raised: one bad turn should not abort a
                # cohort, but it must not pass unnoticed either.
                self.board.annotate(f"pedagogy violation: {violation}", rule=violation.rule)

            moves.append(hint.move)

            # Only remediation teaches. A reflective prompt costs a turn and
            # changes nothing, which is what gives the error-first rule a price.
            if hint.move is TutorMove.REMEDIATE:
                self.learner.receive_hint(hint.targets)


def build_session(
    learner_id: str,
    seed: int,
    domain: Domain,
    config: Config,
) -> Session:
    """Assemble a session from a config, refusing anything not yet built."""
    agents = config.agents

    if agents.tutor.impl != "template":
        raise NotImplementedForModels(
            f"tutor impl {agents.tutor.impl!r} needs the provider layer; use 'template'"
        )
    if agents.planner.impl not in ("deterministic", "oracle"):
        raise NotImplementedForModels(
            f"planner impl {agents.planner.impl!r} needs the provider layer; "
            f"use 'deterministic'"
        )

    if agents.diagnostic.impl == "oracle":
        diagnostic: Diagnostic = OracleDiagnostic()
    elif agents.diagnostic.impl == "noised_oracle":
        diagnostic = NoisedOracleDiagnostic(agents.diagnostic.noise_rate, seed=seed)
    else:
        raise NotImplementedForModels(
            f"diagnostic impl {agents.diagnostic.impl!r} needs the provider layer"
        )

    if config.simulator.surface != "symbolic":
        raise NotImplementedForModels(
            "the model-backed surface renderer needs the provider layer; "
            "use simulator.surface: symbolic"
        )

    # The arm decides the view, and the view decides which planner is even
    # usable: FrontierPlanner cannot run on a view with no frontier.
    planner: Planner = (
        FrontierPlanner()
        if config.arm == "coupled"
        else FixedOrderPlanner(agents.planner.advance_after)
    )

    profile = sample_profile(learner_id, seed, domain.misconceptions, config.simulator)
    return Session(
        learner=SimulatedLearner(profile, domain, config.simulator),
        board=new_blackboard(learner_id, seed, domain.concepts, config),
        planner=planner,
        tutor=TemplateTutor(config.zpd),
        diagnostic=diagnostic,
        surface=SymbolicSurface(),
        domain=domain,
        config=config,
    )
