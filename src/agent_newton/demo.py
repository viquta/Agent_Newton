"""A human-in-the-loop session, with the blackboard visible while it runs.

    uv run agent-newton demo --config experiments/configs/demo.yaml

This drives the real
:class:`~agent_newton.core.orchestration.session.Session` — the same planner,
verifier, arbitration policy and shared state the cohorts run. A person answers
where a simulated learner otherwise would, and a
:class:`~agent_newton.core.orchestration.session.SessionObserver` renders what
the blackboard is doing between steps.

The panel is the point. It shows the things a reader is otherwise asked to take
on trust: per-concept mastery moving as answers arrive, the frontier narrowing,
the goal and how much of the route to it remains, and replanning firing with the
evidence that caused it.

**A person has no injected misconception label**, so the diagnostic agent has to
infer one from the step alone — the configuration check in ``config.py`` rejects
an oracle here. Diagnostic accuracy is therefore not computable and is reported
as unavailable, never as zero.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from agent_newton.config import Config
from agent_newton.core.agents.base import Diagnosis, Hint, Resumable
from agent_newton.core.orchestration.session import Watching, build_session
from agent_newton.core.simulator.human import HumanLearner
from agent_newton.core.evaluation import outcomes
from agent_newton.core.state import bkt, route
from agent_newton.core.state.store import Blackboard
from agent_newton.domains import registry
from agent_newton.manifest import RunManifest
from agent_newton.runs import new_run_dir
from agent_newton.store import LearnerStore
from agent_newton.domains.base import Domain, Item, VerificationResult, Verdict

#: Where a mastery bar sits between "no idea" and "done".
_BAR = 18

QUIT = ":q"


def _bar(value: float, band) -> Text:
    """One concept's mastery, with the band's two thresholds marked."""
    filled = int(round(value * _BAR))
    text = Text()
    for cell in range(_BAR):
        position = (cell + 0.5) / _BAR
        if cell < filled:
            colour = (
                "green"
                if value >= band.theta_upper
                else "yellow"
                if value >= band.theta_lower
                else "cyan"
            )
            text.append("█", style=colour)
        elif abs(position - band.theta_upper) < 0.5 / _BAR:
            text.append("│", style="dim green")
        elif abs(position - band.theta_lower) < 0.5 / _BAR:
            text.append("│", style="dim yellow")
        else:
            text.append("·", style="dim")
    return text


class DemoObserver(Watching):
    """Renders the blackboard between steps.

    Only reads. It cannot change what the session does, which is what keeps the
    demo a view of the real system rather than a variant of it.

    Subclasses :class:`Watching` so a hook added to the session later does not
    break a sitting the first time the loop reaches it.
    """

    def __init__(self, console: Console, domain: Domain, config: Config) -> None:
        self._console = console
        self._domain = domain
        self._config = config
        self._prior = bkt.initial(config.bkt)
        self._seen_versions = 0

    def board_panel(self, board: Blackboard) -> Panel:
        graph = self._domain.concepts
        mastery = dict(board.state.mastery)
        frontier = board.frontier
        plan = board.plan

        table = Table(show_header=False, box=None, padding=(0, 1))
        for concept_id in graph.topological_order():
            value = mastery.get(concept_id, self._prior)
            marker = "▶" if concept_id in frontier else " "
            style = "bold" if concept_id in frontier else "dim"
            table.add_row(
                Text(marker, style="bold cyan"),
                Text(graph.get(concept_id).name[:34], style=style),
                _bar(value, self._config.zpd),
                Text(f"{value:.2f}", style=style),
            )

        header = Text()
        if plan is not None:
            remaining = route.remaining(
                plan.goal, mastery, graph, self._config.zpd, self._prior
            )
            header.append("goal  ", style="dim")
            header.append(graph.get(plan.goal).name, style="bold magenta")
            header.append(f"   {len(remaining)} concept(s) to go", style="dim")
            header.append(f"   ({plan.emphasis.value})\n", style="dim")

        errors = board.state.error_trace[-4:]
        trace = Text()
        if errors:
            trace.append("\nrecent errors\n", style="dim")
            for event in errors:
                trace.append(f"  {event.concept_id}", style="red")
                if event.misconception_label:
                    trace.append(f" — {event.misconception_label}", style="dim red")
                trace.append("\n")

        return Panel(
            Group(header, table, trace),
            title="shared learner state",
            border_style="blue",
        )

    def item_started(self, item: Item, board: Blackboard) -> None:
        self._console.print()
        self._console.print(self.board_panel(board))
        for record in board.audit_log[self._seen_versions :]:
            if record.cause in ("replan", "plan"):
                self._console.print(
                    Text("  ↻ ", style="magenta")
                    + Text(record.summary, style="dim magenta")
                )
        self._seen_versions = len(board.audit_log)

    def step_graded(
        self,
        item: Item,
        response: str,
        result: VerificationResult,
        diagnosis: Diagnosis,
    ) -> None:
        if result.verdict is Verdict.CORRECT:
            self._console.print(Text("  ✓ correct", style="bold green"))
            return
        if result.verdict is Verdict.UNPARSEABLE:
            # Not scored against the learner: the verifier could not measure.
            self._console.print(
                Text(f"  ? could not read that — {result.detail}", style="yellow")
            )
            return
        line = Text("  ✗ not right", style="bold red")
        if diagnosis.misconception_id:
            line.append(f"  (diagnosed: {diagnosis.misconception_id})", style="dim")
        self._console.print(line)

    def item_finished(self, item: Item, solved: bool) -> None:
        if solved:
            return
        # Running out of attempts used to look exactly like nothing happening:
        # the next question appeared and there was no way to tell whether the
        # last one had been right. Say so, and give the answer — the item is
        # over, so nothing is given away.
        self._console.print(
            Panel(
                Text.from_markup(
                    f"That is all the attempts for this one. The answer was "
                    f"[bold]{item.answer}[/bold].\n"
                    f"You will see this idea again — the next question on it "
                    f"will use different numbers.",
                ),
                title="moving on",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    def reflection_recorded(self, item: Item, text: str) -> None:
        self._console.print(
            Text("  ✎ noted — ", style="magenta") + Text(text, style="dim magenta")
        )

    def working_recorded(self, item: Item, text: str) -> None:
        self._console.print(
            Text("  ✎ your working — ", style="magenta") + Text(text, style="dim magenta")
        )

    def session_resumed(self, elapsed_days: float, concepts_decayed: int) -> None:
        # Said plainly, because a returning learner will otherwise see estimates
        # lower than the ones they left with and read it as the system having
        # forgotten *them*, rather than as it being less sure after a gap.
        days = f"{elapsed_days:.0f} day" + ("" if 0.5 <= elapsed_days < 1.5 else "s")
        self._console.print()
        self._console.print(
            Panel(
                Text(
                    f"Welcome back — it has been about {days}.\n"
                    f"I have become less certain about {concepts_decayed} thing(s) "
                    f"you had shown me, so some of them may come round again. "
                    f"Nothing you did was forgotten; the estimates just went stale.",
                ),
                title="picking up where you left off",
                border_style="magenta",
                padding=(0, 2),
            )
        )

    def training_finished(self, reason: str, items: int) -> None:
        # Without this the post-test simply appeared, and a sitting that ended
        # on a failed item read as the system giving up rather than as the
        # budget running out.
        said = {
            "budget_spent": (
                f"That is the {items}-question training budget spent. It is a "
                f"setting, not a judgement — you were mid-concept and there was "
                f"more to do."
            ),
            "every_goal_reached": (
                f"Every goal reached, in {items} questions. There was nothing "
                f"left to work toward."
            ),
            "nothing_left_to_select": (
                f"Training ran out of material after {items} questions — nothing "
                f"selectable was left on the way to the goal."
            ),
        }.get(reason, f"Training finished after {items} questions.")
        self._console.print()
        self._console.print(
            Panel(Text(said), title="training over", border_style="yellow", padding=(0, 2))
        )

    def phase_started(self, phase: str, total: int) -> None:
        label = {"pretest": "Pre-test", "posttest": "Post-test"}.get(phase, phase)
        body = Text(
            f"{label} — {total} questions.\n\n"
            "No feedback and no hints: this measures what you can do unaided, so "
            "telling you the answers would change what it measures.\n"
        )
        if phase == "pretest":
            # Whether it steers has to match what the config actually does. The
            # two sentences are opposites, and a person who has just answered
            # fifteen questions will notice which one was true.
            body.append(
                "What you get wrong here decides what the training works on "
                "first.\n"
                if self._config.cohort.seed_from_pretest
                else "It does not steer the training that follows — the tutoring "
                "starts from scratch and learns from your practice answers, not "
                "from these.\n"
            )
        body.append("Type :q at any point to stop.")
        self._console.print()
        self._console.print(
            Panel(body, title=f"{label.lower()} · {total} questions", border_style="yellow")
        )

    def phase_answer(self, phase: str, index: int, total: int, item: Item) -> None:
        self._console.print(
            Text(f"  recorded  ({index + 1} of {total})", style="dim")
        )

    def phase_finished(self, phase: str, result) -> None:  # noqa: ANN001
        label = {"pretest": "Pre-test", "posttest": "Post-test"}.get(phase, phase)
        body = Text()
        # Out of what could be read, not out of what was asked. The two differ
        # only when the verifier failed to parse something, and the sentence
        # below has always promised those are not held against the learner —
        # while the percentage beside it counted every one of them as wrong.
        body.append(f"{result.correct} of {result.measured} correct", style="bold")
        body.append(f"   ({result.measured_score:.0%})\n", style="bold")
        if result.unmeasurable:
            body.append(
                f"{result.unmeasurable} of the {result.total} answer(s) could not "
                f"be read by the verifier, so they are left out of that score "
                f"rather than counted against you.\n",
                style="dim",
            )
        if phase == "pretest":
            missed = result.concepts_missed
            if missed:
                # Aggregates are not what someone who has just sat a test wants
                # back. Naming the concepts is the part they asked for; whether
                # it actually steers is the flag above.
                body.append("\nWhat you got wrong:\n", style="bold")
                for concept_id in missed:
                    body.append(f"  · {self._domain.concepts.get(concept_id).name}\n")
                body.append(
                    "\nThe training will start with these.\n"
                    if self._config.cohort.seed_from_pretest
                    else "\nThe training does not use this — it starts from "
                    "scratch.\n",
                    style="dim",
                )
            else:
                body.append(
                    "\nNothing missed — the training will work through the "
                    "syllabus from the start.\n",
                    style="dim",
                )
            body.append(
                "\nTraining starts now. From here you get feedback, hints, and "
                "the panel showing what the system believes about you.",
                style="dim",
            )
        self._console.print(
            Panel(body, title=f"{label.lower()} result", border_style="yellow")
        )

    def tutor_replied(self, item: Item, hint: Hint) -> None:
        self._console.print(
            Panel(
                Text(hint.text),
                title=f"tutor · {hint.move.value} · {hint.level.label}",
                border_style="green",
                padding=(0, 2),
            )
        )


class Quit(Exception):
    """The person asked to stop."""


def _days_since(store: LearnerStore, learner_id: str, config: Config) -> float:
    """Real time since this learner's last sitting, in days.

    Wall-clock, because for a person the gap is a fact about their life rather
    than a parameter. A cohort declares its gaps instead, which is why the
    sequence runner passes them in and this only applies to the demo.

    Zero for a first sitting, and zero when the previous session was never
    closed — an unfinished record should not be read as a gap of unknown length.
    """
    sessions = store.sessions(learner_id, config.arm)
    finished = [s for s in sessions if s["ended_at"]]
    if not finished:
        return 0.0
    last = datetime.fromisoformat(finished[-1]["ended_at"])
    return max(0.0, (datetime.now(timezone.utc) - last).total_seconds() / 86_400)


def _what_to_work_on(session, config: Config, domain: Domain) -> Panel:  # noqa: ANN001
    """What the sitting says is still outstanding.

    Asked for after a sitting ended mid-concept: *"make a note that this skill
    is something to work more on in the future"*. Derived from the state — the
    unmastered concepts on the way to the goal, and the misconceptions that
    actually recurred — rather than from whatever the planner happened to be
    holding when the session stopped.
    """
    graph = domain.concepts
    mastery = dict(session.board.state.mastery)
    prior = bkt.initial(config.bkt)

    goal = route.next_goal(graph.goals(), mastery, config.zpd, prior)
    body = Text()

    if goal is None:
        body.append("Every declared goal is mastered. Nothing is outstanding.\n")
        return Panel(body, title="what to work on next", border_style="magenta")

    remaining = route.remaining(goal, mastery, graph, config.zpd, prior)
    body.append("Next goal  ", style="dim")
    body.append(f"{graph.get(goal).name}\n\n", style="bold magenta")

    if remaining:
        body.append("Still to master on the way there:\n", style="bold")
        for concept_id in remaining:
            value = mastery.get(concept_id, prior)
            body.append(f"  · {graph.get(concept_id).name}")
            body.append(f"   {value:.2f}\n", style="dim")

    # Asked for directly: *"if I get 3 times wrong on a question, it should skip
    # it, note it as a weakness, and move on"*. Only populated when the sitting
    # sets a visit cap; without one nothing is ever set aside.
    if session.board.weaknesses:
        body.append("\nSet aside for now, and worth coming back to:\n", style="bold")
        for concept_id in sorted(session.board.weaknesses):
            body.append(f"  · {graph.get(concept_id).name}")
            body.append(
                f"   worked {session.board.visits(concept_id)} time(s)\n", style="dim"
            )

    # The error trace is a rolling window, so a repeat here means the
    # misconception kept coming back rather than merely appearing once.
    repeated = Counter(
        event.misconception_label
        for event in session.board.state.error_trace
        if event.misconception_label
    )
    persistent = [(label, n) for label, n in repeated.most_common() if n > 1]
    if persistent:
        body.append("\nThese kept coming back:\n", style="bold")
        for label, count in persistent:
            described = " ".join(domain.misconceptions.get(label).description.split())
            body.append(f"  · {described}")
            body.append(f"   ({count} times)\n", style="dim")

    return Panel(body, title="what to work on next", border_style="magenta")


def _did_this_teach(session, outcome, domain: Domain) -> Panel:  # noqa: ANN001
    """What the sitting actually moved, concept by concept.

    A pre-to-post percentage cannot answer the question a sitting is for. One
    sitting read −13% and looked like the system harming the learner; the cause
    was that 21 of its 24 training steps went to concepts the pre-test had
    already shown were fine, so nothing it taught was anything that needed
    teaching. That took a hand count of the transcript to find. It prints
    itself now.
    """
    body = Text()
    graph = domain.concepts

    aimed = outcomes.dose_on_gap(session.board.audit_log, outcome.pretest)
    if aimed is not None:
        body.append("Aimed at what you got wrong  ", style="dim")
        body.append(f"{aimed:.0%}", style="bold")
        body.append(" of the training\n", style="dim")
    spent = outcomes.dose_by_concept(session.board.audit_log)
    if spent:
        gaps = set(outcome.pretest.concepts_missed)
        body.append("\nWhere the time went:\n", style="bold")
        for concept_id, steps in sorted(spent.items(), key=lambda kv: -kv[1]):
            mark = "·" if concept_id in gaps else " "
            body.append(f"  {mark} {graph.get(concept_id).name}")
            body.append(f"   {steps} step(s)\n", style="dim")
        if gaps:
            body.append(
                "  · marks something the pre-test showed you had not got yet\n",
                style="dim",
            )

    if not (outcome.pretest.administered and outcome.posttest.administered):
        return Panel(body, title="did this teach you anything", border_style="green")

    changed = outcomes.per_concept_change(outcome.pretest, outcome.posttest)
    by_state: dict[str, list[str]] = {}
    for change in changed:
        by_state.setdefault(change.state, []).append(change.concept_id)

    labels = {
        "fixed": ("Now right, and were wrong before", "green"),
        "lost": ("Were right before, and slipped", "red"),
        "still_wrong": ("Still to get", "yellow"),
        "unmeasured": ("Could not be read at one end or the other", "dim"),
    }
    body.append("\nBetween the two tests:\n", style="bold")
    for state, (label, colour) in labels.items():
        found = by_state.get(state, [])
        if not found:
            continue
        body.append(f"  {label}: ", style=colour)
        body.append(f"{len(found)}\n", style=f"bold {colour}")
        for concept_id in found:
            body.append(f"      {graph.get(concept_id).name}\n", style="dim")

    normalised = outcome.normalised_gain
    if normalised is None:
        body.append(
            "\nYou had the pre-test entirely right, so there was no room to "
            "improve on it — the training was working on what comes after.\n",
            style="dim",
        )
    else:
        # Raw gain is bounded by what was not already known, and that bound
        # differs between people. Stating the share makes a small-looking figure
        # readable rather than discouraging.
        available = 1.0 - outcome.pretest.measured_score
        body.append(
            f"\nThat is {normalised:.0%} of the {available:.0%} you had left to "
            f"gain.\n",
            style="dim",
        )

    return Panel(body, title="did this teach you anything", border_style="green")


def _store(config: Config, domain: Domain, session, learner, outcome) -> Path:  # noqa: ANN001
    """Write the sitting to disk. Called however the sitting ended.

    ``outcome`` is None when the person stopped part-way, in which case the
    outcome-derived figures are absent rather than zero — the session did not
    finish, so it has no pre/post gain and no final distance to a goal, and
    writing zeros would put numbers into the record that mean the opposite of
    what they say.
    """
    run_id, run_dir = new_run_dir(config)
    manifest = RunManifest.create(config, run_id)
    for field, value in domain.content_hashes().items():
        setattr(manifest, field, value)
    manifest.write(run_dir)

    record: dict = {
        "learner_id": learner.learner_id,
        "completed": outcome is not None,
        "responses": [{"item_id": i, "response": r} for i, r in learner.responses],
        # What the tutor said back. Already in the audit log below, and lifted
        # out because a transcript of a sitting is unreadable if half of the
        # conversation has to be recovered by filtering on a cause string.
        "turns": [
            r.evidence | {"version": r.version}
            for r in session.board.audit_log
            if r.cause == "tutor"
        ],
        "said": [u.model_dump() for u in session.board.state.reflections],
        "mastery": dict(session.board.state.mastery),
        "error_trace": [e.model_dump() for e in session.board.state.error_trace],
        "unmeasurable_steps": session.board.unmeasurable,
        "audit_log": [
            {
                "version": r.version,
                "cause": r.cause,
                "summary": r.summary,
                "evidence": r.evidence,
            }
            for r in session.board.audit_log
        ],
    }
    if outcome is not None:
        record.update(
            {
                "stop_reason": outcome.stop_reason,
                "goal": outcome.goal,
                "goals_mastered": outcome.goals_mastered,
                "distance_to_goal": outcome.distance_to_goal,
                "items_attempted": outcome.items_attempted,
                "pretest": {
                    "correct": outcome.pretest.correct,
                    "total": outcome.pretest.total,
                    # Raw and measured are both kept. They differ exactly when
                    # the verifier could not read an answer, and a transcript
                    # showing only one cannot say which of the two a figure was.
                    "unmeasurable": outcome.pretest.unmeasurable,
                    "score": outcome.pretest.score,
                    "measured_score": outcome.pretest.measured_score,
                    "administered": outcome.pretest.administered,
                    "concepts_missed": list(outcome.pretest.concepts_missed),
                    "seeded_the_learner_model": config.cohort.seed_from_pretest,
                },
                "posttest": {
                    "correct": outcome.posttest.correct,
                    "total": outcome.posttest.total,
                    "unmeasurable": outcome.posttest.unmeasurable,
                    "score": outcome.posttest.score,
                    "measured_score": outcome.posttest.measured_score,
                    "administered": outcome.posttest.administered,
                    "concepts_missed": list(outcome.posttest.concepts_missed),
                },
                "gain": outcome.gain,
                "normalised_gain": outcome.normalised_gain,
                # What the sitting moved and where its time went. Derived here
                # rather than left to be recomputed, because the last sitting's
                # finding came from doing this by hand off the transcript.
                "per_concept_change": [
                    {
                        "concept_id": change.concept_id,
                        "before": change.before.value if change.before else None,
                        "after": change.after.value if change.after else None,
                        "state": change.state,
                    }
                    for change in outcomes.per_concept_change(
                        outcome.pretest, outcome.posttest
                    )
                ],
                "dose_by_concept": outcomes.dose_by_concept(session.board.audit_log),
                "dose_on_gap": outcomes.dose_on_gap(
                    session.board.audit_log, outcome.pretest
                ),
                "cross_concept_diagnoses": outcome.cross_concept_diagnoses,
            }
        )

    (run_dir / "transcript.json").write_text(json.dumps(record, indent=2) + "\n")
    return run_dir


def run_demo(
    config_path: Path,
    console: Console | None = None,
    learner_id: str = "human",
    elapsed_days: float | None = None,
) -> None:
    """One sitting for one learner.

    ``learner_id`` is who is sitting down. The same id twice picks up where that
    learner left off; a new id starts fresh. ``elapsed_days`` overrides the real
    gap since their last sitting, which is how decay can be exercised without
    waiting weeks for it.
    """
    console = console or Console()
    config = Config.from_yaml(config_path)
    domain = registry.load_domain(config.domain)

    if config.simulator.learner != "human":
        console.print(
            "[red]this config is not a human session[/red] — set "
            "simulator.learner: human"
        )
        raise SystemExit(1)

    observer = DemoObserver(console, domain, config)

    def _asked(prompt: str, *, optional: bool = False) -> str:
        """Read one line, honouring :q wherever it is typed.

        Every prompt goes through here. The intro says ":q stops at any point",
        and it only checked the answer prompt — so typing it at the working or
        reflection prompt was silently recorded as prose and the sitting carried
        on. A stated affordance that works in one place out of three is worse
        than not offering it.
        """
        said = (
            Prompt.ask(prompt, default="", show_default=False)
            if optional
            else Prompt.ask(prompt)
        ).strip()
        if said == QUIT:
            raise Quit
        return said

    def ask(item: Item, attempt: int) -> str:
        console.print()
        console.print(
            Panel(
                Text(" ".join(item.prompt.split()), style="bold"),
                title=f"{domain.name} · {item.concept_id}"
                + (f" · attempt {attempt + 1}" if attempt else ""),
                border_style="cyan",
            )
        )
        return _asked(f"  your answer  [dim]({QUIT} to stop)[/dim]")

    def ask_reflection(item: Item, prompt: str) -> str:
        # Prose, not an answer. It never reaches the verifier and costs no
        # attempt — see Session._work_item.
        return _asked(
            "  [magenta]in your own words[/magenta]  [dim](enter to skip)[/dim]",
            optional=True,
        )

    def ask_working(item: Item, response: str) -> str:
        # Asked after the answer has been graded and recorded, so it cannot
        # become a hint the learner writes for themselves. Optional every time:
        # a channel you have to fill in is a tax on answering.
        return _asked(
            "  [magenta]your working[/magenta]  [dim](optional — enter to skip)[/dim]",
            optional=True,
        )

    learner = HumanLearner(
        ask, learner_id=learner_id,
        ask_reflection=ask_reflection, ask_working=ask_working,
    )

    store = LearnerStore(config.paths.store_path)
    store.ensure_learner(learner_id, "human", config.domain)
    previous = store.latest_state(learner_id, config.arm)
    gap = elapsed_days if elapsed_days is not None else _days_since(store, learner_id, config)
    session_id = store.open_session(
        learner_id=learner_id,
        arm=config.arm,
        config_hash=config.content_hash(),
        elapsed_days=gap,
        decay_half_life_days=config.decay.half_life_days,
    )

    session = build_session(
        learner_id, config.seed, domain, config, learner=learner,
        observer=observer,
        state=previous,
        planner_state=store.latest_planner_state(learner_id, config.arm),
        elapsed_days=gap,
    )

    console.print(
        Panel(
            Text.from_markup(
                "There are three parts: a [yellow]pre-test[/yellow], then "
                "[cyan]training[/cyan], then a [yellow]post-test[/yellow].\n\n"
                "[bold]Answers[/bold] are mathematical expressions — [dim]5*x**4[/dim], "
                "[dim]5x^4[/dim] and [dim]x**4*5[/dim] are all accepted, because a "
                "symbolic verifier checks equivalence rather than spelling. For "
                "several roots, separate them with commas: [dim]0, 2[/dim].\n"
                "[bold]When the tutor asks you a question in words[/bold], answer in "
                "words — that reply is kept and is not graded.\n"
                "[bold]After every answer you may show your working[/bold], in words or "
                "maths. It is never graded and never costs an attempt; the tutor reads "
                "it, so a hint can address your actual steps. Press enter to skip.\n"
                "[bold]:q[/bold] stops at any point.\n\n"
                "During training the panel shows what the system believes about you "
                "and why it chooses what it chooses. Nothing here knows the right "
                "answer in advance: the verifier grades every step symbolically, and "
                "the diagnostic agent infers your misconception from the step alone.",
            ),
            title="Agent_Newton",
            border_style="magenta",
        )
    )

    outcome = None
    try:
        outcome = session.run()
    except (Quit, KeyboardInterrupt):
        console.print("\n[dim]stopped — the state so far:[/dim]")

    console.print()
    console.print(observer.board_panel(session.board))
    console.print(_what_to_work_on(session, config, domain))

    # Written on **every** exit path. Returning early on an interrupt is what
    # this used to do, and it meant four human sittings produced no record at
    # all: no transcript, no working, no reflections, no audit log. A sitting
    # that was stopped part-way is still the only data of its kind this project
    # has, and it is unrepeatable.
    run_dir = _store(config, domain, session, learner, outcome)

    # And into the store, so the next sitting can pick this learner up. Also on
    # every exit path: someone who stops half-way has still done the work, and
    # making them lose it would teach them to never stop.
    store.close_session(
        session_id,
        state=session.board.state,
        audit_log=session.board.audit_log,
        planner_state=(
            session.planner.snapshot()
            if isinstance(session.planner, Resumable)
            else None
        ),
        stop_reason=outcome.stop_reason if outcome else "interrupted",
    )
    store.close()

    if outcome is None:
        console.print(
            f"\n[dim]saved to {run_dir} — stopped part-way, so there is no "
            f"pre/post figure[/dim]"
        )
        return

    console.print(_did_this_teach(session, outcome, domain))

    summary = Table(title="session", show_header=False, box=None)
    summary.add_row("items attempted", str(outcome.items_attempted))
    if outcome.pretest.administered and outcome.posttest.administered:
        # Out of what the verifier could read. An answer it could not parse is
        # not a wrong one, and the panel has always said so — the score did not,
        # which made the two disagree in front of the person they were about.
        summary.add_row(
            "pre-test",
            f"{outcome.pretest.measured_score:.0%}"
            f" of {outcome.pretest.measured} read",
        )
        summary.add_row(
            "post-test",
            f"{outcome.posttest.measured_score:.0%}"
            f" of {outcome.posttest.measured} read",
        )
        summary.add_row("gain", f"{outcome.gain:+.0%}")
        if outcome.normalised_gain is not None:
            summary.add_row(
                "of what was there to gain", f"{outcome.normalised_gain:.0%}"
            )
    else:
        # A skipped bank and a bank scored zero both read 0.0, and they mean
        # opposite things.
        summary.add_row("pre/post", "not administered")
    summary.add_row("goals mastered", str(outcome.goals_mastered))
    summary.add_row("concepts to the next goal", str(outcome.distance_to_goal))
    summary.add_row("steps the verifier could not read", str(outcome.unmeasurable_steps))
    said = session.board.state.reflections
    if said:
        summary.add_row(
            "things you said", str(sum(1 for u in said if u.kind == "reflection"))
        )
        summary.add_row("working you showed", str(sum(1 for u in said if u.kind == "working")))
    if outcome.cross_concept_diagnoses:
        # Visible rather than quietly dropped: the diagnostic may name a
        # misconception from a concept other than the one being worked.
        summary.add_row(
            "diagnoses from another concept", str(outcome.cross_concept_diagnoses)
        )
    # There is no injected label for a person, so there is nothing to score the
    # diagnostic against. Unavailable, not zero.
    summary.add_row("diagnostic accuracy", "unavailable (no ground truth)")
    summary.add_row("remediation ratio", "unavailable (no ground truth)")
    console.print(Columns([summary]))
    console.print(f"\n[dim]saved to {run_dir}[/dim]")
