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
from agent_newton.core.orchestration.session import (
    StopTraining,
    Watching,
    build_session,
)
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
#: Ends training and goes straight to the post-test. Distinct from `:q`, which
#: ends the sitting where it stands: someone who has had enough of the questions
#: has still done the work, and the post-test is what turns it into a measured
#: result rather than a transcript.
END_TRAINING = ":e"
#: Declines a prompt that would otherwise insist. A refusal that can be recorded
#: is worth more than a field somebody filled with a full stop to get past it.
DECLINE = ":s"


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


def _readable(expression: str) -> str:
    """An answer in the notation the learner was asked to use.

    The bank stores what the verifier parses, which is Python's: ``x**6``,
    ``2*x``. The intro tells the learner to write ``x^6`` and the tutor is told
    the same, so the one place the system wrote sympy back at a person was the
    line revealing the answer.

    Deliberately only cosmetic — it never goes near the verifier, which keeps
    reading the stored form.
    """
    return expression.replace("**", "^").replace("*", "")


def question_title(
    domain: Domain, item: Item, attempt: int, *, testing: bool
) -> str:
    """The heading above a question.

    ⚠️ **The concept is named during training and withheld during a test.**
    Every question used to be titled ``calculus · chain_rule``, including in the
    held-out banks — which told the learner which rule to reach for on a test
    meant to measure what they can do unaided. A person noticed: *"I'm kind of
    cheating here cause I can see what I need to use to solve it."* Every
    absolute pre- and post-test score recorded before this is inflated by it.

    During training it stays, because there the panel showing what the system
    believes and why is the entire point of the demo.
    """
    if testing:
        return domain.name
    return f"{domain.name} · {item.concept_id}" + (
        f" · attempt {attempt + 1}" if attempt else ""
    )


class DemoObserver(Watching):
    """Renders the blackboard between steps.

    Only reads. It cannot change what the session does, which is what keeps the
    demo a view of the real system rather than a variant of it.

    Subclasses :class:`Watching` so a hook added to the session later does not
    break a sitting the first time the loop reaches it.
    """

    def __init__(
        self,
        console: Console,
        domain: Domain,
        config: Config,
        carried: tuple = (),
    ) -> None:
        self._console = console
        self._domain = domain
        self._config = config
        self._prior = bkt.initial(config.bkt)
        self._seen_versions = 0
        #: What this learner said in *earlier* sittings, by concept. Captured
        #: before the session starts, because the state carries every word ever
        #: said and there is otherwise no way to tell last month's from this
        #: minute's — the utterances have no timestamp.
        self._before: dict[str, list[str]] = {}
        for utterance in carried:
            self._before.setdefault(utterance.concept_id, []).append(utterance.text)
        #: Concepts already reminded about this sitting. Once each: the point is
        #: to be reminded, not nagged.
        self._reminded: set[str] = set()
        #: True while a held-out bank is being administered. The front end reads
        #: it to withhold the concept name — see :func:`question_title`. Held
        #: here rather than passed around because the session already tells the
        #: observer when a phase starts and ends, and adding a second channel
        #: for the same fact would let the two disagree.
        self.testing = False

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
        self._remind(item)
        for record in board.audit_log[self._seen_versions :]:
            if record.cause in ("replan", "plan"):
                self._console.print(
                    Text("  ↻ ", style="magenta")
                    + Text(record.summary, style="dim magenta")
                )
        self._seen_versions = len(board.audit_log)

    def _remind(self, item: Item) -> None:
        """What this learner said about this concept before today.

        Shown once per concept per sitting, when it comes round again. A learner
        who wrote down how they were thinking about something a month ago is the
        only one who can say whether they still think it — the system cannot,
        and should not pretend to by summarising it.

        Quoted flat, without comment. It may well be a misconception stated in
        their own words, and dressing it up as advice would be teaching it back
        to them.
        """
        said = self._before.get(item.concept_id)
        if not said or item.concept_id in self._reminded:
            return
        self._reminded.add(item.concept_id)
        body = Text()
        # Named, because it was not: the panel appeared with two quotes and no
        # subject, and the learner asked *"are these notes from the quotient
        # rule? or the product rule? Why are these relevant?"* Words carry no
        # timestamp, so "before today" is as precise as this can honestly be.
        body.append("Before today, on ", style="dim")
        body.append(self._domain.concepts.get(item.concept_id).name, style="bold")
        body.append(", you wrote:\n\n", style="dim")
        for text in said[-2:]:
            body.append(f"  “{text}”\n", style="italic")
        body.append(
            "\nWorth a look — do you still think that, or has it changed?",
            style="dim",
        )
        self._console.print(
            Panel(body, title="from an earlier sitting", border_style="magenta",
                  padding=(0, 2))
        )

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

    def item_finished(
        self, item: Item, solved: bool, reason: str = "attempts_spent"
    ) -> None:
        if solved:
            return
        # Running out of attempts used to look exactly like nothing happening:
        # the next question appeared and there was no way to tell whether the
        # last one had been right. Say so, and give the answer — the item is
        # over, so nothing is given away, and the next question on this concept
        # carries different numbers.
        #
        # The two ways of not finishing are said differently, because they are
        # not the same thing and the learner is the one who can tell: after
        # three unreadable answers nobody's attempts were spent and nothing
        # about them was measured, and "that is all the attempts for this one"
        # was simply false.
        # Written the way the learner is asked to write, not the way sympy
        # prints: `2*x*(x**6 + 2) + (x**2 + 1)*6*x**5` was shown to someone who
        # had been told to type `x^6`.
        answer = _readable(item.answer)
        said = (
            f"None of that could be read, so no attempts were used and nothing "
            f"was recorded about you. The answer was [bold]{answer}[/bold]."
            if reason == "unreadable"
            else f"That is all the attempts for this one. The answer was "
            f"[bold]{answer}[/bold]."
        )
        self._console.print(
            Panel(
                Text.from_markup(
                    f"{said}\n"
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
            "learner_ended_it": (
                f"You ended training after {items} question(s). The post-test "
                f"still runs — it is what turns the work you did into a measured "
                f"result rather than a transcript."
            ),
        }.get(reason, f"Training finished after {items} questions.")
        self._console.print()
        self._console.print(
            Panel(Text(said), title="training over", border_style="yellow", padding=(0, 2))
        )

    def phase_started(self, phase: str, total: int) -> None:
        # The bank is held out, so the concept names come off the questions
        # until it is over.
        self.testing = True
        label = {"pretest": "Pre-test", "posttest": "Post-test"}.get(phase, phase)
        body = Text(
            f"{label} — {total} questions.\n\n"
            "No feedback and no hints: this measures what you can do unaided, so "
            "telling you the answers would change what it measures.\n"
        )
        whole = len(self._domain.items.bank(phase))  # type: ignore[arg-type]
        if total < whole:
            # Read off the bank rather than from the config, so what is said
            # here cannot disagree with what is actually being administered.
            # Said at all because a returning learner has sat the long version
            # and would otherwise be left wondering what happened to it — and
            # because a score out of a shorter bank is not the same measurement
            # as the one they got last time.
            body.append(
                f"Short, because you have been here before: these are the {total} "
                f"of {whole} that are on the way to your next goal. The rest is "
                f"either behind you or too far ahead to be worth asking about "
                f"today.\n",
                style="dim",
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
        self.testing = False
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
            elif result.total < len(self._domain.items.bank("pretest")):
                # A clean *re-check* is not a blank slate. Saying "from the
                # start" to someone who has just demonstrated the whole route
                # describes a different sitting from the one about to happen —
                # the training goes on past what was measured, which is why the
                # post-test now asks about that too.
                body.append(
                    "\nNothing missed on the re-check, so the training moves "
                    "past it — on to what comes next on the way to your goal. "
                    "The post-test will ask about whatever you work on today.\n",
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


def _ask_what_to_practise(
    console: Console, domain: Domain, board, config: Config, asked
) -> frozenset[str]:  # noqa: ANN001
    """Let the learner say what they came for, before anything is measured.

    Asked first because it is the one thing the system cannot infer. Everything
    else on the blackboard is an inference *about* the learner; this is the
    learner talking.

    What it does is stated plainly, including what it does not do. The request
    moves which **goal** is aimed at, so the route runs through what was asked
    for — it does not skip the prerequisites on the way, and it does not
    reopen a concept the model already believes is mastered. Saying so here is
    the difference between a preference and a promise.
    """
    graph = domain.concepts
    prior = bkt.initial(config.bkt)
    mastery = dict(board.state.mastery)

    table = Table(show_header=False, box=None, padding=(0, 1))
    ids = list(graph.topological_order())
    for number, concept_id in enumerate(ids, start=1):
        value = mastery.get(concept_id, prior)
        table.add_row(
            Text(f"{number:>2}", style="cyan"),
            Text(graph.get(concept_id).name[:38]),
            Text(f"{value:.2f}", style="dim"),
        )
    console.print()
    console.print(
        Panel(
            Group(
                Text(
                    "What would you like to work on? The numbers beside each are "
                    "what the system currently believes.\n",
                    style="dim",
                ),
                table,
            ),
            title="what shall we practise",
            border_style="magenta",
        )
    )
    said = asked(
        "  [magenta]numbers, comma separated[/magenta]  "
        "[dim](enter for no preference)[/dim]",
        optional=True,
    )
    chosen: list[str] = []
    for part in said.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(ids):
            chosen.append(ids[int(part) - 1])

    requested = board.record_request(chosen)
    if not requested:
        console.print(
            "  [dim]no preference — the route to the next goal decides[/dim]"
        )
        return requested

    body = Text()
    for concept_id in sorted(requested, key=ids.index):
        name = graph.get(concept_id).name
        if route.reached(concept_id, mastery, config.zpd, prior):
            # Honest rather than accommodating: the band is what decides what is
            # offered, and a request cannot waive it without making mastery mean
            # something different for the person who asked.
            body.append(f"  {name}", style="bold")
            body.append(
                f" — the model has you at {mastery.get(concept_id, prior):.2f} "
                f"here, so it will not come round unless that goes stale.\n",
                style="dim",
            )
            continue
        needed = [
            c
            for c in graph.all_prerequisites(concept_id)
            if mastery.get(c, prior) < config.zpd.theta_lower
        ]
        body.append(f"  {name}", style="bold")
        if needed:
            body.append(
                f" — {len(needed)} thing(s) come first: "
                + ", ".join(graph.get(c).name for c in needed)
                + ".\n",
                style="dim",
            )
        else:
            body.append(" — ready to work on now.\n", style="dim")
    console.print(
        Panel(body, title="what that means", border_style="magenta", padding=(0, 2))
    )
    return requested


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
        covered = outcome.pretest.covered
        # Only when the pre-test was narrowed to the route. Then some training
        # went to concepts it never asked about, and those are outside the
        # percentage above rather than counted against it — which has to be
        # visible, or the share reads as covering everything below it.
        unasked = [c for c in spent if outcome.pretest.administered and c not in covered]
        body.append("\nWhere the time went:\n", style="bold")
        for concept_id, steps in sorted(spent.items(), key=lambda kv: -kv[1]):
            mark = "·" if concept_id in gaps else "?" if concept_id in unasked else " "
            body.append(f"  {mark} {graph.get(concept_id).name}")
            body.append(f"   {steps} step(s)\n", style="dim")
        if gaps:
            body.append(
                "  · marks something the pre-test showed you had not got yet\n",
                style="dim",
            )
        if unasked:
            body.append(
                "  ? marks something the pre-test did not ask about, so it is "
                "outside the share above\n",
                style="dim",
            )

    if not (outcome.pretest.administered and outcome.posttest.administered):
        return Panel(body, title="did this teach you anything", border_style="green")

    # Matched against the pre-test's own concepts. The post-test also covers
    # what the sitting taught, and a concept with no "before" cannot be reported
    # as having changed — it gets its own block below.
    changed = outcomes.per_concept_change(outcome.pretest, outcome.matched_posttest)
    by_state: dict[str, list[str]] = {}
    for change in changed:
        by_state.setdefault(change.state, []).append(change.concept_id)

    labels = {
        "fixed": ("Now right, and were wrong before", "green"),
        "lost": ("Were right before, and slipped", "red"),
        "still_wrong": ("Still to get", "yellow"),
        "unmeasured": ("Could not be read at one end or the other", "dim"),
    }
    # `still_right` is deliberately absent from `labels`: a concept that was
    # fine at both ends is not something the sitting moved. So the heading is
    # written only when something falls under it — otherwise a learner who
    # passed the whole re-check gets a heading with nothing beneath it, which
    # reads as a figure that failed to render.
    if any(by_state.get(state) for state in labels):
        body.append("\nBetween the two tests:\n", style="bold")
    for state, (label, colour) in labels.items():
        found = by_state.get(state, [])
        if not found:
            continue
        body.append(f"  {label}: ", style=colour)
        body.append(f"{len(found)}\n", style=f"bold {colour}")
        for concept_id in found:
            body.append(f"      {graph.get(concept_id).name}\n", style="dim")

    # ⚠️ The part a sitting used to say nothing about. Every step of one went to
    # the product rule, which neither bank asked about, and the figures reported
    # a gain of zero over six other concepts. True, and silent about the only
    # thing that was taught.
    beyond = outcome.taught_beyond_the_pretest
    if beyond.administered:
        body.append("\nWhat you were taught today, tested afterwards:\n", style="bold")
        for result in beyond.per_item:
            mark = "✓" if result.verdict is Verdict.CORRECT else "·"
            body.append(f"  {mark} {graph.get(result.concept_id).name}")
            body.append(f"   {result.verdict.value}\n", style="dim")
        body.append(
            "  no before-and-after for these — the pre-test could not know they "
            "were coming, so this is what you could do afterwards and not a "
            "gain.\n",
            style="dim",
        )

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


def _across_sittings(  # noqa: ANN001
    store: LearnerStore, learner_id: str, config: Config, domain: Domain
) -> Panel | None:
    """What has been tried on each concept over this learner's whole history.

    None for a first sitting, where there is no history to be across.

    A learner who never grasps something despite sustained teaching is an
    ordinary case rather than a failure. What can be shown either way is what
    was tried — and, where the repertoire was not exhausted, what was not. That
    last part is the one that can come back against the system.
    """
    from agent_newton.core.evaluation.teaching import records

    history = [r for r in records(store, learner_id, config.arm) if r.sittings_spanned > 1]
    if not history:
        return None

    graph = domain.concepts
    body = Text()
    body.append(
        "Concepts you have worked in more than one sitting.\n\n", style="dim"
    )
    for record in sorted(history, key=lambda r: -r.attempts)[:6]:
        body.append(f"  {graph.get(record.concept_id).name}\n", style="bold")
        body.append(
            f"      {record.attempts} question(s) over {record.sittings_spanned} "
            f"sittings, {record.days_spanned:.0f} days\n",
            style="dim",
        )
        if record.movement is not None:
            direction = "up" if record.movement > 0 else "down" if record.movement < 0 else "flat"
            body.append(
                f"      the estimate has gone {direction} "
                f"({record.movement:+.2f}) over that time\n",
                style="dim",
            )
        outstanding = record.not_attempted
        if outstanding is None:
            # Sittings before turns were kept. Absent, not untried.
            body.append(
                "      what was said was not kept for some of these sittings\n",
                style="dim",
            )
        elif outstanding:
            body.append(
                f"      not yet tried here: "
                f"{', '.join(sorted(x.split(':', 1)[1] for x in outstanding))}\n",
                style="dim yellow",
            )
        else:
            body.append(
                "      every kind of help available has been tried on this\n",
                style="dim green",
            )

    return Panel(body, title="across your sittings", border_style="magenta")


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
                    # What the bank asked about at all. A score over a narrowed
                    # bank is not comparable to one over the whole of it, and a
                    # transcript recording only the score cannot say which it
                    # was — nor which concepts a gap could have been found in.
                    "concepts_measured": sorted(outcome.pretest.covered),
                    "scope": config.cohort.pretest_scope,
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
                    "concepts_measured": sorted(outcome.posttest.covered),
                    # The two halves kept apart, because only the first has a
                    # baseline and a reader cannot tell them apart from a score.
                    "concepts_matched_to_pretest": sorted(outcome.pretest.covered),
                    "concepts_taught_beyond_pretest": sorted(
                        outcome.taught_beyond_the_pretest.covered
                    ),
                },
                # Over the concepts both banks asked about. See SessionOutcome.
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

    store = LearnerStore(config.paths.store_path)
    store.ensure_learner(learner_id, "human", config.domain)
    previous = store.latest_state(learner_id, config.arm)

    # ⚠️ Say so when the subject matter has moved under a resumed learner.
    #
    # Mastery is keyed by concept id and the error trace by misconception label,
    # so renaming or removing either leaves stored state pointing at content that
    # no longer exists — and `_cross_concept` looks every trace label up in the
    # catalogue, where a missing id raises *after* the sitting. `assert_poolable`
    # already refuses to pool runs across a content change; resuming across one
    # was unguarded on the same risk.
    #
    # A warning rather than a refusal: adding content is the common case and is
    # harmless, and a person who has sat down to work should not be turned away
    # by a hash. What must not happen is that nobody noticed.
    if previous is not None:
        drift = store.content_drift(learner_id, config.arm, domain.content_hashes())
        if drift:
            console.print(
                Panel(
                    Text(
                        "The subject matter has changed since this learner's last "
                        "sitting:\n"
                        + "\n".join(
                            f"  {field.removesuffix('_hash').replace('_', ' ')}: "
                            f"{was[:8]} -> {now[:8]}"
                            for field, (was, now) in sorted(drift.items())
                        )
                        + "\n\nStored progress is kept. Anything it refers to that no "
                        "longer exists will not resolve, so read this sitting's "
                        "figures with that in mind.",
                    ),
                    title="content changed",
                    border_style="yellow",
                )
            )
    observer = DemoObserver(
        console, domain, config,
        carried=tuple(previous.reflections) if previous else (),
    )

    def _asked(prompt: str, *, optional: bool = False, insist: bool = False) -> str:
        """Read one line, honouring the control words wherever they are typed.

        Every prompt goes through here. The intro says ":q stops at any point",
        and it only checked the answer prompt — so typing it at the working or
        reflection prompt was silently recorded as prose and the sitting carried
        on. A stated affordance that works in one place out of three is worse
        than not offering it.

        ``insist`` re-asks on an empty line instead of taking it. It is not a
        trap: :s declines, and the loop below says so every time it comes round,
        because a person who is out of words needs a way past that does not
        involve inventing some.
        """
        while True:
            said = (
                Prompt.ask(prompt, default="", show_default=False)
                if optional or insist
                else Prompt.ask(prompt)
            ).strip()
            if said == QUIT:
                raise Quit
            if said == END_TRAINING:
                if observer.testing:
                    # The banks are the measurement. Stopping inside one would
                    # leave a score that looks like an answer to a question
                    # nobody finished asking.
                    console.print(
                        f"  [dim]{END_TRAINING} works during training, not inside "
                        f"a test. {QUIT} stops the sitting.[/dim]"
                    )
                    continue
                raise StopTraining
            if said == DECLINE:
                return ""
            if said or not insist:
                return said
            console.print(
                f"  [dim]a line about how you got there, or {DECLINE} to skip "
                f"it[/dim]"
            )

    def ask(item: Item, attempt: int) -> str:
        console.print()
        console.print(
            Panel(
                Text(" ".join(item.prompt.split()), style="bold"),
                title=question_title(domain, item, attempt, testing=observer.testing),
                border_style="cyan",
            )
        )
        hint = (
            f"[dim]({QUIT} to stop)[/dim]"
            if observer.testing
            else f"[dim]({QUIT} to stop, {END_TRAINING} to end training and go "
            f"to the post-test)[/dim]"
        )
        return _asked(f"  your answer  {hint}")

    def ask_reflection(item: Item, prompt: str) -> str:
        # Prose, not an answer. It never reaches the verifier and costs no
        # attempt — see Session._work_item.
        return _asked(
            "  [magenta]in your own words[/magenta]  [dim](enter to skip)[/dim]",
            optional=True,
        )

    def _asked_at_length(prompt: str) -> str:
        """Several lines, ended by a blank one.

        Working is written the way it is done — a line per step. Read one line
        at a time, a paste out of a notes app arrives as a single run-on with
        the newlines gone: *"Differentiate: y = ...f(x) is (x^2 + 1)g(x) is
        (x^6 + 2)What was the product rule?"* was one field's worth. The
        structure a person put in is most of what makes it readable, and it was
        being thrown away at the door.
        """
        console.print(f"  {prompt}")
        lines: list[str] = []
        while True:
            said = Prompt.ask("  ", default="", show_default=False).rstrip()
            if said.strip() == QUIT:
                raise Quit
            if said.strip() == END_TRAINING:
                if observer.testing:
                    console.print(
                        f"  [dim]{END_TRAINING} works during training, not inside "
                        f"a test. {QUIT} stops the sitting.[/dim]"
                    )
                    continue
                raise StopTraining
            if said.strip() == DECLINE:
                return ""
            if not said:
                if lines:
                    return "\n".join(lines)
                console.print(
                    f"  [dim]a line or two about how you got there, or "
                    f"{DECLINE} to skip it[/dim]"
                )
                continue
            lines.append(said)

    def ask_working(item: Item, response: str, required: bool = False) -> str:
        """The reasoning behind the step, insisted on when it did not come out.

        Asked before the verdict is shown on a failed step, so what arrives is
        the reasoning that was used rather than an account of an error the
        learner has just been told about — and before the diagnostic sees the
        step, which is the only moment the words can affect what happens next.

        Optional after a correct answer, where it is volunteered rather than
        asked for. Kept there rather than dropped with the rest of the burden:
        "I guessed" written under a right answer is the one thing that can tell
        a lucky guess from knowing it.
        """
        if not required:
            return _asked(
                "  [magenta]your working[/magenta]  "
                "[dim](optional — enter to skip)[/dim]",
                optional=True,
            )
        return _asked_at_length(
            "[magenta]before I say — how did you get there?[/magenta]  "
            f"[dim](as many lines as you like; blank line when done, "
            f"{DECLINE} to skip)[/dim]"
        )

    learner = HumanLearner(
        ask, learner_id=learner_id,
        ask_reflection=ask_reflection, ask_working=ask_working,
    )

    gap = elapsed_days if elapsed_days is not None else _days_since(store, learner_id, config)
    session_id = store.open_session(
        learner_id=learner_id,
        arm=config.arm,
        config_hash=config.content_hash(),
        elapsed_days=gap,
        decay_half_life_days=config.decay.half_life_days,
        content_hashes=domain.content_hashes(),
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
                "[bold]After a correct answer you may still show your working[/bold], "
                "in words or maths — including \"I guessed\", which is worth more "
                "than it sounds. Never graded, never an attempt. Press enter to "
                "skip.\n"
                "[bold]When an answer does not come out right[/bold] you are "
                "asked how you got there, before you are told what went wrong. "
                "That reply is kept and never graded; it is what lets the system "
                "tell a slip from a misunderstanding, and it is read back to you "
                "next time this idea comes round. [dim]:s[/dim] skips it.\n"
                "[bold]:e[/bold] ends the training whenever you have had enough "
                "and goes straight to the post-test. [bold]:q[/bold] stops the "
                "sitting altogether, at any point.\n\n"
                "During training the panel shows what the system believes about you "
                "and why it chooses what it chooses. Nothing here knows the right "
                "answer in advance: the verifier grades every step symbolically, and "
                "the diagnostic agent infers your misconception from the step alone.",
            ),
            title="Agent_Newton",
            border_style="magenta",
        )
    )

    # Before anything is measured, and before any question is asked. The one
    # thing on the blackboard that is the learner talking rather than the system
    # inferring, so it is taken first and recorded like everything else.
    #
    # Decay is applied first, though, or the estimates shown beside each concept
    # are not the ones the sitting will use — a learner would be choosing from a
    # picture of where they were before the gap.
    #
    # ⚠️ And the gap is then spent, so the session must not apply it again.
    # `decay.relax` is exponential in the elapsed time: applying one day twice
    # ages the model by two, which would be a gap nobody had.
    decayed = session.board.apply_decay(session.elapsed_days)
    if decayed:
        observer.session_resumed(session.elapsed_days, decayed)
    session.elapsed_days = 0.0
    try:
        _ask_what_to_practise(console, domain, session.board, config, _asked)
    except Quit:
        console.print("\n[dim]stopped before the pre-test[/dim]")

    outcome = None
    failed: BaseException | None = None
    try:
        outcome = session.run()
    except (Quit, KeyboardInterrupt):
        console.print("\n[dim]stopped — the state so far:[/dim]")
    except Exception as exc:  # noqa: BLE001 - re-raised below, after storing
        # ⚠️ Anything at all, and it is stored before it is re-raised.
        #
        # This caught only Quit and KeyboardInterrupt, so a failure from anywhere
        # else propagated straight past `_store` and the sitting was lost — which
        # is exactly the defect `27c3324` fixed for interrupts, arriving through a
        # different door. The live case: the model backend goes down or times out
        # mid-sitting, `ProviderError` comes up through the agents, and forty
        # minutes of a person's work leaves no record.
        #
        # The agents now tolerate that particular failure, so this is the backstop
        # rather than the fix. It exists because the next unhandled failure will be
        # one nobody predicted, and the record must survive it.
        failed = exc
        console.print(
            Panel(
                Text(
                    f"{type(exc).__name__}: {exc}\n\n"
                    "The sitting is being saved before this is raised — the work "
                    "so far is not lost.",
                ),
                title="the sitting stopped on an error",
                border_style="red",
            )
        )

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
        stop_reason=(
            outcome.stop_reason
            if outcome
            else ("failed" if failed is not None else "interrupted")
        ),
    )
    # Read *after* closing the session, so this sitting is part of the history
    # rather than the one thing missing from it.
    across = _across_sittings(store, learner_id, config, domain)
    store.close()

    if across is not None:
        console.print(across)

    if failed is not None:
        # Everything is written; now let the failure be seen. Swallowing it would
        # leave a stored sitting that looks merely short, with no way to tell a
        # person who stopped from a backend that fell over.
        raise failed

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
