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

from pathlib import Path

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from agent_newton.config import Config
from agent_newton.core.agents.base import Diagnosis, Hint
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.simulator.human import HumanLearner
from agent_newton.core.state import bkt, route
from agent_newton.core.state.store import Blackboard
from agent_newton.domains import registry
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


class DemoObserver:
    """Renders the blackboard between steps.

    Only reads. It cannot change what the session does, which is what keeps the
    demo a view of the real system rather than a variant of it.
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


def run_demo(config_path: Path, console: Console | None = None) -> None:
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
        answer = Prompt.ask(f"  your answer  [dim]({QUIT} to stop)[/dim]").strip()
        if answer == QUIT:
            raise Quit
        return answer

    learner = HumanLearner(ask)
    session = build_session(
        learner.learner_id, config.seed, domain, config, learner=learner,
        observer=observer,
    )

    console.print(
        Panel(
            Text(
                "Answer each exercise. The panel above shows what the system "
                "believes about you and why it is choosing what it chooses.\n"
                "Nothing here knows the right answer in advance — a symbolic "
                "verifier grades every step, and the diagnostic agent has to "
                "infer your misconception from the step alone.",
            ),
            title="Agent_Newton",
            border_style="magenta",
        )
    )

    try:
        outcome = session.run()
    except (Quit, KeyboardInterrupt):
        console.print("\n[dim]stopped — the state so far:[/dim]")
        console.print(observer.board_panel(session.board))
        return

    console.print()
    console.print(observer.board_panel(session.board))

    summary = Table(title="session", show_header=False, box=None)
    summary.add_row("items attempted", str(outcome.items_attempted))
    if outcome.pretest.administered and outcome.posttest.administered:
        summary.add_row("pre-test", f"{outcome.pretest.score:.0%}")
        summary.add_row("post-test", f"{outcome.posttest.score:.0%}")
        summary.add_row("gain", f"{outcome.gain:+.0%}")
    else:
        # A skipped bank and a bank scored zero both read 0.0, and they mean
        # opposite things.
        summary.add_row("pre/post", "not administered")
    summary.add_row("goals mastered", str(outcome.goals_mastered))
    summary.add_row("concepts to the next goal", str(outcome.distance_to_goal))
    summary.add_row("steps the verifier could not read", str(outcome.unmeasurable_steps))
    # There is no injected label for a person, so there is nothing to score the
    # diagnostic against. Unavailable, not zero.
    summary.add_row("diagnostic accuracy", "unavailable (no ground truth)")
    summary.add_row("remediation ratio", "unavailable (no ground truth)")
    console.print(Columns([summary]))
