"""Command-line entry points."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent_newton import __version__
from agent_newton.config import Config
from agent_newton.domains import registry
from agent_newton.domains.base import DomainError
from agent_newton.domains.validate import validate

app = typer.Typer(add_completion=False, help="Agent_Newton — multi-agent ITS.")
domain_app = typer.Typer(help="Inspect and validate teaching domains.")
app.add_typer(domain_app, name="domain")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"agent-newton {__version__}")


@app.command("config-check")
def config_check(path: Path = typer.Argument(..., help="Path to a run config YAML.")) -> None:
    """Validate a run config and show how it resolves.

    Exercises the design invariants — the circularity control, the ZPD band, BKT
    identifiability — so a bad config fails here rather than eight hours into a
    cohort run.
    """
    try:
        config = Config.from_yaml(path)
    except Exception as exc:  # pydantic ValidationError or a YAML error
        console.print(f"[red]invalid config:[/red] {path}")
        console.print(str(exc))
        raise typer.Exit(code=1)

    table = Table(title=f"{path}", show_header=False, box=None)
    table.add_row("run name", config.run_name)
    table.add_row("domain", config.domain)
    table.add_row("arm", config.arm)
    table.add_row("seed", str(config.seed))
    table.add_row("cohort", f"{config.cohort.n_learners} learners")
    table.add_row("simulator", config.simulator.surface)
    table.add_row("tutor", f"{config.agents.tutor.impl}")
    table.add_row("diagnostic", f"{config.agents.diagnostic.impl}")
    table.add_row("planner", f"{config.agents.planner.impl}")
    table.add_row("ZPD band", f"({config.zpd.theta_lower}, {config.zpd.theta_upper})")
    table.add_row("replan theta", str(config.arbitration.theta))
    table.add_row("uses LLM", "yes" if config.uses_llm() else "no (deterministic)")
    table.add_row("config hash", config.content_hash())
    console.print(table)


@domain_app.command("list")
def domain_list() -> None:
    """List registered domains."""
    for name in registry.available():
        console.print(f"  {name}")


@domain_app.command("validate")
def domain_validate(
    name: str = typer.Argument(..., help="Domain name, or 'all'."),
) -> None:
    """Check a domain's content for internal consistency.

    Run after any edit to a domain's YAML or buggy rules. Checks referential
    integrity, coverage, held-out separation, that every stated answer verifies,
    and that every probed misconception actually produces a wrong answer.
    """
    names = registry.available() if name == "all" else (name,)
    failed = False

    for domain_name in names:
        try:
            domain = registry.load_domain(domain_name)
        except DomainError as exc:
            console.print(f"[red]✗ {domain_name}[/red] — failed to load")
            console.print(f"    {exc}")
            failed = True
            continue

        report = validate(domain)
        stats = ", ".join(f"{v} {k}" for k, v in report.stats.items())
        if report.ok:
            console.print(f"[green]✓ {domain_name}[/green] — {stats}")
        else:
            failed = True
            console.print(
                f"[red]✗ {domain_name}[/red] — {len(report.problems)} problem(s); {stats}"
            )
            for problem in report.problems:
                console.print(f"    {problem}")

    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
