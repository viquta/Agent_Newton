"""Command-line entry points."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent_newton import __version__
from agent_newton.config import Config, ModelSpec
from agent_newton.domains import registry
from agent_newton.domains.base import DomainError
from agent_newton.domains.validate import validate

app = typer.Typer(add_completion=False, help="Agent_Newton — multi-agent ITS.")
domain_app = typer.Typer(help="Inspect and validate teaching domains.")
app.add_typer(domain_app, name="domain")
eval_app = typer.Typer(help="Component evaluations.")
app.add_typer(eval_app, name="evaluate")
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

        # Warnings never fail the check: the content is internally consistent,
        # but something about it is provisional and should stay visible.
        for warning in report.warnings:
            console.print(f"    [yellow]! {warning}[/yellow]")

    if failed:
        raise typer.Exit(code=1)


@eval_app.command("diagnostic")
def evaluate_diagnostic(
    domain_name: str = typer.Option("calculus", "--domain", help="Domain to evaluate on."),
    model: str = typer.Option("gemma4:12b", "--model", help="Model to evaluate."),
    provider: str = typer.Option("ollama", "--provider"),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N cases."),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List the cases and exit."),
) -> None:
    """Score a diagnostic agent against the item bank's injected labels.

    Every (item, misconception) pair the bank declares becomes one case, so the
    confusion matrix is balanced by construction rather than shaped by whatever
    a session happened to produce.

    Safe to interrupt. Identical prompts hit the response cache, so restarting
    resumes without repeating work or spending anything.
    """
    from agent_newton.core.agents.llm import LLMDiagnostic
    from agent_newton.core.evaluation.diagnostic import DiagnosticReport, cases, evaluate
    from agent_newton.llm.factory import build_provider

    try:
        domain = registry.load_domain(domain_name)
    except DomainError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    todo = cases(domain)[:limit]
    if dry_run:
        console.print(f"{len(todo)} cases on {domain_name}, {len({m for _, m, _ in todo})} labels")
        for item_id, misconception, wrong in todo:
            console.print(f"  {item_id:18} {misconception:36} -> {wrong}")
        return

    spec = ModelSpec(provider=provider, model=model)  # pyright: ignore[reportArgumentType]
    agent = LLMDiagnostic(build_provider(spec, Path(".cache/llm")))

    directory = out or Path("results") / f"diagnostic_{domain_name}_{model.replace(':', '-')}"
    directory.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]{len(todo)} cases[/bold] on {domain_name} with {provider}/{model}\n"
        f"writing to {directory}  ·  safe to interrupt, cached runs resume\n"
    )

    report = DiagnosticReport()
    rows_path = directory / "predictions.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["item_id", "concept_id", "injected", "inferred", "correct",
             "wrong_answer", "prompt", "seconds"]
        )
        for n, prediction in enumerate(evaluate(domain, agent, limit=limit), start=1):
            report.predictions.append(prediction)
            writer.writerow([
                prediction.item_id, prediction.concept_id, prediction.injected,
                prediction.inferred or "", prediction.correct,
                prediction.wrong_answer, prediction.prompt, f"{prediction.seconds:.1f}",
            ])
            handle.flush()  # partial results survive an interrupt
            mark = "[green]OK  [/green]" if prediction.correct else "[red]MISS[/red]"
            console.print(
                f"  {n:>3}/{len(todo)} {mark} {prediction.seconds:5.1f}s  "
                f"{prediction.injected[:34]:36} -> {prediction.inferred or '<abstained>'}"
            )

    summary = {
        "domain": domain_name,
        "model": f"{provider}/{model}",
        "cases": report.total,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "abstentions": report.abstentions,
        "failed_calls": agent.failures,
        "seconds": report.seconds,
        "per_label": report.per_label(),
        "confusion": {f"{a} -> {b}": n for (a, b), n in report.confusion().items()},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    console.print()
    table = Table(title="diagnostic accuracy", show_header=False, box=None)
    table.add_row("cases", str(report.total))
    table.add_row("accuracy", f"{report.accuracy:.1%}")
    table.add_row("macro F1", f"{report.macro_f1:.3f}")
    table.add_row("abstained", str(report.abstentions))
    table.add_row("failed calls", str(agent.failures))
    table.add_row("total time", f"{report.seconds / 60:.1f} min")
    console.print(table)

    worst = report.worst_confusions()
    if worst:
        console.print("\n[bold]most frequent confusions[/bold]")
        for injected, inferred, count in worst:
            console.print(f"  {count}x  {injected}  ->  {inferred}")
    else:
        console.print("\n[green]no confusions[/green]")


if __name__ == "__main__":
    app()
