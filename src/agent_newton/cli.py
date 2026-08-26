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


@app.command("demo")
def demo(
    config_path: Path = typer.Option(
        Path("experiments/configs/demo.yaml"), "--config", help="A human session config."
    ),
    learner: str = typer.Option(
        "human",
        "--learner",
        help="Who is sitting down. The same name twice picks up where that "
        "learner left off; a new name starts fresh.",
    ),
    elapsed_days: float | None = typer.Option(
        None,
        "--elapsed-days",
        help="Pretend this many days have passed since this learner's last "
        "sitting, instead of using the real gap. For exercising decay without "
        "waiting weeks for it.",
    ),
) -> None:
    """Work through a session yourself, with the blackboard visible.

    Drives the same session the cohorts run — same planner, verifier,
    arbitration policy and shared state — with a person answering where a
    simulated learner otherwise would. The panel shows mastery moving, the
    frontier narrowing, the goal and what remains of the route, and replanning
    firing with the evidence that caused it.

    Needs a model for the diagnostic: a person carries no injected misconception
    label, so it has to be inferred from the step.
    """
    from agent_newton.demo import run_demo

    run_demo(config_path, console, learner_id=learner, elapsed_days=elapsed_days)


@eval_app.command("verifier")
def evaluate_verifier(
    domain_name: str = typer.Option("calculus", "--domain", help="Domain to evaluate."),
    gold: Path | None = typer.Option(None, "--gold", help="Gold-set YAML."),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
    show_all: bool = typer.Option(False, "--all", help="List every case, not just misses."),
) -> None:
    """Score a domain's verifier against its hand-labelled gold set.

    No model is involved and nothing is cached: the verifier is symbolic, so the
    whole set runs in well under a second and the numbers are exact.

    Exits non-zero when a case disagrees with its label without a stated reason,
    or when a stated reason no longer applies.
    """
    from agent_newton.core.evaluation.verifier import load, score

    try:
        domain = registry.load_domain(domain_name)
    except DomainError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    path = gold or Path("tests/fixtures/gold") / f"{domain_name}_verifier_cases.yaml"
    if not path.exists():
        console.print(f"[red]no gold set at {path}[/red]")
        raise typer.Exit(code=1)

    try:
        report = score(domain, load(path, domain))
    except DomainError as exc:
        console.print(f"[red]{path} is not usable:[/red] {exc}")
        raise typer.Exit(code=1)

    directory = out or Path("results") / f"verifier_{domain_name}"
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "cases.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case_id", "item_id", "kind", "response", "expected", "actual",
             "agrees", "known_limitation", "detail"]
        )
        for scored in report.scored:
            writer.writerow([
                scored.case.id, scored.case.item_id, scored.case.kind,
                scored.case.response, scored.case.expected.value, scored.actual.value,
                scored.agrees, scored.case.known_limitation, scored.detail,
            ])

    summary = {
        "domain": domain_name,
        "gold_set": str(path),
        "item_bank_hash": domain.items.content_hash(),
        "cases": report.total,
        "accuracy": report.accuracy,
        "by_kind": {k: len(report.of_kind(k)) for k in ("canonical", "equivalent",
                                                        "wrong", "unreadable")},
        "false_negative_rate": report.false_negative_rate,
        "false_negatives_scored_incorrect": [s.case.id for s in report.scored_incorrect()],
        "false_negatives_unmeasured": [s.case.id for s in report.unmeasured()],
        "false_accept_rate": report.false_accept_rate,
        "false_accepts": [s.case.id for s in report.false_accepts()],
        "hidden_errors": [s.case.id for s in report.hidden_errors()],
        "surprises": [s.case.id for s in report.surprises()],
        "resolved": [s.case.id for s in report.resolved()],
        "confusion": {f"{a.value} -> {b.value}": n for (a, b), n in report.confusion().items()},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    console.print(
        f"[bold]{report.total} cases[/bold] on {domain_name} from {path}\n"
        f"writing to {directory}\n"
    )

    for scored in report.scored:
        if scored.agrees and not show_all:
            continue
        if scored.agrees:
            mark = "[green]OK  [/green]"
        elif scored.case.known_limitation:
            mark = "[yellow]KNOWN[/yellow]"
        else:
            mark = "[red]MISS[/red]"
        console.print(
            f"  {mark} {scored.case.id:34} {scored.case.kind:11} "
            f"{scored.case.expected.value} -> {scored.actual.value}"
        )
        if not scored.agrees:
            console.print(f"        {scored.case.response!r} on {scored.case.item_id}")

    table = Table(title="verifier accuracy", show_header=False, box=None)
    table.add_row("cases", str(report.total))
    table.add_row("accuracy", f"{report.accuracy:.1%}")
    table.add_row(
        "false negatives",
        f"{report.false_negative_rate:.1%} of {len(report.of_kind('equivalent'))} "
        f"equivalent forms  "
        f"({len(report.scored_incorrect())} scored wrong, "
        f"{len(report.unmeasured())} unmeasured)",
    )
    table.add_row(
        "false accepts",
        f"{report.false_accept_rate:.1%} of {len(report.of_kind('wrong'))} wrong answers"
        + (f"  ({len(report.hidden_errors())} unmeasured)" if report.hidden_errors() else ""),
    )
    table.add_row("elapsed", f"{report.seconds * 1000:.0f} ms")
    console.print()
    console.print(table)

    # A correct answer scored as an error is the one that reaches the learner
    # model, so it is called out apart from the rate it sits inside.
    if report.scored_incorrect():
        console.print(
            f"\n[red]{len(report.scored_incorrect())} correct answer(s) scored as "
            f"errors[/red] — these write error events about learners who made none"
        )

    if report.surprises() or report.resolved():
        for scored in report.resolved():
            console.print(
                f"\n[red]{scored.case.id}[/red] now agrees with its label; its "
                f"known_limitation describes the past and should be removed"
            )
        raise typer.Exit(code=1)

    if any(s.case.known_limitation for s in report.scored):
        console.print(
            f"\n[yellow]{sum(1 for s in report.scored if s.case.known_limitation)} "
            f"stated limitation(s)[/yellow] — see known_limitation in {path.name}"
        )


@eval_app.command("planner")
def evaluate_planner(
    config_path: Path = typer.Option(
        ..., "--config", help="Run config; the planner and arm it names are what is scored."
    ),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
) -> None:
    """Score a planner's choices against a policy holding the true profile.

    The planner under test drives a real session; the reference watches and
    answers the same question from the same options. Regret is the remaining
    misconception probability the chosen item could not bring to the surface and
    the reference's could — zero when a different choice was equally good.

    Run it per arm: the decoupled planner is the interesting subject, since the
    coupled one selects from the frontier by construction.
    """
    from agent_newton.core.evaluation.planning import evaluate
    from agent_newton.core.orchestration.session import build_session

    try:
        config = Config.from_yaml(config_path)
    except Exception as exc:  # pydantic ValidationError or a YAML error
        console.print(f"[red]invalid config:[/red] {config_path}\n{exc}")
        raise typer.Exit(code=1)

    if config.uses_llm():
        console.print(
            "[yellow]note:[/yellow] this config calls a model, so every learner "
            "costs inference. A model-free config scores the same planner logic "
            "in seconds."
        )

    domain = registry.load_domain(config.domain)
    learners = [f"L{n:04d}" for n in range(config.cohort.n_learners)]
    report = evaluate(learners, domain, config, build_session)

    directory = out or Path("results") / f"planner_{config.domain}_{config.arm}"
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "decisions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["learner_id", "goal", "chosen_item", "chosen_concept", "reference_item",
             "reference_concept", "chosen_value", "reference_value", "regret",
             "in_frontier", "options"]
        )
        for d in report.decisions:
            writer.writerow([
                d.learner_id, d.goal or "", d.chosen_item or "", d.chosen_concept or "",
                d.reference_item or "", d.reference_concept or "",
                f"{d.chosen_value:.4f}", f"{d.reference_value:.4f}", f"{d.regret:.4f}",
                d.in_frontier, d.options,
            ])

    summary = {
        "config": str(config_path),
        "domain": config.domain,
        "arm": config.arm,
        "planner": config.agents.planner.impl,
        "emphasis": config.agents.planner.emphasis.value,
        "decisions": report.total,
        "item_agreement": report.agreement,
        "concept_agreement": report.concept_agreement,
        "mean_regret": report.mean_regret,
        "mean_reference_value": report.reference_value,
        "regret_share": report.regret_share,
        "costly_disagreements": report.costly_disagreements,
        "in_frontier_rate": report.in_frontier_rate,
        "no_selection": report.no_selection,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = Table(title=f"planner vs oracle — {config.arm}", show_header=False, box=None)
    table.add_row("decisions", str(report.total))
    table.add_row("item agreement", f"{report.agreement:.1%}")
    table.add_row("concept agreement", f"{report.concept_agreement:.1%}")
    table.add_row(
        "mean regret",
        f"{report.mean_regret:.4f} of {report.reference_value:.4f} available",
    )
    # The comparable figure: the arms face different states, so one planner may
    # simply have been offered more to leave behind.
    table.add_row("regret share", f"{report.regret_share:.1%} of what was available")
    table.add_row("costly disagreements", str(report.costly_disagreements))
    table.add_row(
        "selections in frontier",
        f"{report.in_frontier_rate:.1%} of {report.total - report.no_selection} made",
    )
    table.add_row("nothing to select", str(report.no_selection))
    console.print(table)
    console.print(f"\nwritten to {directory}")

    worst = report.worst()
    if any(d.regret > 1e-9 for d in worst):
        console.print("\n[bold]largest regrets[/bold]")
        for d in worst:
            if d.regret <= 1e-9:
                continue
            console.print(
                f"  {d.regret:.3f}  {d.learner_id}  chose {d.chosen_concept} "
                f"where the reference took {d.reference_concept}"
            )


@eval_app.command("diagnostic")
def evaluate_diagnostic(
    domain_name: str = typer.Option("calculus", "--domain", help="Domain to evaluate on."),
    model: str = typer.Option("gemma4:12b", "--model", help="Model to evaluate."),
    provider: str = typer.Option("ollama", "--provider"),
    think: bool | None = typer.Option(
        None,
        "--think/--no-think",
        help="Whether a reasoning model deliberates before answering. Unset "
        "leaves the backend default.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N cases."),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List the cases and exit."),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Tokens the model may generate per call, deliberation included. "
        "Unset keeps the provider default (1024). Raise it with --think, or a "
        "reasoning model spends the budget before it answers.",
    ),
    context_tokens: int | None = typer.Option(
        None,
        "--context-tokens",
        help="Context window: prompt, deliberation and answer together. Unset "
        "leaves the server's own. Raise it alongside --max-tokens -- Ollama "
        "truncates silently rather than refusing, so a prompt that does not fit "
        "produces a confident answer to a question it only partly saw.",
    ),
    timeout_seconds: float | None = typer.Option(
        None,
        "--timeout",
        help="Seconds allowed for one call. Unset keeps the provider default "
        "(120). Deliberation needs wall-clock as well as tokens.",
    ),
    label_space: str = typer.Option(
        "concept",
        "--label-space",
        help="Labels offered per case: 'concept' (the item's own concept) or "
        "'catalogue' (all of them). Not interchangeable as measurements — the "
        "narrow space is the easier task, so run both and report both.",
    ),
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

    if label_space not in ("concept", "catalogue"):
        console.print(f"[red]unknown label space {label_space!r}[/red]")
        raise typer.Exit(code=1)

    spec = ModelSpec(
        provider=provider,  # pyright: ignore[reportArgumentType]
        model=model,
        think=think,
        max_tokens=max_tokens,
        context_tokens=context_tokens,
        timeout_seconds=timeout_seconds,
    )
    agent = LLMDiagnostic(
        build_provider(spec, Path(".cache/llm")),
        label_space=label_space,  # pyright: ignore[reportArgumentType]
    )

    # The reasoning mode is part of the run's identity, not a flag on it: the
    # same model answering with and without deliberation gives two different
    # measurements, and they must not overwrite each other. The label space is
    # the same kind of thing: a narrower one is an easier task, so the two
    # figures must not land in the same directory either.
    suffix = "" if think is None else f"_think-{str(think).lower()}"
    # The call limits join it for the same reason. A budget that lets a model
    # finish deliberating and one that cuts it off mid-thought are two different
    # measurements of the same model, and landing them in one directory would
    # overwrite the first with the second.
    if max_tokens is not None:
        suffix += f"_predict-{max_tokens}"
    if context_tokens is not None:
        suffix += f"_ctx-{context_tokens}"
    suffix += f"_labels-{label_space}"
    directory = (
        out
        or Path("results") / f"diagnostic_{domain_name}_{model.replace(':', '-')}{suffix}"
    )
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
        "model": spec.label(),
        # Part of the measurement's identity. An accuracy figure under one label
        # space cannot be compared against the other, and a summary that did not
        # say which it was would invite exactly that comparison.
        "label_space": label_space,
        "cases": report.total,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "abstentions": report.abstentions,
        "failed_calls": agent.failures,
        "seconds": report.seconds,
        "accuracy_by_bank": report.accuracy_by_bank(),
        "per_label": report.per_label(),
        "confusion": {f"{a} -> {b}": n for (a, b), n in report.confusion().items()},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    console.print()
    table = Table(title="diagnostic accuracy", show_header=False, box=None)
    table.add_row("cases", str(report.total))
    table.add_row("accuracy", f"{report.accuracy:.1%}")
    for bank, row in report.accuracy_by_bank().items():
        # Only practice items are diagnosed in a session, so that row is the
        # rate a running system is exposed to.
        table.add_row(f"  on {bank}", f"{row['accuracy']:.1%} of {int(row['cases'])}")
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


@eval_app.command("tutor")
def evaluate_tutor(
    domain_name: str = typer.Option("calculus", "--domain", help="Domain to evaluate on."),
    model: str = typer.Option("gemma4:12b", "--model", help="Model writing the hints."),
    provider: str = typer.Option("ollama", "--provider"),
    think: bool | None = typer.Option(None, "--think/--no-think"),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Tokens the model may generate per call, deliberation included. "
        "Unset keeps the provider default (1024). Raise it with --think, or a "
        "reasoning model spends the budget before it answers.",
    ),
    context_tokens: int | None = typer.Option(
        None,
        "--context-tokens",
        help="Context window: prompt, deliberation and answer together. Unset "
        "leaves the server's own. Raise it alongside --max-tokens -- Ollama "
        "truncates silently rather than refusing, so a prompt that does not fit "
        "produces a confident answer to a question it only partly saw.",
    ),
    timeout_seconds: float | None = typer.Option(
        None,
        "--timeout",
        help="Seconds allowed for one call. Unset keeps the provider default "
        "(120). Deliberation needs wall-clock as well as tokens.",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Second model, for the checks no predicate can decide. Must differ "
        "from --model: a model grading its own replies measures its taste, not "
        "its faithfulness.",
    ),
    judge_think: bool | None = typer.Option(
        None,
        "--judge-think/--no-judge-think",
        help="Reasoning mode for the judge. Left unset rather than inherited "
        "from --think: the two roles are different jobs, and a mode chosen to "
        "keep a tutor responsive at a keyboard has no bearing on how carefully "
        "a judge should read.",
    ),
    gold: Path | None = typer.Option(
        None, "--gold", help="Hand-labelled set the judge is calibrated against."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Stop after N turns. For smoke-testing: the bank is "
        "ordered, so a truncated run is not a balanced sample."
    ),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List the cases and exit."),
) -> None:
    """Score a tutor on the turns a learner would read.

    The deterministic checks decide what has a right answer — whether a hint
    gives the answer away is settled by the domain's verifier, so equivalence
    rather than spelling. They need no model beyond the one writing the hints.

    ``--judge-model`` adds the two checks that are judgements: whether a reply
    keeps to what the student's step shows, and whether the assigned support
    levels are visible in the text. The judge is scored against the hand-labelled
    set first, and its agreement is reported beside its verdicts.

    Safe to interrupt. Identical prompts hit the response cache.
    """
    from agent_newton.config import ZPDConfig
    from agent_newton.core.agents.llm import LLMTutor
    from agent_newton.core.evaluation import tutor as evaluation
    from agent_newton.llm.factory import build_provider

    try:
        domain = registry.load_domain(domain_name)
    except DomainError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    band = ZPDConfig()
    todo = evaluation.cases(domain, band)[:limit]
    if dry_run:
        console.print(f"{len(todo)} turns on {domain_name}")
        for case in todo:
            console.print(f"  {case.id:64} P={case.mastery:.2f}  -> {case.wrong_answer}")
        return

    if judge_model is not None and judge_model == model:
        console.print(
            "[red]the judge must differ from the tutor[/red] — a model grading "
            "its own replies measures its taste rather than their faithfulness"
        )
        raise typer.Exit(code=1)

    cache = Path(".cache/llm")
    spec = ModelSpec(
        provider=provider,  # pyright: ignore[reportArgumentType]
        model=model,
        think=think,
        max_tokens=max_tokens,
        context_tokens=context_tokens,
        timeout_seconds=timeout_seconds,
    )
    agent = LLMTutor(build_provider(spec, cache), band)

    suffix = "" if think is None else f"_think-{str(think).lower()}"
    if max_tokens is not None:
        suffix += f"_predict-{max_tokens}"
    if context_tokens is not None:
        suffix += f"_ctx-{context_tokens}"
    directory = (
        out or Path("results") / f"tutor_{domain_name}_{model.replace(':', '-')}{suffix}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]{len(todo)} turns[/bold] on {domain_name} with {provider}/{model}\n"
        f"writing to {directory}  ·  safe to interrupt, cached runs resume\n"
    )

    collected: list[evaluation.Turn] = []
    with (directory / "turns.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case_id", "item_id", "concept_id", "misconception_id", "move", "level",
             "targets", "wrong_answer", "text", "violations", "seconds"]
        )
        for n, turn in enumerate(evaluation.ask(domain, agent, band, todo), start=1):
            item = domain.items.get(turn.case.item_id)
            violations = evaluation.check_turn(turn, item, domain)
            collected.append(turn)
            writer.writerow([
                turn.case.id, turn.case.item_id, turn.case.concept_id,
                turn.case.misconception_id, turn.move, turn.level, turn.targets or "",
                turn.case.wrong_answer, turn.text,
                "; ".join(v.rule for v in violations), f"{turn.seconds:.1f}",
            ])
            handle.flush()  # partial results survive an interrupt
            mark = "[green]OK  [/green]" if not violations else "[red]FAIL[/red]"
            console.print(
                f"  {n:>3}/{len(todo)} {mark} {turn.seconds:5.1f}s  "
                f"{turn.move:10} {turn.level:12} {turn.case.item_id}"
            )
            for violation in violations:
                console.print(f"        [red]{violation}[/red]")

    report = evaluation.score(domain, collected)

    summary: dict = {
        "domain": domain_name,
        "model": spec.label(),
        "turns": report.total,
        "clean": report.clean,
        "clean_rate": report.clean_rate,
        "fallbacks": report.fallbacks,
        "violations_by_rule": report.by_rule(),
        "rate_by_rule": report.rate_by_rule(),
        "by_level": report.by_level(),
        "by_move": report.by_move(),
        # Only practice turns are ever read by a learner — the test banks are
        # administered without hints — so that row is the rate a running system
        # exposes anyone to.
        "by_bank": report.by_bank(domain),
        "seconds": report.seconds,
    }

    if judge_model is not None:
        judge_spec = ModelSpec(provider=provider, model=judge_model, think=judge_think)  # pyright: ignore[reportArgumentType]
        judge = build_provider(judge_spec, cache)
        path = gold or Path("tests/fixtures/gold") / f"{domain_name}_tutor_cases.yaml"
        try:
            labelled = evaluation.load_gold(path, domain)
        except (DomainError, FileNotFoundError) as exc:
            console.print(f"[red]{path} is not usable:[/red] {exc}")
            raise typer.Exit(code=1)

        console.print(f"\ncalibrating the judge on {len(labelled)} hand-labelled cases")
        judged = evaluation.calibrate(judge, domain, labelled)
        console.print(f"  agreement with the hand labels: {judged.agreement:.1%}")
        for case_id, hand, read in judged.disagreements():
            console.print(f"  [yellow]{case_id}[/yellow]: labelled {hand}, judged {read}")

        console.print(f"\njudging {len(collected)} turns")
        evaluation.judge_turns(judge, domain, collected, judged)
        ranking = evaluation.rank_levels(judge, domain, collected)

        summary["judge"] = {
            "model": judge_spec.label(),
            "gold_set": str(path),
            "calibration_cases": len(labelled),
            # Reported first, and deliberately: the grounded rate below is only
            # as good as this figure, and a reader who takes one without the
            # other has a rate whose error is unknown.
            "agreement_with_hand_labels": judged.agreement,
            "disagreements": [c for c, _, _ in judged.disagreements()],
            "grounded_rate": judged.grounded_rate,
            "unobtainable": judged.unobtainable,
            "level_order_agreement": ranking.agreement,
            "level_pairs": len(ranking.scored),
        }

    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    console.print()
    table = Table(title="tutor turns", show_header=False, box=None)
    table.add_row("turns", str(report.total))
    table.add_row("clean", f"{report.clean_rate:.1%}")
    for rule, count in sorted(report.by_rule().items(), key=lambda kv: -kv[1]):
        table.add_row(f"  {rule}", f"{count} ({count / report.total:.1%})")
    for level, row in report.by_level().items():
        table.add_row(f"  at {level}", f"{row['clean_rate']:.1%} of {int(row['turns'])}")
    table.add_row("fallbacks", str(report.fallbacks))
    table.add_row("total time", f"{report.seconds / 60:.1f} min")
    if "judge" in summary:
        table.add_row("judge agreement", f"{summary['judge']['agreement_with_hand_labels']:.1%}")
        table.add_row("grounded", f"{summary['judge']['grounded_rate']:.1%}")
        table.add_row("level order seen", f"{summary['judge']['level_order_agreement']:.1%}")
    console.print(table)

    if report.failures():
        console.print(
            f"\n[red]{len(report.failures())} turn(s) broke a stated rule[/red] — "
            f"see turns.csv"
        )


@eval_app.command("lessons")
def evaluate_lessons(
    learner: str = typer.Option(..., "--learner", help="Whose sittings to read."),
    arm: str = typer.Option("coupled", "--arm"),
    judge_model: str = typer.Option(
        "gemma4:26b",
        "--judge-model",
        help="The model doing the judging. Should differ from the one that "
        "wrote the turns: a model grading its own replies measures its taste, "
        "not its faithfulness.",
    ),
    provider: str = typer.Option("ollama", "--provider"),
    gold: Path = typer.Option(
        Path("tests/fixtures/gold/calculus_lesson_grounding_cases.yaml"),
        "--gold",
        help="Hand-labelled set the judge is calibrated against.",
    ),
    store_path: Path = typer.Option(Path("results/learners.db"), "--store"),
) -> None:
    """Score a learner's lesson turns for faithfulness to what they said.

    The same question `evaluate tutor` asks of hints, over the part of the
    system it was never pointed at. A sitting is why: a learner wrote
    `x2 + h - 3^2 / x + h - x` and the tutor replied "You've set up the
    calculation perfectly!" — which is a claim about their work that their work
    does not support.

    ⚠️ The agreement figure is not decoration. It measures the *judge*, and a
    verdict rate quoted without it states a number whose error is unknown. Read
    the disagreements before the rate.
    """
    import yaml

    from agent_newton.core.evaluation.tutor import (
        JudgeReport,
        LessonExchange,
        judge_lesson_grounded,
        judge_lesson_turns,
        lesson_exchanges,
    )
    from agent_newton.llm.factory import build_provider
    from agent_newton.store import LearnerStore

    spec = ModelSpec(provider=provider, model=judge_model, think=False)  # pyright: ignore[reportArgumentType]
    judge = build_provider(spec, Path(".cache/llm"))

    report = JudgeReport()
    for case in yaml.safe_load(gold.read_text())["cases"]:
        report.calibration.append(
            (
                case["id"],
                bool(case["grounded"]),
                judge_lesson_grounded(
                    judge,
                    LessonExchange(
                        case["concept_id"],
                        " ".join(str(case["said"]).split()),
                        " ".join(str(case["reply"]).split()),
                    ),
                ),
            )
        )

    with LearnerStore(store_path) as store:
        exchanges = lesson_exchanges(store.audit(learner, arm))
    answering = [e for e in exchanges if e.said.strip()]
    if not answering:
        console.print(
            f"[yellow]{learner} has no lesson turns answering anything. "
            f"Openings are not judged — there is nothing to be faithful to "
            f"yet.[/yellow]"
        )
        raise typer.Exit()

    console.print(
        f"[bold]{len(answering)} lesson turn(s)[/bold] from {learner}, judged by "
        f"{provider}/{judge_model}\n"
    )
    judge_lesson_turns(judge, answering, report)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("turns judged", str(len(report.verdicts)))
    table.add_row("grounded", f"{report.grounded_rate:.1%}")
    table.add_row("unobtainable", str(report.unobtainable))
    table.add_row("judge agreement", f"{report.agreement:.1%} of {len(report.scored)}")
    console.print(table)

    if report.disagreements():
        console.print("\n[yellow]the judge read these differently to the hand "
                      "labels — read them before the rate above[/yellow]")
        for case_id, hand, judged in report.disagreements():
            console.print(f"  {case_id}: hand={hand} judge={judged}")

    ungrounded = [i for i, j in report.verdicts if j is False]
    if ungrounded:
        console.print(
            f"\n[yellow]{len(ungrounded)} turn(s) claimed more than the learner "
            f"showed[/yellow]"
        )
        for case_id in ungrounded[:10]:
            console.print(f"  {case_id}")


@eval_app.command("recall")
def evaluate_recall(
    gold: Path = typer.Option(
        Path("tests/fixtures/gold/calculus_recall_cases.yaml"), "--gold"
    ),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model"),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Similarity below which a match is dropped. Higher returns less "
        "and means it more; a strategy that always fills its quota looks good "
        "on recall and bad on precision.",
    ),
    limit: int = typer.Option(3, "--limit", help="Utterances returned per query."),
) -> None:
    """Compare recall strategies on hand-labelled cases.

    Two are built, so that which one this system should use is measured rather
    than argued about. `bonus_lesson_idea.md` closed retrieval for *lesson
    content* and that argument stands — fifteen lessons keyed by concept id is a
    dict lookup. This is the other case the same note names as the one that
    would earn an index: a corpus nobody keyed, queried in the learner's own
    words.

    ⚠️ Precision and recall are reported apart and never averaged. An unrelated
    remark handed to a tutor as context is worse than silence, because the tutor
    will try to use it.
    """
    from agent_newton.core.evaluation.recall import load_gold, score
    from agent_newton.core.recall import EmbeddedRecall, KeyedRecall
    from agent_newton.llm.embed import CachedEmbedder, OllamaEmbedder

    from agent_newton.core.recall import Recall

    cases = load_gold(gold)
    strategies: list[Recall] = [KeyedRecall()]
    try:
        embedder = CachedEmbedder(
            OllamaEmbedder(embed_model), Path(".cache/embed")
        )
        embedder.embed(["probe"])
        strategies.append(EmbeddedRecall(embedder, threshold))
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]no embedding model reachable ({type(exc).__name__}); "
            f"scoring the keyed strategy only[/yellow]\n"
        )

    console.print(
        f"[bold]{len(cases.cases)} case(s)[/bold] over {len(cases.corpus)} stored "
        f"utterances\n"
    )
    table = Table(box=None, padding=(0, 2))
    table.add_column("strategy")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("noise", justify="right")
    table.add_column("right to say nothing", justify="right")

    reports = []
    for strategy in strategies:
        report = score(cases, strategy, limit)
        reports.append(report)
        table.add_row(
            report.label,
            f"{report.precision:.1%}",
            f"{report.recall:.1%}",
            str(report.noise),
            f"{report.returned_nothing_correctly}/"
            f"{sum(1 for c in cases.cases if not c.relevant)}",
        )
    console.print(table)

    for report in reports:
        missed = report.missed()
        if missed:
            console.print(f"\n[yellow]{report.label} missed[/yellow]")
            for case_id, want in missed:
                console.print(f"  {case_id}: {', '.join(sorted(want))}")


@app.command("sitting")
def sitting(
    run: str = typer.Argument(
        "latest", help="A run directory, or 'latest' for the most recent one."
    ),
    results_dir: Path = typer.Option(
        Path("results"), "--results", help="Where run directories live."
    ),
    write: bool = typer.Option(
        True, "--write/--no-write", help="Also write sitting.md beside the transcript."
    ),
) -> None:
    """Read a stored sitting back as prose.

    Every figure quoted about the human sittings so far came out of a script
    written for the occasion, because the record is tens of kilobytes of JSON.
    This renders the audit log in order — question, answer, verdict, what the
    tutor said, and what the support level was chosen from.

    Reads only.
    """
    import json

    from agent_newton.core.evaluation.sitting import narrate, summarise
    from agent_newton.domains import registry

    if run == "latest":
        found = sorted(
            (d for d in results_dir.glob("*/") if (d / "transcript.json").exists()),
            key=lambda d: d.name,
        )
        if not found:
            console.print(f"[red]no sitting with a transcript under {results_dir}[/red]")
            raise typer.Exit(code=1)
        run_dir = found[-1]
    else:
        run_dir = Path(run) if Path(run).exists() else results_dir / run

    transcript_path = run_dir / "transcript.json"
    if not transcript_path.exists():
        console.print(f"[red]no transcript at {transcript_path}[/red]")
        raise typer.Exit(code=1)

    record = json.loads(transcript_path.read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    domain = registry.load_domain(manifest["domain"])

    counts = summarise(record["audit_log"])
    header = [
        f"run `{run_dir.name}`",
        f"domain {manifest['domain']}, arm {manifest['arm']}",
        f"tutor {manifest['models']['tutor']}",
        f"diagnostic {manifest['models']['diagnostic']}",
        f"completed: {record.get('completed')}"
        + (f", stopped because {record['stop_reason']}" if "stop_reason" in record else ""),
        f"support levels: {counts['levels'] or 'no tutor turns'}",
        f"verdicts: {counts['verdicts'] or 'none'}",
    ]
    figures = {}
    for key in ("items_attempted", "gain", "normalised_gain", "dose_on_gap"):
        if key in record:
            figures[key] = record[key]
    if "dose_by_concept" in record:
        figures["where the time went"] = record["dose_by_concept"]

    text = narrate(
        record["audit_log"],
        domain,
        learner_id=record.get("learner_id", ""),
        header=header,
        figures=figures,
    )
    console.print(text)
    if write:
        (run_dir / "sitting.md").write_text(text)
        console.print(f"[dim]written to {run_dir / 'sitting.md'}[/dim]")


@app.command("history")
def history(
    learner: str = typer.Argument(..., help="Whose history to read."),
    arm: str = typer.Option("coupled", "--arm", help="Histories are per arm."),
    concept: list[str] | None = typer.Option(
        None, "--concept", help="Restrict to these concepts. Repeatable."
    ),
    store_path: Path = typer.Option(
        Path("results/learners.db"), "--store", help="Learner store to read."
    ),
    out: Path | None = typer.Option(None, "--out", help="Output directory."),
) -> None:
    """What the system did about each skill, across every sitting.

    A learner who never grasps a concept despite sustained, appropriate teaching
    is an ordinary pedagogical case rather than a failed experiment. What can be
    established either way is whether the instruction was appropriate — which is
    a claim about the system's behaviour, so it holds for a person, for whom
    remediation and diagnostic accuracy are both unavailable.

    Reads only. Nothing here writes to the store.
    """
    from agent_newton.core.evaluation.teaching import records, repertoire, summarise
    from agent_newton.store import LearnerStore, check_learner_id

    if not store_path.exists():
        console.print(f"[red]no learner store at {store_path}[/red]")
        raise typer.Exit(code=1)

    with LearnerStore(store_path) as store:
        found = records(store, learner, arm, concepts=concept or None)

    if not found:
        # Distinct from a learner who was taught nothing: this one has no
        # history in this arm at all.
        console.print(
            f"[yellow]no history for {learner!r} in the {arm} arm[/yellow] — "
            f"histories are kept per arm, so check the other one"
        )
        raise typer.Exit(code=1)

    # Checked here too, not only in the store: `history` can be asked about a
    # learner who does not exist, so it may never touch `ensure_learner` — and
    # this is the line that turns an id into a path.
    check_learner_id(learner)
    directory = out or Path("results") / f"history_{learner}_{arm}"
    directory.mkdir(parents=True, exist_ok=True)

    # Rows as CSV, aggregate as JSON — the shape every row-level evaluation here
    # uses. Flat on purpose: this is what a figure is drawn from, so nesting it
    # would only have to be undone again.
    with (directory / "records.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["learner_id", "arm", "concept_id", "seq", "elapsed_days", "attempts",
             "correct", "unmeasurable", "nudge", "targeted", "worked_step", "hints",
             "reflects", "remediates", "remediation_targets", "distinct_items",
             "seeded", "decayed", "mastery_before", "mastery_after",
             "instruction_recorded"]
        )
        for record in found:
            for s in record.sittings:
                writer.writerow([
                    s.learner_id, s.arm, s.concept_id, s.seq, f"{s.elapsed_days:g}",
                    s.attempts, s.correct, s.unmeasurable,
                    s.levels.get("nudge", 0), s.levels.get("targeted", 0),
                    s.levels.get("worked_step", 0),
                    s.moves.get("hint", 0), s.moves.get("reflect", 0),
                    s.moves.get("remediate", 0),
                    "; ".join(s.remediation_targets), s.distinct_items,
                    s.seeded, s.decayed,
                    "" if s.mastery_before is None else f"{s.mastery_before:.4f}",
                    "" if s.mastery_after is None else f"{s.mastery_after:.4f}",
                    s.instruction_recorded,
                ])

    summary = summarise(found)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    console.print(
        f"[bold]{len(found)} concept(s)[/bold] over {summary['sittings']} sitting(s) "
        f"for {learner} ({arm})\nwriting to {directory}\n"
    )

    table = Table(title="teaching history", box=None)
    table.add_column("concept")
    table.add_column("sittings", justify="right")
    table.add_column("days", justify="right")
    table.add_column("attempts", justify="right")
    table.add_column("moved", justify="right")
    table.add_column("never tried")
    for record in sorted(found, key=lambda r: -r.attempts):
        outstanding = record.not_attempted
        table.add_row(
            record.concept_id,
            str(record.sittings_spanned),
            f"{record.days_spanned:g}",
            str(record.attempts),
            "—" if record.movement is None else f"{record.movement:+.2f}",
            # None is unavailable, not "nothing outstanding" — the sittings
            # simply did not keep what the tutor said.
            "[dim]not recorded[/dim]" if outstanding is None
            else "[green]nothing[/green]" if not outstanding
            else ", ".join(sorted(x.split(":", 1)[1] for x in outstanding)),
        )
    console.print(table)

    if not summary["instruction_recorded"]:
        console.print(
            "\n[yellow]No sitting in this history recorded what the tutor said.[/yellow]\n"
            "Turns have only been kept since 2026-08-12, so the instruction half "
            "is unavailable rather than empty — this is not evidence that nothing "
            "was taught."
        )
        return

    # The case the record exists for: sustained teaching, nothing moving.
    stuck = [
        r for r in found
        if r.attempts >= 10 and (r.movement or 0.0) <= 0.0
    ]
    for record in stuck:
        outstanding = record.not_attempted
        console.print(
            f"\n[yellow]{record.concept_id}[/yellow]: {record.attempts} attempts over "
            f"{record.sittings_spanned} sitting(s) and {record.days_spanned:g} days, "
            f"and the estimate did not rise."
        )
        if outstanding:
            console.print(
                f"  [red]The repertoire was not exhausted[/red] — never reached: "
                f"{', '.join(sorted(outstanding))}. That is a gap in the teaching, "
                f"not in the learner."
            )
        elif outstanding is not None:
            console.print(
                "  Every instructional move available was used. Whatever else is "
                "true, the system did not fail to try."
            )


if __name__ == "__main__":
    app()
