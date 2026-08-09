"""How diagnostic error becomes an outcome difference.

    uv run python experiments/run_propagation.py \
        --config experiments/configs/calculus.yaml \
        --diagnostic-summary results/diagnostic_calculus_gemma4-12b/summary.json

Runs the same cohort under three diagnostic conditions, in both arms:

===========  ============================================================
``oracle``   reads the injected label; diagnosis is perfect
``noised``   the same oracle, corrupted at the *measured* error rate
``llm``      the real agent, inferring from the student's step alone
===========  ============================================================

Everything else is held identical — seeds, items, tutor, planner, simulator — so
the only thing that moves between conditions is how good the diagnosis is.

The chain the study traces is short and has one gate in it. A hint remediates a
misconception only when it names one the learner actually holds; a misdiagnosis
produces a hint aimed elsewhere, and that hint does no work. So the gap between
``oracle`` and ``llm`` is the outcome cost of the diagnostic's error rate, and
``noised`` sits between them as a check on the explanation: if error rate alone
accounts for the gap, the noised condition should land near the real one. Where
it does not, the real agent's errors are structured rather than random — it
confuses particular labels for particular others, and that costs more or less
than the same number of uniform mistakes.

The noise rate is read from the offline evaluation's ``summary.json`` rather
than typed in, so the condition cannot silently drift from the measurement it is
meant to represent.

``oracle`` and ``noised`` call no model and take seconds. ``llm`` calls one per
incorrect step; identical prompts hit the response cache, so running the offline
diagnostic evaluation first makes much of it free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.manifest import RunManifest, assert_poolable  # noqa: E402

CONDITIONS = ("oracle", "noised", "llm")
ARMS = ("coupled", "decoupled")

#: The outcome the comparison is read on. Pre/post gain is reported beside it
#: but is ceiling-limited: a learner's two or three misconceptions touch only a
#: few bank items, so the test scores start high and move little.
PRIMARY = "mean_remediation"


def condition_config(base: Config, condition: str, arm: str, noise_rate: float) -> Config:
    """The base config with only the diagnostic and the arm changed."""
    diagnostic = base.agents.diagnostic
    if condition == "oracle":
        diagnostic = diagnostic.model_copy(update={"impl": "oracle", "noise_rate": 0.0})
    elif condition == "noised":
        diagnostic = diagnostic.model_copy(
            update={"impl": "noised_oracle", "noise_rate": noise_rate}
        )
    elif condition == "llm":
        diagnostic = diagnostic.model_copy(update={"impl": "llm", "noise_rate": 0.0})
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(f"unknown condition {condition!r}")

    agents = base.agents.model_copy(update={"diagnostic": diagnostic})
    return base.model_copy(
        update={
            "agents": agents,
            "arm": arm,
            "run_name": f"{base.run_name}_prop_{condition}",
        }
    )


def measured_noise_rate(path: Path) -> float:
    """1 - accuracy, from the offline diagnostic evaluation.

    Read from the artifact rather than accepted as a number, so the condition
    and the measurement it stands for cannot drift apart.
    """
    summary = json.loads(path.read_text())
    accuracy = float(summary["accuracy"])
    if summary.get("cases", 0) < 1:
        raise ValueError(f"{path} scored no cases")
    return round(1.0 - accuracy, 4)


def paired_differences(coupled: dict, decoupled: dict, key: str) -> list[float]:
    """Per-learner coupled - decoupled, matched by learner id.

    The same seed produces the same learner in both arms, so this is a paired
    comparison. Matching by id rather than by position means a change to the
    cohort's ordering cannot silently pair the wrong learners.
    """
    left = {row["learner_id"]: row[key] for row in coupled["per_learner"]}
    right = {row["learner_id"]: row[key] for row in decoupled["per_learner"]}
    shared = sorted(set(left) & set(right))
    if len(shared) != len(left) or len(shared) != len(right):
        raise ValueError("the two arms did not run the same learners")
    return [left[learner] - right[learner] for learner in shared]


def summarise(differences: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(differences),
        "sd": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        "favour_coupled": sum(1 for d in differences if d > 0),
        "ties": sum(1 for d in differences if d == 0),
        "favour_decoupled": sum(1 for d in differences if d < 0),
        "n": len(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-summary",
        type=Path,
        help="summary.json from `agent-newton evaluate diagnostic`; supplies the "
        "noise rate for the noised condition.",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        help="Override the measured rate. Recorded as an override in the summary.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset to run, e.g. 'oracle,noised' to skip the model.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(sorted(unknown))}")

    noise_rate = args.noise_rate
    noise_source = "override" if noise_rate is not None else None
    if "noised" in conditions and noise_rate is None:
        if not args.diagnostic_summary:
            parser.error(
                "the noised condition needs a measured error rate: pass "
                "--diagnostic-summary, or --noise-rate to override it"
            )
        noise_rate = measured_noise_rate(args.diagnostic_summary)
        noise_source = str(args.diagnostic_summary)

    base = Config.from_yaml(args.config)
    results: dict[str, dict[str, dict]] = {}
    manifests: list[RunManifest] = []

    for condition in conditions:
        results[condition] = {}
        for arm in ARMS:
            config = condition_config(base, condition, arm, noise_rate or 0.0)
            print(f"\n=== {condition} / {arm} " + "=" * 40, flush=True)
            metrics = run(config)
            results[condition][arm] = metrics
            manifests.append(
                RunManifest.read(base.paths.results_dir / metrics["run_id"])
            )

    # Nothing here is comparable if the domain content moved between runs.
    assert_poolable(manifests)

    report = {
        "config": str(args.config),
        "domain": base.domain,
        "n_learners": base.cohort.n_learners,
        "max_items": base.cohort.max_items,
        "seed": base.seed,
        "primary_outcome": PRIMARY,
        "noise_rate": noise_rate,
        "noise_rate_source": noise_source,
        "conditions": {},
    }

    for condition in conditions:
        coupled = results[condition]["coupled"]
        decoupled = results[condition]["decoupled"]
        report["conditions"][condition] = {
            "run_ids": {arm: results[condition][arm]["run_id"] for arm in ARMS},
            "diagnostic_accuracy": {
                arm: results[condition][arm]["diagnostic_accuracy"] for arm in ARMS
            },
            "mean_remediation": {arm: results[condition][arm]["mean_remediation"] for arm in ARMS},
            "mean_gain": {arm: results[condition][arm]["mean_gain"] for arm in ARMS},
            "unmeasurable_steps": {
                arm: results[condition][arm]["unmeasurable_steps"] for arm in ARMS
            },
            "paired_remediation": summarise(
                paired_differences(coupled, decoupled, "remediation")
            ),
            "paired_gain": summarise(paired_differences(coupled, decoupled, "gain")),
        }

    directory = args.out or base.paths.results_dir / f"propagation_{base.domain}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n\nwritten to {directory / 'summary.json'}\n")
    header = f"{'condition':10} {'diag acc':>9} {'remediation c/d':>22} {'paired mean':>12} {'c>d':>5}"
    print(header)
    print("-" * len(header))
    for condition in conditions:
        entry = report["conditions"][condition]
        accuracy = entry["diagnostic_accuracy"]["coupled"]
        paired = entry["paired_remediation"]
        print(
            f"{condition:10} "
            f"{'n/a' if accuracy is None else f'{accuracy:.3f}':>9} "
            f"{entry['mean_remediation']['coupled']:>10.3f} / "
            f"{entry['mean_remediation']['decoupled']:<9.3f} "
            f"{paired['mean']:>+12.4f} "
            f"{paired['favour_coupled']:>3}/{paired['n']}"
        )


if __name__ == "__main__":
    main()
