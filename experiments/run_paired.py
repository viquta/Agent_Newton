"""The paired cohort comparison, at a size fixed in advance.

    uv run python experiments/run_paired.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

The same seed produces the same learner in both arms, so every learner is their
own control and the unit of analysis is the difference between the two
architectures for one person.

**The seed must differ from the one the power analysis used.** Learner profiles
are drawn from ``(seed, learner_id)``, so re-using it would hand the
confirmatory run the very learners the sample size was chosen from — sizing a
study on data and then reporting that same data as its result. The script
refuses a seed matching the config's.

Outcomes are reported together with a Holm correction. One cohort answers
several questions, and without adjusting, the chance that *something* among them
clears 0.05 is well above 0.05.

``--dose-matched`` reruns the coupled arm with its item budget cut to what the
decoupled arm actually used. The decoupled planner walks off the end of its list
and stops, so it attempts fewer items; that difference is a consequence of the
manipulation rather than an imposed cap, but a reader is entitled to ask whether
the effect is just more practice. This answers it with a number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import (  # noqa: E402
    ALPHA,
    compare,
    holm_bonferroni,
)
from agent_newton.manifest import RunManifest, assert_poolable  # noqa: E402

#: Reported together, primary first. Order fixes the reporting order, not the
#: correction, which treats them as one family.
OUTCOMES = ("remediation", "gain", "goals_mastered", "distance_to_goal")
ARMS = ("coupled", "decoupled")


def cohort(config: Config, arm: str, n: int, seed: int, suffix: str = "") -> dict:
    tuned = config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_paired{suffix}",
        }
    )
    return run(tuned)


def by_learner(metrics: dict) -> dict[str, dict]:
    return {row["learner_id"]: row for row in metrics["per_learner"]}


def analyse(coupled: dict, decoupled: dict, rng: np.random.Generator) -> list:
    results = [compare(o, by_learner(coupled), by_learner(decoupled), rng) for o in OUTCOMES]
    adjusted = holm_bonferroni([r.sign_p for r in results])
    return list(zip(results, adjusted))


def show(rows: list, title: str) -> None:
    print(f"\n{title}")
    header = (
        f"  {'outcome':18}{'mean diff':>12}{'95% CI':>22}{'ties':>7}"
        f"{'c/d':>9}{'r':>8}{'sign p':>10}{'holm':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result, adjusted in rows:
        ci = f"[{result.ci95[0]:+.3f}, {result.ci95[1]:+.3f}]"
        mark = " *" if adjusted < ALPHA else ""
        print(
            f"  {result.outcome:18}{result.mean_difference:>+12.4f}{ci:>22}"
            f"{result.ties:>7}"
            f"{f'{result.favouring_first}/{result.favouring_second}':>9}"
            f"{result.rank_biserial:>+8.3f}"
            f"{result.sign_p:>10.2e}{adjusted:>9.2e}{mark}"
        )
    print("\n  ties = learners the two architectures treated identically;")
    print("  c/d  = discordant pairs favouring coupled / decoupled;")
    print("  r    = rank-biserial; * = significant after Holm correction.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Learners per arm.")
    parser.add_argument(
        "--seed", type=int, required=True, help="Must differ from the config's."
    )
    parser.add_argument("--dose-matched", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which is the seed the "
            f"power analysis sized from. Learner profiles come from "
            f"(seed, learner_id), so this would report the same learners the "
            f"sample size was chosen on. Pick another."
        )

    metrics = {arm: cohort(config, arm, args.n, args.seed) for arm in ARMS}
    manifests = [
        RunManifest.read(config.paths.results_dir / metrics[arm]["run_id"]) for arm in ARMS
    ]
    assert_poolable(manifests)

    rng = np.random.default_rng(args.seed)
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "power_sized_on_seed": config.seed,
        "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
        "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
        "primary_outcome": OUTCOMES[0],
        "alpha": ALPHA,
        "results": [
            {
                "outcome": r.outcome,
                "mean_difference": r.mean_difference,
                "ci95": list(r.ci95),
                "n_pairs": r.n_pairs,
                "ties": r.ties,
                "favouring_coupled": r.favouring_first,
                "favouring_decoupled": r.favouring_second,
                "rank_biserial": r.rank_biserial,
                "sign_p": r.sign_p,
                "wilcoxon_p": r.wilcoxon_p,
                "holm_p": adjusted,
                "significant": adjusted < ALPHA,
            }
            for r, adjusted in rows
        ],
    }
    show(rows, f"paired comparison — {args.n} learners, seed {args.seed}")
    print(
        f"\n  items attempted: coupled {metrics['coupled']['mean_items']:.1f}, "
        f"decoupled {metrics['decoupled']['mean_items']:.1f}"
    )

    if args.dose_matched:
        budget = max(1, math.floor(metrics["decoupled"]["mean_items"]))
        capped = config.model_copy(
            update={"cohort": config.cohort.model_copy(update={"max_items": budget})}
        )
        matched = cohort(capped, "coupled", args.n, args.seed, suffix="_dosematched")
        matched_rows = analyse(matched, metrics["decoupled"], rng)
        show(matched_rows, f"dose-matched — coupled capped at {budget} items")
        report["dose_matched"] = {
            "budget": budget,
            "run_id": matched["run_id"],
            "mean_items": matched["mean_items"],
            "results": [
                {
                    "outcome": r.outcome,
                    "mean_difference": r.mean_difference,
                    "sign_p": r.sign_p,
                    "holm_p": adjusted,
                    "significant": adjusted < ALPHA,
                }
                for r, adjusted in matched_rows
            ],
        }

    directory = args.out or config.paths.results_dir / f"paired_{config.domain}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
