"""Does a slipping learner give the dwelling cap something to bite on?

    uv run python experiments/probe_slip.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260926

`cohort.max_visits_per_concept` sets a concept aside once it has been worked that
many times without clearing the band, and it has never been swept. The stated
reason is that it had no subject: a simulated learner is wrong only when a bug of
theirs fires, so once the bug is taught they stop erring and no concept is ever
worked enough times to hit a cap. Nothing could get stuck.

`simulator.slip_rate` is meant to change that — a learner can now be wrong on a
question no misconception of theirs would have spoiled, so a concept can go on
failing after it has been taught. This measures whether that is true, by counting
how often the cap actually fires.

**One arm.** This is about whether a mechanism has a subject, not about the
architecture, so both conditions run coupled and nothing is compared between
arms.

⚠️ **Zero is the claim under test.** If the cap fires at slip 0.0 then it always
had a subject and the deferred note was wrong about why it was never swept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402
from run_paired import spent_seeds  # noqa: E402

from agent_newton.config import Config  # noqa: E402

RATES = (0.0, 0.05, 0.10, 0.20, 0.40)

#: Visits before a concept is set aside. Three, because it is the number the
#: sitting note asked for — "if I get 3 times wrong on a question, it should skip
#: it, note it as a weakness, and move on".
CAP = 3


def tuned(config: Config, rate: float, cap: int | None) -> Config:
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(update={"slip_rate": rate}),
            "cohort": config.cohort.model_copy(
                update={"max_visits_per_concept": cap}
            ),
            "arm": "coupled",
            "seed": config.seed,
            "run_name": f"{config.run_name}_slip{rate}",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(f"--seed {args.seed} is the config's, which sized the study.")
    spent = spent_seeds(config.paths.results_dir)
    if args.seed in spent:
        parser.error(f"--seed {args.seed} is spent by {', '.join(sorted(spent[args.seed]))}.")

    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "max_visits_per_concept": CAP,
        "standing": "exploratory",
        "points": [],
    }

    print(f"\ncoupled arm, cap at {CAP} visits, N = {args.n}")
    header = (
        f"  {'slip':>6}{'set aside':>12}{'per learner':>13}{'unlabelled':>12}"
        f"{'remediation':>13}{'items':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rate in RATES:
        settings = tuned(config, rate, CAP).model_copy(
            update={
                "seed": args.seed,
                "cohort": config.cohort.model_copy(
                    update={"n_learners": args.n, "max_visits_per_concept": CAP}
                ),
            }
        )
        metrics = run(settings)
        aside = metrics["replans_by_trigger"].get("concept_set_aside", 0)
        point = {
            "slip_rate": rate,
            "run_id": metrics["run_id"],
            "concept_set_aside": aside,
            "per_learner": aside / args.n,
            "unlabelled_errors": metrics["unlabelled_errors"],
            "mean_remediation": metrics["mean_remediation"],
            "mean_items": metrics["mean_items"],
            "replans_by_trigger": metrics["replans_by_trigger"],
        }
        report["points"].append(point)
        print(
            f"  {rate:>6.2f}{aside:>12}{point['per_learner']:>13.3f}"
            f"{point['unlabelled_errors']:>12}{point['mean_remediation']:>13.4f}"
            f"{point['mean_items']:>8.1f}"
        )

    print("\n  `set aside` counts the cap firing across the whole cohort.")
    print("  ⚠ If the row at 0.00 is zero, the cap had no subject before this.")

    directory = args.out or config.paths.results_dir / "probe_slip"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
