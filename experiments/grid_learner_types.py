"""Both arms against each kind of simulated learner, and the paths they take.

    uv run python experiments/grid_learner_types.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260920

⚠️ **Exploratory, and it may not revise anything already declared.** The
confirmatory comparison was pre-declared on one population and held to on it.
These are *additional* populations, and the `basic` row is that same population —
so it is the baseline the others are read against, not a re-run of the result.
A category that came out better is not a better answer to the declared question.

Every category is a point in the dial space of `SimulatorConfig`, and the dials
are off in `basic`, which is why `basic` reproduces what is already stored.

⚠️ `exponential` is not a separate category. It is what `remediation_factor`
already does, so it *is* `basic`; naming it twice would run one configuration
under two labels and invite a reader to compare a thing with itself.

Each category is run in both arms on the same seed, so a learner meets both
architectures — the pairing the whole comparison rests on — and the paired
statistics are the same ones `run_paired` applies, correction included.

⚠️ **Framing B is run for every category, and for forgetting it is not
optional.** The coupled arm attempts about 1.37 times as many items, so under a
mechanism that advances with work done it meets about 1.37 times as many
forgetting events. A difference measured in framing A alone therefore cannot be
told apart from the coupled arm being charged for doing more. §8.5 asks for the
framings anyway; here one of them is load-bearing.

Two series come back per arm. `remediation` is ground truth, moved by teaching
and by forgetting. `accuracy` is what was observed, moved by those and by a slip,
which changes no belief at all. Slipping is invisible in the first and obvious in
the second, which is the reason for reporting both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import (  # noqa: E402
    ARMS,
    OUTCOMES,
    analyse,
    by_learner,
    cohort,
    record,
    spent_seeds,
)

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

#: Name → what it changes about `SimulatorConfig`. `basic` changes nothing and is
#: the population every stored result was produced on.
#:
#: The settings are round numbers chosen to be legible, not tuned. ⚠️ A value
#: picked because it made an arm win would assume the conclusion — the sweeps
#: are what say whether a finding survives the choice, and these are one point
#: each, for a picture.
CATEGORIES: dict[str, dict] = {
    "basic": {},
    "linear": {"remediation_curve": "linear"},
    "forgetful": {"forgetting_rate": 0.5, "forgetting_period": 10},
    #: 0.10 is `BKTConfig.p_slip`'s default — the rate the belief model already
    #: assumes of a learner the generator could not make slip.
    "slipper": {"slip_rate": 0.10},
}


def populated(config: Config, category: str) -> Config:
    """The config with one category's dials set."""
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(update=CATEGORIES[category]),
            "run_name": f"{config.run_name}_{category}",
        }
    )


def paths(metrics: dict) -> list[dict]:
    """The cohort's two series, as `run_cohort` averaged them."""
    return metrics.get("mean_trajectory", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Learners per arm.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from. Pick another."
        )
    spent = spent_seeds(config.paths.results_dir)
    if args.seed in spent:
        parser.error(
            f"--seed {args.seed} has already been used by "
            f"{', '.join(sorted(spent[args.seed]))}. Profiles come from "
            f"(seed, learner_id), so this would re-report learners an existing "
            f"analysis has already seen."
        )

    rng = np.random.default_rng(args.seed)
    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "alpha": ALPHA,
        "primary_outcome": OUTCOMES[0],
        "standing": "exploratory",
        "categories": {},
    }

    for category in CATEGORIES:
        tuned = populated(config, category)
        metrics = {
            arm: cohort(tuned, arm, args.n, args.seed, suffix=f"_{category}")
            for arm in ARMS
        }
        rows = analyse(metrics["coupled"], metrics["decoupled"], rng)
        report["categories"][category] = {
            "dials": CATEGORIES[category],
            # The provenance chain: a summary that names its runs can be checked
            # against them, and a run no summary names is a byproduct.
            "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
            "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
            "unlabelled_errors": {
                arm: metrics[arm]["unlabelled_errors"] for arm in ARMS
            },
            "outcomes": [record(result, adjusted) for result, adjusted in rows],
            "paths": {arm: paths(metrics[arm]) for arm in ARMS},
        }

        # Framing B. The dose difference is endogenous, but a mechanism that
        # advances with items turns it into exposure, and exposure is not
        # architecture. Capping the coupled arm at the decoupled arm's budget is
        # what separates the two.
        budget = max(1, math.floor(metrics["decoupled"]["mean_items"]))
        capped = tuned.model_copy(
            update={"cohort": tuned.cohort.model_copy(update={"max_items": budget})}
        )
        matched = cohort(
            capped, "coupled", args.n, args.seed, suffix=f"_{category}_dosematched"
        )
        matched_rows = analyse(matched, metrics["decoupled"], rng)
        report["categories"][category]["dose_matched"] = {
            "budget": budget,
            "run_id": matched["run_id"],
            "mean_items": matched["mean_items"],
            "outcomes": [record(result, adjusted) for result, adjusted in matched_rows],
        }

        primary = report["categories"][category]["outcomes"][0]
        print(f"\n{category}  {CATEGORIES[category] or 'dials off — today'}")
        print(
            f"  {OUTCOMES[0]}: {primary['mean_difference']:+.4f} "
            f"holm {primary['holm_p']:.2e} "
            f"ties {primary['ties']}/{args.n} "
            f"c/d {primary['favouring_coupled']}/{primary['favouring_decoupled']}"
        )
        matched_primary = report["categories"][category]["dose_matched"]["outcomes"][0]
        print(
            f"  dose-matched at {budget}: "
            f"{matched_primary['mean_difference']:+.4f} "
            f"holm {matched_primary['holm_p']:.2e} "
            f"c/d {matched_primary['favouring_coupled']}"
            f"/{matched_primary['favouring_decoupled']}"
        )

    directory = args.out or config.paths.results_dir / "grid_learner_types"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
