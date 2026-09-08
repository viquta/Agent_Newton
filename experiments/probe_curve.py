"""Does the comparison survive the shape of the remediation curve?

    uv run python experiments/probe_curve.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260927

`simulator.remediation_curve` is the one dial of the three with no note asking
for it. Exponential is today — a hint takes a constant *share* of what is left,
so a misconception approaches zero without arriving. Linear takes the same
*amount* every time, from what the learner started with, so a misconception can
be finished off.

Which of the two describes practice is a long-running question in the
learning-curve literature and the artifact has no way to prefer one. That is the
whole reason it is here: a result that holds under both is a result that does not
depend on the answer.

Swept across `remediation_factor` as well as the shape, because the shapes are
defined in terms of that factor — they agree exactly on the first hint and
separate only afterwards, so a single factor would show the shape at one point on
its own axis rather than across it.

Both arms at every point. The question is whether the *comparison* moves, not
whether the learners do — they certainly do, and the right-hand column says by
how much so a stable comparison cannot be mistaken for an inert dial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import ARMS, OUTCOMES, analyse, cohort, spent_seeds  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

CURVES = ("exponential", "linear")
FACTORS = (0.35, 0.55, 0.75)


def tuned(config: Config, curve: str, factor: float) -> Config:
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={"remediation_curve": curve, "remediation_factor": factor}
            ),
            "run_name": f"{config.run_name}_{curve}{factor}",
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

    rng = np.random.default_rng(args.seed)
    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "outcome": OUTCOMES[0],
        "standing": "exploratory",
        "points": [],
    }

    print(f"\ncoupled minus decoupled on the declared primary, N = {args.n}")
    header = (
        f"  {'curve':>12}{'factor':>8}{'difference':>12}{'holm':>10}{'c/d':>9}"
        f"{'ties':>6}{'remediation c/d':>18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for curve in CURVES:
        for factor in FACTORS:
            settings = tuned(config, curve, factor)
            metrics = {
                arm: cohort(
                    settings, arm, args.n, args.seed, suffix=f"_{curve}{factor}"
                )
                for arm in ARMS
            }
            rows = analyse(metrics["coupled"], metrics["decoupled"], rng)
            primary, adjusted = rows[0]
            point = {
                "curve": curve,
                "remediation_factor": factor,
                "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
                "mean_remediation": {
                    arm: metrics[arm]["mean_remediation"] for arm in ARMS
                },
                "mean_difference": primary.mean_difference,
                "ties": primary.ties,
                "favouring_coupled": primary.favouring_first,
                "favouring_decoupled": primary.favouring_second,
                "rank_biserial": primary.rank_biserial,
                "sign_p": primary.sign_p,
                "holm_p": adjusted,
                "significant": adjusted < ALPHA,
            }
            report["points"].append(point)
            rem = point["mean_remediation"]
            split = f"{primary.favouring_first}/{primary.favouring_second}"
            pair = f"{rem['coupled']:.3f}/{rem['decoupled']:.3f}"
            print(
                f"  {curve:>12}{factor:>8.2f}{primary.mean_difference:>+12.4f}"
                f"{adjusted:>10.2e}{split:>9}{primary.ties:>6}{pair:>18}"
            )

    print("\n  positive = the coupled arm ahead. The right-hand column is what")
    print("  the learners achieved, so a flat comparison cannot be read as a")
    print("  dial that did nothing.")

    directory = args.out or config.paths.results_dir / "probe_curve"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
