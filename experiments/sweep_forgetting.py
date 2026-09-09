"""The paired comparison across forgetting strengths, swept from zero.

    uv run python experiments/sweep_forgetting.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260924

The grid measured one point and found the declared primary reverses against a
learner who forgets. This is the shape of that: whether the reversal arrives
gradually, at a threshold, or not at all below some strength.

⚠️ **Swept, never chosen.** A strength picked because it produced a result would
assume the conclusion. That is `simulator.prerequisite_dependence`'s rule and it
applies here for the same reason — with the sign reversed, since here the dial
happens to favour the arm the study is *not* arguing for. Reporting the curve is
what makes the choice of one point on it uninteresting.

⚠️ **Zero is checked, not trusted, and the check is internal.**
`sweep_prerequisites.py` can compare its zero against the published confirmatory
figures because it re-uses the confirmatory seed. This runs on a fresh one — that
seed is spent — so those figures do not apply. Instead zero is run *twice*: once
with the dial at 0.0 and a period set, and once with the dial removed entirely.
The two must agree exactly, or the new code path is doing something when it is
switched off, and the sweep refuses to continue.

Both framings at every point. The grid found the dose confound does not explain
the reversal, but the whole curve is a stronger statement than one point of it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import ARMS, OUTCOMES, analyse, cohort, spent_seeds  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PERIOD = 10

#: Exact. The two zero runs differ only in whether the dial is present, and a
#: dial that is off must be indistinguishable from a dial that is absent.
TOLERANCE = 1e-12


def tuned(config: Config, rate: float | None, label: str) -> Config:
    """``rate=None`` removes the dial rather than setting it to zero."""
    update = {} if rate is None else {
        "forgetting_rate": rate,
        "forgetting_period": PERIOD,
    }
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(update=update),
            "run_name": f"{config.run_name}_{label}",
        }
    )


def measure(config: Config, rate: float | None, label: str, args, rng) -> dict:
    """One point: both arms, both framings."""
    settings = tuned(config, rate, label)
    metrics = {
        arm: cohort(settings, arm, args.n, args.seed, suffix=f"_{label}")
        for arm in ARMS
    }
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    budget = max(1, math.floor(metrics["decoupled"]["mean_items"]))
    capped = settings.model_copy(
        update={"cohort": settings.cohort.model_copy(update={"max_items": budget})}
    )
    matched = cohort(capped, "coupled", args.n, args.seed, suffix=f"_{label}_dose")
    matched_rows = analyse(matched, metrics["decoupled"], rng)

    primary, adjusted = rows[0]
    matched_primary, matched_adjusted = matched_rows[0]
    return {
        "forgetting_rate": rate,
        "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
        "dose_matched_run_id": matched["run_id"],
        "dose_matched_budget": budget,
        # What the dial does to the learner, regardless of the arms. If this does
        # not fall as the rate rises, the dial is not biting and nothing else on
        # the row means anything.
        "mean_remediation": {
            arm: metrics[arm]["mean_remediation"] for arm in ARMS
        },
        "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
        "framing_a": {
            "mean_difference": primary.mean_difference,
            "ties": primary.ties,
            "favouring_coupled": primary.favouring_first,
            "favouring_decoupled": primary.favouring_second,
            "rank_biserial": primary.rank_biserial,
            "sign_p": primary.sign_p,
            "holm_p": adjusted,
            "significant": adjusted < ALPHA,
        },
        "framing_b": {
            "mean_difference": matched_primary.mean_difference,
            "sign_p": matched_primary.sign_p,
            "holm_p": matched_adjusted,
            "significant": matched_adjusted < ALPHA,
        },
        "all_outcomes_framing_a": [
            {
                "outcome": r.outcome,
                "mean_difference": r.mean_difference,
                "holm_p": adj,
                "significant": adj < ALPHA,
            }
            for r, adj in rows
        ],
    }


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
        parser.error(
            f"--seed {args.seed} is already spent by "
            f"{', '.join(sorted(spent[args.seed]))}."
        )

    rng = np.random.default_rng(args.seed)

    # Zero, twice: dial off against dial absent. Run before anything else,
    # because a leak here makes every row below meaningless.
    absent = measure(config, None, "absent", args, rng)
    zero = measure(config, 0.0, "zero", args, rng)
    drift = abs(
        zero["framing_a"]["mean_difference"] - absent["framing_a"]["mean_difference"]
    )
    if drift > TOLERANCE:
        raise SystemExit(
            f"\n⚠️  the dial does something when it is switched off.\n"
            f"    dial absent: {absent['framing_a']['mean_difference']:+.6f}\n"
            f"    dial at 0.0: {zero['framing_a']['mean_difference']:+.6f}\n"
            f"    drift {drift:.2e} over a tolerance of {TOLERANCE:.0e}.\n"
            f"    Every row of this sweep would be measured against a moved "
            f"baseline, so it is not run."
        )
    print(
        f"\nzero checks out — dial absent and dial at 0.0 agree to "
        f"{drift:.1e}; continuing."
    )

    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "period": PERIOD,
        "outcome": OUTCOMES[0],
        "standing": "exploratory",
        "zero_check": {
            "dial_absent": absent["framing_a"]["mean_difference"],
            "dial_at_zero": zero["framing_a"]["mean_difference"],
            "drift": drift,
            "tolerance": TOLERANCE,
        },
        "points": [],
    }

    print("\ncoupled minus decoupled on the declared primary")
    header = (
        f"  {'rate':>6}{'framing A':>12}{'holm':>10}{'c/d':>10}{'ties':>6}"
        f"{'framing B':>12}{'remediation c/d':>18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rate in RATES:
        found = zero if rate == 0.0 else measure(config, rate, f"r{rate}", args, rng)
        report["points"].append(found)
        a, b = found["framing_a"], found["framing_b"]
        rem = found["mean_remediation"]
        split = f"{a['favouring_coupled']}/{a['favouring_decoupled']}"
        pair = f"{rem['coupled']:.3f}/{rem['decoupled']:.3f}"
        print(
            f"  {rate:>6.2f}{a['mean_difference']:>+12.4f}{a['holm_p']:>10.2e}"
            f"{split:>10}{a['ties']:>6}{b['mean_difference']:>+12.4f}{pair:>18}"
        )

    print("\n  positive = the coupled arm ahead.")
    print("  `remediation c/d` is what each arm achieved — the dial biting,")
    print("  independent of any comparison between them.")

    directory = args.out or config.paths.results_dir / "sweep_forgetting"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
