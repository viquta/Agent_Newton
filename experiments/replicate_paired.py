"""Replicate the confirmatory paired design over fresh seeds.

    uv run python experiments/replicate_paired.py \
        --config experiments/configs/calculus.yaml --n 160

The declared primary outcome reads +0.0259 on the confirmatory seed, while the
pilot puts the same effect at less than half that and gives N = 160 a power of
0.26. An estimate more than twice the pilot's, obtained at a quarter of the power
it was sized for, is the winner's-curse signature: where power is low, the
estimates that clear significance are the ones that overshot. As it stands the
significant result cannot be told apart from a favourable draw.

**This is a replication check on an estimate, not a new test of the hypothesis.**
It cannot make the primary powered — ten runs of an underpowered design is still
an underpowered design — and nothing it produces licenses changing the primary
outcome in either direction. What it can do is say whether the estimate is
*stable* across independent populations, which is the question the single
confirmatory run cannot answer about itself.

Three commitments, all declared here rather than decided afterwards:

1. **The seeds are fixed in advance and every one is reported.** None is dropped,
   re-rolled or added after a result is seen. They are consecutive and mechanical
   so that nobody, the author included, could have chosen them to flatter the
   outcome.
2. **The pre-registered quantity is the median and spread of the framing-A
   remediation difference** — deliberately *not* a count of how many seeds clear
   correction. "k of 10 were significant" would re-import the very lottery this
   exists to characterise: at 143 ties out of 160, significance at 0.26 power is
   close to a coin toss, while the point estimate behaves.
3. **The prediction was written down first**, in the private handover, before
   this file had been run once.

⚠️ The design is imported from ``run_paired`` rather than restated. A replication
that drew its cohorts differently would not be one, and the failure would look
like a result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import ARMS, OUTCOMES, analyse, cohort  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

#: Fixed in advance. Consecutive on purpose: an arbitrary-looking list invites
#: the question of how it was chosen, and the answer has to be "mechanically".
SEEDS = tuple(range(20260901, 20260911))

#: The seed the confirmatory figures were read off. Run first, through this
#: script's own code path, and required to reproduce before any fresh seed is
#: touched — a replication harness that quietly measures something else would
#: produce a beautifully consistent answer to the wrong question.
BASELINE_SEED = 20260811
RECORDED_N = 160
RECORDED = {
    "remediation": +0.0259,
    "gain": +0.0021,
    "goals_mastered": +3.7750,
    "distance_to_goal": -5.9625,
}
TOLERANCE = 5e-4

#: What the pilot estimated on an independent population, and what the
#: confirmatory seed returned. Printed beside the replication so the comparison
#: the whole exercise is about does not have to be assembled by hand.
PILOT_ESTIMATE = +0.0115
CONFIRMATORY_ESTIMATE = +0.0259

PRIMARY = "remediation"


def one_seed(config: Config, n: int, seed: int, dose_matched: bool) -> dict:
    """Framing A for this seed, with framing B beside it when asked for."""
    metrics = {arm: cohort(config, arm, n, seed) for arm in ARMS}
    rng = np.random.default_rng(seed)
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    point: dict = {
        "seed": seed,
        "outcomes": {
            result.outcome: {
                "mean_difference": result.mean_difference,
                "ci95": list(result.ci95),
                "discordant": [result.favouring_first, result.favouring_second],
                "ties": result.ties,
                "sign_p": result.sign_p,
                "holm": adjusted,
                "significant": adjusted < ALPHA,
            }
            for result, adjusted in rows
        },
    }
    if dose_matched:
        # ⚠️ floor, matching `run_paired` exactly. A replication that capped
        # the budget by a different rule would not be replicating framing B.
        capped = max(1, math.floor(metrics["decoupled"]["mean_items"]))
        matched = cohort(
            config.model_copy(
                update={"cohort": config.cohort.model_copy(update={"max_items": capped})}
            ),
            "coupled",
            n,
            seed,
            suffix="_dosematched",
        )
        rng = np.random.default_rng(seed)
        point["dose_matched"] = {
            "capped_at": capped,
            "outcomes": {
                result.outcome: {
                    "mean_difference": result.mean_difference,
                    "discordant": [result.favouring_first, result.favouring_second],
                    "holm": adjusted,
                    "significant": adjusted < ALPHA,
                }
                for result, adjusted in analyse(matched, metrics["decoupled"], rng)
            },
        }
    return point


def check_baseline(point: dict, n: int) -> None:
    """Refuse to continue unless the confirmatory seed still reproduces."""
    if n != RECORDED_N:
        print(
            f"\nbaseline not checked: the recorded figures are at N = {RECORDED_N} "
            f"and this run is at N = {n}.\n"
            f"    The replication is still internally comparable; it is the "
            f"published baseline that is not.\n"
        )
        return
    drift = {
        outcome: point["outcomes"][outcome]["mean_difference"] - recorded
        for outcome, recorded in RECORDED.items()
        if abs(point["outcomes"][outcome]["mean_difference"] - recorded) > TOLERANCE
    }
    if drift:
        print(f"\n⚠️  seed {BASELINE_SEED} does not reproduce the recorded numbers:")
        for outcome, delta in drift.items():
            print(f"      {outcome}: off by {delta:+.4f}")
        print(
            "\n    Every fresh seed below would be read against those figures, so "
            "this\n    cannot continue. Either the design has moved, or the "
            "recorded\n    numbers need re-recording first."
        )
        raise SystemExit(1)
    print(f"seed {BASELINE_SEED} reproduces the recorded numbers — continuing.\n")


def distribution(points: list[dict], outcome: str) -> dict:
    """The pre-registered summary: where the estimate sits, and how much it moves."""
    values = sorted(p["outcomes"][outcome]["mean_difference"] for p in points)
    return {
        "n_seeds": len(values),
        "median": stats.median(values),
        "mean": stats.fmean(values),
        "min": values[0],
        "max": values[-1],
        "stdev": stats.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
        # Reported because a reader will want it, and labelled so it cannot be
        # mistaken for the pre-registered quantity — see the module docstring.
        "seeds_clearing_correction": sum(
            p["outcomes"][outcome]["significant"] for p in points
        ),
    }


def show(points: list[dict], primary: dict) -> None:
    print(f"\nframing A, per seed — primary outcome ({PRIMARY})\n")
    header = f"  {'seed':>10}{'mean diff':>12}{'ties':>7}{'c/d':>9}{'holm':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in points:
        o = p["outcomes"][PRIMARY]
        mark = " *" if o["significant"] else ""
        coupled, decoupled = o["discordant"]
        ratio = f"{coupled}/{decoupled}"
        print(
            f"  {p['seed']:>10}{o['mean_difference']:>+12.4f}{o['ties']:>7}"
            f"{ratio:>9}{o['holm']:>11.2e}{mark}"
        )

    spread = f"{primary['min']:+.4f} to {primary['max']:+.4f}"
    print(f"\n  {'median':>10}{primary['median']:>+12.4f}")
    print(f"  {'mean':>10}{primary['mean']:>+12.4f}")
    print(f"  {'range':>10}{spread:>26}")
    print(f"  {'stdev':>10}{primary['stdev']:>+12.4f}")

    print(
        f"\n  for comparison — pilot estimate {PILOT_ESTIMATE:+.4f}, "
        f"confirmatory seed {CONFIRMATORY_ESTIMATE:+.4f}"
    )
    print(
        "\n  ⚠ The pre-registered quantity is the median and spread above, not "
        "the\n    number of seeds marked *. At this tie rate significance is "
        "close to a\n    lottery; the point estimate is what behaves."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=RECORDED_N, help="Learners per arm.")
    parser.add_argument(
        "--dose-matched",
        action="store_true",
        help="Also run framing B per seed. Reported beside A, never instead of it.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    print(f"checking the baseline: seed {BASELINE_SEED}, N = {args.n}")
    baseline = one_seed(config, args.n, BASELINE_SEED, dose_matched=False)
    check_baseline(baseline, args.n)

    points = []
    for seed in SEEDS:
        print(f"  seed {seed} …", flush=True)
        points.append(one_seed(config, args.n, seed, args.dose_matched))

    primary = distribution(points, PRIMARY)
    show(points, primary)

    directory = args.out or Path("results") / "replication_paired"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(args.config),
        "n_learners": args.n,
        "seeds": list(SEEDS),
        "baseline_seed": BASELINE_SEED,
        "baseline": baseline["outcomes"],
        "primary_outcome": PRIMARY,
        "pilot_estimate": PILOT_ESTIMATE,
        "confirmatory_estimate": CONFIRMATORY_ESTIMATE,
        "distribution": {o: distribution(points, o) for o in OUTCOMES},
        "per_seed": points,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwritten to {directory}")


if __name__ == "__main__":
    main()
