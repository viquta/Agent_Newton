"""Is doubting a prerequisite worth doing, and where?

    uv run python experiments/sweep_doubt.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260819

`bkt.prerequisite_doubt` charges part of a repeated failure back to the
prerequisites the model already calls mastered. It exists because the graph
decided what may be *selected* and never informed what is *believed*: a concept
above `theta_upper` cannot be offered again, so the belief that closed it is
unfalsifiable, and failing what is built on it is the only evidence that can
still arrive.

Two dials, and the second is why one is not enough.

`bkt.prerequisite_doubt` (alpha)
    how much of a repeated failure reaches the prerequisites.

`simulator.prerequisite_dependence` (k)
    whether shaky foundations make a hint take less well. **At k = 0 the
    generator has no path by which strengthening a prerequisite helps
    anything.** So a sweep of alpha alone measures nothing but the cost of items
    spent off-route, and reads as "the mechanism is harmful" — a fact about the
    generator, not about the idea.

⚠️ **The primary comparison is within the coupled arm.** The decoupled arm cannot
compute doubt at all, so a coupled−decoupled difference at alpha > 0 moves only
because the coupled arm moved; reporting that as an arm comparison would dress a
within-arm effect as a between-arm one. The headline here is coupled@(alpha, k)
against coupled@(0, k), paired on the learner, which is a claim about the
mechanism with no arm comparison in it. The paired figure is printed beside it as
context and carries the caveat.

Two guards:

* **(0, 0) must reproduce the recorded numbers.** Run first, compared, and the
  sweep refuses to continue on a mismatch — a curve read against figures the
  code no longer produces means nothing.
* **A gain at k = 0 is flagged.** It was written expecting nothing could
  produce one, on the reasoning that reopening a prerequisite can only spend
  items when hints land in full regardless. ⚠️ **That reasoning was wrong and
  the first run said so.** A concept above `theta_upper` can still carry
  *unremediated misconceptions* — mastery is BKT over correct/incorrect, and a
  misconception can sit there unfired — so reopening it reaches them, and
  `remediation_ratio` rises for reasons that have nothing to do with
  dependence. The check is kept, as a signpost to that path rather than as an
  alarm about a defect.

Exploratory. It characterises a mechanism that stays at 0.0 in every experiment
config, and no point on it licenses turning it on for the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402
from run_paired import ARMS, analyse, by_learner  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA, compare  # noqa: E402

#: Both dials off. Not a parameter: the claim is that zero is today's system.
BASELINE = 0.0

#: §7b as re-recorded 2026-08-17, natural framing. These hold at **one seed and
#: one N** — profiles come from (seed, learner_id), so another seed is another
#: population and comparing against it would be reading the curve against
#: figures that were never about these learners. Checked only where they apply;
#: skipped, loudly, everywhere else.
RECORDED_SEED = 20260811
RECORDED_N = 160
RECORDED = {
    "remediation": +0.0259,
    "gain": +0.0021,
    "goals_mastered": +3.7750,
    "distance_to_goal": -5.9625,
}
TOLERANCE = 5e-4


def tuned(
    config: Config, alpha: float, dependence: float, arm: str, n: int, seed: int
) -> Config:
    return config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "bkt": config.bkt.model_copy(update={"prerequisite_doubt": alpha}),
            "simulator": config.simulator.model_copy(
                update={"prerequisite_dependence": dependence}
            ),
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_a{alpha:g}_k{dependence:g}",
        }
    )


def coupled_at(config: Config, alpha: float, k: float, n: int, seed: int) -> dict:
    return run(tuned(config, alpha, k, "coupled", n, seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--doubt", default="0,0.25,0.5,1.0")
    parser.add_argument("--dependence", default="0,0.5,1.0")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run (0, 0) and check it against the recorded numbers, then stop.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from; profiles come from (seed, learner_id)."
        )

    alphas = [float(a) for a in args.doubt.split(",")]
    ks = [float(k) for k in args.dependence.split(",")]
    for name, values in (("--doubt", alphas), ("--dependence", ks)):
        if values[0] != BASELINE:
            parser.error(
                f"{name} must start at 0: every other point is read against the "
                f"system without the mechanism at all."
            )

    rng = np.random.default_rng(args.seed)

    # --- guard 1: zero reproduces what is on record -------------------------
    base_metrics = {
        arm: run(tuned(config, BASELINE, BASELINE, arm, args.n, args.seed))
        for arm in ARMS
    }
    baseline_paired = {
        outcome.outcome: outcome.mean_difference
        for outcome, _ in analyse(base_metrics["coupled"], base_metrics["decoupled"], rng)
    }
    if (args.n, args.seed) != (RECORDED_N, RECORDED_SEED):
        print(
            f"\nzero not checked against the recorded numbers: those are "
            f"N = {RECORDED_N} on seed {RECORDED_SEED}, and this run is "
            f"N = {args.n} on seed {args.seed}.\n"
            f"    The curve stays internally comparable — every point is read "
            f"against zero\n    on these learners. It is the published "
            f"baseline that this cannot confirm,\n    which is what "
            f"`--baseline-only` on seed {RECORDED_SEED} is for.\n"
        )
    else:
        drift = {
            outcome: baseline_paired[outcome] - recorded
            for outcome, recorded in RECORDED.items()
            if abs(baseline_paired[outcome] - recorded) > TOLERANCE
        }
        if drift:
            print("\n⚠️  zero does not reproduce the recorded numbers:")
            for outcome, delta in drift.items():
                print(f"      {outcome}: off by {delta:+.4f}")
            print(
                "\n    The curve is read against those figures, so it cannot\n"
                "    continue. Either the mechanism is not inert at zero, or\n"
                "    something else moved and §7b needs re-recording first."
            )
            raise SystemExit(1)
        print("\nzero reproduces the recorded numbers — continuing.\n")

    if args.baseline_only:
        return

    # --- the grid -----------------------------------------------------------
    curve: list[dict] = []
    suspicious: list[str] = []
    for k in ks:
        reference = coupled_at(config, BASELINE, k, args.n, args.seed)
        decoupled = run(tuned(config, BASELINE, k, "decoupled", args.n, args.seed))
        for alpha in alphas:
            treated = (
                reference if alpha == BASELINE
                else coupled_at(config, alpha, k, args.n, args.seed)
            )
            # The headline: the coupled arm against itself, same learners.
            within = compare(
                "remediation", by_learner(treated), by_learner(reference), rng
            )
            # Context only. Moves solely because the coupled arm moved.
            across = compare(
                "remediation", by_learner(treated), by_learner(decoupled), rng
            )
            point = {
                "alpha": alpha,
                "k": k,
                "within_arm": {
                    "mean_difference": within.mean_difference,
                    "sign_p": within.sign_p,
                    "discordant": [within.favouring_first, within.favouring_second],
                },
                "across_arms": {"mean_difference": across.mean_difference},
            }
            curve.append(point)

            # --- guard 2: a gain at k = 0 comes from somewhere else --------
            if k == BASELINE and alpha > BASELINE and within.mean_difference > TOLERANCE:
                suspicious.append(
                    f"alpha={alpha:g}: {within.mean_difference:+.4f} at k = 0"
                )

    header = (
        f"  {'k':>5}{'alpha':>8}{'within-arm':>14}{'p':>10}{'c/d':>9}"
        f"{'coupled-decoupled':>20}"
    )
    print("the effect of doubting a prerequisite, against how much prerequisites matter")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in curve:
        within = point["within_arm"]
        mark = " *" if within["sign_p"] < ALPHA else ""
        cd = "{}/{}".format(*within["discordant"])
        print(
            f"  {point['k']:>5g}{point['alpha']:>8g}"
            f"{within['mean_difference']:>+14.4f}{within['sign_p']:>10.2e}{cd:>9}"
            f"{point['across_arms']['mean_difference']:>+20.4f}{mark}"
        )
    print(
        "\n  within-arm        = coupled at (alpha, k) minus coupled at (0, k),\n"
        "                      the same learners. This is the claim.\n"
        "  coupled-decoupled = context only. ⚠️ It moves solely because the\n"
        "                      coupled arm moved — the decoupled arm cannot do\n"
        "                      this at all — so it is not evidence about the\n"
        "                      architecture and must not be quoted as such."
    )

    if suspicious:
        print("\n⚠️  doubt helps at k = 0, so the benefit is not about dependence:")
        for line in suspicious:
            print(f"      {line}")
        print(
            "    At k = 0 a hint lands in full however shaky the foundations, so\n"
            "    none of this can be the dependence mechanism. The path that is\n"
            "    left: a concept above theta_upper can still hold *unremediated\n"
            "    misconceptions*, and reopening it reaches them. Read the curve\n"
            "    as being about the band closing off work, not about foundations."
        )

    out = args.out or config.paths.results_dir / "sweep_doubt"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "recorded_baseline": RECORDED,
                "baseline_paired": baseline_paired,
                "curve": curve,
                "suspicious_at_zero": suspicious,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
