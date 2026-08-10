"""How many learners the paired comparison needs, decided before running it.

    uv run python experiments/power_analysis.py --config experiments/configs/calculus.yaml

The comparison is paired: the same seed produces the same learner in both arms,
so every learner is their own control. What that pairing produces here is not a
bell curve. Most learners record **exactly zero** difference between the arms —
their misconceptions were either reached and remediated in both, or in neither —
and the effect lives entirely in a minority of discordant pairs.

That rules out the usual machinery. A paired t-test assumes a distribution this
plainly is not, and a formula-based power calculation would be quoting a number
derived from an assumption the data contradicts. So power is estimated by
**simulation**: run one large pilot pool, resample cohorts of each candidate
size from it, apply the test that will actually be used, and count how often it
rejects.

The test is the **exact sign test** on discordant pairs. It assumes nothing about
the size of a difference, only its direction, which is all a tie-heavy paired
sample supports. Wilcoxon's signed-rank is reported beside it: it uses the
magnitudes and so has more power when they are informative, at the cost of
assuming the differences are comparable across learners.

Resampling with replacement from a pilot pool stands in for drawing fresh
cohorts. That holds because learners are independent draws — the profile is
sampled from `(seed, learner_id)` alone — and it costs one pilot run rather than
hundreds of cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import (  # noqa: E402
    ALPHA,
    bootstrap_ci,
    paired_differences,
    rank_biserial,
    sign_test,
    wilcoxon,
)

#: Outcomes to size for. The first is the declared primary.
OUTCOMES = ("remediation", "gain", "goals_mastered", "distance_to_goal")

#: Cohort sizes to evaluate.
SIZES = (20, 30, 40, 60, 80, 120, 160, 200)

TARGET_POWER = 0.80


def power_at(
    differences: np.ndarray, size: int, replicates: int, rng: np.random.Generator
) -> dict:
    """Empirical rejection rate for cohorts of ``size`` drawn from the pilot."""
    picks = rng.integers(0, differences.size, size=(replicates, size))
    samples = differences[picks]
    sign = np.array([sign_test(row) for row in samples])
    rank = np.array([wilcoxon(row) for row in samples])
    return {
        "n_learners": size,
        "power_sign_test": float((sign < ALPHA).mean()),
        "power_wilcoxon": float((rank < ALPHA).mean()),
        "mean_discordant_pairs": float((samples != 0).sum(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=200, help="Pilot pool size.")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.uses_llm():
        parser.error(
            "the pilot runs every learner in both arms; use a model-free config "
            "or size the cohort from a run that costs hours"
        )

    pilot = config.model_copy(
        update={
            "cohort": config.cohort.model_copy(update={"n_learners": args.pilot}),
            "run_name": f"{config.run_name}_pilot",
        }
    )
    arms = {
        arm: {
            row["learner_id"]: row
            for row in run(pilot.model_copy(update={"arm": arm}))["per_learner"]
        }
        for arm in ("coupled", "decoupled")
    }

    rng = np.random.default_rng(config.seed)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    report = {
        "config": str(args.config),
        "pilot_learners": args.pilot,
        "replicates": args.replicates,
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "primary_outcome": OUTCOMES[0],
        "outcomes": {},
    }

    for outcome in OUTCOMES:
        differences = paired_differences(arms["coupled"], arms["decoupled"], outcome)
        curve = [power_at(differences, n, args.replicates, rng) for n in sizes]
        sufficient = [row for row in curve if row["power_sign_test"] >= TARGET_POWER]
        report["outcomes"][outcome] = {
            "pilot_mean_difference": float(differences.mean()),
            "pilot_ci95": bootstrap_ci(differences, rng, args.replicates),
            "pilot_discordant": int((differences != 0).sum()),
            "pilot_favouring_coupled": int((differences > 0).sum()),
            "pilot_favouring_decoupled": int((differences < 0).sum()),
            "rank_biserial": rank_biserial(differences),
            "sign_test_p": sign_test(differences),
            "wilcoxon_p": wilcoxon(differences),
            "required_n": sufficient[0]["n_learners"] if sufficient else None,
            "curve": curve,
        }

    directory = args.out or config.paths.results_dir / f"power_{config.domain}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "power.json").write_text(json.dumps(report, indent=2) + "\n")

    for outcome in OUTCOMES:
        entry = report["outcomes"][outcome]
        primary = " (primary)" if outcome == OUTCOMES[0] else ""
        print(f"\n{outcome}{primary}")
        print(
            f"  pilot n={args.pilot}: mean diff {entry['pilot_mean_difference']:+.4f} "
            f"95% CI [{entry['pilot_ci95'][0]:+.4f}, {entry['pilot_ci95'][1]:+.4f}]"
        )
        print(
            f"  discordant {entry['pilot_discordant']}/{args.pilot} "
            f"({entry['pilot_favouring_coupled']} coupled, "
            f"{entry['pilot_favouring_decoupled']} decoupled), "
            f"rank-biserial {entry['rank_biserial']:+.3f}"
        )
        required = entry["required_n"]
        print(
            f"  N for {TARGET_POWER:.0%} power (sign test): "
            + (f"{required}" if required else f"more than {max(sizes)}")
        )
        print(f"  {'N':>6} {'sign':>7} {'wilcoxon':>9} {'discordant':>11}")
        for row in entry["curve"]:
            print(
                f"  {row['n_learners']:>6} {row['power_sign_test']:>7.2f} "
                f"{row['power_wilcoxon']:>9.2f} {row['mean_discordant_pairs']:>11.1f}"
            )

    print(f"\nwritten to {directory / 'power.json'}")


if __name__ == "__main__":
    main()
