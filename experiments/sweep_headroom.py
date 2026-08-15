"""How much of the paired comparison is limited by there being nothing to move?

    uv run python experiments/sweep_headroom.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

``simulator.misconceptions_per_learner`` decides how much a learner has wrong at
the start. At the configured 2 against a 15-entry catalogue the mean pre-test
sits near 0.91, so the gain outcome has under a tenth of its range available and
some learners begin with none at all — they are in the mean and can only lose.
This sweeps that setting and reports what each population leaves for the
comparison to detect.

**This dial changes the population, not the pedagogy.** Both arms draw from the
same profiles at every point, the planners are untouched, and nothing in the
simulator's response to teaching moves. A point where the arms separate says the
effect was there and the measurement lacked room; a flat curve says the null is
not a ceiling artifact. Neither reading depends on the generator rewarding the
architecture, which is what separates this from the sweep in
``sweep_prerequisites.py`` and is why it needs no equivalent falsification
column.

Reported per point:

``headroom``
    ``1 - mean_pretest``, averaged over the arms. What the gain outcome could
    move through even in principle.

``ceiling``
    Learners whose pre-test was already perfect. ``normalised_gain`` is
    undefined for them and the raw mean still counts them.

``discordant``
    Pairs where the arms differed at all, split by which they favour. **The
    column to read.** The sign test's real sample size is this, not ``--n``, so
    it is what a null at this size actually rests on.

Two is run first and checked against the figures already recorded. A sweep whose
baseline no longer reproduces them is measuring a different system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402
from run_paired import ARMS, OUTCOMES, analyse  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

#: Today's configured value, and the point every other is read against.
BASELINE = 2

#: What §7b recorded under the natural framing, seed 20260811, **at this N**.
#: Shared with ``sweep_prerequisites.py``; kept as a literal in both so a sweep
#: cannot silently start agreeing with a moved baseline.
RECORDED_N = 160
RECORDED = {
    "remediation": +0.0136,
    "gain": +0.0017,
    "goals_mastered": +3.9062,
    "distance_to_goal": -5.8125,
}
TOLERANCE = 5e-4


def tuned(config: Config, held: int, arm: str, n: int, seed: int) -> Config:
    simulator = config.simulator.model_copy(
        update={"misconceptions_per_learner": held}
    )
    return config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "simulator": simulator,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_m{held}",
        }
    )


def paired_at(config: Config, held: int, n: int, seed: int, rng) -> dict:
    """Coupled against decoupled, at one population difficulty."""
    metrics = {arm: run(tuned(config, held, arm, n, seed)) for arm in ARMS}
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    return {
        "outcomes": {
            result.outcome: {
                "mean_difference": result.mean_difference,
                "sign_p": result.sign_p,
                "holm": adjusted,
                "discordant": [result.favouring_first, result.favouring_second],
                "ties": result.ties,
            }
            for result, adjusted in rows
        },
        "population": {
            arm: {
                "mean_pretest": metrics[arm]["mean_pretest"],
                "mean_gain": metrics[arm]["mean_gain"],
                "mean_normalised_gain": metrics[arm]["mean_normalised_gain"],
                "learners_at_ceiling": metrics[arm]["learners_at_ceiling"],
                "mean_remediation": metrics[arm]["mean_remediation"],
                "mean_items": metrics[arm]["mean_items"],
            }
            for arm in ARMS
        },
    }


def headroom(point: dict) -> float:
    """``1 - mean_pretest``, averaged over the arms."""
    pretests = [point["population"][arm]["mean_pretest"] for arm in ARMS]
    return 1.0 - sum(pretests) / len(pretests)


def check_baseline(point: dict, n: int) -> None:
    """Refuse to continue if the configured value no longer reproduces §7b."""
    if n != RECORDED_N:
        print(
            f"\nbaseline not checked against the recorded numbers: those are at "
            f"N = {RECORDED_N} and this run is at N = {n}.\n"
            f"    The curve is still internally comparable; it is the published "
            f"baseline that is not.\n"
        )
        return

    drift = {
        outcome: point["outcomes"][outcome]["mean_difference"] - recorded
        for outcome, recorded in RECORDED.items()
        if abs(point["outcomes"][outcome]["mean_difference"] - recorded) > TOLERANCE
    }
    if drift:
        print(f"\n⚠️  m = {BASELINE} does not reproduce the recorded numbers:")
        for outcome, delta in drift.items():
            print(f"      {outcome}: off by {delta:+.4f}")
        print(
            "\n    The sweep is read against those figures, so it cannot "
            "continue.\n    Either this dial is not inert at the configured "
            "value, or something\n    else moved and §7b needs re-recording "
            "first."
        )
        raise SystemExit(1)
    print(f"m = {BASELINE} reproduces the recorded numbers — continuing.\n")


def show(curve: list[dict]) -> None:
    header = (
        f"  {'held':>5}{'headroom':>10}{'ceiling':>9}{'remediation':>14}"
        f"{'holm':>10}{'c/d':>9}{'norm. gain c/d':>17}"
    )
    print("\nthe comparison against how much the population has wrong")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in curve:
        remediation = point["outcomes"]["remediation"]
        mark = " *" if remediation["holm"] < ALPHA else ""
        ceiling = sum(point["population"][arm]["learners_at_ceiling"] for arm in ARMS)
        normalised = "/".join(
            f"{point['population'][arm]['mean_normalised_gain']:.3f}" for arm in ARMS
        )
        print(
            f"  {point['held']:>5}{headroom(point):>10.3f}{ceiling:>9}"
            f"{remediation['mean_difference']:>+14.4f}{remediation['holm']:>10.2e}"
            f"{'{}/{}'.format(*remediation['discordant']):>9}{normalised:>17}{mark}"
        )
    print(
        "\n  held        = misconceptions_per_learner, against a 15-entry catalogue.\n"
        "  headroom    = 1 - mean pre-test, averaged over the arms.\n"
        "  ceiling     = learners starting at 100%, summed over both arms.\n"
        "  remediation = coupled minus decoupled, the declared primary.\n"
        "  c/d         = discordant pairs favouring coupled / decoupled. The\n"
        "                sign test's real sample size, and what a null rests on."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--held",
        default="2,4,6",
        help=f"Comma separated, and it must begin at {BASELINE}.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from; profiles come from (seed, learner_id)."
        )

    held = [int(m) for m in args.held.split(",")]
    if held[0] != BASELINE:
        parser.error(
            f"the sweep must start at {BASELINE}: every other point is read "
            f"against the numbers already recorded, and those were produced at "
            f"the configured value."
        )

    rng = np.random.default_rng(args.seed)
    curve: list[dict] = []
    for count in held:
        point = paired_at(config, count, args.n, args.seed, rng)
        point["held"] = count
        curve.append(point)
        if count == BASELINE:
            check_baseline(point, args.n)

    show(curve)

    out = args.out or config.paths.results_dir / "sweep_headroom"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "baseline_held": BASELINE,
                "recorded_baseline": RECORDED,
                "outcomes": list(OUTCOMES),
                "curve": curve,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
