"""Does the architecture's advantage depend on prerequisites mattering?

    uv run python experiments/sweep_prerequisites.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

§7c measured something uncomfortable: prerequisite order makes no difference to
what a simulated learner learns. Matched on coverage, the arm that teaches every
dependant *before* its prerequisite ties the one that respects the order —
+0.0008, p = 1.0, one discordant pair in 137. The generator has no notion of
sequencing at all, so the outcome the architecture is judged on is one nothing
in it can move.

``simulator.prerequisite_dependence`` supplies the missing notion: a correct
hint takes less well when the learner's own foundations under that concept are
shaky. This sweeps its strength and reports the whole curve.

**Why a sweep and not a setting.** A strength chosen because the coupled arm
wins at it assumes the conclusion. Swept from zero it supports a claim worth
making — *the advantage appears above this much prerequisite dependence* — and
it states the assumption in the same breath as the result.

Two things are checked at every point, and the second is the one that can
falsify the whole exercise:

``paired``
    coupled against decoupled, the same learners in both, as in P11.

``ordering``
    ``goal_directed`` against ``shuffled`` — same material, arbitrary order.
    **If arbitrary order still ties at high strength, the dial is not making
    sequencing matter and whatever the paired comparison shows is something
    else.** It is reported beside every point rather than assumed from one.

Zero must reproduce the recorded numbers exactly. It is run first and compared,
and the sweep refuses to continue if it does not.
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

#: The point every other point is read against. Not a parameter: the claim is
#: that zero is today, and a sweep that could start elsewhere could not make it.
BASELINE = 0.0

#: What §7b recorded (re-recorded 2026-08-17 after the 16th catalogue entry), under the natural framing, seed 20260811 — **at this N**.
#: Checked rather than trusted: if the code no longer reproduces it, the sweep is
#: measuring a different system and its curve means nothing. A run at another N
#: says so and skips the check rather than failing it, since a mean over twenty
#: learners is not the recorded figure and never was.
RECORDED_N = 160
RECORDED = {
    "remediation": +0.0259,
    "gain": +0.0021,
    "goals_mastered": +3.7750,
    "distance_to_goal": -5.9625,
}
TOLERANCE = 5e-4


def tuned(config: Config, dependence: float, arm: str, n: int, seed: int, **over) -> Config:
    simulator = config.simulator.model_copy(
        update={"prerequisite_dependence": dependence}
    )
    return config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "simulator": simulator,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_k{dependence:g}",
            **over,
        }
    )


def paired_at(config: Config, dependence: float, n: int, seed: int, rng) -> dict:
    """Coupled against decoupled, at one strength."""
    metrics = {
        arm: run(tuned(config, dependence, arm, n, seed)) for arm in ARMS
    }
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)
    return {
        outcome.outcome: {
            "mean_difference": outcome.mean_difference,
            "sign_p": outcome.sign_p,
            "holm": adjusted,
            "discordant": [outcome.favouring_first, outcome.favouring_second],
        }
        for outcome, adjusted in rows
    }


def ordering_at(config: Config, dependence: float, n: int, seed: int, rng) -> dict:
    """Prerequisite order against arbitrary order, at one strength.

    The falsification check. Both cover the same material — the goal's
    unmastered closure — so a difference is order and nothing else. A dial that
    leaves this null has not made sequencing matter, whatever it did to the
    paired comparison.
    """
    ordered = run(
        tuned(
            config, dependence, "coupled", n, seed,
            agents=config.agents.model_copy(
                update={"planner": config.agents.planner.model_copy(
                    update={"impl": "goal_directed"}
                )}
            ),
        )
    )
    arbitrary = run(
        tuned(
            config, dependence, "coupled", n, seed,
            agents=config.agents.model_copy(
                update={"planner": config.agents.planner.model_copy(
                    update={"impl": "shuffled"}
                )}
            ),
        )
    )
    result = compare("remediation", by_learner(ordered), by_learner(arbitrary), rng)
    return {
        "mean_difference": result.mean_difference,
        "sign_p": result.sign_p,
        "discordant": [result.favouring_first, result.favouring_second],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--strengths",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma separated, and it must begin at 0.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from; profiles come from (seed, learner_id)."
        )

    strengths = [float(k) for k in args.strengths.split(",")]
    if strengths[0] != BASELINE:
        parser.error(
            "the sweep must start at 0: every other point is read against the "
            "numbers already recorded, and those were produced without the "
            "mechanism at all."
        )

    rng = np.random.default_rng(args.seed)
    curve: list[dict] = []
    for dependence in strengths:
        paired = paired_at(config, dependence, args.n, args.seed, rng)
        ordering = ordering_at(config, dependence, args.n, args.seed, rng)
        curve.append({"k": dependence, "paired": paired, "ordering": ordering})

        if dependence == BASELINE and args.n != RECORDED_N:
            print(
                f"\nzero not checked against the recorded numbers: those are at "
                f"N = {RECORDED_N} and this run is at N = {args.n}.\n"
                f"    The curve is still internally comparable; it is the "
                f"published baseline that is not.\n"
            )
        elif dependence == BASELINE:
            drift = {
                outcome: paired[outcome]["mean_difference"] - recorded
                for outcome, recorded in RECORDED.items()
                if abs(paired[outcome]["mean_difference"] - recorded) > TOLERANCE
            }
            if drift:
                print("\n⚠️  zero does not reproduce the recorded numbers:")
                for outcome, delta in drift.items():
                    print(f"      {outcome}: off by {delta:+.4f}")
                print(
                    "\n    The sweep is read against those figures, so it cannot "
                    "continue.\n    Either the mechanism is not inert at zero, or "
                    "something else moved\n    and §7b needs re-recording first."
                )
                raise SystemExit(1)
            print("zero reproduces the recorded numbers — continuing.\n")

    header = (
        f"  {'k':>5}{'remediation':>14}{'holm':>10}{'c/d':>9}"
        f"{'ordered-arbitrary':>20}{'p':>10}{'c/d':>9}"
    )
    print("\nthe advantage against the strength of prerequisite dependence")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in curve:
        remediation = point["paired"]["remediation"]
        ordering = point["ordering"]
        mark = " *" if remediation["holm"] < ALPHA else ""
        paired_cd = "{}/{}".format(*remediation["discordant"])
        ordering_cd = "{}/{}".format(*ordering["discordant"])
        print(
            f"  {point['k']:>5g}{remediation['mean_difference']:>+14.4f}"
            f"{remediation['holm']:>10.2e}{paired_cd:>9}"
            f"{ordering['mean_difference']:>+20.4f}{ordering['sign_p']:>10.2e}"
            f"{ordering_cd:>9}{mark}"
        )
    print(
        "\n  remediation      = coupled minus decoupled, the declared primary.\n"
        "  ordered-arbitrary = goal_directed minus shuffled, same material.\n"
        "  ⚠️ Read the second column first. Where it is null the dial has not\n"
        "     made sequencing matter, and the first column is measuring\n"
        "     something other than what this sweep is named after."
    )

    out = args.out or config.paths.results_dir / "sweep_prerequisites"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "recorded_baseline": RECORDED,
                "curve": curve,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
