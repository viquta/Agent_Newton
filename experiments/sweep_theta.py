"""How readily should evidence be allowed to reopen the plan?

    uv run python experiments/sweep_theta.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

``arbitration.theta`` is the mastery movement that reopens a plan. Low, and the
planner reconsiders on every wobble; high, and long-horizon planning stops
hearing what the learner is doing. The threshold is a design parameter of the
architecture rather than a fact about it, so the honest form is a curve.

**⚠️ Read the trigger breakdown, never the replan total.** The triggers compete:
raising ``theta`` suppresses ``mastery_delta`` and lets ``frontier_crossed`` and
``misconception_repeat`` take up the slack, so the total can sit still while the
threshold is doing a great deal. A sweep reading totals would conclude the
parameter does nothing. Setting a goal is audited under ``plan`` rather than
``replan``, so it stays out of these counts by construction.

**And read it beside the prerequisite dependence.** §7l measured that
sequencing does not affect what a simulated learner learns at ``k = 0``, so a
parameter governing *when the sequence is reconsidered* has nothing to move
there either — a flat outcome curve at k = 0 is the expected reading rather than
a finding about the threshold. ``--dependence`` runs the same sweep in a regime
where order does matter, which is the only regime in which this parameter's
effect on outcomes is interpretable. Run both.

Paired against the default threshold, learner by learner: the same seed produces
the same learner at every point, so each is their own control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import run  # noqa: E402
from run_paired import analyse, by_learner  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402

#: The configured threshold every other point is compared against.
DEFAULT_THETA = 0.15


def tuned(config: Config, theta: float, n: int, seed: int, dependence: float) -> Config:
    return config.model_copy(
        update={
            "arm": "coupled",
            "seed": seed,
            "arbitration": config.arbitration.model_copy(update={"theta": theta}),
            "simulator": config.simulator.model_copy(
                update={"prerequisite_dependence": dependence}
            ),
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_theta{theta:g}",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--thetas",
        default="0.05,0.10,0.15,0.25,0.40",
        help="Comma separated. The configured default must be among them.",
    )
    parser.add_argument(
        "--dependence",
        type=float,
        default=0.0,
        help=(
            "simulator.prerequisite_dependence for the whole sweep. At 0 the "
            "learner is indifferent to sequencing, so the outcome columns have "
            "nothing to move — see §7l."
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from; profiles come from (seed, learner_id)."
        )

    thetas = [float(t) for t in args.thetas.split(",")]
    if DEFAULT_THETA not in thetas:
        parser.error(
            f"the sweep must include the configured threshold {DEFAULT_THETA}: "
            f"every point is read as a difference from it."
        )

    rng = np.random.default_rng(args.seed)
    metrics = {
        theta: run(tuned(config, theta, args.n, args.seed, args.dependence))
        for theta in thetas
    }
    baseline = metrics[DEFAULT_THETA]

    points: list[dict] = []
    for theta in thetas:
        rows = analyse(metrics[theta], baseline, rng)
        points.append(
            {
                "theta": theta,
                "replans_by_trigger": metrics[theta]["replans_by_trigger"],
                "suppressed": metrics[theta]["suppressed_triggers"],
                "mean_items": metrics[theta]["mean_items"],
                "against_default": {
                    result.outcome: {
                        "mean_difference": result.mean_difference,
                        "holm": adjusted,
                        "discordant": [result.favouring_first, result.favouring_second],
                    }
                    for result, adjusted in rows
                },
            }
        )

    triggers = sorted({t for p in points for t in p["replans_by_trigger"]})
    header = (
        f"  {'theta':>6}" + "".join(f"{t[:14]:>16}" for t in triggers)
        + f"{'suppressed':>12}{'items':>8}{'remediation':>14}{'holm':>10}"
    )
    print(f"\nreplanning against the threshold that governs it "
          f"(prerequisite_dependence = {args.dependence:g})")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in points:
        remediation = point["against_default"]["remediation"]
        mark = " *" if remediation["holm"] < ALPHA else ""
        counts = "".join(
            f"{point['replans_by_trigger'].get(t, 0):>16}" for t in triggers
        )
        print(
            f"  {point['theta']:>6g}{counts}{point['suppressed']:>12}"
            f"{point['mean_items']:>8.1f}"
            f"{remediation['mean_difference']:>+14.4f}{remediation['holm']:>10.2e}"
            f"{mark}"
        )
    print(
        "\n  Counts are replans by trigger over the whole cohort, never a total:\n"
        "  raising theta suppresses one pathway and lets another take up the\n"
        "  slack, so a total would sit still while the threshold did plenty.\n"
        "  remediation = difference from the configured threshold, same learners."
    )
    if args.dependence == 0.0:
        print(
            "\n  ⚠️ At dependence 0 the learner is indifferent to sequencing (§7l),\n"
            "     so a flat remediation column is the expected reading and not a\n"
            "     finding about theta. Re-run with --dependence 1 to interpret it."
        )

    out = args.out or config.paths.results_dir / "sweep_theta"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_k{args.dependence:g}.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "prerequisite_dependence": args.dependence,
                "default_theta": DEFAULT_THETA,
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out / f'summary_k{args.dependence:g}.json'}")


if __name__ == "__main__":
    main()
