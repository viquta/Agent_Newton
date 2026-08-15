"""How readily should evidence be allowed to reopen the plan?

    uv run python experiments/sweep_arbitration.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811 \\
        --parameter min_items_between_replans

P12. The arbitration policy has three knobs and the plan named ``theta`` — the
mastery movement that reopens a plan. Each is a design parameter of the
architecture rather than a fact about it, so the honest form is a curve.

⚠️ **``theta`` turned out to be the wrong one to single out, and the sweep is
what showed it.** Across an eightfold change, 0.05 to 0.40, **not one learner in
160 differed** — zero discordant pairs on every outcome, in both prerequisite
regimes. The reason is in the trigger counts: ``_find_trigger`` tests
``mastery_delta`` first and ``misconception_repeat`` second, so suppressing the
first at a high threshold simply hands the same replan to the second at the same
step. The two counts trade off *exactly* — 2276 between them at every point —
while ``frontier_crossed`` sits flat at 2221. The threshold decides which
trigger is credited, not whether the plan reopens.

What does gate replanning is the rate limit: 2489 suppressions against ~4657
replans. Hence ``--parameter``, and hence running it on
``min_items_between_replans`` as well. A sensitivity analysis is only worth
reporting on a parameter the system is actually sensitive to, and which one that
is was a measurement rather than a guess.

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

#: The arbitration knobs this can sweep, with the points to try when none are
#: named. Each list must contain the configured default: every column is read as
#: a difference from it.
SWEEPS: dict[str, str] = {
    "theta": "0.05,0.10,0.15,0.25,0.40",
    "min_items_between_replans": "0,1,2,4,8",
    "k_repeats": "1,2,3,5",
}


def tuned(
    config: Config, parameter: str, value: float, n: int, seed: int, dependence: float
) -> Config:
    setting = int(value) if parameter != "theta" else value
    return config.model_copy(
        update={
            "arm": "coupled",
            "seed": seed,
            "arbitration": config.arbitration.model_copy(update={parameter: setting}),
            "simulator": config.simulator.model_copy(
                update={"prerequisite_dependence": dependence}
            ),
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_{parameter}{value:g}",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--parameter",
        default="theta",
        choices=sorted(SWEEPS),
        help="Which arbitration knob to sweep.",
    )
    parser.add_argument(
        "--points",
        default=None,
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

    points_wanted = args.points or SWEEPS[args.parameter]
    values = [float(v) for v in points_wanted.split(",")]
    configured = float(getattr(config.arbitration, args.parameter))
    if configured not in values:
        parser.error(
            f"the sweep must include the configured {args.parameter} "
            f"({configured:g}): every point is read as a difference from it."
        )

    rng = np.random.default_rng(args.seed)
    metrics = {
        value: run(
            tuned(config, args.parameter, value, args.n, args.seed, args.dependence)
        )
        for value in values
    }
    baseline = metrics[configured]

    points: list[dict] = []
    for value in values:
        rows = analyse(metrics[value], baseline, rng)
        points.append(
            {
                args.parameter: value,
                "replans_by_trigger": metrics[value]["replans_by_trigger"],
                "suppressed": metrics[value]["suppressed_triggers"],
                "mean_items": metrics[value]["mean_items"],
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
        f"  {args.parameter[:8]:>8}" + "".join(f"{t[:14]:>16}" for t in triggers)
        + f"{'suppressed':>12}{'items':>8}{'remediation':>14}{'holm':>10}"
    )
    print(f"\nreplanning against {args.parameter} "
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
            f"  {point[args.parameter]:>8g}{counts}{point['suppressed']:>12}"
            f"{point['mean_items']:>8.1f}"
            f"{remediation['mean_difference']:>+14.4f}{remediation['holm']:>10.2e}"
            f"{mark}"
        )
    print(
        "\n  Counts are replans by trigger over the whole cohort, never a total:\n"
        "  suppressing one pathway lets another take up the slack at the same\n"
        "  step, so a total sits still while the parameter does plenty.\n"
        "  remediation = difference from the configured value, same learners."
    )
    if args.dependence == 0.0:
        print(
            f"\n  ⚠️ At dependence 0 the learner is indifferent to sequencing "
            f"(§7l),\n     so a flat remediation column is the expected reading "
            f"and not a\n     finding about {args.parameter}. Re-run with "
            f"--dependence 1 to interpret it."
        )

    out = args.out or config.paths.results_dir / "sweep_arbitration"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{args.parameter}_k{args.dependence:g}.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "prerequisite_dependence": args.dependence,
                "parameter": args.parameter,
                "configured": configured,
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"\nwritten to "
        f"{out / f'summary_{args.parameter}_k{args.dependence:g}.json'}"
    )


if __name__ == "__main__":
    main()
