"""Massed against spaced practice, as forgetting is turned up.

    uv run python experiments/probe_spacing.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260921

**A mechanism probe, not a comparison of the arms.** Both conditions run in the
*coupled* arm with the same planner, the same seed and the same item budget, so
the two differ only in how long the planner is allowed to stay on a concept:

* **massed** — `max_visits_per_concept` unset, which is every configuration
  measured so far. The planner works a concept until its posterior clears the
  band and only then moves on.
* **spaced** — the same planner capped at one visit, so a concept is set aside
  and ranked last after each visit and comes back only when nothing else on the
  way to the goal is available. The same material, spread out.

⚠️ **What this is for.** The paired grid found the declared primary *reverses*
against a learner who forgets, and not because of the dose — capping the coupled
arm's budget left the reversal where it was. One explanation on offer is that the
coupled arm dwells, so what it taught early decays while it is elsewhere, while
the decoupled walk spreads the same budget thinner and loses less. That is a
claim about massing, and it is testable without either arm: if it is right,
spacing should gain on massing as forgetting rises, and the two should be level
when forgetting is off.

⚠️ **Zero is the control and it has to come out null**, or the probe is measuring
the cap rather than the forgetting. `max_visits_per_concept` has never been swept
before, so nothing says in advance what it costs on its own.

⚠️ **Coverage is not matched by construction**, which is the trap the ordering
probe fell into: an arm that reaches less material looks worse for a reason that
is not the mechanism. The goal measures are reported at every point precisely
because they are the ones that move when coverage does — a spacing condition that
had simply seen less would show it there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import OUTCOMES, by_learner, cohort, spent_seeds  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import compare  # noqa: E402

#: Forgetting strengths. Zero is today and is the control.
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Steps between forgetting events, held fixed so the sweep moves one thing.
PERIOD = 10

#: name → `max_visits_per_concept`. None is today.
CONDITIONS: dict[str, int | None] = {"massed": None, "spaced": 1}


def tuned(config: Config, rate: float, visits: int | None) -> Config:
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={"forgetting_rate": rate, "forgetting_period": PERIOD}
            ),
            "cohort": config.cohort.model_copy(
                update={"max_visits_per_concept": visits}
            ),
            "run_name": f"{config.run_name}_spacing",
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
        parser.error(
            f"--seed {args.seed} is already spent by "
            f"{', '.join(sorted(spent[args.seed]))}."
        )

    rng = np.random.default_rng(args.seed)
    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "forgetting_period": PERIOD,
        "conditions": {name: visits for name, visits in CONDITIONS.items()},
        "standing": "exploratory",
        "note": "massed minus spaced, both in the coupled arm",
        "points": [],
    }

    header = (
        f"  {'forgetting':>11}{'massed − spaced':>18}{'sign p':>10}{'c/d':>9}"
        f"{'items m/s':>14}{'goals m/s':>14}"
    )
    print("\nmassed minus spaced, coupled arm, same seed and budget")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rate in RATES:
        runs = {
            name: cohort(
                tuned(config, rate, visits),
                "coupled",
                args.n,
                args.seed,
                suffix=f"_{name}_{rate}",
            )
            for name, visits in CONDITIONS.items()
        }
        rows = {
            outcome: compare(
                outcome, by_learner(runs["massed"]), by_learner(runs["spaced"]), rng
            )
            for outcome in OUTCOMES
        }
        primary = rows[OUTCOMES[0]]
        point = {
            "forgetting_rate": rate,
            "run_ids": {name: runs[name]["run_id"] for name in CONDITIONS},
            "mean_items": {n: runs[n]["mean_items"] for n in CONDITIONS},
            "mean_goals_mastered": {
                n: runs[n]["mean_goals_mastered"] for n in CONDITIONS
            },
            "outcomes": {
                outcome: {
                    "mean_difference": result.mean_difference,
                    "sign_p": result.sign_p,
                    "ties": result.ties,
                    "favouring_massed": result.favouring_first,
                    "favouring_spaced": result.favouring_second,
                    "sign_effect_size": result.sign_effect_size,
                }
                for outcome, result in rows.items()
            },
        }
        report["points"].append(point)
        items = point["mean_items"]
        goals = point["mean_goals_mastered"]
        split = f"{primary.favouring_first}/{primary.favouring_second}"
        item_pair = f"{items['massed']:.1f}/{items['spaced']:.1f}"
        goal_pair = f"{goals['massed']:.2f}/{goals['spaced']:.2f}"
        print(
            f"  {rate:>11.2f}{primary.mean_difference:>+18.4f}"
            f"{primary.sign_p:>10.2e}{split:>9}{item_pair:>14}{goal_pair:>14}"
        )

    print("\n  positive = massed ahead; negative = spacing ahead.")
    print("  ⚠ The row at 0.00 is the control: a difference there is the visit cap")
    print("    costing something on its own, not forgetting doing anything.")

    directory = args.out or config.paths.results_dir / "probe_spacing"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
