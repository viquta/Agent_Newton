"""Is there a strength at which ageing the belief helps rather than swamps?

    uv run python experiments/sweep_belief_decay.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260923

`probe_staleness.py` matched the belief's decay to the learner's forgetting and
found the coupled arm did *worse*, not better. But its control also found that
ageing a belief that was correct costs the coupled arm −0.30 on its own — ten
times the effect being explained — so the instrument was too blunt to settle
anything. This sweeps the strength.

Two rows at every point, and the pair is the test:

* **forgetting on** — the learner regains half of what was taught every ten
  items, the grid's setting. The belief ages on the same beat.
* **forgetting off** — the learner keeps what was taught. The belief ages anyway,
  so this is what the dial costs when there is nothing for it to track.

If the belief needs to age *because the learner forgets*, the two rows must part:
ageing should cost less where there is something to track than where there is
not. If they move together, the dial is simply damaging and the staleness reading
gets no support at any strength.

`half_life_days` is **inverse strength** — a posterior closes half the distance
to the prior in that many days, and one application is one day. So 1.0 halves the
gap each time and 50.0 barely moves it. `None` is off, and off is today.

⚠️ **Framing A only.** The grid and the staleness probe both carry framing B, and
both found it does not change the direction under forgetting. This is a mechanism
sweep rather than a headline, and running three cohorts per point to re-establish
that would triple it for no answer.
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

#: Inverse strength. None is off; 1.0 is what the staleness probe used and found
#: overwhelming; the rest fill in between.
HALF_LIVES: tuple[float | None, ...] = (None, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0)

#: The beat, shared by the learner and the belief so the two are on the same
#: clock and only the strength is swept.
PERIOD = 10

FORGETTING = {"forgetting_rate": 0.5, "forgetting_period": PERIOD}


def tuned(config: Config, half_life: float | None, forgets: bool, label: str) -> Config:
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update=FORGETTING if forgets else {}
            ),
            "decay": config.decay.model_copy(
                update={
                    "half_life_days": half_life,
                    "within_session_period": None if half_life is None else PERIOD,
                }
            ),
            "run_name": f"{config.run_name}_{label}",
        }
    )


def point(config: Config, half_life: float | None, forgets: bool, args, rng) -> dict:
    label = f"hl{half_life or 'off'}_{'forgets' if forgets else 'steady'}"
    settings = tuned(config, half_life, forgets, label)
    metrics = {
        arm: cohort(settings, arm, args.n, args.seed, suffix=f"_{label}")
        for arm in ARMS
    }
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)
    primary, adjusted = rows[0]
    return {
        "half_life_days": half_life,
        "forgetting": forgets,
        "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
        "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
        "mean_difference": primary.mean_difference,
        "ties": primary.ties,
        "favouring_coupled": primary.favouring_first,
        "favouring_decoupled": primary.favouring_second,
        "rank_biserial": primary.rank_biserial,
        "sign_p": primary.sign_p,
        "holm_p": adjusted,
        "significant": adjusted < ALPHA,
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
    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "period": PERIOD,
        "forgetting": FORGETTING,
        "outcome": OUTCOMES[0],
        "standing": "exploratory",
        "framing": "A only — see the module docstring",
        "note": "coupled minus decoupled, per point",
        "points": [],
    }

    print("\ncoupled minus decoupled on the declared primary, framing A")
    header = (
        f"  {'half-life':>10}{'forgets':>12}{'steady':>12}{'difference':>13}"
        f"{'c/d (forgets)':>16}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for half_life in HALF_LIVES:
        forgets = point(config, half_life, True, args, rng)
        steady = point(config, half_life, False, args, rng)
        report["points"].extend([forgets, steady])
        split = f"{forgets['favouring_coupled']}/{forgets['favouring_decoupled']}"
        name = "off" if half_life is None else f"{half_life:g}"
        print(
            f"  {name:>10}{forgets['mean_difference']:>+12.4f}"
            f"{steady['mean_difference']:>+12.4f}"
            f"{forgets['mean_difference'] - steady['mean_difference']:>+13.4f}"
            f"{split:>16}"
        )

    print("\n  positive = the coupled arm ahead. `difference` is forgets − steady:")
    print("  above zero means ageing the belief costs less where there is")
    print("  something for it to track, which is what the staleness reading needs.")

    directory = args.out or config.paths.results_dir / "sweep_belief_decay"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
