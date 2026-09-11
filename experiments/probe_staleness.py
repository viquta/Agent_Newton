"""Is the reversal under forgetting about routing, or about a stale model?

    uv run python experiments/probe_staleness.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260922

The paired grid found the declared primary reverses against a learner who
forgets, and not because of the dose. `probe_spacing.py` then ruled out the first
explanation offered: the coupled arm does not lose by dwelling — dwelling is
worth something.

**The reading this tests.** Belief ages only across a gap between sittings —
`apply_decay(elapsed_days)` runs once, at the top of `Session.run` — while a
forgetting learner now decays *within* one. So the coupled arm routes on an
estimate that is stale by construction, and the decoupled arm cannot be misled by
an estimate it never reads. If that is what the reversal is, then ageing the
belief on the same beat as the learner should recover some of it.

Three conditions, each a paired comparison of the arms:

* **forgetting only** — the grid's row, re-run here. It should reproduce, and if
  it does not, nothing below can be read.
* **forgetting + matched decay** — the belief ages half the distance to the prior
  every ten items, which is the shape and cadence the learner forgets at. The
  test.
* **decay only** — no forgetting, belief ages anyway. ⚠️ **The control that
  matters**, because decaying a belief that was *correct* should cost the coupled
  arm something on its own. Without it, an improvement in the second condition
  cannot be told from the dial simply being benign.

⚠️ Whatever comes out, it is a statement about *this* belief model and *this*
forgetting mechanism meeting each other. Neither is the artifact's claim.
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

#: Held fixed so the learner forgets at the same strength everywhere it forgets
#: at all — the grid's setting.
FORGETTING = {"forgetting_rate": 0.5, "forgetting_period": 10}

#: The belief ages by one day every ten items at a one-day half-life, so it loses
#: half the distance to the prior on the same beat the learner regains half of
#: what was taught. Matched on purpose: a belief staler than the learner would
#: answer a different question.
DECAY = {"half_life_days": 1.0, "within_session_period": 10}

CONDITIONS: dict[str, tuple[dict, dict]] = {
    "forgetting_only": (FORGETTING, {}),
    "forgetting_and_decay": (FORGETTING, DECAY),
    "decay_only": ({}, DECAY),
}


def tuned(config: Config, simulator: dict, decay: dict, label: str) -> Config:
    return config.model_copy(
        update={
            "simulator": config.simulator.model_copy(update=simulator),
            "decay": config.decay.model_copy(update=decay),
            "run_name": f"{config.run_name}_{label}",
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
        "forgetting": FORGETTING,
        "decay": DECAY,
        "standing": "exploratory",
        "note": "coupled minus decoupled, per condition",
        "conditions": {},
    }

    print("\ncoupled minus decoupled, on the declared primary")
    header = f"  {'condition':24}{'framing A':>12}{'holm':>10}{'c/d':>10}{'framing B':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, (simulator, decay) in CONDITIONS.items():
        settings = tuned(config, simulator, decay, label)
        metrics = {
            arm: cohort(settings, arm, args.n, args.seed, suffix=f"_{label}")
            for arm in ARMS
        }
        rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

        budget = max(1, math.floor(metrics["decoupled"]["mean_items"]))
        capped = settings.model_copy(
            update={"cohort": settings.cohort.model_copy(update={"max_items": budget})}
        )
        matched = cohort(
            capped, "coupled", args.n, args.seed, suffix=f"_{label}_dosematched"
        )
        matched_rows = analyse(matched, metrics["decoupled"], rng)

        report["conditions"][label] = {
            "simulator": simulator,
            "decay": decay,
            "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
            "dose_matched_run_id": matched["run_id"],
            "dose_matched_budget": budget,
            "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
            "framing_a": [
                {
                    "outcome": r.outcome,
                    "mean_difference": r.mean_difference,
                    "ties": r.ties,
                    "favouring_coupled": r.favouring_first,
                    "favouring_decoupled": r.favouring_second,
                    "sign_effect_size": r.sign_effect_size,
                    "sign_p": r.sign_p,
                    "holm_p": adjusted,
                    "significant": adjusted < ALPHA,
                }
                for r, adjusted in rows
            ],
            "framing_b": [
                {
                    "outcome": r.outcome,
                    "mean_difference": r.mean_difference,
                    "sign_p": r.sign_p,
                    "holm_p": adjusted,
                    "significant": adjusted < ALPHA,
                }
                for r, adjusted in matched_rows
            ],
        }

        a = report["conditions"][label]["framing_a"][0]
        b = report["conditions"][label]["framing_b"][0]
        split = f"{a['favouring_coupled']}/{a['favouring_decoupled']}"
        print(
            f"  {label:24}{a['mean_difference']:>+12.4f}{a['holm_p']:>10.2e}"
            f"{split:>10}{b['mean_difference']:>+12.4f}"
        )

    print("\n  positive = the coupled arm ahead.")
    print("  ⚠ `decay_only` is the control: ageing a belief that was correct")
    print("    should cost the coupled arm something by itself.")

    directory = args.out or config.paths.results_dir / "probe_staleness"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
