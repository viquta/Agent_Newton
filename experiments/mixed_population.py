"""One cohort whose learners are not all the same kind.

    uv run python experiments/mixed_population.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260925

Every study so far gives all 160 learners the same dials — including the grid,
which runs four populations one after another rather than one population of four
kinds. This runs the mixture: each learner is assigned a kind, and the cohort
holds all of them at once.

⚠️ **Face validity, not a result.** It answers "what if learners differ from each
other?", which is a question about whether the design behaves sensibly, and it
answers it about *one* mixture chosen for being even rather than for resembling
anybody. It is reported beside the sweeps and revises nothing.

**The assignment is a pure function of the learner's id**, so the same person is
the same kind in both arms and the pairing survives. It cannot be otherwise: an
assignment that differed between arms would compare two different populations and
call the difference an architecture.

⚠️ **And it does not change who the learners are.** `sample_profile` draws from
`misconceptions_per_learner` and `p_fire_range`, neither of which any kind
touches, so a mixed cohort is the *same people* as a uniform one, behaving
differently. That is what makes the per-kind breakdown below comparable to the
grid's separate runs.

Two things are reported. The **cohort** figure is what the paired comparison says
about a mixed population as a whole. The **per-kind** figures are the same
learners split by what they were assigned, which is where a mixture can be told
apart from an average — ⚠️ each on a quarter of the cohort, so they are read for
direction and not for significance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_learner_types import CATEGORIES  # noqa: E402
from run_cohort import run  # noqa: E402
from run_paired import ARMS, OUTCOMES, by_learner, spent_seeds  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import (  # noqa: E402
    ALPHA,
    compare,
    holm_bonferroni,
)

#: The kinds, in a fixed order so the assignment does not depend on dict
#: ordering. Taken from the grid so there is one declaration of what a kind is.
KINDS = tuple(CATEGORIES)


def kind_of(learner_id: str, seed: int) -> str:
    """Which kind this learner is. Pure in ``(seed, learner_id)``.

    Hashed rather than taken from the id's number so the mixture does not line up
    with anything else that is ordered by learner index — the ids are handed out
    in sequence, and a kind that tracked that would be a kind that tracked
    whatever else does.
    """
    digest = hashlib.sha256(f"{seed}|{learner_id}|kind".encode()).digest()
    return KINDS[int.from_bytes(digest[:8], "big") % len(KINDS)]


def mixed(seed: int):
    """A per-learner hook that gives each learner their kind's dials."""

    def assign(learner_id: str, config: Config) -> Config:
        return config.model_copy(
            update={
                "simulator": config.simulator.model_copy(
                    update=CATEGORIES[kind_of(learner_id, seed)]
                )
            }
        )

    return assign


def cohort(config: Config, arm: str, n: int, seed: int, suffix: str = "") -> dict:
    """As ``run_paired.cohort``, but with the mixture applied per learner."""
    tuned = config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "run_name": f"{config.run_name}_mixed{suffix}",
        }
    )
    return run(tuned, per_learner=mixed(seed))


def analyse(coupled: dict, decoupled: dict, rng, keep=None) -> list:
    """Paired comparison over the outcome family, optionally on a subset."""
    first, second = by_learner(coupled), by_learner(decoupled)
    if keep is not None:
        first = {k: v for k, v in first.items() if k in keep}
        second = {k: v for k, v in second.items() if k in keep}
    results = [compare(o, first, second, rng) for o in OUTCOMES]
    return list(zip(results, holm_bonferroni([r.sign_p for r in results])))


def row(result, adjusted) -> dict:
    return {
        "outcome": result.outcome,
        "mean_difference": result.mean_difference,
        "n_pairs": result.n_pairs,
        "ties": result.ties,
        "favouring_coupled": result.favouring_first,
        "favouring_decoupled": result.favouring_second,
        "sign_effect_size": result.sign_effect_size,
        "sign_p": result.sign_p,
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
    metrics = {arm: cohort(config, arm, args.n, args.seed) for arm in ARMS}
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    budget = max(1, math.floor(metrics["decoupled"]["mean_items"]))
    capped = config.model_copy(
        update={"cohort": config.cohort.model_copy(update={"max_items": budget})}
    )
    matched = cohort(capped, "coupled", args.n, args.seed, suffix="_dose")
    matched_rows = analyse(matched, metrics["decoupled"], rng)

    assigned: dict[str, list[str]] = {kind: [] for kind in KINDS}
    for learner in by_learner(metrics["coupled"]):
        assigned[kind_of(learner, args.seed)].append(learner)

    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "kinds": {kind: CATEGORIES[kind] for kind in KINDS},
        "standing": "exploratory — face validity, not a result",
        "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
        "dose_matched_run_id": matched["run_id"],
        "dose_matched_budget": budget,
        "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
        "assignment": {kind: len(ids) for kind, ids in assigned.items()},
        "cohort_framing_a": [row(r, a) for r, a in rows],
        "cohort_framing_b": [row(r, a) for r, a in matched_rows],
        "by_kind": {},
    }

    print(f"\nmixed cohort, N = {args.n}, seed {args.seed}")
    print("  assignment: " + ", ".join(f"{k} {len(v)}" for k, v in assigned.items()))
    primary = report["cohort_framing_a"][0]
    matched_primary = report["cohort_framing_b"][0]
    print(
        f"\n  whole cohort   {primary['mean_difference']:+.4f}  "
        f"holm {primary['holm_p']:.2e}  "
        f"c/d {primary['favouring_coupled']}/{primary['favouring_decoupled']}  "
        f"ties {primary['ties']}  |  framing B "
        f"{matched_primary['mean_difference']:+.4f}"
    )

    print(f"\n  split by kind — ⚠ about {args.n // len(KINDS)} learners each")
    header = f"  {'kind':12}{'difference':>12}{'sign p':>10}{'c/d':>9}{'ties':>7}{'n':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kind, ids in assigned.items():
        subset = analyse(metrics["coupled"], metrics["decoupled"], rng, keep=set(ids))
        report["by_kind"][kind] = [row(r, a) for r, a in subset]
        one = report["by_kind"][kind][0]
        split = f"{one['favouring_coupled']}/{one['favouring_decoupled']}"
        print(
            f"  {kind:12}{one['mean_difference']:>+12.4f}{one['sign_p']:>10.2e}"
            f"{split:>9}{one['ties']:>7}{one['n_pairs']:>5}"
        )

    print("\n  ⚠ The per-kind rows carry no correction and a quarter of the")
    print("    cohort each. They are the direction, not a set of findings.")

    directory = args.out or config.paths.results_dir / "mixed_population"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
