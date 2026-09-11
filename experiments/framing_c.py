"""Framing C — the decoupled arm revises instead of stopping.

    uv run python experiments/framing_c.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

§8.5 declares three framings and this is the third. `FixedOrderPlanner` walks a
topological ordering and, on reaching the end, stops — so the decoupled arm
attempts about 32 items against the coupled arm's 45, and every outcome that
grows with practice is confounded by the difference. `on_exhaustion: cycle`
starts the walk again, and the session's own item budget is what ends the run.

**The charity runs the other way from framing B, which is the point of having
both.** B removes the dose difference by taking work away from the coupled arm;
C removes it by giving the decoupled arm more. A reader who suspects the effect
is just more practice is answered by B; a reader who objects that B handicaps
the treatment is answered by C. Neither alone settles it.

⚠️ **Stopping was never forced by the missing learner model.** A planner seeing
only item correctness has no reason to give up rather than revise, so `stop` is
a choice about the baseline and not a consequence of the ablation. That is why
this framing is owed rather than optional.

⚠️ **This runs on the confirmatory seed on purpose, and the guard enforces it
rather than merely allowing it.** A framing is a re-analysis of one cohort under
a different accounting, so it is only meaningful on the cohort the other two
framings used — a fresh seed would produce a third population and answer nothing
about the reported result. The script therefore refuses any seed but the one
`results/paired_<domain>/summary.json` reports, which is the opposite of
`run_paired`'s spent-seed guard and for the same reason: the seed must match the
question. `sweep_arbitration.py` and `sweep_headroom.py` reuse this seed on the
same grounds.

Both arms are re-run rather than read from the stored summary, because the paired
statistics need per-learner rows and the summary keeps only aggregates. The
coupled arm's configuration is untouched, so its cohort reproduces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired import (  # noqa: E402
    ARMS,
    OUTCOMES,
    analyse,
    cohort,
    record,
    show,
)

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import ALPHA  # noqa: E402
from agent_newton.manifest import RunManifest, assert_poolable  # noqa: E402


def cycling(config: Config) -> Config:
    """The same configuration, with the decoupled planner revising at the end.

    Only `on_exhaustion` moves. `advance_after` decides *when* the walk steps
    forward and would change what the arm does on every item; leaving it alone is
    what keeps this a framing rather than a second manipulation.
    """
    planner = config.agents.planner.model_copy(update={"on_exhaustion": "cycle"})
    return config.model_copy(
        update={"agents": config.agents.model_copy(update={"planner": planner})}
    )


def confirmatory_seed(config: Config) -> tuple[int, Path]:
    """The seed the reported comparison used, read from its own summary."""
    summary = config.paths.results_dir / f"paired_{config.domain}" / "summary.json"
    if not summary.exists():
        raise SystemExit(
            f"framing C has nothing to re-frame — {summary} does not exist. "
            f"Run experiments/run_paired.py first."
        )
    stored = json.loads(summary.read_text())
    return int(stored["seed"]), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Learners per arm.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    declared, summary = confirmatory_seed(config)
    if args.seed != declared:
        parser.error(
            f"--seed {args.seed} is not the seed the reported comparison used. "
            f"{summary.name} reports {declared}, and a framing is an alternative "
            f"accounting of that cohort — on any other seed it would be a third "
            f"population and would say nothing about the result. Pass {declared}."
        )
    if args.seed == config.seed:
        parser.error(
            f"--seed {args.seed} matches the config's, which the power analysis "
            f"sized from. The stored comparison should not have used it either."
        )

    metrics = {
        "coupled": cohort(config, "coupled", args.n, args.seed, suffix="_framingc"),
        "decoupled": cohort(
            cycling(config), "decoupled", args.n, args.seed, suffix="_framingc"
        ),
    }
    manifests = [
        RunManifest.read(config.paths.results_dir / metrics[arm]["run_id"])
        for arm in ARMS
    ]
    assert_poolable(manifests)

    rng = np.random.default_rng(args.seed)
    rows = analyse(metrics["coupled"], metrics["decoupled"], rng)

    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "framing": "C — the decoupled arm cycles its walk instead of stopping",
        "on_exhaustion": "cycle",
        "re_frames": str(summary.relative_to(config.paths.results_dir.parent)),
        "arms": {arm: metrics[arm]["run_id"] for arm in ARMS},
        "mean_items": {arm: metrics[arm]["mean_items"] for arm in ARMS},
        "primary_outcome": OUTCOMES[0],
        "alpha": ALPHA,
        "results": [record(r, adjusted) for r, adjusted in rows],
    }

    show(rows, f"framing C — decoupled cycles, {args.n} learners, seed {args.seed}")
    print(
        f"\n  items attempted: coupled {metrics['coupled']['mean_items']:.1f}, "
        f"decoupled {metrics['decoupled']['mean_items']:.1f}"
    )
    # The framing only does its job if the counts actually converged; a decoupled
    # arm still well short of the coupled one has not removed the difference it
    # exists to remove, and the number says so rather than the intention.
    gap = abs(metrics["coupled"]["mean_items"] - metrics["decoupled"]["mean_items"])
    print(f"  gap closed to {gap:.1f} items (framing A leaves about 12)")

    directory = args.out or config.paths.results_dir / "framing_c"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
