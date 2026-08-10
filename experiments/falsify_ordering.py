"""Does prerequisite order affect what a simulated learner learns?

    uv run python experiments/falsify_ordering.py \\
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

The prerequisite graph constrains which item a planner may select. Whether it
does anything *else* — whether working a prerequisite before its dependant
changes what the learner ends up knowing — is a property of the simulated
learner, not of the planner, and it has never been measured.

The reason to doubt it: nothing in ``core/simulator/`` reads mastery, the
concept graph or the prerequisite relation. A step is a function of the
learner's profile, the misconceptions the item probes, and a seeded roll. If
that is the whole story then sequencing cannot matter, and an architecture
evaluated on its ability to sequence well is being credited for something the
generator cannot reward.

**The test holds material constant and varies only order.** Each condition
selects from the same set — the current goal's unmastered prerequisite closure —
so no condition can lose simply by never meeting the learner's misconceptions. A
null result is only informative if coverage is equal, which is why coverage is
reported beside the outcomes rather than assumed.

===============  =================================================
``goal_directed``  order from the learner model, prerequisites respected
``reverse``        reverse topological: every dependant before its prerequisite
``shuffled``       seeded arbitrary order
===============  =================================================

**Read it as a contrast.** If remediation is flat across conditions while
``goals_mastered`` and ``distance_to_goal`` move, then the manipulation is
moving the measures computed *from* the graph and not what the learner does.
That is the validity threat stated as a number.

Order is not entirely inert even so: remediation lowers a misconception's firing
probability, so meeting one earlier changes later rolls. Small differences are
expected. The claim under test is narrower — that they do not favour
prerequisite-respecting sequencing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_coverage import coverage  # noqa: E402
from run_cohort import learner_ids  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.evaluation.statistics import (  # noqa: E402
    ALPHA,
    compare,
    holm_bonferroni,
)
from agent_newton.core.orchestration.session import build_session  # noqa: E402
from agent_newton.domains import registry  # noqa: E402

BASELINE = "goal_directed"
PROBES = ("reverse", "shuffled")
OUTCOMES = ("remediation", "gain", "goals_mastered", "distance_to_goal")

#: Outcomes about what the learner ends up knowing, as against outcomes computed
#: from the graph. The whole point of the test is that these two groups should
#: behave differently if the graph is doing nothing.
LEARNING = ("remediation", "gain")


def run_condition(config: Config, impl: str, n: int, seed: int) -> dict[str, dict]:
    """One planner over the whole cohort, keyed by learner."""
    tuned = config.model_copy(
        update={
            "arm": "coupled",
            "seed": seed,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
            "agents": config.agents.model_copy(
                update={"planner": config.agents.planner.model_copy(update={"impl": impl})}
            ),
        }
    )
    domain = registry.load_domain(tuned.domain)
    rows: dict[str, dict] = {}
    for learner_id in learner_ids(tuned):
        session = build_session(learner_id, tuned.seed, domain, tuned)
        outcome = session.run()
        met, fired, _ = coverage(session, outcome, domain)
        rows[learner_id] = {
            "remediation": outcome.remediation_ratio,
            "gain": outcome.gain,
            "goals_mastered": outcome.goals_mastered,
            "distance_to_goal": outcome.distance_to_goal,
            "items": outcome.items_attempted,
            "opportunity": met,
            "exhibition": fired,
        }
    return rows


def mean(rows: dict[str, dict], key: str) -> float:
    values = [r[key] for r in rows.values() if r[key] is not None]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.uses_llm():
        parser.error("use a model-free config; this runs three cohorts")

    conditions = {
        impl: run_condition(config, impl, args.n, args.seed)
        for impl in (BASELINE, *PROBES)
    }

    rng = np.random.default_rng(args.seed)
    report = {
        "config": str(args.config),
        "n_learners": args.n,
        "seed": args.seed,
        "baseline": BASELINE,
        "alpha": ALPHA,
        "conditions": {
            impl: {
                "mean_items": mean(rows, "items"),
                "opportunity_coverage": mean(rows, "opportunity"),
                "exhibition_coverage": mean(rows, "exhibition"),
                **{f"mean_{o}": mean(rows, o) for o in OUTCOMES},
            }
            for impl, rows in conditions.items()
        },
        "comparisons": {},
    }

    print(f"\n{'condition':16}{'items':>8}{'coverage':>10}" + "".join(f"{o:>18}" for o in OUTCOMES))
    print("-" * (34 + 18 * len(OUTCOMES)))
    for impl, rows in conditions.items():
        print(
            f"{impl:16}{mean(rows, 'items'):>8.1f}{mean(rows, 'opportunity'):>9.1%}"
            + "".join(f"{mean(rows, o):>18.4f}" for o in OUTCOMES)
        )

    for probe in PROBES:
        results = [
            compare(o, conditions[BASELINE], conditions[probe], rng) for o in OUTCOMES
        ]
        adjusted = holm_bonferroni([r.sign_p for r in results])
        report["comparisons"][probe] = [
            {
                "outcome": r.outcome,
                "mean_difference": r.mean_difference,
                "ci95": list(r.ci95),
                "ties": r.ties,
                "favouring_baseline": r.favouring_first,
                "favouring_probe": r.favouring_second,
                "rank_biserial": r.rank_biserial,
                "sign_p": r.sign_p,
                "holm_p": p,
                "significant": p < ALPHA,
            }
            for r, p in zip(results, adjusted)
        ]

        print(f"\n{BASELINE} minus {probe}")
        print(f"  {'outcome':18}{'mean diff':>12}{'ties':>7}{'base/probe':>12}{'holm p':>11}  sig")
        for r, p in zip(results, adjusted):
            group = "learning" if r.outcome in LEARNING else "graph-derived"
            print(
                f"  {r.outcome:18}{r.mean_difference:>+12.4f}{r.ties:>7}"
                f"{f'{r.favouring_first}/{r.favouring_second}':>12}{p:>11.2e}"
                f"  {'yes' if p < ALPHA else 'no':4} ({group})"
            )

    directory = args.out or config.paths.results_dir / f"ordering_{config.domain}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
