"""How much of a learner's misconception profile a session actually reaches.

    uv run python experiments/measure_coverage.py --config experiments/configs/calculus.yaml

This is what sets ``cohort.max_items``. A learner whose misconceptions are never
exercised contributes a zero difference to both arms and dilutes any effect
toward null, so the budget has to be large enough that most profiles are
reached — and no larger, since every extra item costs time in every run.

Two measures, and the difference between them matters:

**Opportunity** — the learner was given at least one practice item that probes
the misconception. This is the one the budget controls: it is a question about
whether the session ever arrives at the material.

**Exhibition** — the misconception actually fired. Bounded above by opportunity
and additionally governed by the profile's firing probabilities, so it plateaus
lower and no budget can drive it to 1.0.

Reported per arm, because the two route differently and may not reach the same
material in the same time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cohort import learner_ids  # noqa: E402

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.orchestration.session import build_session  # noqa: E402
from agent_newton.domains import registry  # noqa: E402
from agent_newton.domains.base import Domain  # noqa: E402

BUDGETS = (12, 20, 30, 45, 60, 80)
ARMS = ("coupled", "decoupled")


def items_given(session) -> list[str]:
    """Practice items the learner was given, from the audit log.

    Read from the log rather than tracked separately so this measures what the
    session actually did, not what a parallel bookkeeping counter believed.
    """
    return [
        record.evidence["item_id"]
        for record in session.board.audit_log
        if record.cause == "observation" and "item_id" in record.evidence
    ]


def coverage(session, outcome, domain: Domain) -> tuple[float, float, int]:
    """(opportunity, exhibition, profile size) for one learner."""
    held = set(session.learner.profile.initial)
    if not held:
        return 1.0, 1.0, 0

    probed: set[str] = set()
    for item_id in items_given(session):
        probed |= set(domain.items.get(item_id).probes)

    exhibited = {injected for injected, _ in outcome.diagnoses if injected}
    return len(held & probed) / len(held), len(held & exhibited) / len(held), len(held)


def run(config: Config, budget: int, arm: str) -> dict:
    domain = registry.load_domain(config.domain)
    tuned = config.model_copy(
        update={
            "arm": arm,
            "cohort": config.cohort.model_copy(update={"max_items": budget}),
        }
    )

    opportunity, exhibition, items = [], [], []
    for learner_id in learner_ids(tuned):
        session = build_session(learner_id, tuned.seed, domain, tuned)
        outcome = session.run()
        met, fired, _ = coverage(session, outcome, domain)
        opportunity.append(met)
        exhibition.append(fired)
        items.append(outcome.items_attempted)

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {
        "max_items": budget,
        "arm": arm,
        "opportunity": mean(opportunity),
        "exhibition": mean(exhibition),
        "mean_items_used": mean(items),
        "learners_fully_covered": sum(1 for x in opportunity if x == 1.0),
        "n_learners": len(opportunity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        default=",".join(str(b) for b in BUDGETS),
        help="Comma-separated item budgets to sweep.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.uses_llm():
        parser.error(
            "this sweep runs every budget on every learner in both arms; use a "
            "model-free config or it will take hours"
        )

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    rows = [run(config, budget, arm) for budget in budgets for arm in ARMS]

    directory = args.out or config.paths.results_dir / f"coverage_{config.domain}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coverage.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "domain": config.domain,
                "n_learners": config.cohort.n_learners,
                "seed": config.seed,
                "emphasis": config.agents.planner.emphasis.value,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    header = f"{'budget':>7} {'arm':10} {'opportunity':>12} {'exhibition':>11} {'items used':>11} {'fully covered':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['max_items']:>7} {row['arm']:10} "
            f"{row['opportunity']:>11.1%} {row['exhibition']:>10.1%} "
            f"{row['mean_items_used']:>11.1f} "
            f"{row['learners_fully_covered']:>7}/{row['n_learners']}"
        )
    print(f"\nwritten to {directory / 'coverage.json'}")


if __name__ == "__main__":
    main()
