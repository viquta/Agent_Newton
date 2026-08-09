"""Run a cohort and write its artifacts.

    uv run python experiments/run_cohort.py --config experiments/configs/smoke.yaml

Learner ids are derived from the run seed, so the same config produces the same
cohort — and the *same learners* appear in both arms, which is what makes the
comparison paired rather than between groups.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent_newton.config import Config
from agent_newton.core.orchestration.session import build_session
from agent_newton.domains import registry
from agent_newton.logging_setup import log_event, setup_logging
from agent_newton.manifest import RunManifest
from agent_newton.runs import new_run_dir

log = logging.getLogger(__name__)


def learner_ids(config: Config) -> list[str]:
    """Stable ids, so both arms draw the same learners."""
    return [f"L{n:04d}" for n in range(config.cohort.n_learners)]


def run(config: Config) -> dict:
    domain = registry.load_domain(config.domain)
    run_id, run_dir = new_run_dir(config)
    setup_logging(run_dir)

    manifest = RunManifest.create(config, run_id)
    for field, value in domain.content_hashes().items():
        setattr(manifest, field, value)
    manifest.write(run_dir)

    log_event(
        log,
        "cohort.start",
        f"{config.cohort.n_learners} learners on {config.domain} ({config.arm})",
        run_id=run_id,
        uses_llm=config.uses_llm(),
    )

    outcomes = []
    for learner_id in learner_ids(config):
        session = build_session(learner_id, config.seed, domain, config)
        outcome = session.run()
        outcomes.append(outcome)
        log_event(
            log,
            "learner.done",
            f"{learner_id}: {outcome.pretest.score:.2f} -> {outcome.posttest.score:.2f}",
            learner_id=learner_id,
            gain=outcome.gain,
            items=outcome.items_attempted,
            remediation=outcome.remediation_ratio,
        )

    metrics = {
        "run_id": run_id,
        "arm": config.arm,
        "domain": config.domain,
        "n_learners": len(outcomes),
        "mean_pretest": _mean(o.pretest.score for o in outcomes),
        "mean_posttest": _mean(o.posttest.score for o in outcomes),
        "mean_gain": _mean(o.gain for o in outcomes),
        "mean_items": _mean(o.items_attempted for o in outcomes),
        "mean_remediation": _mean(o.remediation_ratio for o in outcomes),
        "unmeasurable_steps": sum(o.unmeasurable_steps for o in outcomes),
        "diagnostic_accuracy": _diagnostic_accuracy(outcomes),
        "replans_by_trigger": _triggers(outcomes),
        "suppressed_triggers": sum(o.suppressed for o in outcomes),
        "per_learner": [
            {
                "learner_id": o.learner_id,
                "pretest": o.pretest.score,
                "posttest": o.posttest.score,
                "gain": o.gain,
                "items": o.items_attempted,
                "items_to_exhaustion": o.items_to_exhaustion,
                "remediation": o.remediation_ratio,
            }
            for o in outcomes
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _triggers(outcomes) -> dict[str, int]:
    """Replans by trigger, summed over the cohort."""
    totals: dict[str, int] = {}
    for outcome in outcomes:
        for trigger, count in outcome.triggers.items():
            totals[trigger] = totals.get(trigger, 0) + count
    return dict(sorted(totals.items()))


def _diagnostic_accuracy(outcomes) -> float | None:
    """Agreement between the injected label and the inferred one.

    None when nothing was diagnosed. Meaningless for an oracle by construction,
    which is the point of reporting it: it should read 1.0 there, and anything
    else means the ground-truth channel is wired wrongly.
    """
    pairs = [pair for o in outcomes for pair in o.diagnoses]
    if not pairs:
        return None
    return sum(1 for injected, inferred in pairs if injected == inferred) / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("coupled", "decoupled"))
    parser.add_argument("--domain")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.arm:
        config = config.model_copy(update={"arm": args.arm})
    if args.domain:
        config = config.model_copy(update={"domain": args.domain})

    metrics = run(config)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_learner"}, indent=2))


if __name__ == "__main__":
    main()
