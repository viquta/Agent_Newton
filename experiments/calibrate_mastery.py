"""Does a mastery estimate predict held-out performance?

    uv run python experiments/calibrate_mastery.py \
        --config experiments/configs/calculus.yaml --n 160 --seed 20260811

Open item 2. The estimate has been *shown* wrong once — a sitting ended with
three concepts above ``theta_upper`` that the held-out post-test said the learner
could not do — and never checked since. Two later findings point the same way:
the band closes off concepts that can still hold unremediated misconceptions
(§7u), and a repeated attempt at one item moves the posterior further than the
evidence warrants (`research_private/tools/bkt_attempts.py`).

It matters because of where it lands. ``goals_mastered`` and
``distance_to_goal`` — the result that separates the arms under every framing,
and the one the artifact was built to show — are **derived from mastery**. The
headline inherits whatever the estimate is worth.

The post-test is already held out and already administered, so the check is a
pairing rather than a new experiment: for every learner and concept, the
posterior at the end of training against the post-test item probing that concept.

⚠️ **The reference line is not the identity, and using it would be a mistake.**
BKT does not claim P(correct) = P(knows). It claims

    P(correct) = P * (1 - p_slip) + (1 - P) * p_guess

so a concept at 1.00 should still be answered correctly only ``1 - p_slip`` of
the time, and one at 0.00 should be right ``p_guess`` of the time by luck.
Scoring against ``p`` itself would report a well-behaved estimate as badly
calibrated at both ends. Both curves are printed.

⚠️ **What this can and cannot close.** It calibrates the estimate against the
*simulated* learner, which is the population every headline figure is computed
over — so it speaks directly to those figures. It says nothing about a person:
§7i's failure was human, and calibrating against a generator whose mechanism is
not BKT's cannot stand in for that. The human half of open item 2 stays open.

Unreadable answers are excluded throughout. The verifier failing to measure is
not evidence about the learner, and counting it as a wrong answer here would
charge our own failure to the estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_newton.config import Config  # noqa: E402
from agent_newton.core.orchestration.session import build_session  # noqa: E402
from agent_newton.domains import registry  # noqa: E402
from agent_newton.domains.base import Verdict  # noqa: E402

#: Posterior bands the pairs are reported in. The last two straddle
#: ``theta_upper``, because "the model called it mastered and it was not" is the
#: failure this exists to size.
EDGES = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)


def predicted(p: float, config: Config) -> float:
    """What BKT says the chance of a correct answer is at posterior ``p``."""
    return p * (1.0 - config.bkt.p_slip) + (1.0 - p) * config.bkt.p_guess


def pairs_for(config: Config, arm: str, n: int, seed: int) -> list[tuple[str, str, float, bool]]:
    """(learner, concept, posterior at the end of training, answered correctly)."""
    domain = registry.load_domain(config.domain)
    tuned = config.model_copy(
        update={
            "arm": arm,
            "seed": seed,
            "cohort": config.cohort.model_copy(update={"n_learners": n}),
        }
    )
    collected: list[tuple[str, str, float, bool]] = []
    for index in range(n):
        learner_id = f"L{index:04d}"
        session = build_session(learner_id, tuned.seed, domain, tuned)
        outcome = session.run()
        mastery = dict(session.board.state.mastery)
        prior = session.board.probability  # honours the configured prior
        for item in outcome.posttest.per_item:
            if item.verdict is Verdict.UNPARSEABLE:
                continue
            estimate = mastery.get(item.concept_id)
            if estimate is None:
                # Never observed during training. The prior is not a claim about
                # this learner, so it is not an estimate to hold to account.
                continue
            collected.append(
                (learner_id, item.concept_id, estimate, item.verdict is Verdict.CORRECT)
            )
        del prior
    return collected


def report(pairs: list[tuple[str, str, float, bool]], config: Config, arm: str) -> dict:
    print(f"\n{arm}: {len(pairs)} (learner, concept) pairs with a held-out answer")
    header = f"  {'posterior':>14}{'n':>7}{'mean P':>9}{'observed':>10}{'BKT says':>10}{'gap':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    bins = []
    for low, high in zip(EDGES, EDGES[1:]):
        inside = [p for p in pairs if low <= p[2] < high or (high == 1.0 and p[2] >= 1.0)]
        if not inside:
            continue
        mean_p = sum(p[2] for p in inside) / len(inside)
        observed = sum(1 for p in inside if p[3]) / len(inside)
        expected = predicted(mean_p, config)
        bins.append(
            {
                "low": low,
                "high": high,
                "n": len(inside),
                "mean_posterior": mean_p,
                "observed": observed,
                "predicted": expected,
            }
        )
        print(
            f"  {f'{low:.1f}–{high:.1f}':>14}{len(inside):>7}{mean_p:>9.3f}"
            f"{observed:>10.3f}{expected:>10.3f}{observed - expected:>+8.3f}"
        )

    upper = config.zpd.theta_upper
    mastered = [p for p in pairs if p[2] >= upper]
    failed = [p for p in mastered if not p[3]]
    share = len(failed) / len(mastered) if mastered else 0.0
    print(
        f"\n  above theta_upper ({upper}): {len(mastered)} pairs, "
        f"{len(failed)} answered wrongly ({share:.1%})"
    )
    print(
        f"  BKT expects {1 - predicted(1.0, config):.1%} wrong there on slips alone."
    )
    concepts = sorted({p[1] for p in failed})
    if concepts:
        print(f"  concepts involved: {', '.join(concepts)}")

    return {
        "arm": arm,
        "pairs": len(pairs),
        "bins": bins,
        "above_theta_upper": {
            "pairs": len(mastered),
            "wrong": len(failed),
            "share_wrong": share,
            "expected_wrong": 1 - predicted(1.0, config),
            "concepts": concepts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arms", default="coupled,decoupled")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    findings = [
        report(pairs_for(config, arm, args.n, args.seed), config, arm)
        for arm in args.arms.split(",")
    ]

    print(
        "\n  observed = share of held-out items answered correctly.\n"
        "  BKT says = P*(1-p_slip) + (1-P)*p_guess, which is what the model\n"
        "             actually claims. ⚠️ Not the identity line.\n"
        "  A positive gap means the estimate is pessimistic in that band, a\n"
        "  negative one that it is optimistic — the direction §7i found."
    )

    out = args.out or config.paths.results_dir / "calibration_mastery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "n_learners": args.n,
                "seed": args.seed,
                "bkt": config.bkt.model_dump(),
                "theta_upper": config.zpd.theta_upper,
                "arms": findings,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
