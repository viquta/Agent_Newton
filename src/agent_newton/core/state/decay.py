"""Forgetting, as it applies to what the system *believes*.

Between sessions the learner model becomes less certain. A concept last
evidenced weeks ago is not known as well as the posterior says, and a concept
last failed weeks ago is not necessarily still failed. Both estimates relax
toward the BKT prior, which is what "we have no current evidence" means.

Two things this module deliberately is not:

* **It is not the learner forgetting.** That happens to the simulated learner's
  profile, is a separate mechanism with its own swept rate, and is ground truth
  rather than belief. Conflating them would let the system's estimate be right
  about forgetting by construction.
* **It is not a scheduler.** Nothing here decides to review anything. Decay
  lowers a posterior, the concept falls back below ``theta_upper``, it re-enters
  the goal's relevant set, and the route passes through it again. Spaced review
  is a consequence of the frontier and the route, not a rule added beside them —
  which is the whole reason it is worth showing that a planner without the
  posteriors cannot do it.
"""

from __future__ import annotations


def relax(estimate: float, prior: float, elapsed_days: float, half_life_days: float) -> float:
    """Move one estimate toward ``prior``, halving the gap every half-life.

    ``prior + (estimate - prior) * 0.5 ** (elapsed / half_life)``.

    Three properties, each tested:

    * **Zero elapsed time is the identity.** A sequence run with no gaps must
      reproduce the same sessions run back to back, or the mechanism is
      manufacturing an effect rather than modelling one.
    * **The prior is the limit, never crossed.** Belief decays to "no current
      evidence", not to zero. A concept the learner was measured *worse* than
      the prior at therefore relaxes *upward*: time erodes bad news exactly as
      it erodes good news, because both are evidence going stale.
    * **Monotone in elapsed time.** More time never moves an estimate further
      from the prior.
    """
    if half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    if elapsed_days <= 0.0:
        return estimate
    return prior + (estimate - prior) * 0.5 ** (elapsed_days / half_life_days)
