"""The rule engine: what a simulated learner actually does.

Deterministic given a seed. The engine decides whether a step is correct and,
if not, which misconception produced it; a surface renderer may later phrase
that decision in natural language, but it never changes it.

Randomness is drawn per decision from a hash of
``(seed, learner, item, repetition, attempt, misconception)`` rather than from a
running generator. Three consequences:

* A decision does not depend on how many rolls happened on *other* items, so
  reordering unrelated work cannot perturb it.
* **The same learner meeting the same item for the nth time draws the same
  number in both architectures.** Only the tutoring differs, so the comparison
  is not also absorbing the difference between two random streams. This is the
  common random numbers technique, and it lowers the variance of the paired
  comparison at no cost in bias.
* ``repetition`` is part of the key, so practising an item again is a fresh
  draw. Without it a repeated item reproduces its own past verbatim: a learner
  who once answered correctly would do so forever, mastering the concept
  without ever demonstrating anything, and one who erred could never stop.
  Common random numbers then survive only while the two arms have given the
  item the same number of times — which is exactly as long as their histories
  agree, and is the most the technique can offer once they genuinely diverge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent_newton.config import SimulatorConfig
from agent_newton.core.simulator.profile import MisconceptionProfile
from agent_newton.domains.base import Domain, Item


@dataclass(frozen=True, slots=True)
class SimulatedStep:
    """One step, with the ground truth that produced it.

    ``fired`` is the injected label the diagnostic agent is scored against. It
    is never exposed to any agent.
    """

    response: str
    fired: str | None
    correct: bool

    @property
    def label(self) -> str:
        return self.fired or ("correct" if self.correct else "unlabelled-error")


def _roll(
    seed: int,
    learner_id: str,
    item_id: str,
    repetition: int,
    attempt: int,
    misconception_id: str,
) -> float:
    """A stable number in [0, 1) for one specific decision."""
    key = f"{seed}|{learner_id}|{item_id}|{repetition}|{attempt}|{misconception_id}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class SimulatedLearner:
    """A learner defined by a misconception profile and a domain's buggy rules."""

    def __init__(
        self,
        profile: MisconceptionProfile,
        domain: Domain,
        config: SimulatorConfig,
    ) -> None:
        self._profile = profile
        self._domain = domain
        self._config = config

    @property
    def profile(self) -> MisconceptionProfile:
        """Ground truth. For the evaluation harness only — never for an agent."""
        return self._profile

    def answer(self, item: Item, attempt: int = 0, repetition: int = 0) -> SimulatedStep:
        """Produce a step for this item.

        Misconceptions this learner holds *and* which the item can elicit are
        considered in a fixed order; the first to fire produces the response. If
        none fires, the learner answers correctly.

        ``attempt`` counts steps within one visit; ``repetition`` counts how
        many times the item has been given before. Both enter the draw, so
        neither a retry nor a later revisit repeats an earlier outcome.
        """
        applicable = sorted(m for m in item.probes if self._profile.holds(m))

        for misconception_id in applicable:
            probability = self._profile.probability(misconception_id)
            if _roll(
                self._profile.seed,
                self._profile.learner_id,
                item.id,
                repetition,
                attempt,
                misconception_id,
            ) >= probability:
                continue

            rule = self._domain.buggy_rule(misconception_id)
            if rule is None:
                continue
            wrong = rule.apply(item)
            if wrong is None:
                # The item declares it probes this misconception but the rule
                # cannot act on it. `domain validate` rejects that combination,
                # so this is a guard rather than an expected path.
                continue

            return SimulatedStep(response=wrong, fired=misconception_id, correct=False)

        return SimulatedStep(response=item.answer, fired=None, correct=True)

    def receive_hint(self, targeted_misconception: str | None) -> bool:
        """Apply a hint. Returns whether it changed anything.

        A hint weakens a misconception only if it names one the learner actually
        holds. Aiming at the wrong one — or at nothing — leaves the learner
        exactly as they were, which is the mechanism that gives diagnostic
        accuracy its consequences for learning outcomes.
        """
        if targeted_misconception is None:
            return False
        return self._profile.remediate(
            targeted_misconception, self._config.remediation_factor
        )
