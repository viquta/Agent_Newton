"""Diagnostic agents.

Three conditions, differing only in how the misconception label is arrived at.
Holding everything else fixed and varying this alone is what isolates the effect
of diagnostic error on learning outcomes:

* :class:`OracleDiagnostic` — always right. The upper bound.
* :class:`NoisedOracleDiagnostic` — right at a set rate, wrong at random
  otherwise. Isolates the error *rate* from the error *pattern*.
* the model-backed agent (later) — wrong in whatever way a model is wrong.

If the real agent matches the noised oracle at the same accuracy, only its rate
matters. If it does worse, its errors are systematic — it confuses particular
misconceptions — and the confusion matrix says which.
"""

from __future__ import annotations

import hashlib

from agent_newton.core.agents.base import Diagnosis
from agent_newton.domains.base import Domain, Item


class OracleDiagnostic:
    """Reads the injected label. Perfect by construction."""

    def __init__(self) -> None:
        self._label: str | None = None

    def observe_ground_truth(self, label: str | None) -> None:
        self._label = label

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis:  # noqa: ARG002
        return Diagnosis(self._label, confidence=1.0)


class NoisedOracleDiagnostic:
    """The injected label, corrupted at a fixed rate.

    Corruption is deterministic in ``(seed, item, response, label)`` rather than
    drawn from a running generator, so a learner meeting the same situation is
    misdiagnosed the same way in both architectures. Otherwise the arms would
    differ by their noise streams as well as by their tutoring.
    """

    def __init__(self, noise_rate: float, seed: int = 0) -> None:
        self._noise_rate = noise_rate
        self._seed = seed
        self._label: str | None = None

    def observe_ground_truth(self, label: str | None) -> None:
        self._label = label

    def _draw(self, item_id: str, response: str) -> float:
        key = f"{self._seed}|{item_id}|{response}|{self._label}"
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis:
        if self._label is None:
            return Diagnosis(None)

        if self._draw(item.id, response) >= self._noise_rate:
            return Diagnosis(self._label, confidence=1.0 - self._noise_rate)

        # Wrong, but plausibly wrong: a label from the same catalogue rather
        # than nonsense, since that is the shape of a real agent's mistakes.
        alternatives = [m for m in sorted(domain.misconceptions.ids()) if m != self._label]
        if not alternatives:
            return Diagnosis(None)
        index = int(self._draw(item.id, response + "!") * len(alternatives))
        return Diagnosis(alternatives[min(index, len(alternatives) - 1)],
                         confidence=1.0 - self._noise_rate)
