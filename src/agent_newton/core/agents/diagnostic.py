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
from collections import Counter

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

    Corruption is deterministic in ``(seed, item, response, label, occurrence)``
    rather than drawn from a running generator, so a learner meeting the same
    situation for the nth time is misdiagnosed the same way in both
    architectures. Otherwise the arms would differ by their noise streams as
    well as by their tutoring.

    ``occurrence`` is load-bearing, and for the same reason ``repetition`` is in
    the simulator's rule engine. Without it a repeated situation reproduces its
    own past verbatim, and the *realised* error rate stops resembling the
    nominal one: a session revisits a handful of situations many times over — on
    calculus, 331 diagnoses across 12 distinct labels, one of them 169 times —
    so whether that one situation's single draw fell below the threshold decides
    half the run. Measured before this was keyed on occurrence, a nominal rate
    of 0.10 realised as 0.69.

    Counting the occurrence restores the rate while keeping the arms aligned for
    as long as their histories agree, which is the most common random numbers
    can offer once two runs genuinely diverge.

    **The realised rate still runs below the nominal one, and that is not a
    defect.** A corrupted diagnosis misses the remediation, so the learner keeps
    making the error and meets the same situation again — where the next draw is
    usually correct. Corruption manufactures extra, mostly-correct encounters
    and dilutes itself. Measured over a 20-learner calculus cohort: nominal 0.10
    realises 0.906 accuracy against a target of 0.900, nominal 0.25 realises
    0.856 against 0.750, nominal 0.50 realises 0.600 against 0.500 — the gap
    widening with the rate, as the feedback predicts.

    So compare conditions on the **realised** accuracy the run reports, never on
    the nominal rate it was configured with.
    """

    def __init__(self, noise_rate: float, seed: int = 0) -> None:
        self._noise_rate = noise_rate
        self._seed = seed
        self._label: str | None = None
        self._seen: Counter[tuple[str, str, str | None]] = Counter()

    def observe_ground_truth(self, label: str | None) -> None:
        self._label = label

    def _draw(self, item_id: str, response: str, occurrence: int, salt: str = "") -> float:
        key = f"{self._seed}|{item_id}|{response}|{self._label}|{occurrence}|{salt}"
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis:
        if self._label is None:
            return Diagnosis(None)

        situation = (item.id, response, self._label)
        occurrence = self._seen[situation]
        self._seen[situation] += 1

        if self._draw(item.id, response, occurrence) >= self._noise_rate:
            return Diagnosis(self._label, confidence=1.0 - self._noise_rate)

        # Wrong, but plausibly wrong: a label from the same catalogue rather
        # than nonsense, since that is the shape of a real agent's mistakes.
        alternatives = [m for m in sorted(domain.misconceptions.ids()) if m != self._label]
        if not alternatives:
            return Diagnosis(None)
        index = int(self._draw(item.id, response, occurrence, salt="which") * len(alternatives))
        return Diagnosis(alternatives[min(index, len(alternatives) - 1)],
                         confidence=1.0 - self._noise_rate)
