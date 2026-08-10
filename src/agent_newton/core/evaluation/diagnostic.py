"""Diagnostic accuracy, measured offline against the item bank.

Every ``(item, misconception)`` pair the bank declares is turned into the wrong
answer that misconception produces, handed to the agent, and scored against the
label that generated it.

Offline rather than from session transcripts, for two reasons:

* **Balance.** A session over-samples whatever sits near the start of the
  syllabus, so a macro-F1 taken from transcripts is dominated by early concepts.
  The bank gives every entry its declared share.
* **Separation.** Diagnostic accuracy should not depend on how a planner
  happened to route a learner. Measuring it here keeps the component figure
  independent of the system it sits in — which is the point of measuring it at
  the component level at all.

The run is **resumable**: identical prompts hit the response cache, so
interrupting and restarting costs nothing and repeats nothing.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

from agent_newton.core.agents.base import Diagnostic
from agent_newton.domains.base import Domain


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored case."""

    item_id: str
    concept_id: str
    prompt: str
    wrong_answer: str
    injected: str
    inferred: str | None
    seconds: float
    #: Which bank the item came from. Carried because only ``practice`` items
    #: are ever diagnosed during a session — the pre- and post-tests are
    #: administered without hints, state updates or diagnosis — so the accuracy
    #: a running system is exposed to is the practice subset, not the whole set.
    bank: str = "practice"

    @property
    def correct(self) -> bool:
        return self.inferred == self.injected

    @property
    def abstained(self) -> bool:
        """The agent declined to name anything.

        Counted apart from a wrong answer: an admission and a confident error
        are different failures, and a tutor can act on the first honestly.
        """
        return self.inferred is None


@dataclass
class DiagnosticReport:
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.predictions)

    @property
    def correct(self) -> int:
        return sum(1 for p in self.predictions if p.correct)

    @property
    def abstentions(self) -> int:
        return sum(1 for p in self.predictions if p.abstained)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.predictions)

    def confusion(self) -> dict[tuple[str, str], int]:
        """(injected, inferred) counts. ``None`` becomes ``"<abstained>"``."""
        return dict(
            Counter((p.injected, p.inferred or "<abstained>") for p in self.predictions)
        )

    def per_label(self) -> dict[str, dict[str, float]]:
        """Precision, recall and F1 for each injected label."""
        labels = sorted({p.injected for p in self.predictions})
        scores: dict[str, dict[str, float]] = {}
        for label in labels:
            tp = sum(1 for p in self.predictions if p.injected == label and p.correct)
            fp = sum(
                1 for p in self.predictions if p.inferred == label and p.injected != label
            )
            fn = sum(1 for p in self.predictions if p.injected == label and not p.correct)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores[label] = {
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return scores

    def accuracy_by_bank(self) -> dict[str, dict[str, float]]:
        """Accuracy split by item bank.

        The whole-set figure is the right *component* measure: every declared
        pair gets its share, so it is not shaped by how a planner happened to
        route anyone. But only practice items are diagnosed during a session,
        so the practice row is the error rate a running system is actually
        exposed to. Reporting both keeps a reader from comparing the component
        figure against a system-level one and finding an unexplained gap.
        """
        banks = sorted({p.bank for p in self.predictions})
        rows: dict[str, dict[str, float]] = {}
        for bank in banks:
            scored = [p for p in self.predictions if p.bank == bank]
            correct = sum(1 for p in scored if p.correct)
            rows[bank] = {
                "cases": len(scored),
                "correct": correct,
                "accuracy": correct / len(scored) if scored else 0.0,
            }
        return rows

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1.

        Unweighted on purpose: a rare misconception that is never recognised
        should cost as much as a common one, because the tutor fails the learner
        holding it just as completely.
        """
        scores = self.per_label()
        return sum(s["f1"] for s in scores.values()) / len(scores) if scores else 0.0

    def worst_confusions(self, limit: int = 8) -> list[tuple[str, str, int]]:
        """The most frequent mistakes, commonest first."""
        wrong = [
            (injected, inferred, count)
            for (injected, inferred), count in self.confusion().items()
            if injected != inferred
        ]
        return sorted(wrong, key=lambda row: -row[2])[:limit]


def cases(domain: Domain) -> list[tuple[str, str, str]]:
    """Every ``(item_id, misconception_id, wrong_answer)`` the bank can produce.

    Skips pairs whose rule declines the item; ``domain validate`` already
    rejects those, so it should find none.
    """
    found = []
    for item in domain.items.all():
        for misconception_id in item.probes:
            rule = domain.buggy_rule(misconception_id)
            if rule is None:
                continue
            wrong = rule.apply(item)
            if wrong is not None:
                found.append((item.id, misconception_id, wrong))
    return sorted(found)


def evaluate(
    domain: Domain,
    agent: Diagnostic,
    *,
    limit: int | None = None,
    on_result: Callable[[Prediction], None] | None = None,
) -> Iterator[Prediction]:
    """Score the agent over the bank, yielding each case as it completes.

    A generator so a caller can print progress and write partial results — an
    hour-long run should not be all-or-nothing.
    """
    for item_id, injected, wrong in cases(domain)[:limit]:
        item = domain.items.get(item_id)
        started = time.time()
        diagnosis = agent.diagnose(item, wrong, domain)
        prediction = Prediction(
            item_id=item_id,
            concept_id=item.concept_id,
            prompt=item.prompt,
            wrong_answer=wrong,
            injected=injected,
            inferred=diagnosis.misconception_id,
            seconds=time.time() - started,
            bank=item.bank,
        )
        if on_result is not None:
            on_result(prediction)
        yield prediction
