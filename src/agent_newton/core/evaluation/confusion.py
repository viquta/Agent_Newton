"""Scoring the confusion detector against hand labels.

The detector answers one question about one string: does what the learner wrote
say they do not know what the concept *is*, as opposed to showing a mistake in
applying it. That is the third of the three things that can buy a lesson, and it
is the one place a model is permitted to decide something — so a figure for it
has to exist, and until this module it did not. The gold set was exercised only
as a pytest gate, which checks the detector but cannot produce a number anyone
can store, cite or compare a later run against.

⚠️ **The floor is reported beside the agreement, and it is not decoration.** The
set is balanced, so a detector answering "confused" to everything scores exactly
half. An agreement figure quoted without that number next to it says nothing —
the same reason a judged tutor rate is meaningless without the judge's
calibration beside it.

⚠️ **The two halves are reported apart, because the false half is the hard one.**
Hedging, uncertainty about an answer, and "this was confusing" all describe
someone who is doing the work and must *not* fire. A detector that fires on
everything gets the true half perfectly, and pooling the halves into one accuracy
would hide exactly that failure.

⚠️ **A stated limit on the instrument, which no figure from it escapes.** The
verdict on a borderline phrasing turns on punctuation: an unpunctuated
"i dont understand implicit differentiation" does not fire on the model this was
built against, while the same sentence with a full stop does. Every case in the
fixture happens to be punctuated, so the agreement reported here is measured over
a set that does not vary the thing known to move the verdict. The case is
recorded in the fixture's header rather than kept as a failing case, because the
calibration gate asserts total agreement and one known-bad case would turn a gate
into noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import yaml


class Detector(Protocol):
    """What this module needs, which is less than ``ConfusionDetector``.

    Declared here rather than imported so the scoring stays independent of the
    agent package: anything answering this one question can be scored, including
    the model-free ``NoConfusion``, which is what establishes the floor.
    """

    def confused(self, concept_id: str, text: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ConfusionCase:
    """One thing a learner wrote, and whether a person read it as confusion."""

    concept_id: str
    text: str
    confused: bool
    source: str


@dataclass(frozen=True, slots=True)
class ConfusionGold:
    """The hand-labelled set, and the properties it is required to have."""

    cases: tuple[ConfusionCase, ...]

    @property
    def positives(self) -> int:
        return sum(1 for case in self.cases if case.confused)

    @property
    def negatives(self) -> int:
        return len(self.cases) - self.positives

    @property
    def balanced(self) -> bool:
        """Whether a constant answer scores exactly half.

        Not a detail of the fixture — it is what makes the agreement figure
        readable at all, so it is a property of the instrument and is reported.
        """
        return self.positives == self.negatives

    @property
    def floor(self) -> float:
        """What answering "confused" to everything would score."""
        return self.positives / len(self.cases) if self.cases else 0.0


def load_gold(path: str | Path) -> ConfusionGold:
    """Read the fixture. The path is a parameter: ``core`` names no domain."""
    data = yaml.safe_load(Path(path).read_text())
    cases = tuple(
        ConfusionCase(
            concept_id=entry["concept_id"],
            text=entry["text"],
            confused=bool(entry["confused"]),
            source=entry.get("source", ""),
        )
        for entry in data["cases"]
    )
    if not cases:
        raise ValueError(f"{path} holds no cases")
    return ConfusionGold(cases=cases)


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One case the detector read differently from the person who labelled it."""

    concept_id: str
    text: str
    expected: bool
    got: str | None

    @property
    def kind(self) -> str:
        """``false positive`` fires on someone doing the work; the reverse misses."""
        return "false positive" if self.expected is False else "false negative"


@dataclass(frozen=True, slots=True)
class ConfusionReport:
    """What a run of the detector over the gold set produced."""

    label: str
    total: int
    agreed: int
    #: Agreement within each half, reported apart — see the module docstring.
    positives: int
    positives_agreed: int
    negatives: int
    negatives_agreed: int
    floor: float
    balanced: bool
    disagreements: tuple[Disagreement, ...]

    @property
    def agreement(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def detected_confusion(self) -> float:
        """Agreement on the half that should fire."""
        return self.positives_agreed / self.positives if self.positives else 0.0

    @property
    def left_work_alone(self) -> float:
        """Agreement on the half that must not fire — the hard one."""
        return self.negatives_agreed / self.negatives if self.negatives else 0.0

    @property
    def beats_the_floor(self) -> bool:
        """Whether the figure says anything a constant answer would not."""
        return self.agreement > self.floor

    def as_dict(self) -> dict:
        """The stored summary. Carries the floor, so the figure cannot travel without it."""
        return {
            "detector": self.label,
            "cases": self.total,
            "agreed": self.agreed,
            "agreement": self.agreement,
            "floor": self.floor,
            "beats_the_floor": self.beats_the_floor,
            "balanced": self.balanced,
            "detected_confusion": self.detected_confusion,
            "left_work_alone": self.left_work_alone,
            "positives": self.positives,
            "negatives": self.negatives,
            "disagreements": [
                {
                    "concept_id": d.concept_id,
                    "text": d.text,
                    "expected": d.expected,
                    "got": d.got,
                    "kind": d.kind,
                }
                for d in self.disagreements
            ],
        }


def score(gold: ConfusionGold, detector: Detector, label: str) -> ConfusionReport:
    """Run the detector over every case and count agreement with the labels.

    The detector returns the *quote* it read rather than a bool, so that a firing
    can be argued with afterwards. Here that is narrowed to whether it fired at
    all — but the quote is kept on each disagreement, since a false positive is
    only interpretable if you can see what was read that way.
    """
    agreed = pos = pos_ok = neg = neg_ok = 0
    disagreements: list[Disagreement] = []
    for case in gold.cases:
        got = detector.confused(case.concept_id, case.text)
        fired = got is not None
        if case.confused:
            pos += 1
            pos_ok += fired
        else:
            neg += 1
            neg_ok += not fired
        if fired == case.confused:
            agreed += 1
        else:
            disagreements.append(
                Disagreement(case.concept_id, case.text, case.confused, got)
            )
    return ConfusionReport(
        label=label,
        total=len(gold.cases),
        agreed=agreed,
        positives=pos,
        positives_agreed=pos_ok,
        negatives=neg,
        negatives_agreed=neg_ok,
        floor=gold.floor,
        balanced=gold.balanced,
        disagreements=tuple(disagreements),
    )
