"""Verifier accuracy, measured against a hand-labelled gold set.

The verifier decides correctness for every student step, and nothing downstream
re-checks it. A verdict it gets wrong is not corrected later: it becomes a BKT
update, an error event, and a diagnosis of a misconception nobody held. So it is
measured against labels written by hand rather than against itself.

Two failure directions, with different consequences:

* **False negative** — a correct answer written differently, judged incorrect.
  The learner is charged with an error they did not make, and the tutor aims a
  hint at a misconception that is not there. This is why the gold set is loaded
  with equivalent-but-differently-written answers; they are the cases a
  string-matching verifier would fail and an equivalence verifier must pass.
* **False accept** — a wrong answer judged correct. The misconception never
  enters the error trace, so nothing remediates it and the effect is invisible.

A third outcome is neither: ``UNPARSEABLE`` means the verifier could not
measure. It records no evidence, so it cannot mislead the learner model — but it
also wastes a step, so it is counted separately rather than folded into either
rate.

Every case carries a class, and the class *is* the hand label — there is no
second field to drift from it:

===============  =======================================================
``canonical``    the bank's own answer, verbatim → ``CORRECT``
``equivalent``   correct, written differently → ``CORRECT``
``wrong``        a genuine mathematical error → ``INCORRECT``
``unreadable``   nothing mathematically decidable → ``UNPARSEABLE``
===============  =======================================================

A case may carry ``known_limitation``: a stated reason why the verifier is
expected to disagree with the hand label today. Those cases stay in the set and
stay counted. :meth:`GoldReport.surprises` holds disagreements nobody predicted
and :meth:`GoldReport.resolved` holds predictions that no longer hold, so the
markers cannot quietly rot in either direction.

Generic over the domain: the gold file names item ids, and the items, verifier
and answers all come from the ``Domain`` passed in.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent_newton.domains.base import Domain, DomainError, Verdict

#: Case class -> the verdict it asserts.
EXPECTED: dict[str, Verdict] = {
    "canonical": Verdict.CORRECT,
    "equivalent": Verdict.CORRECT,
    "wrong": Verdict.INCORRECT,
    "unreadable": Verdict.UNPARSEABLE,
}


@dataclass(frozen=True, slots=True)
class GoldCase:
    """One hand-labelled ``(item, response)`` pair."""

    id: str
    item_id: str
    response: str
    kind: str
    #: Why the verifier is expected to disagree with the label, or "" when it
    #: is expected to agree.
    known_limitation: str = ""
    note: str = ""

    @property
    def expected(self) -> Verdict:
        return EXPECTED[self.kind]


@dataclass(frozen=True, slots=True)
class Scored:
    case: GoldCase
    actual: Verdict
    detail: str

    @property
    def agrees(self) -> bool:
        return self.actual is self.case.expected

    @property
    def surprising(self) -> bool:
        """Disagrees with the hand label, and nobody said it would."""
        return not self.agrees and not self.case.known_limitation

    @property
    def resolved(self) -> bool:
        """Was expected to disagree, and no longer does."""
        return self.agrees and bool(self.case.known_limitation)


@dataclass
class GoldReport:
    scored: list[Scored] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.scored)

    @property
    def agreements(self) -> int:
        return sum(1 for s in self.scored if s.agrees)

    @property
    def accuracy(self) -> float:
        return self.agreements / self.total if self.total else 0.0

    def of_kind(self, kind: str) -> list[Scored]:
        return [s for s in self.scored if s.case.kind == kind]

    def surprises(self) -> list[Scored]:
        return [s for s in self.scored if s.surprising]

    def resolved(self) -> list[Scored]:
        return [s for s in self.scored if s.resolved]

    # --- the two rates that matter -------------------------------------------

    def false_negatives(self) -> list[Scored]:
        """Correct-but-differently-written answers not judged correct.

        Canonical cases are excluded from the denominator on purpose: an answer
        written exactly as the bank states it is not a test of equivalence, and
        including it would dilute the rate with cases that cannot fail.
        """
        return [s for s in self.of_kind("equivalent") if not s.agrees]

    @property
    def false_negative_rate(self) -> float:
        equivalent = self.of_kind("equivalent")
        return len(self.false_negatives()) / len(equivalent) if equivalent else 0.0

    def scored_incorrect(self) -> list[Scored]:
        """False negatives that reached the learner model as an error."""
        return [s for s in self.false_negatives() if s.actual is Verdict.INCORRECT]

    def unmeasured(self) -> list[Scored]:
        """False negatives the verifier declined to judge.

        No evidence is recorded, so the learner model is not corrupted — the
        step is simply wasted.
        """
        return [s for s in self.false_negatives() if s.actual is Verdict.UNPARSEABLE]

    def false_accepts(self) -> list[Scored]:
        """Wrong answers judged correct."""
        return [s for s in self.of_kind("wrong") if s.actual is Verdict.CORRECT]

    @property
    def false_accept_rate(self) -> float:
        wrong = self.of_kind("wrong")
        return len(self.false_accepts()) / len(wrong) if wrong else 0.0

    def hidden_errors(self) -> list[Scored]:
        """Wrong answers the verifier could not read.

        Not a false accept — nothing is scored — but the error still never
        reaches the error trace, so it cannot be remediated either.
        """
        return [s for s in self.of_kind("wrong") if s.actual is Verdict.UNPARSEABLE]

    def confusion(self) -> dict[tuple[Verdict, Verdict], int]:
        """(expected, actual) counts."""
        return dict(Counter((s.case.expected, s.actual) for s in self.scored))


def load(path: str | Path, domain: Domain) -> list[GoldCase]:
    """Read a gold file and check it against the domain it labels.

    Referential integrity is checked here rather than at scoring time so a
    renamed item fails loudly instead of silently shrinking the set.
    """
    path = Path(path)
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}

    raw = data.get("cases")
    if not raw:
        raise DomainError(f"{path} declares no cases")

    cases: list[GoldCase] = []
    seen: set[str] = set()
    for entry in raw:
        case = GoldCase(
            id=str(entry["id"]),
            item_id=str(entry["item_id"]),
            response=str(entry["response"]),
            kind=str(entry["kind"]),
            known_limitation=str(entry.get("known_limitation", "")).strip(),
            note=str(entry.get("note", "")).strip(),
        )
        if case.id in seen:
            raise DomainError(f"{path}: duplicate case id {case.id!r}")
        seen.add(case.id)
        if case.kind not in EXPECTED:
            raise DomainError(
                f"{path}: case {case.id!r} has unknown kind {case.kind!r}; "
                f"known: {', '.join(sorted(EXPECTED))}"
            )
        domain.items.get(case.item_id)  # raises on an unknown id
        cases.append(case)
    return cases


def score(domain: Domain, cases: list[GoldCase]) -> GoldReport:
    """Run the domain's verifier over every case."""
    report = GoldReport()
    started = time.time()
    for case in cases:
        item = domain.items.get(case.item_id)
        result = domain.verifier.verify(item, case.response)
        report.scored.append(Scored(case, result.verdict, result.detail))
    report.seconds = time.time() - started
    return report
