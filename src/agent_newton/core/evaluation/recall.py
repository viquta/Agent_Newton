"""Scoring a recall strategy against hand labels.

Precision and recall, reported separately and never averaged into one number.
They are not interchangeable here: an unrelated remark handed to a tutor as
context is worse than silence, because the tutor will try to use it — so a
strategy that finds everything and half of it is noise is worse for this purpose
than one that finds less and means it.

⚠️ ``returned_nothing_correctly`` is reported beside them because neither
precision nor recall can see it. A case with no relevant utterance has no true
positives to find, so it contributes nothing to recall and cannot lower
precision unless something is returned — and returning nothing there is the
right answer, which ought to be worth stating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import yaml

from agent_newton.core.recall.base import Recall
from agent_newton.core.state.schema import Utterance


def _kind(value: str) -> Literal["reflection", "working", "lesson"]:
    if value not in ("reflection", "working", "lesson"):
        raise ValueError(
            f"unknown utterance kind {value!r}; expected reflection, working "
            f"or lesson"
        )
    return value


@dataclass(frozen=True, slots=True)
class RecallCase:
    """One query, and the utterances a person judged relevant to it."""

    id: str
    concept_id: str
    query: str
    relevant: frozenset[str]
    note: str = ""


@dataclass(frozen=True, slots=True)
class RecallGold:
    """A corpus, and the cases asked over it."""

    corpus: tuple[Utterance, ...]
    #: Utterance id per corpus position, since ``Utterance`` carries no id and
    #: should not gain one for a test fixture's benefit.
    ids: tuple[str, ...]
    cases: tuple[RecallCase, ...]

    def id_of(self, utterance: Utterance) -> str:
        for stored_id, stored in zip(self.ids, self.corpus):
            if stored is utterance:
                return stored_id
        return "?"


def load_gold(path: str | Path) -> RecallGold:
    data = yaml.safe_load(Path(path).read_text())
    corpus, ids = [], []
    for entry in data["corpus"]:
        ids.append(entry["id"])
        corpus.append(
            Utterance(
                text=" ".join(str(entry["text"]).split()),
                item_id="",
                concept_id=entry["concept_id"],
                # Narrowed from the YAML's free string. The fixture is
                # hand-written and a typo there should fail here rather than
                # produce an utterance of a kind nothing else recognises.
                kind=_kind(entry.get("kind", "lesson")),
            )
        )
    known = set(ids)
    cases = []
    for entry in data["cases"]:
        unknown = set(entry["relevant"]) - known
        if unknown:
            # Referential integrity here rather than at scoring time, so a
            # renamed utterance fails loudly instead of silently shrinking the
            # set it is labelled against.
            raise ValueError(f"case {entry['id']} names unknown utterances: {unknown}")
        cases.append(
            RecallCase(
                id=entry["id"],
                concept_id=entry["concept_id"],
                query=" ".join(str(entry["query"]).split()),
                relevant=frozenset(entry["relevant"]),
                note=entry.get("note", ""),
            )
        )
    return RecallGold(tuple(corpus), tuple(ids), tuple(cases))


@dataclass
class RecallReport:
    """What one strategy found, per case and in total."""

    label: str = ""
    #: (case id, what was returned, what should have been)
    rows: list[tuple[str, frozenset[str], frozenset[str]]] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(len(got & want) for _, got, want in self.rows)

    @property
    def returned(self) -> int:
        return sum(len(got) for _, got, _ in self.rows)

    @property
    def relevant(self) -> int:
        return sum(len(want) for _, _, want in self.rows)

    @property
    def precision(self) -> float:
        """Of what it returned, how much was worth returning."""
        return self.true_positives / self.returned if self.returned else 0.0

    @property
    def recall(self) -> float:
        """Of what was there to find, how much it found."""
        return self.true_positives / self.relevant if self.relevant else 0.0

    @property
    def returned_nothing_correctly(self) -> int:
        """Cases with nothing to find, where nothing was returned.

        Invisible to both figures above, and the easiest thing to get wrong: a
        strategy that always fills its quota fails every one of these while its
        recall is untouched.
        """
        return sum(1 for _, got, want in self.rows if not want and not got)

    @property
    def noise(self) -> int:
        """Returned and not wanted. What a tutor would try to use."""
        return sum(len(got - want) for _, got, want in self.rows)

    def missed(self) -> list[tuple[str, frozenset[str]]]:
        return [(case, want - got) for case, got, want in self.rows if want - got]


def score(gold: RecallGold, strategy: Recall, limit: int = 3) -> RecallReport:
    report = RecallReport(label=strategy.label)
    for case in gold.cases:
        found = strategy.about(gold.corpus, case.concept_id, case.query, limit)
        report.rows.append(
            (case.id, frozenset(gold.id_of(u) for u in found), case.relevant)
        )
    return report
