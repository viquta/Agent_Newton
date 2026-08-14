"""What the system did about one skill, across every sitting.

A learner who never grasps a concept despite sustained, appropriate teaching is
an ordinary pedagogical case rather than a failed experiment. What can be
established either way is whether the instruction was appropriate — and that is
a claim about the *system's* behaviour, not the learner's, so it can be made
about a simulated learner whose response to teaching is a rule, and about a
person for whom no ground truth exists at all.

That last point is why this is here. ``remediation_ratio`` is unavailable for a
human and pre/post gain is confounded by a fixed bank seen repeatedly, so the
teaching record is the one account of a person's tutoring that can be given.

Nothing new is recorded to produce it. Every mutation already passes through
one writer and lands in the audit log, and ``LearnerStore`` projects that log
into a table queryable across sittings. This module only reads.

**What it establishes, and what it does not.** It shows fidelity to the stated
pedagogy: which of the instructional moves the system owns were actually used
on a concept, over how long, and what the estimate did meanwhile. It does *not*
show the teaching was good — only that it was, or was not, what the design
prescribes. Written carelessly that becomes "success is compliance with our own
rules", which is why the next paragraph is the load-bearing one.

**The record must be able to say the system did not try everything.** The
repertoire is finite and enumerable, so :attr:`TeachingRecord.not_attempted`
reports what was never reached. A learner stuck on a concept who never received
a worked step, or never a reflective prompt before a correction, is a failure of
the system rather than of the learner, and this has to be able to say so.
Otherwise it is a log that can only report success.

⚠️ **Absent is not zero.** Tutor turns were not recorded before 2026-08-12, so a
history from before then has no instruction half at all. That is reported as
unavailable — :attr:`TeachingRecord.not_attempted` is ``None`` — and never as
"nothing was taught", which is the same distinction the outcome measures draw
for a skipped test bank and a bank scored zero.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from agent_newton.core.evaluation.outcomes import INSTRUCTION_CAUSE
from agent_newton.core.pedagogy import HintLevel, TutorMove

#: The cause carrying what the tutor said. Distinct from instruction: an
#: observation is the learner acting, a turn is the system responding.
TUTOR_CAUSE = "tutor"


def repertoire() -> tuple[str, ...]:
    """Every instructional move the system owns, as stable strings.

    Derived from the enums rather than written down, so the repertoire cannot
    drift from what the tutor can actually do. If a fourth support level is ever
    added, ``not_attempted`` starts reporting it without this being touched.
    """
    return tuple(
        [f"level:{level.label}" for level in HintLevel]
        + [f"move:{move.value}" for move in TutorMove]
    )


@dataclass(frozen=True, slots=True)
class Sitting:
    """One concept's history within one sitting — a row in ``records.csv``.

    Flat on purpose. This is the unit a figure is drawn from, so nesting it
    would only have to be undone again.
    """

    learner_id: str
    arm: str
    concept_id: str
    seq: int
    elapsed_days: float

    attempts: int = 0
    correct: int = 0
    #: Steps the verifier could not read. Not failures — see the invariant at
    #: the top of ``core/state/store.py``.
    unmeasurable: int = 0

    levels: Mapping[str, int] = field(default_factory=dict)
    moves: Mapping[str, int] = field(default_factory=dict)
    #: Misconceptions a remediating turn actually aimed at, in order.
    remediation_targets: tuple[str, ...] = ()
    distinct_items: int = 0

    seeded: int = 0
    decayed: int = 0

    #: The estimate on entering and leaving this sitting, over this concept.
    #: None when nothing touched it — which cannot happen for a row that exists.
    mastery_before: float | None = None
    mastery_after: float | None = None

    #: Whether this sitting recorded tutor turns at all. False for every sitting
    #: before turns were kept, and the reason a missing instruction half is
    #: reported as unavailable rather than as an absence of teaching.
    instruction_recorded: bool = False

    @property
    def movement(self) -> float | None:
        if self.mastery_before is None or self.mastery_after is None:
            return None
        return self.mastery_after - self.mastery_before


@dataclass
class TeachingRecord:
    """Everything the system did about one concept, over a whole history."""

    learner_id: str
    arm: str
    concept_id: str
    sittings: list[Sitting] = field(default_factory=list)

    # --- what happened ----------------------------------------------------

    @property
    def attempts(self) -> int:
        return sum(s.attempts for s in self.sittings)

    @property
    def correct(self) -> int:
        return sum(s.correct for s in self.sittings)

    @property
    def sittings_spanned(self) -> int:
        return len(self.sittings)

    @property
    def days_spanned(self) -> float:
        """Elapsed days between the first sitting that touched this and the last.

        The first sitting's own gap is excluded: it is the time before the
        concept was ever worked, which is not part of how long it has been
        taught.
        """
        return sum(s.elapsed_days for s in self.sittings[1:])

    @property
    def levels(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        for sitting in self.sittings:
            total.update(sitting.levels)
        return dict(total)

    @property
    def moves(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        for sitting in self.sittings:
            total.update(sitting.moves)
        return dict(total)

    @property
    def remediation_targets(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        for sitting in self.sittings:
            total.update(sitting.remediation_targets)
        return dict(total)

    # --- what the estimate did --------------------------------------------

    @property
    def trajectory(self) -> tuple[tuple[int, float], ...]:
        """``(seq, mastery_after)`` per sitting — the line a figure plots."""
        return tuple(
            (s.seq, s.mastery_after)
            for s in self.sittings
            if s.mastery_after is not None
        )

    @property
    def movement(self) -> float | None:
        """Estimate at the end of the last sitting minus the start of the first.

        Spans the gaps, so decay is included: a concept taught, forgotten and
        taught again has moved less than the sittings alone would suggest, and
        that is the honest figure over a history.
        """
        first = next((s.mastery_before for s in self.sittings if s.mastery_before is not None), None)
        last = next(
            (s.mastery_after for s in reversed(self.sittings) if s.mastery_after is not None),
            None,
        )
        if first is None or last is None:
            return None
        return last - first

    # --- what was and was not tried ---------------------------------------

    @property
    def instruction_recorded(self) -> bool:
        """Whether any sitting in this history kept what the tutor said."""
        return any(s.instruction_recorded for s in self.sittings)

    @property
    def attempted(self) -> frozenset[str]:
        """Repertoire entries actually used on this concept."""
        used = {f"level:{name}" for name, n in self.levels.items() if n}
        used |= {f"move:{name}" for name, n in self.moves.items() if n}
        return frozenset(used)

    @property
    def not_attempted(self) -> frozenset[str] | None:
        """Repertoire entries never reached. **None when nothing was recorded.**

        This is the field that makes the record a measure rather than a log: it
        is what allows the answer "the system did not try everything". A learner
        stuck on a concept who never received a worked step is a failure of the
        system.

        None rather than the whole repertoire when no sitting kept its turns.
        Reporting every move as unattempted would be a claim about the teaching,
        when the truth is that the teaching was not written down.
        """
        if not self.instruction_recorded:
            return None
        return frozenset(repertoire()) - self.attempted

    @property
    def exhausted_repertoire(self) -> bool | None:
        """Whether every instructional move was tried. None when unrecorded."""
        remaining = self.not_attempted
        return None if remaining is None else not remaining


# --- derivation --------------------------------------------------------------


def _evidence(row: Any) -> dict[str, Any]:
    """The stored evidence blob, as a mapping.

    Stored as text by the projection; the session's own audit log carries it as
    a dict. Both reach here, so both are accepted.
    """
    raw = row["evidence"]
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw or {})


def records(
    store: Any,
    learner_id: str,
    arm: str,
    concepts: Iterable[str] | None = None,
) -> list[TeachingRecord]:
    """Derive a record per concept from a learner's stored history.

    One pass over the events, grouped by ``(sitting, concept)``. Reads only —
    the store is queried, never written.

    ``concepts`` restricts the result; without it, every concept the history
    touched gets a record. A concept that was never touched gets **no record**
    rather than an empty one, on the same grounds as everywhere else here: the
    absence of teaching and a record of no teaching are different claims.
    """
    wanted = set(concepts) if concepts is not None else None
    gaps = {int(s["seq"]): float(s["elapsed_days"]) for s in store.sessions(learner_id, arm)}

    # (seq, concept) -> the parts a Sitting is built from.
    attempts: dict[tuple[int, str], dict[str, Any]] = {}
    #: Sittings that kept any tutor turn at all, whatever the concept — a
    #: property of when the sitting ran, not of what was taught in it.
    recorded: set[int] = set()

    for row in store.events(learner_id, arm):
        cause = row["cause"]
        if cause == TUTOR_CAUSE:
            recorded.add(int(row["seq"]))
        if cause not in (INSTRUCTION_CAUSE, TUTOR_CAUSE, "seed", "decay"):
            continue

        evidence = _evidence(row)
        concept_id = evidence.get("concept_id")
        if not concept_id or (wanted is not None and concept_id not in wanted):
            continue

        key = (int(row["seq"]), str(concept_id))
        part = attempts.setdefault(
            key,
            {
                "attempts": 0, "correct": 0, "unmeasurable": 0,
                "levels": Counter(), "moves": Counter(), "targets": [],
                "items": set(), "seeded": 0, "decayed": 0,
                "before": None, "after": None,
            },
        )

        # The estimate as it stood entering and leaving this sitting. Taken from
        # whichever entries carry it — an unmeasurable observation does not, on
        # purpose, because nothing moved.
        if "mastery_before" in evidence:
            if part["before"] is None:
                part["before"] = float(evidence["mastery_before"])
            part["after"] = float(evidence["mastery_after"])

        if cause == INSTRUCTION_CAUSE:
            verdict = evidence.get("verdict")
            part["attempts"] += 1
            if verdict == "correct":
                part["correct"] += 1
            elif verdict == "unparseable":
                part["unmeasurable"] += 1
            if evidence.get("item_id"):
                part["items"].add(str(evidence["item_id"]))
        elif cause == TUTOR_CAUSE:
            part["levels"][str(evidence.get("level", ""))] += 1
            part["moves"][str(evidence.get("move", ""))] += 1
            if evidence.get("targets"):
                part["targets"].append(str(evidence["targets"]))
        elif cause == "seed":
            part["seeded"] += 1
        elif cause == "decay":
            part["decayed"] += 1

    by_concept: dict[str, TeachingRecord] = {}
    for (seq, concept_id), part in sorted(attempts.items()):
        record = by_concept.setdefault(
            concept_id, TeachingRecord(learner_id, arm, concept_id)
        )
        record.sittings.append(
            Sitting(
                learner_id=learner_id,
                arm=arm,
                concept_id=concept_id,
                seq=seq,
                elapsed_days=gaps.get(seq, 0.0),
                attempts=part["attempts"],
                correct=part["correct"],
                unmeasurable=part["unmeasurable"],
                levels={k: v for k, v in part["levels"].items() if k},
                moves={k: v for k, v in part["moves"].items() if k},
                remediation_targets=tuple(part["targets"]),
                distinct_items=len(part["items"]),
                seeded=part["seeded"],
                decayed=part["decayed"],
                mastery_before=part["before"],
                mastery_after=part["after"],
                instruction_recorded=seq in recorded,
            )
        )
    return [by_concept[c] for c in sorted(by_concept)]


def summarise(found: Sequence[TeachingRecord]) -> dict[str, Any]:
    """The aggregate that is not rows — for ``summary.json``."""
    return {
        "concepts": len(found),
        "sittings": len({s.seq for r in found for s in r.sittings}),
        "instruction_recorded": any(r.instruction_recorded for r in found),
        "repertoire": list(repertoire()),
        "per_concept": {
            r.concept_id: {
                "sittings_spanned": r.sittings_spanned,
                "days_spanned": r.days_spanned,
                "attempts": r.attempts,
                "correct": r.correct,
                "levels": r.levels,
                "moves": r.moves,
                "remediation_targets": r.remediation_targets,
                "movement": r.movement,
                "trajectory": [list(point) for point in r.trajectory],
                # None where the sittings predate turn recording — unavailable,
                # not "nothing was tried".
                "not_attempted": (
                    None if r.not_attempted is None else sorted(r.not_attempted)
                ),
                "exhausted_repertoire": r.exhausted_repertoire,
            }
            for r in found
        },
    }
