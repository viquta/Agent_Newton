"""A sitting, read back as prose.

    uv run agent-newton sitting

Every number quoted about the human sittings so far came out of a script written
for the occasion, because the record is a 33 KB JSON document and the only way
to answer "what support did it give, and why" was to write a query. That is
fine for a machine and useless for the person who sat there.

This renders the audit log — the whole record, in order — as something a person
can read: the question, what they answered, what the verifier made of it, what
the tutor said, and **what the support level was chosen from**. The last part is
why this reads the log rather than the transcript's summary blocks: the log is
the complete account, and the two figures behind each level are now stored in
it.

Nothing here recomputes a decision. A renderer that re-derived the support level
would agree with the session right up to the moment one of them was wrong, which
is the only moment worth having it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from agent_newton.domains.base import Domain

#: Audit causes this reads, and what each contributes to the account.
_OBSERVATION = "observation"
_TUTOR = "tutor"
_ANNOTATION = "annotation"


def _name(domain: Domain, concept_id: str | None) -> str:
    if not concept_id:
        return "—"
    try:
        return domain.concepts.get(concept_id).name
    except Exception:  # a concept that has since been renamed out of the graph
        return concept_id


def narrate(
    records: Sequence[Mapping[str, Any]],
    domain: Domain,
    *,
    learner_id: str = "",
    header: Sequence[str] = (),
    figures: Mapping[str, Any] | None = None,
) -> str:
    """The sitting as markdown, in the order it happened.

    ``records`` are audit entries as ``{"cause", "summary", "evidence"}`` —
    dictionaries rather than ``AuditRecord``, so a stored transcript and a live
    board render through the same code without one of them being converted into
    the other's shape first.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# Sitting — {learner_id or 'unknown learner'}")
    add("")
    for line in header:
        add(f"- {line}")
    add("")

    current: str | None = None

    for record in records:
        cause = record.get("cause")
        evidence = record.get("evidence") or {}
        concept = evidence.get("concept_id")

        # A section starts at the first record about a new item, whatever kind
        # it is. Anchoring on the graded step alone put the reasoning above its
        # own heading, because the reasoning is now taken *before* the verdict.
        item_id = evidence.get("item_id")
        if item_id and item_id != current and cause in (_OBSERVATION, _TUTOR, _ANNOTATION):
            current = item_id
            add(f"### {_name(domain, concept)} · `{current}`")
            add("")

        if cause == _OBSERVATION:
            verdict = evidence.get("verdict", "?")
            wrote = evidence.get("response")
            if wrote is not None:
                shown = f"`{wrote}`" if str(wrote).strip() else "*(nothing)*"
                add(f"**{learner_id or 'the learner'}** answered {shown}")
                add("")
            moved = ""
            if "mastery_before" in evidence:
                moved = (
                    f" — belief {evidence['mastery_before']:.2f} → "
                    f"{evidence['mastery_after']:.2f}"
                )
            add(f"→ *{verdict}*{moved}")
            if evidence.get("misconception_label"):
                add(f"  <br>diagnosed `{evidence['misconception_label']}`")
            add("")

        elif cause == _TUTOR:
            add(f"**tutor** · {evidence.get('move')} · **{evidence.get('level')}**")
            add("")
            for line in str(evidence.get("text", "")).splitlines():
                add(f"> {line}")
            add("")
            # The two figures the level was chosen from. Absent in sittings
            # recorded before they were stored, and said to be absent rather
            # than defaulted — a zero here would read as a belief of zero.
            if "prior_failures" in evidence:
                add(
                    f"chosen from a belief of {evidence.get('mastery', 0.0):.2f} "
                    f"when the question was posed and "
                    f"{evidence['prior_failures']} earlier readable failure(s) "
                    f"on this item."
                )
            else:
                add("*(this sitting predates the level's inputs being recorded)*")
            add("")

        elif cause == _ANNOTATION and "reflection" in evidence:
            kind = evidence.get("kind", "reflection")
            add(f"**{learner_id or 'the learner'}** ({kind}):")
            for line in str(evidence["reflection"]).splitlines():
                add(f"> {line}")
            add("")

        elif cause == _ANNOTATION and evidence.get("declined"):
            add("*asked for the reasoning; none was given.*")
            add("")

        elif cause in ("plan", "replan"):
            add(f"↻ {record.get('summary')}")
            add("")

        elif cause == _ANNOTATION and evidence.get("requested") is not None:
            asked = ", ".join(
                _name(domain, c) for c in evidence["requested"]
            ) or "nothing in particular"
            add(f"**Asked to work on:** {asked}")
            add("")

    if figures:
        add("## Figures")
        add("")
        for label, value in figures.items():
            add(f"- {label}: {value}")
        add("")

    return "\n".join(lines) + "\n"


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts worth seeing before reading anything.

    Levels are the reason this exists: two sittings ran entirely at
    ``worked_step`` and nobody noticed until a transcript was read by hand.
    """
    levels: dict[str, int] = {}
    moves: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    for record in records:
        evidence = record.get("evidence") or {}
        if record.get("cause") == _TUTOR:
            levels[evidence.get("level", "?")] = levels.get(evidence.get("level", "?"), 0) + 1
            moves[evidence.get("move", "?")] = moves.get(evidence.get("move", "?"), 0) + 1
        elif record.get("cause") == _OBSERVATION:
            verdict = evidence.get("verdict", "?")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    return {"levels": levels, "moves": moves, "verdicts": verdicts}
