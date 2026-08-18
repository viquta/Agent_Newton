"""Persistent learner state, so a plan can outlive the session that made it.

A single-session artifact cannot demonstrate long-horizon planning: a plan that
never survives the sitting has no horizon. This is what lets a learner be picked
up where they were left, with the model's belief aged by however long the gap
was.

**This class holds no ground truth.** The simulated learner's misconception
profile lives in the same database but is reached only through
:mod:`agent_newton.store.ground_truth`, which agents may not import. That split
is the same one the in-memory design already makes — a profile is a Python object
the agents are simply never handed — carried over to persistence, where a shared
database would otherwise make ground truth reachable by anything holding a
connection.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from agent_newton.core.state.schema import AuditRecord, LearnerState

_SCHEMA = Path(__file__).parent / "schema.sql"


#: A learner id is an identity, and it also ends up in a filesystem path —
#: `agent-newton history <learner>` writes `results/history_<learner>_<arm>/`. So
#: `history '../../../tmp/x'` wrote outside `results/`, and an id containing a
#: slash silently created a directory tree instead of a directory.
#:
#: Parameterised queries already make any id safe for SQL — every hostile string
#: tried round-tripped and left the tables intact. Safe for SQL is not the same as
#: safe as a *name*, which is why this is enforced on the identity rather than
#: patched at each place a path is built.
_LEARNER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def check_learner_id(learner_id: str) -> str:
    """Return ``learner_id`` if it is usable as an identity, else raise.

    Permits what the project actually uses — ``victor``, ``L0000``, ``probe`` —
    and refuses anything with a path separator, a leading dot, whitespace, quotes,
    or nothing at all.
    """
    if not _LEARNER_ID.match(learner_id):
        raise ValueError(
            f"learner id {learner_id!r} is not usable: it must start with a letter "
            f"or digit and contain only letters, digits, hyphens and underscores. "
            f"The id names a directory as well as a row, so a separator or a dot "
            f"would write outside the results tree"
        )
    return learner_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LearnerStore:
    """Learners, their sessions, and the state each session left behind.

    Opened per run rather than held globally: a store is a file, and passing one
    around is how the runner keeps it out of the agents' reach.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA.read_text())
        self._migrate()

    def _migrate(self) -> None:
        """Add columns a database created by an earlier schema does not have.

        ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a new
        column never reaches a store that already exists — and this one holds
        sittings that cannot be regenerated. Each add is attempted and its
        duplicate-column error swallowed, which is the whole migration.
        """
        for column in (
            "catalogue_hash",
            "item_bank_hash",
            "concept_graph_hash",
            "resources_hash",
        ):
            try:
                self._db.execute(f"ALTER TABLE session ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass
        self._db.commit()
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> LearnerStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- learners ---------------------------------------------------------

    def ensure_learner(self, learner_id: str, kind: str, domain: str) -> None:
        """Register a learner, or leave an existing one alone.

        The id is checked here because this is the only way a learner comes into
        existence, so nothing downstream has to re-check it.
        """
        check_learner_id(learner_id)
        self._db.execute(
            "INSERT OR IGNORE INTO learner (learner_id, kind, domain, created_at) "
            "VALUES (?, ?, ?, ?)",
            (learner_id, kind, domain, _now()),
        )
        self._db.commit()

    def learners(self, kind: str | None = None) -> list[sqlite3.Row]:
        if kind is None:
            return list(self._db.execute("SELECT * FROM learner ORDER BY learner_id"))
        return list(
            self._db.execute(
                "SELECT * FROM learner WHERE kind = ? ORDER BY learner_id", (kind,)
            )
        )

    # -- sessions ---------------------------------------------------------

    def next_index(self, learner_id: str, arm: str) -> int:
        """Where this learner is up to in *this arm's* sequence.

        Keyed on the arm because the same learner under two architectures is two
        histories. Resuming across them would let one arm inherit the other's
        progress, which is the comparison destroying itself.
        """
        row = self._db.execute(
            "SELECT MAX(seq) AS last FROM session WHERE learner_id = ? AND arm = ?",
            (learner_id, arm),
        ).fetchone()
        return 0 if row["last"] is None else int(row["last"]) + 1

    def open_session(
        self,
        *,
        learner_id: str,
        arm: str,
        config_hash: str,
        elapsed_days: float = 0.0,
        run_id: str | None = None,
        decay_half_life_days: float | None = None,
        content_hashes: Mapping[str, str] | None = None,
    ) -> int:
        seq = self.next_index(learner_id, arm)
        cursor = self._db.execute(
            "INSERT INTO session (learner_id, arm, seq, elapsed_days, run_id, "
            "config_hash, decay_half_life_days, started_at, "
            "catalogue_hash, item_bank_hash, concept_graph_hash, resources_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                learner_id,
                arm,
                seq,
                elapsed_days,
                run_id,
                config_hash,
                decay_half_life_days,
                _now(),
                *(
                    (content_hashes or {}).get(field)
                    for field in (
                        "catalogue_hash",
                        "item_bank_hash",
                        "concept_graph_hash",
                        "resources_hash",
                    )
                ),
            ),
        )
        self._db.commit()
        return int(cursor.lastrowid or 0)

    def close_session(
        self,
        session_id: int,
        *,
        state: LearnerState,
        audit_log: Sequence[AuditRecord] = (),
        planner_state: Mapping[str, object] | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """Store what the session ended with.

        The state blob is what a later session resumes from. The event and
        utterance rows are projections of it, written so a sequence can be
        queried without deserialising every state — where they disagree, the
        blob is authoritative.
        """
        self._db.execute(
            "UPDATE session SET ended_at = ?, stop_reason = ? WHERE session_id = ?",
            (_now(), stop_reason, session_id),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO state (session_id, learner_state, planner_state) "
            "VALUES (?, ?, ?)",
            (
                session_id,
                state.model_dump_json(),
                None if planner_state is None else json.dumps(dict(planner_state)),
            ),
        )
        self._db.executemany(
            "INSERT INTO event (session_id, version, cause, summary, evidence) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (session_id, r.version, r.cause, r.summary, json.dumps(r.evidence, default=str))
                for r in audit_log
            ],
        )
        # ⚠️ Read from the audit log, not from ``state.reflections``.
        #
        # The state carries every word the learner has ever said, because it is
        # resumed whole — so projecting it wrote the entire history under each
        # new session id. One learner's table held 81 rows and 27 distinct
        # texts across three sittings, two of which had said nothing at all.
        # Nothing read the table yet, which is the only reason it never
        # produced a wrong number.
        #
        # The audit log is per sitting, which is exactly what a per-sitting
        # projection needs, and it is the same source the event rows above are
        # built from. Where the two disagreed, this one was the odd one out.
        self._db.executemany(
            "INSERT INTO utterance (session_id, kind, item_id, concept_id, text) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    r.evidence["kind"],
                    r.evidence["item_id"],
                    r.evidence["concept_id"],
                    r.evidence["reflection"],
                )
                for r in audit_log
                if r.cause == "annotation" and "reflection" in r.evidence
            ],
        )
        self._db.commit()

    def sessions(self, learner_id: str, arm: str) -> list[sqlite3.Row]:
        return list(
            self._db.execute(
                "SELECT * FROM session WHERE learner_id = ? AND arm = ? ORDER BY seq",
                (learner_id, arm),
            )
        )

    # -- resuming ---------------------------------------------------------

    def latest_state(self, learner_id: str, arm: str) -> LearnerState | None:
        """The state this learner last left behind in this arm, or None.

        None means a learner who has never sat a session in this arm — which is
        the first session of a sequence, and is deliberately distinct from a
        learner whose state happens to be empty.
        """
        row = self._db.execute(
            "SELECT s.learner_state FROM state s "
            "JOIN session ss ON ss.session_id = s.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ? "
            "ORDER BY ss.seq DESC LIMIT 1",
            (learner_id, arm),
        ).fetchone()
        return None if row is None else LearnerState.model_validate_json(row["learner_state"])

    def content_drift(
        self, learner_id: str, arm: str, current: Mapping[str, str]
    ) -> dict[str, tuple[str, str]]:
        """Domain content that has changed since this learner's last sitting.

        ``{field: (stored, current)}`` for each hash that differs, empty when
        nothing did. Read this before resuming: mastery is keyed by concept id
        and an error trace by misconception label, so renaming or removing either
        leaves stored state pointing at content that no longer exists.

        The concrete failure it guards: ``_cross_concept`` looks up every
        error-trace label in the catalogue, and a missing id raises. That happens
        at *outcome* time, after the sitting, when the work is already done.

        A session written before these columns existed reports nothing, because
        unverifiable and unchanged are different and only one of them is a
        warning worth giving. That is what covers ``resources_hash`` for every
        sitting recorded before resources existed, and it is why adding the
        column does not turn an old learner's history into false mismatches.
        """
        row = self._db.execute(
            "SELECT catalogue_hash, item_bank_hash, concept_graph_hash, "
            "resources_hash FROM session "
            "WHERE learner_id = ? AND arm = ? ORDER BY seq DESC LIMIT 1",
            (learner_id, arm),
        ).fetchone()
        if row is None:
            return {}
        drift: dict[str, tuple[str, str]] = {}
        for field, now in current.items():
            was = row[field] if field in row.keys() else None
            if was is not None and now is not None and was != now:
                drift[field] = (was, now)
        return drift

    def latest_planner_state(self, learner_id: str, arm: str) -> dict[str, Any] | None:
        """The planner bookkeeping this learner last left behind in this arm.

        None for the coupled arm, always: its planner keeps nothing, because
        everything it routes from is on the blackboard.
        """
        row = self._db.execute(
            "SELECT s.planner_state FROM state s "
            "JOIN session ss ON ss.session_id = s.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ? "
            "ORDER BY ss.seq DESC LIMIT 1",
            (learner_id, arm),
        ).fetchone()
        if row is None or row["planner_state"] is None:
            return None
        return json.loads(row["planner_state"])

    # -- reading a history ------------------------------------------------

    def utterances(
        self, learner_id: str, arm: str, concept_id: str | None = None
    ) -> list[sqlite3.Row]:
        """Everything the learner has said, across sessions.

        Here because planning is meant to read it — *"check their notes and
        previous answers before deciding on a new question"*. Within one session
        the state already carries it; this is what makes it survive the sitting.
        """
        sql = (
            "SELECT u.*, ss.seq FROM utterance u "
            "JOIN session ss ON ss.session_id = u.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ?"
        )
        params: list[object] = [learner_id, arm]
        if concept_id is not None:
            sql += " AND u.concept_id = ?"
            params.append(concept_id)
        return list(self._db.execute(sql + " ORDER BY ss.seq, u.utterance_id", params))

    def events(
        self, learner_id: str, arm: str, cause: str | None = None
    ) -> Iterator[sqlite3.Row]:
        sql = (
            "SELECT e.*, ss.seq FROM event e "
            "JOIN session ss ON ss.session_id = e.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ?"
        )
        params: list[object] = [learner_id, arm]
        if cause is not None:
            sql += " AND e.cause = ?"
            params.append(cause)
        yield from self._db.execute(sql + " ORDER BY ss.seq, e.event_id", params)
