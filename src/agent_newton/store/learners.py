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
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from agent_newton.core.state.schema import AuditRecord, LearnerState

_SCHEMA = Path(__file__).parent / "schema.sql"


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
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> LearnerStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- learners ---------------------------------------------------------

    def ensure_learner(self, learner_id: str, kind: str, domain: str) -> None:
        """Register a learner, or leave an existing one alone."""
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
    ) -> int:
        seq = self.next_index(learner_id, arm)
        cursor = self._db.execute(
            "INSERT INTO session (learner_id, arm, seq, elapsed_days, run_id, "
            "config_hash, decay_half_life_days, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                learner_id,
                arm,
                seq,
                elapsed_days,
                run_id,
                config_hash,
                decay_half_life_days,
                _now(),
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
        self._db.executemany(
            "INSERT INTO utterance (session_id, kind, item_id, concept_id, text) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (session_id, u.kind, u.item_id, u.concept_id, u.text)
                for u in state.reflections
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
