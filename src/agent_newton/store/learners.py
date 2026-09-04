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


#: Bumped when a projection changes shape, so ``_backfill`` runs once and then
#: stops. Stored in ``PRAGMA user_version``, which SQLite keeps for exactly this.
_SCHEMA_VERSION = 2

_INSERT_TURN = (
    "INSERT INTO turn (session_id, version, item_id, concept_id, move, level, "
    "targets, text, mastery, prior_failures) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_UTTERANCE = (
    "INSERT INTO utterance (session_id, version, kind, item_id, concept_id, text) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def _utterance_row(session_id: int, version: int, evidence: Mapping[str, Any]) -> tuple:
    """One ``annotation`` record carrying a reflection, flattened into columns."""
    return (
        session_id,
        version,
        str(evidence.get("kind") or "reflection"),
        str(evidence.get("item_id") or ""),
        str(evidence.get("concept_id") or ""),
        str(evidence.get("reflection") or ""),
    )


def _turn_row(session_id: int, version: int, evidence: Mapping[str, Any]) -> tuple:
    """One ``tutor`` audit record, flattened into columns.

    Tolerant of missing keys, because it also runs over rows written before
    those keys existed — ``mastery`` and ``prior_failures`` were added after the
    first sittings, and a backfill that raised on them would refuse to migrate
    exactly the history worth migrating.
    """
    return (
        session_id,
        version,
        str(evidence.get("item_id") or ""),
        str(evidence.get("concept_id") or ""),
        str(evidence.get("move") or ""),
        str(evidence.get("level") or ""),
        evidence.get("targets"),
        str(evidence.get("text") or ""),
        float(evidence.get("mastery") or 0.0),
        int(evidence.get("prior_failures") or 0),
    )


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
        for column in ("concept_id", "item_id"):
            try:
                self._db.execute(f"ALTER TABLE event ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass
        for table in ("turn", "utterance"):
            try:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN version INTEGER")
            except sqlite3.OperationalError:
                pass
        # ⚠️ Here rather than in schema.sql. That file is executed in full on
        # every open and *before* this runs, so an index on a column this
        # migration has yet to add raises on any store an earlier schema
        # created — which is every store holding a sitting worth keeping.
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS event_by_concept ON event(concept_id)"
        )
        # Ordering a conversation means ordering turns against utterances, which
        # is what `version` is for. Dropped and recreated rather than
        # `IF NOT EXISTS`: an earlier schema created `turn_by_session` over
        # `session_id` alone, and that name would keep the narrower index
        # forever.
        self._db.execute("DROP INDEX IF EXISTS turn_by_session")
        self._db.execute(
            "CREATE INDEX turn_by_session ON turn(session_id, version)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS utterance_by_session "
            "ON utterance(session_id, version)"
        )
        self._db.commit()
        self._backfill()

    def _backfill(self) -> None:
        """Fill the new columns and the ``turn`` table from rows already stored.

        A store holds sittings that cannot be regenerated — a person sat at a
        keyboard once and said what they said — so adding a column and leaving
        every existing row NULL would make the new shape useless for exactly the
        history it was added to make readable.

        Guarded by ``PRAGMA user_version`` rather than by looking at whether the
        rows are empty. An emptiness check would re-run on any store that
        genuinely has no turns, and would silently stop being a migration and
        start being a repair that fires at random.

        Reads only ``event.evidence``, which was authoritative all along, and
        writes nothing back to it.
        """
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version >= _SCHEMA_VERSION:
            return

        for row in self._db.execute(
            "SELECT event_id, cause, evidence FROM event"
        ).fetchall():
            try:
                evidence = json.loads(row["evidence"])
            except (TypeError, ValueError):
                continue
            if not isinstance(evidence, dict):
                continue
            self._db.execute(
                "UPDATE event SET concept_id = ?, item_id = ? WHERE event_id = ?",
                (evidence.get("concept_id"), evidence.get("item_id"), row["event_id"]),
            )

        # Both projections are rebuilt from the log, not only `turn`.
        #
        # ⚠️ `utterance` needed it. It held 194 rows against 140 the audit log
        # accounted for — 54 left over from before `close_session` projected per
        # sitting rather than from the resumed state, which carries everything a
        # learner has ever said. They are real utterances filed against sittings
        # they were not made in, and `LearnerStore.utterances` is what recall is
        # meant to read across sittings, so they would have come back as history
        # that did not happen.
        #
        # Nothing is lost by rebuilding: checked before doing it, every utterance
        # in the table is in the log and none exists only in the table.
        self._db.execute("DELETE FROM turn")
        self._db.executemany(
            _INSERT_TURN,
            [
                _turn_row(row["session_id"], row["version"], json.loads(row["evidence"]))
                for row in self._db.execute(
                    "SELECT session_id, version, evidence FROM event "
                    "WHERE cause = 'tutor'"
                ).fetchall()
            ],
        )
        self._db.execute("DELETE FROM utterance")
        self._db.executemany(
            _INSERT_UTTERANCE,
            [
                _utterance_row(row["session_id"], row["version"], evidence)
                for row in self._db.execute(
                    "SELECT session_id, version, evidence FROM event "
                    "WHERE cause = 'annotation'"
                ).fetchall()
                if "reflection" in (evidence := json.loads(row["evidence"]))
            ],
        )
        self._db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
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
            "INSERT INTO event (session_id, version, cause, summary, evidence, "
            "concept_id, item_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    r.version,
                    r.cause,
                    r.summary,
                    json.dumps(r.evidence, default=str),
                    r.evidence.get("concept_id"),
                    r.evidence.get("item_id"),
                )
                for r in audit_log
            ],
        )
        # What the system said back. Same source and same per-sitting scope as
        # the utterances below — the audit log, never the state, for the reason
        # written there.
        self._db.executemany(
            _INSERT_TURN,
            [
                _turn_row(session_id, r.version, r.evidence)
                for r in audit_log
                if r.cause == "tutor"
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
            _INSERT_UTTERANCE,
            [
                _utterance_row(session_id, r.version, r.evidence)
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
        return list(self._db.execute(sql + " ORDER BY ss.seq, u.version, u.utterance_id", params))

    def turns(
        self, learner_id: str, arm: str, concept_id: str | None = None,
        move: str | None = None,
    ) -> list[sqlite3.Row]:
        """Everything the system said to this learner, across sittings.

        The counterpart to :meth:`utterances`. Between them a sitting can be
        read as a conversation rather than as a list of verdicts — which is what
        it was before turns were stored at all, when a transcript held every
        answer the learner gave and nothing the system replied.

        ``move`` narrows to one kind. ``move='explain'`` is what answers "has
        this learner been taught this concept before, and how was it put" — the
        question a second lesson has to ask before repeating the first one.
        """
        sql = (
            "SELECT t.*, ss.seq FROM turn t "
            "JOIN session ss ON ss.session_id = t.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ?"
        )
        params: list[object] = [learner_id, arm]
        if concept_id is not None:
            sql += " AND t.concept_id = ?"
            params.append(concept_id)
        if move is not None:
            sql += " AND t.move = ?"
            params.append(move)
        return list(self._db.execute(sql + " ORDER BY ss.seq, t.version, t.turn_id", params))

    def audit(self, learner_id: str, arm: str) -> list[AuditRecord]:
        """This learner's whole history, back as audit records.

        The rows carry ``evidence`` as JSON text, which is the shape a database
        needs and the wrong shape for anything that reads a sitting back. Here
        rather than at each call site so the parsing happens once — and because
        a caller that has to remember to json.loads a column will eventually
        forget.

        In version order across sittings, so an ordering — who spoke when — is
        still an ordering after the round trip.
        """
        return [
            AuditRecord(
                version=int(row["version"]),
                cause=row["cause"],
                summary=row["summary"],
                evidence=json.loads(row["evidence"]) if row["evidence"] else {},
            )
            for row in self.events(learner_id, arm)
        ]

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
