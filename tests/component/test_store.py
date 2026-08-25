"""Persistence across sessions.

The store exists so a plan can outlive the sitting that made it. What is tested
here is mostly the ways that could go wrong quietly: a learner resuming into the
wrong arm's history, a profile inherited across the comparison, or a state that
round-trips lossily and silently changes what the planner sees.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent_newton.config import Config, SimulatorConfig
from agent_newton.core.simulator import sample_profile
from agent_newton.core.state.schema import (
    AuditRecord,
    LearnerState,
    Plan,
    Utterance,
)
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry
from agent_newton.domains.base import VerificationResult, Verdict
from agent_newton.store import LearnerStore
from agent_newton.store.ground_truth import ProfileStore


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


@pytest.fixture
def store(tmp_path):
    with LearnerStore(tmp_path / "learners.db") as db:
        yield db


def worked_state(toy, **mastery: float) -> LearnerState:
    config = Config.model_validate({"domain": "toy_algebra"})
    board = new_blackboard("L1", seed=1, graph=toy.concepts, config=config)
    board.record_observation(
        item_id="i1",
        concept_id="distribute",
        result=VerificationResult(Verdict.INCORRECT, "a"),
        misconception_label="distribute_first_term_only",
    )
    board.record_reflection("I was unsure", "i1", "distribute", kind="working")
    board.record_plan(Plan(goal="solve_linear"))
    board.state.mastery.update(mastery)
    return board.state


class TestSessionsAndResuming:
    def test_a_learner_with_no_history_resumes_nothing(self, store) -> None:
        # None means "never sat one", which is deliberately not the same as a
        # learner whose state happens to be empty.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert store.latest_state("L1", "coupled") is None

    def test_a_state_round_trips(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        original = worked_state(toy, distribute=0.42)
        store.close_session(session, state=original, stop_reason="budget_spent")

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert restored.mastery == original.mastery
        assert restored.plan == original.plan
        assert restored.error_trace == original.error_trace
        assert restored.reflections == original.reflections
        assert restored.t == original.t

    def test_the_latest_session_is_the_one_resumed(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        for value in (0.2, 0.5, 0.8):
            session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
            store.close_session(session, state=worked_state(toy, distribute=value))

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert restored.mastery["distribute"] == 0.8

    def test_the_sequence_index_advances(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert store.next_index("L1", "coupled") == 0
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        assert store.next_index("L1", "coupled") == 1

    def test_registering_a_learner_twice_is_harmless(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert len(store.learners()) == 1


class TestTheArmsKeepSeparateHistories:
    """The same person under two architectures is two histories.

    If one arm resumed into the other's state, the paired comparison would be
    destroyed — and it would not look broken, it would look like a result.
    """

    def test_state_does_not_cross_arms(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.close_session(session, state=worked_state(toy, distribute=0.9))

        assert store.latest_state("L1", "coupled") is not None
        assert store.latest_state("L1", "decoupled") is None

    def test_sequence_indices_are_per_arm(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        assert store.next_index("L1", "coupled") == 2
        assert store.next_index("L1", "decoupled") == 0

    def test_profiles_do_not_cross_arms(self, store, tmp_path, toy) -> None:
        # A profile remediated in one arm must never be inherited by the other:
        # the decoupled arm would start with the coupled arm's teaching already
        # applied.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        store.close_session(session, state=worked_state(toy))

        with ProfileStore(store.path) as profiles:
            profiles.save(session, profile)
            assert profiles.latest("L1", "coupled") is not None
            assert profiles.latest("L1", "decoupled") is None


class TestProfilesArePersistedWithTheirStartingPoint:
    def test_firing_and_initial_both_survive(self, store, toy) -> None:
        # Remediation lowers firing and forgetting raises it, so without the
        # starting point there is nothing to report remediation as a proportion
        # of — `remediation_ratio` would silently rebase on the current values.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        label = next(iter(profile.firing))
        profile.remediate(label, 0.5)
        store.close_session(session, state=worked_state(toy))

        with ProfileStore(store.path) as profiles:
            profiles.save(session, profile)
            restored = profiles.latest("L1", "coupled")

        assert restored is not None
        assert restored.firing == profile.firing
        assert dict(restored.initial) == dict(profile.initial)
        assert restored.remediation_ratio() == pytest.approx(profile.remediation_ratio())


class TestTheQueryableProjections:
    """⚠️ Each row must belong to the sitting it was said in.

    The utterance rows were projected from ``state.reflections``, and a resumed
    state carries every word the learner has ever said — so each new session
    wrote the whole history again under its own id. One real learner's table
    held 81 rows and 27 distinct texts across three sittings, two of which had
    said nothing at all. Nothing read the table yet, which is the only reason it
    never produced a wrong number.
    """

    def _sitting(self, store, toy, said: str | None) -> int:
        """One sitting that may or may not produce a word of its own.

        Built through the board, because that is how an utterance actually comes
        to exist: recorded on the blackboard, which writes it to the audit log,
        which is per sitting. The state is the resumed one either way.
        """
        config = Config.model_validate({"domain": "toy_algebra"})
        board = new_blackboard("L1", 1, toy.concepts, config)
        if said is not None:
            board.record_reflection(said, "i1", "distribute", kind="working")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.close_session(
            session, state=worked_state(toy), audit_log=board.audit_log
        )
        return session

    def test_utterances_are_queryable_across_sessions(self, store, toy) -> None:
        # The reason a database was chosen: planning is meant to read what the
        # learner said in *previous* sittings, not only this one.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        self._sitting(store, toy, "I was unsure about the second step")
        self._sitting(store, toy, "I think I have it now")

        said = store.utterances("L1", "coupled")
        assert len(said) == 2
        assert {row["concept_id"] for row in said} == {"distribute"}
        assert [row["text"] for row in said] == [
            "I was unsure about the second step",
            "I think I have it now",
        ]
        assert store.utterances("L1", "coupled", concept_id="nothing_here") == []

    def test_a_sitting_that_said_nothing_contributes_nothing(self, store, toy) -> None:
        # The resumed state still carries the earlier sitting's words, and they
        # must not be written a second time under this session's id.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        self._sitting(store, toy, "I was unsure about the second step")
        second = self._sitting(store, toy, None)

        said = store.utterances("L1", "coupled")
        assert len(said) == 1
        assert [row for row in said if row["session_id"] == second] == []

    def test_the_same_words_are_not_counted_twice(self, store, toy) -> None:
        # The shape the defect took: rows kept growing while distinct texts did
        # not, so anything counting what a learner said would have multiplied it
        # by the number of sittings since.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        for _ in range(3):
            self._sitting(store, toy, None)
        self._sitting(store, toy, "the only thing I ever said")

        said = store.utterances("L1", "coupled")
        assert len(said) == len({row["text"] for row in said}) == 1

    def test_events_are_queryable_by_cause(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        state = worked_state(toy)
        board = new_blackboard(
            "L1", 1, toy.concepts, Config.model_validate({"domain": "toy_algebra"})
        )
        store.close_session(session, state=state, audit_log=board.audit_log)

        assert list(store.events("L1", "coupled", cause="annotation"))

    def test_the_state_blob_is_authoritative(self, store, toy) -> None:
        # The projections are for querying. Anything the session resumes from
        # comes back out of the blob, so a projection that drifted could not
        # silently change what the planner sees.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        state = worked_state(toy)
        state.reflections.append(
            Utterance(text="only in the blob", item_id="i9", concept_id="distribute")
        )
        store.close_session(session, state=state)

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert "only in the blob" in [u.text for u in restored.reflections]


class TestTheStoreIsAFile:
    def test_it_survives_being_closed_and_reopened(self, tmp_path, toy) -> None:
        path = tmp_path / "nested" / "learners.db"
        with LearnerStore(path) as first:
            first.ensure_learner("L1", "human", "toy_algebra")
            session = first.open_session(learner_id="L1", arm="coupled", config_hash="h")
            first.close_session(session, state=worked_state(toy, distribute=0.66))

        with LearnerStore(path) as second:
            restored = second.latest_state("L1", "coupled")
            assert restored is not None
            assert restored.mastery["distribute"] == 0.66

    def test_humans_and_simulated_learners_are_distinguishable(self, store) -> None:
        # So the two can be compared later without inferring the kind from
        # whether a profile row happens to exist.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.ensure_learner("H1", "human", "toy_algebra")
        assert [r["learner_id"] for r in store.learners(kind="human")] == ["H1"]
        assert [r["learner_id"] for r in store.learners(kind="simulated")] == ["L1"]


class TestResumingAcrossAContentChange:
    """`assert_poolable` refuses to pool runs across a content change. Nothing
    refused to *resume* a learner across one, and it is the same risk.

    Mastery is keyed by concept id and the error trace by misconception label, so
    renaming or removing either leaves stored state pointing at content that no
    longer exists. The concrete failure: `Session._cross_concept` looks every
    trace label up in the catalogue and a missing id raises — at outcome time,
    after the sitting, when the work is already done.
    """

    def _store(self, tmp_path):  # noqa: ANN001
        from agent_newton.store import LearnerStore

        return LearnerStore(tmp_path / "learners.db")

    def _sat(self, store, hashes):  # noqa: ANN001
        from agent_newton.core.state.schema import LearnerState

        store.ensure_learner("v", "human", "calculus")
        session_id = store.open_session(
            learner_id="v", arm="coupled", config_hash="c0", content_hashes=hashes
        )
        store.close_session(session_id, state=LearnerState(learner_id="v", seed=1))
        return session_id

    def test_a_changed_catalogue_is_reported(self, tmp_path) -> None:
        store = self._store(tmp_path)
        self._sat(store, {"catalogue_hash": "aaa", "item_bank_hash": "bbb",
                          "concept_graph_hash": "ccc"})
        drift = store.content_drift("v", "coupled", {
            "catalogue_hash": "ZZZ", "item_bank_hash": "bbb",
            "concept_graph_hash": "ccc",
        })
        assert "catalogue_hash" in drift
        assert drift["catalogue_hash"] == ("aaa", "ZZZ")
        store.close()

    def test_unchanged_content_reports_nothing(self, tmp_path) -> None:
        store = self._store(tmp_path)
        hashes = {"catalogue_hash": "aaa", "item_bank_hash": "bbb",
                  "concept_graph_hash": "ccc"}
        self._sat(store, hashes)
        assert store.content_drift("v", "coupled", hashes) == {}
        store.close()

    def test_a_session_written_before_the_columns_reports_nothing(self, tmp_path) -> None:
        # Unverifiable and unchanged are different, and only one is worth warning
        # about. A learner whose history predates these columns must not be told
        # the subject matter moved when nobody recorded whether it did.
        store = self._store(tmp_path)
        self._sat(store, None)
        assert store.content_drift("v", "coupled", {"catalogue_hash": "ZZZ"}) == {}
        store.close()

    def test_a_learner_who_never_sat_reports_nothing(self, tmp_path) -> None:
        store = self._store(tmp_path)
        assert store.content_drift("nobody", "coupled", {"catalogue_hash": "ZZZ"}) == {}
        store.close()

    def test_the_migration_adds_the_columns_to_an_existing_store(self, tmp_path) -> None:
        # `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so without
        # the migration a new column never reaches a store that already holds
        # sittings — and those cannot be regenerated.
        import sqlite3

        path = tmp_path / "old.db"
        db = sqlite3.connect(path)
        db.executescript(
            "CREATE TABLE learner (learner_id TEXT PRIMARY KEY, kind TEXT, "
            "domain TEXT, created_at TEXT);"
            "CREATE TABLE session (session_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "learner_id TEXT, arm TEXT, seq INTEGER, elapsed_days REAL, run_id TEXT, "
            "config_hash TEXT, decay_half_life_days REAL, started_at TEXT, "
            "ended_at TEXT, stop_reason TEXT);"
        )
        db.commit()
        db.close()

        store = self._store(tmp_path.joinpath())  # same dir, different file below
        store.close()
        from agent_newton.store import LearnerStore

        migrated = LearnerStore(path)
        columns = {r[1] for r in migrated._db.execute("PRAGMA table_info(session)")}
        assert {"catalogue_hash", "item_bank_hash", "concept_graph_hash"} <= columns
        migrated.close()

    def test_the_real_failure_it_guards(self) -> None:
        # Not a hypothetical: a label the catalogue no longer has raises, and
        # `_cross_concept` calls exactly this on every error-trace entry — at
        # outcome time, after the sitting.
        from agent_newton.domains import registry
        from agent_newton.domains.base import DomainError

        calculus = registry.load_domain("calculus")
        with pytest.raises(DomainError):
            calculus.misconceptions.get("a_label_that_was_removed")


class TestALearnerIdIsAlsoADirectoryName:
    """Safe for SQL is not the same as safe as a name.

    Parameterised queries make any id safe against injection — every hostile
    string was tried and all of them round-tripped with the tables intact. But
    `agent-newton history <learner>` writes `results/history_<learner>_<arm>/`, so
    `history '../../../tmp/x'` wrote *outside* the results tree, and an id
    containing a slash silently created a directory tree rather than a directory.

    Enforced on the identity rather than patched where each path is built, so a
    learner that cannot be named cannot be created.
    """

    @pytest.mark.parametrize("learner_id", ["victor", "L0000", "probe", "a-b_1", "X9"])
    def test_the_ids_the_project_uses_are_accepted(self, learner_id: str) -> None:
        from agent_newton.store import check_learner_id

        assert check_learner_id(learner_id) == learner_id

    @pytest.mark.parametrize(
        "learner_id",
        [
            "../../../tmp/escaped",  # the traversal
            "a/b",                   # a nested directory
            ".hidden",               # a dotfile
            "",                      # nothing at all
            " x",                    # leading whitespace
            "o'brien",               # a quote
            "a b",                   # a space
            "x\x00y",                # a nul byte
        ],
    )
    def test_ids_that_are_not_usable_as_names_are_refused(self, learner_id: str) -> None:
        from agent_newton.store import check_learner_id

        with pytest.raises(ValueError, match="not usable"):
            check_learner_id(learner_id)

    def test_the_store_refuses_to_create_one(self, tmp_path) -> None:
        from agent_newton.store import LearnerStore

        store = LearnerStore(tmp_path / "t.db")
        with pytest.raises(ValueError, match="not usable"):
            store.ensure_learner("../../etc/passwd", "human", "calculus")
        assert store.learners() == []
        store.close()

    def test_the_traversal_it_prevents(self) -> None:
        # The concrete escape, asserted rather than described.
        from pathlib import Path

        base = Path("results").resolve()
        escaped = (Path("results") / "history_../../../tmp/escaped_coupled").resolve()
        assert base not in escaped.parents, "this path no longer escapes; test is stale"

    def test_sql_was_never_the_problem(self, tmp_path) -> None:
        # Recorded so nobody 'fixes' this by escaping quotes: the queries are
        # parameterised and an injection attempt is stored as a literal string.
        from agent_newton.store import LearnerStore

        store = LearnerStore(tmp_path / "t.db")
        store.ensure_learner("dropper", "human", "calculus")
        rows = store._db.execute(
            "SELECT learner_id FROM learner WHERE learner_id = ?",
            ['"; DROP TABLE learner;--'],
        ).fetchall()
        assert rows == []
        assert store.learners(), "the table survived the lookup"
        store.close()


class TestTheEventTableCanBeRead:
    """``evidence`` is a JSON blob, and a table you have to un-JSON is a table
    nobody reads.

    The record of a sitting is where defects get found — the scaffolding
    collapse in a human sitting was found by asking which support levels a
    learner had ever been given, and answering that took a wrapper around the
    tutor. The keys almost every cause carries are columns now, and the blob is
    still there and still authoritative.
    """

    def _sitting(self, store: LearnerStore, log) -> int:
        store.ensure_learner("L1", "human", "calculus")
        session_id = store.open_session(
            learner_id="L1", arm="coupled", config_hash="h"
        )
        store.close_session(
            session_id, state=LearnerState(learner_id="L1", seed=1), audit_log=log
        )
        return session_id

    def _turn(self, **evidence) -> AuditRecord:
        base = {
            "item_id": "ca_pow_p1",
            "concept_id": "power_rule",
            "move": "hint",
            "level": "nudge",
            "targets": None,
            "text": "look again at the exponent",
            "mastery": 0.4,
            "prior_failures": 1,
        }
        base.update(evidence)
        return AuditRecord(
            version=1, cause="tutor", summary="hint at nudge", evidence=base
        )

    def test_the_common_keys_are_columns(self, store) -> None:
        self._sitting(store, [self._turn()])
        row = store._db.execute(
            "SELECT concept_id, item_id FROM event WHERE cause = 'tutor'"
        ).fetchone()
        assert row["concept_id"] == "power_rule"
        assert row["item_id"] == "ca_pow_p1"

    def test_the_blob_is_still_there(self, store) -> None:
        # Nothing is dropped. An audit record may carry anything, and a schema
        # keeping only the columns someone thought of would quietly lose the
        # rest.
        self._sitting(store, [self._turn(something_unusual=7)])
        stored = json.loads(
            store._db.execute("SELECT evidence FROM event").fetchone()["evidence"]
        )
        assert stored["something_unusual"] == 7

    def test_a_record_with_no_concept_leaves_the_column_empty(self, store) -> None:
        # Decay names a concept; an exhausted item budget names neither. NULL is
        # the honest value, and a placeholder would read as a real id.
        self._sitting(
            store,
            [
                AuditRecord(
                    version=1,
                    cause="annotation",
                    summary="item budget spent",
                    evidence={"items_given": 10},
                )
            ],
        )
        row = store._db.execute("SELECT concept_id, item_id FROM event").fetchone()
        assert row["concept_id"] is None
        assert row["item_id"] is None


class TestTurnsAreProjectedLikeUtterances:
    """The counterpart to the utterance table, and it closes the same gap.

    A transcript once held every answer the learner gave and nothing the system
    replied. Turns are recorded now, but only inside the blob — so reading a
    sitting back still meant parsing JSON.
    """

    def _sat(self, store: LearnerStore, *turns) -> None:
        store.ensure_learner("L1", "human", "calculus")
        session_id = store.open_session(
            learner_id="L1", arm="coupled", config_hash="h"
        )
        store.close_session(
            session_id,
            state=LearnerState(learner_id="L1", seed=1),
            audit_log=list(turns),
        )

    def _turn(self, **evidence) -> AuditRecord:
        base = {
            "item_id": "ca_pow_p1",
            "concept_id": "power_rule",
            "move": "hint",
            "level": "nudge",
            "targets": None,
            "text": "look again",
            "mastery": 0.4,
            "prior_failures": 1,
        }
        base.update(evidence)
        return AuditRecord(version=1, cause="tutor", summary="a turn", evidence=base)

    def test_a_turn_becomes_a_row(self, store) -> None:
        self._sat(store, self._turn())
        [row] = store.turns("L1", "coupled")
        assert row["move"] == "hint"
        assert row["level"] == "nudge"
        assert row["text"] == "look again"
        assert row["mastery"] == pytest.approx(0.4)
        assert row["prior_failures"] == 1

    def test_nothing_but_a_tutor_record_becomes_one(self, store) -> None:
        self._sat(
            store,
            AuditRecord(version=1, cause="annotation", summary="something", evidence={}),
        )
        assert store.turns("L1", "coupled") == []

    def test_it_can_be_narrowed_to_one_move(self, store) -> None:
        # `move='explain'` is what answers "has this learner been taught this
        # concept before, and how was it put" — the question a second lesson has
        # to ask before repeating the first one.
        self._sat(
            store,
            self._turn(),
            self._turn(move="explain", level="plain", text="a derivative is..."),
        )
        [lesson] = store.turns("L1", "coupled", move="explain")
        assert lesson["level"] == "plain"

    def test_it_can_be_narrowed_to_one_concept(self, store) -> None:
        self._sat(store, self._turn(), self._turn(concept_id="chain_rule"))
        assert len(store.turns("L1", "coupled", concept_id="power_rule")) == 1

    def test_only_this_sitting_is_projected(self, store) -> None:
        """⚠️ The 81-row bug, which the utterance table already carries a
        warning about.

        The state is resumed whole and carries everything the learner has ever
        said, so projecting *it* wrote the entire history under each new session
        id. The audit log is per sitting, which is what a per-sitting projection
        needs — and it is the same source the event rows come from.
        """
        store.ensure_learner("L1", "human", "calculus")
        for _ in range(3):
            session_id = store.open_session(
                learner_id="L1", arm="coupled", config_hash="h"
            )
            store.close_session(
                session_id,
                state=LearnerState(learner_id="L1", seed=1),
                audit_log=[self._turn()],
            )
        assert len(store.turns("L1", "coupled")) == 3

    def test_a_remediation_target_survives_and_nothing_else_carries_one(
        self, store
    ) -> None:
        # Load-bearing rather than incidental: `remediation_ratio` counts what a
        # hint aimed at, so a target on a lesson would credit it with
        # remediation it did not do.
        self._sat(
            store,
            self._turn(move="remediate", targets="power_rule_forgets_decrement"),
            self._turn(move="explain", level="plain"),
        )
        by_move = {row["move"]: row["targets"] for row in store.turns("L1", "coupled")}
        assert by_move["remediate"] == "power_rule_forgets_decrement"
        assert by_move["explain"] is None


class TestTheBackfillRunsOnceOverHistoryThatCannotBeRegenerated:
    """A store holds sittings a person produced once, at a keyboard.

    Adding a column and leaving every existing row NULL would make the new shape
    useless for exactly the history it was added to make readable.
    """

    def _store_with_an_old_row(self, tmp_path) -> Path:
        """A store shaped the way one written before this change would be."""
        path = tmp_path / "old.db"
        db = sqlite3.connect(path)
        db.executescript(
            """
            CREATE TABLE learner (learner_id TEXT PRIMARY KEY, kind TEXT,
                domain TEXT, created_at TEXT);
            CREATE TABLE session (session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT, arm TEXT, seq INTEGER, elapsed_days REAL
                DEFAULT 0, run_id TEXT, config_hash TEXT, decay_half_life_days
                REAL, started_at TEXT, ended_at TEXT, stop_reason TEXT,
                UNIQUE (learner_id, arm, seq));
            CREATE TABLE state (session_id INTEGER PRIMARY KEY,
                learner_state TEXT, planner_state TEXT);
            CREATE TABLE event (event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, version INTEGER, cause TEXT, summary TEXT,
                evidence TEXT);
            CREATE TABLE utterance (utterance_id INTEGER PRIMARY KEY
                AUTOINCREMENT, session_id INTEGER, kind TEXT, item_id TEXT,
                concept_id TEXT, text TEXT);
            CREATE TABLE profile (session_id INTEGER PRIMARY KEY, firing TEXT,
                initial TEXT);
            """
        )
        db.execute(
            "INSERT INTO learner VALUES ('L1', 'human', 'calculus', 'then')"
        )
        db.execute(
            "INSERT INTO session (learner_id, arm, seq, config_hash, started_at) "
            "VALUES ('L1', 'coupled', 0, 'h', 'then')"
        )
        db.execute(
            "INSERT INTO event (session_id, version, cause, summary, evidence) "
            "VALUES (1, 1, 'tutor', 'a turn', ?)",
            (
                json.dumps(
                    {
                        "item_id": "ca_pow_p1",
                        "concept_id": "power_rule",
                        "move": "remediate",
                        "level": "worked_step",
                        "targets": "power_rule_forgets_decrement",
                        "text": "bring the exponent down",
                    }
                ),
            ),
        )
        db.commit()
        db.close()
        return path

    def test_an_old_turn_becomes_readable(self, tmp_path) -> None:
        store = LearnerStore(self._store_with_an_old_row(tmp_path))
        [row] = store.turns("L1", "coupled")
        assert row["move"] == "remediate"
        assert row["level"] == "worked_step"
        assert row["targets"] == "power_rule_forgets_decrement"
        store.close()

    def test_an_old_event_gains_its_concept(self, tmp_path) -> None:
        store = LearnerStore(self._store_with_an_old_row(tmp_path))
        row = store._db.execute("SELECT concept_id FROM event").fetchone()
        assert row["concept_id"] == "power_rule"
        store.close()

    def test_keys_that_did_not_exist_yet_do_not_stop_it(self, tmp_path) -> None:
        # `mastery` and `prior_failures` were added after the first sittings. A
        # backfill that raised on them would refuse to migrate exactly the
        # history worth migrating.
        store = LearnerStore(self._store_with_an_old_row(tmp_path))
        [row] = store.turns("L1", "coupled")
        assert row["mastery"] == pytest.approx(0.0)
        assert row["prior_failures"] == 0
        store.close()

    def test_reopening_does_not_duplicate_anything(self, tmp_path) -> None:
        """Guarded on ``PRAGMA user_version`` rather than on the rows being
        empty.

        An emptiness check would re-run on any store that genuinely has no
        turns, and would stop being a migration and start being a repair that
        fires at random.
        """
        path = self._store_with_an_old_row(tmp_path)
        for _ in range(3):
            store = LearnerStore(path)
            assert len(store.turns("L1", "coupled")) == 1
            store.close()

    def test_a_fresh_store_is_already_at_the_current_version(self, tmp_path) -> None:
        store = LearnerStore(tmp_path / "fresh.db")
        assert store._db.execute("PRAGMA user_version").fetchone()[0] >= 1
        store.close()
