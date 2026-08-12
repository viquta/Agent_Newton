-- Persistent learner state, across sessions.
--
-- The `state` blob is authoritative: it is the serialised LearnerState the
-- session resumes from. The `event` and `utterance` tables are *projections* of
-- it, written at session end so a sequence can be queried without deserialising
-- every state — which is the whole reason a database was chosen over a file per
-- learner. Where they disagree, the blob is right.
--
-- `profile` is ground truth and is read through a different class in a different
-- module. See store/ground_truth.py for why.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS learner (
    learner_id  TEXT PRIMARY KEY,
    -- `simulated` or `human`. Kept so the two can be compared later without
    -- inferring the kind from whether a profile row happens to exist.
    kind        TEXT NOT NULL,
    domain      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- One sitting. A learner is run through both arms, so the sequence is keyed on
-- (learner, arm): the same person under two architectures is two histories, and
-- resuming must never cross them.
CREATE TABLE IF NOT EXISTS session (
    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id    TEXT    NOT NULL REFERENCES learner(learner_id),
    arm           TEXT    NOT NULL,
    seq           INTEGER NOT NULL,
    -- Gap before this session. Drives decay; 0 for the first.
    elapsed_days  REAL    NOT NULL DEFAULT 0,
    run_id        TEXT,
    config_hash   TEXT    NOT NULL,
    -- Recorded per session because pooling across different decay is refused,
    -- and a sequence may not silently change it midway.
    decay_half_life_days REAL,
    started_at    TEXT    NOT NULL,
    ended_at      TEXT,
    stop_reason   TEXT,
    UNIQUE (learner_id, arm, seq)
);

CREATE TABLE IF NOT EXISTS state (
    session_id    INTEGER PRIMARY KEY REFERENCES session(session_id),
    learner_state TEXT NOT NULL,
    -- A planner's own bookkeeping, for the ones that declare `Resumable`. Only
    -- the decoupled planner has any: its position in the syllabus walk is the
    -- only progress signal it has, and without carrying it a returning learner
    -- would restart at the first concept every session. The coupled planner
    -- stores nothing here, because everything it needs is in learner_state.
    planner_state TEXT
);

CREATE TABLE IF NOT EXISTS event (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES session(session_id),
    version    INTEGER NOT NULL,
    cause      TEXT    NOT NULL,
    summary    TEXT    NOT NULL,
    evidence   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS utterance (
    utterance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES session(session_id),
    kind         TEXT    NOT NULL,
    item_id      TEXT    NOT NULL,
    concept_id   TEXT    NOT NULL,
    text         TEXT    NOT NULL
);

-- GROUND TRUTH. What the simulated learner actually holds, and how strongly.
-- No agent may read this; see store/ground_truth.py.
CREATE TABLE IF NOT EXISTS profile (
    session_id INTEGER PRIMARY KEY REFERENCES session(session_id),
    firing     TEXT NOT NULL,
    initial    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS event_by_session     ON event(session_id);
CREATE INDEX IF NOT EXISTS event_by_cause       ON event(session_id, cause);
CREATE INDEX IF NOT EXISTS utterance_by_concept ON utterance(concept_id);
CREATE INDEX IF NOT EXISTS session_by_learner   ON session(learner_id, arm, seq);
