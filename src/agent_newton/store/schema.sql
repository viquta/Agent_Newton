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
    -- Domain content the state was written against. Denormalised on purpose,
    -- for the same reason `decay_half_life_days` is: what matters at read time
    -- is the value actually used, not a pointer to a config that may since have
    -- changed on disk. `assert_poolable` already refuses to *pool* runs across a
    -- content change; without these, nothing refuses to *resume* a learner
    -- across one, and the identical risk went unguarded on the other path.
    -- NULL means a session written before these existed: unverifiable rather
    -- than mismatched, and treated as such.
    catalogue_hash     TEXT,
    item_bank_hash     TEXT,
    concept_graph_hash TEXT,
    -- What may be shown beside a question. NULL for a session written before
    -- resources existed, and NULL for a domain that offers none -- the two are
    -- indistinguishable here and neither is a mismatch, so `content_drift`
    -- treats both as unverifiable rather than as changed.
    resources_hash     TEXT,
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
    -- The whole evidence dict, as JSON. Authoritative, and never dropped: an
    -- audit record may carry anything, and a schema that only kept the columns
    -- someone thought of would quietly lose the rest.
    evidence   TEXT    NOT NULL,
    -- The two keys almost every cause carries, lifted out so the table can be
    -- read. Reading `evidence` meant parsing JSON to answer "what happened on
    -- the chain rule", which is the question this table exists for. NULL where
    -- the record genuinely has no such key -- decay names a concept, an item
    -- budget names neither.
    concept_id TEXT,
    item_id    TEXT
);

-- What the system said back, projected out of the `tutor` cause.
--
-- The same shape as `utterance`, and for the same reason: the record of a
-- sitting is what a defect gets found in, and one that has to be un-JSONed
-- first does not get read. The scaffolding collapse in a human sitting was
-- found by asking which levels a learner had ever been given, and answering
-- that took a wrapper around the tutor; it is a SELECT now.
--
-- `evidence` on the event row remains authoritative. Where the two disagree,
-- this one is the projection.
CREATE TABLE IF NOT EXISTS turn (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL REFERENCES session(session_id),
    item_id        TEXT    NOT NULL,
    concept_id     TEXT    NOT NULL,
    -- hint / reflect / remediate / present / explain.
    move           TEXT    NOT NULL,
    -- The support level for a hint, the style for a lesson, the depth for
    -- material shown beside a question. One column because they answer the
    -- same question -- "which of the things this move can be was it" -- and
    -- three columns would have been three mostly-NULL ones.
    level          TEXT    NOT NULL,
    -- The misconception a remediation aimed at. NULL for every other move, and
    -- that is load-bearing rather than incidental: `remediation_ratio` counts
    -- what a hint aimed at, so a target on a lesson or a reflection would
    -- credit it with remediation it did not do.
    targets        TEXT,
    text           TEXT    NOT NULL,
    -- The two inputs the level was chosen from. Stored because the level alone
    -- could not be argued with: two sittings ran entirely at `worked_step` and
    -- the transcripts could say only that.
    mastery        REAL    NOT NULL DEFAULT 0,
    prior_failures INTEGER NOT NULL DEFAULT 0
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
CREATE INDEX IF NOT EXISTS turn_by_session      ON turn(session_id);
CREATE INDEX IF NOT EXISTS turn_by_concept      ON turn(concept_id, move);
CREATE INDEX IF NOT EXISTS event_by_cause       ON event(session_id, cause);
CREATE INDEX IF NOT EXISTS utterance_by_concept ON utterance(concept_id);
CREATE INDEX IF NOT EXISTS session_by_learner   ON session(learner_id, arm, seq);
