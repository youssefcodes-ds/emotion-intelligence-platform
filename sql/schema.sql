-- ============================================================================
-- Emotion Intelligence Platform - analytical schema
--
-- Target: SQLite (no server, file lives at data/emotion.db).
-- Every statement is portable to PostgreSQL apart from the AUTOINCREMENT
-- keyword, which would become SERIAL / IDENTITY there.
--
-- Design notes
--   * emotions is a lookup table rather than a free-text column, so a typo in
--     a label name cannot enter the database at all.
--   * predictions stores one row per (message, model) so several models can be
--     compared over identical rows - that is what the comparison table needs.
--   * message_topics is a junction table: a message can carry several topics
--     with different weights, which a single topic_id column could not express.
-- ============================================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS inference_logs;
DROP TABLE IF EXISTS message_topics;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS topics;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS emotions;

-- ---------------------------------------------------------------- lookup ----
CREATE TABLE emotions (
    label_id    INTEGER PRIMARY KEY,
    label_name  TEXT    NOT NULL UNIQUE,
    valence     TEXT    NOT NULL CHECK (valence IN ('positive', 'negative', 'ambiguous'))
);

-- ----------------------------------------------------------------- core -----
CREATE TABLE messages (
    id            TEXT    PRIMARY KEY,
    split         TEXT    NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    text          TEXT    NOT NULL,
    text_clean    TEXT    NOT NULL,
    label         INTEGER NOT NULL REFERENCES emotions (label_id),
    n_words       INTEGER NOT NULL CHECK (n_words >= 0),
    n_chars       INTEGER NOT NULL CHECK (n_chars >= 0),
    has_negation  INTEGER NOT NULL CHECK (has_negation IN (0, 1))
);

CREATE INDEX idx_messages_split ON messages (split);
CREATE INDEX idx_messages_label ON messages (label);

-- ------------------------------------------------------------ registry ------
CREATE TABLE models (
    model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL,
    family       TEXT NOT NULL CHECK (family IN ('classical', 'deep', 'transformer', 'baseline')),
    version      TEXT NOT NULL,
    trained_at   TEXT NOT NULL DEFAULT (datetime('now')),
    notes        TEXT,
    UNIQUE (model_name, version)
);

CREATE TABLE predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id       TEXT    NOT NULL REFERENCES messages (id),
    model_id         INTEGER NOT NULL REFERENCES models (model_id),
    predicted_label  INTEGER NOT NULL REFERENCES emotions (label_id),
    confidence       REAL    CHECK (confidence BETWEEN 0 AND 1),
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (message_id, model_id)
);

CREATE INDEX idx_predictions_model ON predictions (model_id);
CREATE INDEX idx_predictions_message ON predictions (message_id);

-- -------------------------------------------------------------- topics ------
CREATE TABLE topics (
    topic_id     INTEGER PRIMARY KEY,
    topic_label  TEXT NOT NULL,
    keywords     TEXT
);

CREATE TABLE message_topics (
    message_id  TEXT    NOT NULL REFERENCES messages (id),
    topic_id    INTEGER NOT NULL REFERENCES topics (topic_id),
    weight      REAL    NOT NULL CHECK (weight BETWEEN 0 AND 1),
    PRIMARY KEY (message_id, topic_id)
);

-- ---------------------------------------------------------------- logs ------
CREATE TABLE inference_logs (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    model_id      INTEGER REFERENCES models (model_id),
    source        TEXT    NOT NULL CHECK (source IN ('streamlit_text', 'streamlit_csv', 'batch', 'test')),
    input_chars   INTEGER NOT NULL,
    latency_ms    REAL    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'ok'
);

CREATE INDEX idx_logs_model ON inference_logs (model_id);

-- ---------------------------------------------------------------- seed ------
INSERT INTO emotions (label_id, label_name, valence) VALUES
    (0, 'sadness',  'negative'),
    (1, 'joy',      'positive'),
    (2, 'love',     'positive'),
    (3, 'anger',    'negative'),
    (4, 'fear',     'negative'),
    (5, 'surprise', 'ambiguous');
