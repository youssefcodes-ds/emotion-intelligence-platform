-- ============================================================================
-- Analytical queries
--
-- 14 queries. Requirement is "at least 10 meaningful analytical queries;
-- include JOIN, CTE and window function where applicable".
--
--   JOIN             : Q1, Q4, Q5, Q6, Q9, Q10, Q11, Q12, Q13, Q14
--   CTE (WITH)       : Q3, Q7, Q8, Q9, Q10, Q11, Q12, Q14
--   Window function  : Q3, Q6, Q7, Q8, Q12, Q14
--
-- Queries 9-14 read the predictions / topics / logs tables and return nothing
-- until Members 2, 3 and 4 populate them. That is expected - the schema is
-- designed ahead of the data it will hold.
--
-- Each query is delimited by a "-- name:" comment so run_queries.py can split
-- this file and export one CSV per query.
-- ============================================================================


-- name: q01_class_distribution_by_split
-- Class balance per split. The headline fact of the whole project.
SELECT
    m.split,
    e.label_name,
    e.valence,
    COUNT(*)                                                        AS n_messages,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY m.split), 2) AS pct_of_split
FROM messages m
JOIN emotions e ON e.label_id = m.label
GROUP BY m.split, e.label_name, e.valence
ORDER BY m.split, n_messages DESC;


-- name: q02_corpus_summary
-- One-row overview for the executive summary of the report.
SELECT
    COUNT(*)                        AS total_messages,
    COUNT(DISTINCT split)           AS n_splits,
    COUNT(DISTINCT label)           AS n_classes,
    ROUND(AVG(n_words), 2)          AS avg_words,
    MIN(n_words)                    AS min_words,
    MAX(n_words)                    AS max_words,
    ROUND(AVG(n_chars), 2)          AS avg_chars,
    ROUND(100.0 * SUM(has_negation) / COUNT(*), 2) AS pct_with_negation
FROM messages;


-- name: q03_length_percentiles_by_emotion
-- Median and p90 message length per emotion, via NTILE windows.
-- Tells you whether some emotions are expressed more verbosely than others.
WITH ranked AS (
    SELECT
        m.label,
        m.n_words,
        NTILE(100) OVER (PARTITION BY m.label ORDER BY m.n_words) AS pct_bucket
    FROM messages m
)
SELECT
    e.label_name,
    COUNT(*)                                              AS n_messages,
    MIN(CASE WHEN pct_bucket = 50 THEN n_words END)       AS p50_words,
    MIN(CASE WHEN pct_bucket = 90 THEN n_words END)       AS p90_words,
    MAX(n_words)                                          AS max_words
FROM ranked
JOIN emotions e ON e.label_id = ranked.label
GROUP BY e.label_name
ORDER BY p50_words DESC;


-- name: q04_negation_rate_by_emotion
-- Negation flips sentiment and is a known source of classifier errors.
SELECT
    e.label_name,
    e.valence,
    COUNT(*)                                          AS n_messages,
    SUM(m.has_negation)                               AS n_with_negation,
    ROUND(100.0 * SUM(m.has_negation) / COUNT(*), 2)  AS pct_with_negation
FROM messages m
JOIN emotions e ON e.label_id = m.label
GROUP BY e.label_name, e.valence
ORDER BY pct_with_negation DESC;


-- name: q05_valence_breakdown
-- Aggregate to positive / negative / ambiguous - the view a support manager
-- actually wants before drilling into individual emotions.
SELECT
    m.split,
    e.valence,
    COUNT(*)                AS n_messages,
    ROUND(AVG(m.n_words), 1) AS avg_words
FROM messages m
JOIN emotions e ON e.label_id = m.label
GROUP BY m.split, e.valence
ORDER BY m.split, n_messages DESC;


-- name: q06_longest_message_per_emotion
-- ROW_NUMBER partitioned by emotion: the top 3 longest examples of each class.
-- Useful for qualitative inspection in the EDA section.
SELECT label_name, rn AS rank_in_class, n_words, text
FROM (
    SELECT
        e.label_name,
        m.n_words,
        m.text,
        ROW_NUMBER() OVER (PARTITION BY m.label ORDER BY m.n_words DESC) AS rn
    FROM messages m
    JOIN emotions e ON e.label_id = m.label
)
WHERE rn <= 3
ORDER BY label_name, rn;


-- name: q07_imbalance_ratio
-- Every class expressed as a ratio against the majority class.
-- This number is the justification for macro F1 over accuracy.
WITH counts AS (
    SELECT label, COUNT(*) AS n
    FROM messages
    WHERE split = 'train'
    GROUP BY label
),
with_max AS (
    SELECT
        label,
        n,
        MAX(n) OVER ()                               AS max_n,
        SUM(n) OVER ()                               AS total_n,
        RANK() OVER (ORDER BY n DESC)                AS size_rank
    FROM counts
)
SELECT
    e.label_name,
    w.n                                   AS n_train,
    w.size_rank,
    ROUND(100.0 * w.n / w.total_n, 2)     AS pct_of_train,
    ROUND(1.0 * w.max_n / w.n, 2)         AS times_smaller_than_majority
FROM with_max w
JOIN emotions e ON e.label_id = w.label
ORDER BY w.n DESC;


-- name: q08_cumulative_class_share
-- Running share as classes are added largest-first. Shows how few classes
-- account for most of the corpus.
WITH counts AS (
    SELECT label, COUNT(*) AS n FROM messages GROUP BY label
)
SELECT
    e.label_name,
    c.n,
    ROUND(100.0 * SUM(c.n) OVER (ORDER BY c.n DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / SUM(c.n) OVER (), 2) AS cumulative_pct
FROM counts c
JOIN emotions e ON e.label_id = c.label
ORDER BY c.n DESC;


-- name: q09_model_accuracy_comparison
-- Overall accuracy per model on the test split. Needs predictions.
WITH scored AS (
    SELECT
        p.model_id,
        CASE WHEN p.predicted_label = m.label THEN 1 ELSE 0 END AS correct
    FROM predictions p
    JOIN messages m ON m.id = p.message_id
    WHERE m.split = 'test'
)
SELECT
    mo.model_name,
    mo.family,
    mo.version,
    COUNT(*)                                   AS n_scored,
    SUM(s.correct)                             AS n_correct,
    ROUND(100.0 * SUM(s.correct) / COUNT(*), 2) AS accuracy_pct
FROM scored s
JOIN models mo ON mo.model_id = s.model_id
GROUP BY mo.model_name, mo.family, mo.version
ORDER BY accuracy_pct DESC;


-- name: q10_per_class_recall_by_model
-- Recall per emotion per model. This is the table that exposes whether a model
-- is simply ignoring the rare classes, which accuracy would hide.
WITH scored AS (
    SELECT
        p.model_id,
        m.label                                                   AS true_label,
        CASE WHEN p.predicted_label = m.label THEN 1 ELSE 0 END   AS correct
    FROM predictions p
    JOIN messages m ON m.id = p.message_id
    WHERE m.split = 'test'
)
SELECT
    mo.model_name,
    e.label_name,
    COUNT(*)                                    AS n_true,
    SUM(s.correct)                              AS n_recalled,
    ROUND(100.0 * SUM(s.correct) / COUNT(*), 2) AS recall_pct
FROM scored s
JOIN models mo   ON mo.model_id = s.model_id
JOIN emotions e  ON e.label_id  = s.true_label
GROUP BY mo.model_name, e.label_name
ORDER BY mo.model_name, recall_pct ASC;


-- name: q11_most_confused_pairs
-- Which emotions get mistaken for which. Expect love/joy and fear/surprise.
WITH errors AS (
    SELECT
        p.model_id,
        m.label            AS true_label,
        p.predicted_label  AS pred_label,
        COUNT(*)           AS n
    FROM predictions p
    JOIN messages m ON m.id = p.message_id
    WHERE m.split = 'test'
      AND p.predicted_label <> m.label
    GROUP BY p.model_id, m.label, p.predicted_label
)
SELECT
    mo.model_name,
    et.label_name AS true_emotion,
    ep.label_name AS predicted_as,
    er.n          AS n_errors
FROM errors er
JOIN models mo  ON mo.model_id = er.model_id
JOIN emotions et ON et.label_id = er.true_label
JOIN emotions ep ON ep.label_id = er.pred_label
ORDER BY er.n DESC
LIMIT 20;


-- name: q12_low_confidence_review_queue
-- Operational query: the least confident predictions, for human review.
-- NTILE splits predictions into confidence deciles.
WITH ranked AS (
    SELECT
        p.prediction_id,
        p.message_id,
        p.model_id,
        p.predicted_label,
        p.confidence,
        NTILE(10) OVER (PARTITION BY p.model_id ORDER BY p.confidence) AS confidence_decile
    FROM predictions p
)
SELECT
    mo.model_name,
    r.confidence_decile,
    e.label_name AS predicted_as,
    ROUND(r.confidence, 3) AS confidence,
    m.text
FROM ranked r
JOIN messages m  ON m.id = r.message_id
JOIN models mo   ON mo.model_id = r.model_id
JOIN emotions e  ON e.label_id = r.predicted_label
WHERE r.confidence_decile = 1
ORDER BY r.confidence ASC
LIMIT 25;


-- name: q13_emotion_topic_matrix
-- Emotion x topic cross-tab. Fills in once Member 3 loads topic assignments.
SELECT
    t.topic_label,
    e.label_name,
    COUNT(*)            AS n_messages,
    ROUND(AVG(mt.weight), 3) AS avg_topic_weight
FROM message_topics mt
JOIN messages m ON m.id = mt.message_id
JOIN topics t   ON t.topic_id = mt.topic_id
JOIN emotions e ON e.label_id = m.label
GROUP BY t.topic_label, e.label_name
ORDER BY t.topic_label, n_messages DESC;


-- name: q14_inference_latency_by_model
-- Monitoring query for the MLOps section: throughput and tail latency.
WITH ranked AS (
    SELECT
        model_id,
        latency_ms,
        NTILE(100) OVER (PARTITION BY model_id ORDER BY latency_ms) AS pct_bucket
    FROM inference_logs
    WHERE status = 'ok'
)
SELECT
    mo.model_name,
    COUNT(*)                                          AS n_calls,
    ROUND(AVG(r.latency_ms), 2)                       AS avg_ms,
    MIN(CASE WHEN r.pct_bucket = 50 THEN r.latency_ms END) AS p50_ms,
    MIN(CASE WHEN r.pct_bucket = 95 THEN r.latency_ms END) AS p95_ms,
    ROUND(MAX(r.latency_ms), 2)                       AS max_ms
FROM ranked r
JOIN models mo ON mo.model_id = r.model_id
GROUP BY mo.model_name
ORDER BY avg_ms;
