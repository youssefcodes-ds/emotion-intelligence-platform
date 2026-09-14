# Handoff 

## What is ready

| artifact | path |
|---|---|
| Processed splits | `data/processed/{train,validation,test}.csv` |
| SQLite warehouse | `data/emotion.db` (run `python -m src.data.load_sql`) |
| SQL schema and queries | `sql/schema.sql`, `sql/queries.sql` |
| EDA report | `reports/eda_report.md` + `reports/figures/` |
| Data dictionary | `reports/data_dictionary.md` |

## How to load the data

```python
import pandas as pd
train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/validation.csv")
test  = pd.read_csv("data/processed/test.csv")
```

## Rules that are not negotiable

1. **Use `text_clean` for TF-IDF, `text` for transformers.** Both columns are in
   every file. Do not re-clean the text.
2. **Import the label mapping.** `from src.data.schema import ID2LABEL, LABEL_NAMES`.
   Never retype the list — the order is 0=sadness, 1=joy, 2=love, 3=anger,
   4=fear, 5=surprise, and getting it wrong corrupts every confusion matrix.
3. **Do not touch the test split** until your model is final. Tune on
   `validation.csv`.
4. **Do not re-split the data.** Cross-split leakage has already been removed;
   re-splitting reintroduces it.

## Storing your results

Register the model, then write predictions:

```sql
INSERT INTO models (model_name, family, version, notes)
VALUES ('tfidf_logreg', 'classical', 'v1', 'C=1.0, ngram 1-2');

INSERT INTO predictions (message_id, model_id, predicted_label, confidence)
VALUES ('test_000001', 1, 3, 0.87);
```

`family` must be one of: classical, deep, transformer, baseline.
`UNIQUE(message_id, model_id)` means one prediction per model per message.

Once predictions are loaded, `python -m src.data.run_queries` fills in the model
comparison, per-class recall, confusion pairs and review-queue tables
automatically.

## What the data tells you before you start

- Class imbalance is roughly **9.4:1**
  (joy vs surprise). **Report macro F1 and per-class
  recall, not accuracy.**
- Messages are short — median 17 words.
- 22.7% contain a negation. Do not strip
  stopwords; "not" carries the signal.
- See section 8 of the EDA report for the emotion pair most likely to be
  confused.

## Rows dropped during cleaning

```json
{
  "train": {
    "rows_in": 16000,
    "rows_expected_from_source": 16000,
    "dropped_empty_after_cleaning": 0,
    "dropped_exact_duplicates": 1,
    "pct_dropped_at_clean_stage": 0.01,
    "dropped_leaking_into_heldout": 16,
    "rows_out": 15983,
    "class_counts": {
      "anger": 2159,
      "fear": 1934,
      "joy": 5356,
      "love": 1298,
      "sadness": 4665,
      "surprise": 571
    },
    "class_share": {
      "anger": 0.1351,
      "fear": 0.121,
      "joy": 0.3351,
      "love": 0.0812,
      "sadness": 0.2919,
      "surprise": 0.0357
    },
    "median_words": 17,
    "path": "data/processed/train.csv"
  },
  "validation": {
    "rows_in": 2000,
    "rows_expected_from_source": 2000,
    "dropped_empty_after_cleaning": 0,
    "dropped_exact_duplicates": 0,
    "pct_dropped_at_clean_stage": 0.0,
    "rows_out": 2000,
    "class_counts": {
      "anger": 275,
      "fear": 212,
      "joy": 704,
      "love": 178,
      "sadness": 550,
      "surprise": 81
    },
    "class_share": {
      "anger": 0.1375,
      "fear": 0.106,
      "joy": 0.352,
      "love": 0.089,
      "sadness": 0.275,
      "surprise": 0.0405
    },
    "median_words": 17,
    "path": "data/processed/validation.csv"
  },
  "test": {
    "rows_in": 2000,
    "rows_expected_from_source": 2000,
    "dropped_empty_after_cleaning": 0,
    "dropped_exact_duplicates": 0,
    "pct_dropped_at_clean_stage": 0.0,
    "rows_out": 2000,
    "class_counts": {
      "anger": 275,
      "fear": 224,
      "joy": 695,
      "love": 159,
      "sadness": 581,
      "surprise": 66
    },
    "class_share": {
      "anger": 0.1375,
      "fear": 0.112,
      "joy": 0.3475,
      "love": 0.0795,
      "sadness": 0.2905,
      "surprise": 0.033
    },
    "median_words": 17,
    "path": "data/processed/test.csv"
  }
}
```