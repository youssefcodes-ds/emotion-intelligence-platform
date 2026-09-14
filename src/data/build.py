"""Stage 2 of the pipeline: raw -> cleaned -> feature-ready.

Reads ``data/raw/*.csv``, cleans the text, removes duplicates and cross-split
leakage, adds lightweight features, and writes ``data/processed/*.csv`` plus a
manifest documenting every row that was dropped and why.

Run from the project root::

    python -m src.data.build

Output columns
--------------
id           stable primary key, e.g. ``train_000042`` - used by SQL and logs
split        train | validation | test
text         ORIGINAL text, untouched (transformer track uses this)
text_clean   normalised text (TF-IDF track uses this)
label        integer 0-5
label_name   string label
n_words      word count of the original text
n_chars      character count of the original text
has_negation 1 if the cleaned text contains a negation token
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.data.schema import EXPECTED_ROWS, ID2LABEL, LABEL_NAMES, SPLITS
from src.features.text_cleaning import (
    char_count,
    clean_text,
    has_negation,
    is_usable,
    word_count,
)
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs

OUTPUT_COLUMNS = [
    "id", "split", "text", "text_clean",
    "label", "label_name", "n_words", "n_chars", "has_negation",
]


def load_raw() -> dict[str, pd.DataFrame]:
    frames = {}
    for split in SPLITS:
        path = RAW_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.ingest` first."
            )
        frames[split] = pd.read_csv(path, encoding="utf-8")
    return frames


def transform(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(ID2LABEL)
    df["n_words"] = df["text"].map(word_count)
    df["n_chars"] = df["text"].map(char_count)
    df["has_negation"] = df["text_clean"].map(has_negation)
    return df


def assign_ids(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["id"] = [f"{split}_{i:06d}" for i in range(len(df))]
    return df


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    report: dict = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_order": LABEL_NAMES,
        "decisions": {
            "stopwords_removed": False,
            "stemming_or_lemmatisation": False,
            "original_text_preserved": True,
            "exact_duplicates_removed_within_split": True,
            "train_rows_leaking_into_val_or_test_removed": True,
            "official_split_boundaries_respected": True,
        },
        "splits": {},
    }

    print("cleaning")
    frames = {}
    for split, df in raw.items():
        before = len(df)
        out = transform(df, split)

        empty = (~out["text_clean"].map(is_usable)).sum()
        out = out[out["text_clean"].map(is_usable)]

        dupes = int(out.duplicated(subset=["text_clean", "label"]).sum())
        out = out.drop_duplicates(subset=["text_clean", "label"], keep="first")

        frames[split] = out
        pct_dropped = 100 * (before - len(out)) / before if before else 0.0
        report["splits"][split] = {
            "rows_in": int(before),
            "rows_expected_from_source": EXPECTED_ROWS[split],
            "dropped_empty_after_cleaning": int(empty),
            "dropped_exact_duplicates": dupes,
            "pct_dropped_at_clean_stage": round(pct_dropped, 2),
        }
        print(
            f"  {split:<11} {before:>6,} -> {len(out):>6,}  "
            f"(empty {empty}, duplicates {dupes}, {pct_dropped:.1f}% dropped)"
        )
        if pct_dropped > 20:
            print(
                f"    !! WARNING: {pct_dropped:.1f}% of {split} removed as duplicates.\n"
                f"    !! That is high. Inspect data/raw/{split}.csv before trusting this.\n"
                f"    !! Do not just accept it - a shrunken training set changes every result."
            )

    # ---- cross-split leakage ------------------------------------------------
    # A message appearing in both train and test makes every score optimistic.
    # The held-out sets are the ground truth, so training rows are the ones that
    # go. Split boundaries themselves are never altered.
    held_out = set(frames["validation"]["text_clean"]) | set(frames["test"]["text_clean"])
    leaking = frames["train"]["text_clean"].isin(held_out)
    n_leaking = int(leaking.sum())

    frames["train"] = frames["train"][~leaking]
    report["splits"]["train"]["dropped_leaking_into_heldout"] = n_leaking
    print(f"\nleakage    {n_leaking:,} train rows also appear in validation/test -> dropped")

    overlap_vt = len(set(frames["validation"]["text_clean"]) & set(frames["test"]["text_clean"]))
    report["validation_test_overlap_remaining"] = overlap_vt
    if overlap_vt:
        print(f"  note: {overlap_vt} texts appear in BOTH validation and test (left as-is)")

    # ---- write --------------------------------------------------------------
    print("\nwriting")
    for split, df in frames.items():
        df = assign_ids(df, split)[OUTPUT_COLUMNS]

        path = PROCESSED_DIR / f"{split}.csv"
        df.to_csv(path, index=False, encoding="utf-8")

        counts = df["label_name"].value_counts()
        report["splits"][split].update(
            {
                "rows_out": int(len(df)),
                "class_counts": counts.sort_index().to_dict(),
                "class_share": (counts / len(df)).round(4).sort_index().to_dict(),
                "median_words": int(df["n_words"].median()),
                "path": f"data/processed/{split}.csv",
            }
        )
        print(f"  {path.name:<16} {len(df):>6,} rows")

    manifest_path = PROCESSED_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  {manifest_path.name}")

    # ---- final assertions ---------------------------------------------------
    print("\nchecks")
    all_ids = pd.concat([pd.read_csv(PROCESSED_DIR / f"{s}.csv")["id"] for s in SPLITS])
    assert all_ids.is_unique, "id collision across splits"
    print("  ids unique across all splits")

    train_texts = set(pd.read_csv(PROCESSED_DIR / "train.csv")["text_clean"])
    for split in ("validation", "test"):
        other = set(pd.read_csv(PROCESSED_DIR / f"{split}.csv")["text_clean"])
        assert not (train_texts & other), f"leakage remains between train and {split}"
        print(f"  no train/{split} overlap")

    print("\nBuild complete. Next: python -m src.data.load_sql")
    return 0


if __name__ == "__main__":
    sys.exit(main())
