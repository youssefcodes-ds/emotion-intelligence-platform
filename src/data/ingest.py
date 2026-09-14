"""Stage 1 of the pipeline: fetch the raw dataset.

Downloads the official `split` configuration of `dair-ai/emotion` and writes it
to ``data/raw/`` as UTF-8 CSV, plus a manifest recording provenance (source,
timestamp, row counts, SHA-256 of every file). The manifest is what lets anyone
prove the pipeline is reproducible.

Run from the project root::

    python -m src.data.ingest

Three sources are tried in order, so the script still works offline once a
teammate has the files:

1. Parquet over HTTPS from Hugging Face  (needs pandas + pyarrow only)
2. The ``datasets`` library              (only if it is already installed)
3. Manual CSVs dropped in ``data/raw/_manual/``
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from src.data.schema import EXPECTED_ROWS, ID2LABEL, LABEL_NAMES, SPLITS
from src.utils.paths import MANUAL_DIR, RAW_DIR, ensure_dirs

HF_BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split"

# The HF split is named "validation"; the parquet file is named the same.
PARQUET_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def _from_parquet() -> dict[str, pd.DataFrame]:
    """Primary source: download the parquet files directly."""
    frames = {}
    for split, filename in PARQUET_FILES.items():
        url = f"{HF_BASE}/{filename}"
        print(f"  downloading {split:<10} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "emotion-platform/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = response.read()
        frames[split] = pd.read_parquet(io.BytesIO(payload))
    return frames


def _from_datasets_library() -> dict[str, pd.DataFrame]:
    """Fallback: use the `datasets` library if it happens to be installed."""
    from datasets import load_dataset  # imported lazily on purpose

    dataset = load_dataset("dair-ai/emotion", "split")
    return {split: dataset[split].to_pandas() for split in SPLITS}


def _from_manual_csv() -> dict[str, pd.DataFrame]:
    """Last resort: read CSVs a teammate placed in data/raw/_manual/."""
    frames = {}
    for split in SPLITS:
        path = MANUAL_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"  reading   {split:<10} <- {path}")
        frames[split] = pd.read_csv(path, encoding="utf-8")
    return frames


SOURCES = [
    ("huggingface-parquet", _from_parquet),
    ("datasets-library", _from_datasets_library),
    ("manual-csv", _from_manual_csv),
]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of human-readable problems. Empty list means clean."""
    problems: list[str] = []

    for split, df in frames.items():
        prefix = f"[{split}]"

        missing = {"text", "label"} - set(df.columns)
        if missing:
            problems.append(f"{prefix} missing columns: {sorted(missing)}")
            continue

        expected = EXPECTED_ROWS[split]
        if len(df) != expected:
            problems.append(f"{prefix} expected {expected} rows, got {len(df)}")

        if df["text"].isna().any():
            problems.append(f"{prefix} {int(df['text'].isna().sum())} null texts")

        if df["label"].isna().any():
            problems.append(f"{prefix} {int(df['label'].isna().sum())} null labels")

        bad_labels = set(df["label"].dropna().unique()) - set(ID2LABEL)
        if bad_labels:
            problems.append(f"{prefix} labels outside 0-5: {sorted(bad_labels)}")

        if (df["text"].astype(str).str.strip() == "").any():
            problems.append(f"{prefix} contains empty-string texts")

    return problems


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ensure_dirs()
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] | None = None
    source_used = None

    for name, fetch in SOURCES:
        print(f"\nTrying source: {name}")
        try:
            frames = fetch()
            source_used = name
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  network problem: {exc}")
        except ImportError as exc:
            print(f"  not available: {exc}")
        except FileNotFoundError as exc:
            print(f"  file not found: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and try the next source
            print(f"  failed: {type(exc).__name__}: {exc}")

    if frames is None:
        print(
            "\nAll sources failed.\n"
            "Fix: download the three parquet/CSV files by hand from\n"
            "  https://huggingface.co/datasets/dair-ai/emotion/tree/main/split\n"
            f"convert them to CSV named train.csv / validation.csv / test.csv\n"
            f"with columns 'text' and 'label', put them in {MANUAL_DIR}\n"
            "and run this script again."
        )
        return 1

    print(f"\nSource used: {source_used}")

    problems = validate(frames)
    if problems:
        print("\nData quality problems found:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nStopping. Fix the source before continuing.")
        return 1
    print("Validation passed.")

    manifest = {
        "dataset": "dair-ai/emotion",
        "configuration": "split",
        "source_used": source_used,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_order": LABEL_NAMES,
        "files": {},
    }

    print()
    for split, df in frames.items():
        df = df[["text", "label"]].copy()
        df["label_name"] = df["label"].map(ID2LABEL)

        out_path = RAW_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

        manifest["files"][split] = {
            "path": f"data/raw/{split}.csv",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "class_counts": df["label_name"].value_counts().sort_index().to_dict(),
            "sha256": _sha256(out_path),
        }
        print(f"  wrote {out_path.name:<16} {len(df):>6,} rows")

    manifest_path = RAW_DIR / "raw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {manifest_path.name}")

    print("\nIngestion complete. Next: python -m src.data.build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
