"""Stage 3 of the pipeline: processed CSVs -> SQLite warehouse.

Creates ``data/emotion.db`` from ``sql/schema.sql`` and loads the processed
splits into the ``messages`` table. The prediction, topic and log tables are
created empty - Members 2, 3 and 4 write into them.

Run from the project root::

    python -m src.data.load_sql
"""

from __future__ import annotations

import sqlite3
import sys

import pandas as pd

from src.data.schema import SPLITS
from src.utils.paths import DB_PATH, PROCESSED_DIR, SQL_DIR, ensure_dirs

MESSAGE_COLUMNS = [
    "id", "split", "text", "text_clean",
    "label", "n_words", "n_chars", "has_negation",
]


def build_schema(conn: sqlite3.Connection) -> None:
    ddl = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(ddl)
    conn.commit()


def load_messages(conn: sqlite3.Connection) -> int:
    frames = []
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.build` first."
            )
        frames.append(pd.read_csv(path, encoding="utf-8"))

    df = pd.concat(frames, ignore_index=True)[MESSAGE_COLUMNS]
    df.to_sql("messages", conn, if_exists="append", index=False)
    conn.commit()
    return len(df)


def verify(conn: sqlite3.Connection) -> None:
    checks = [
        ("row count", "SELECT COUNT(*) FROM messages"),
        ("distinct splits", "SELECT COUNT(DISTINCT split) FROM messages"),
        ("distinct labels", "SELECT COUNT(DISTINCT label) FROM messages"),
        (
            "orphan labels (must be 0)",
            "SELECT COUNT(*) FROM messages m "
            "LEFT JOIN emotions e ON e.label_id = m.label WHERE e.label_id IS NULL",
        ),
        ("duplicate ids (must be 0)",
         "SELECT COUNT(*) FROM (SELECT id FROM messages GROUP BY id HAVING COUNT(*) > 1)"),
    ]
    print("\nverification")
    for name, query in checks:
        (value,) = conn.execute(query).fetchone()
        print(f"  {name:<28} {value:,}")
        if "must be 0" in name and value != 0:
            raise AssertionError(f"integrity check failed: {name} = {value}")


def main() -> int:
    ensure_dirs()

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"removed existing {DB_PATH.name}")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        print("creating schema from sql/schema.sql")
        build_schema(conn)

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        print(f"  tables: {', '.join(tables)}")

        print("\nloading messages")
        n = load_messages(conn)
        print(f"  inserted {n:,} rows")

        verify(conn)
    finally:
        conn.close()

    print(f"\nwrote {DB_PATH}")
    print("Next: python -m src.data.run_queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
