"""Execute every query in sql/queries.sql and export the results.

Produces ``reports/sql_results/<query_name>.csv`` plus a markdown summary. 

Run from the project root::

    python -m src.data.run_queries
"""

from __future__ import annotations

import re
import sqlite3
import sys

import pandas as pd

from src.utils.paths import DB_PATH, REPORTS_DIR, SQL_DIR

NAME_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)

OUT_DIR = REPORTS_DIR / "sql_results"


def split_queries(text: str) -> list[tuple[str, str]]:
    """Split queries.sql on '-- name: xxx' markers."""
    matches = list(NAME_RE.finditer(text))
    queries = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            queries.append((match.group(1), body))
    return queries


def main() -> int:
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found. Run `python -m src.data.load_sql` first.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = (SQL_DIR / "queries.sql").read_text(encoding="utf-8")
    queries = split_queries(text)
    print(f"found {len(queries)} queries in sql/queries.sql\n")

    conn = sqlite3.connect(DB_PATH)
    lines = ["# SQL query results", ""]
    empty = []

    try:
        for name, body in queries:
            try:
                df = pd.read_sql_query(body, conn)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED  {name}: {exc}")
                lines += [f"## {name}", "", f"FAILED: `{exc}`", ""]
                continue

            out = OUT_DIR / f"{name}.csv"
            df.to_csv(out, index=False, encoding="utf-8")

            status = "empty" if df.empty else f"{len(df)} rows"
            print(f"  {'ok':<7} {name:<38} {status}")
            if df.empty:
                empty.append(name)

            lines += [f"## {name}", ""]
            if df.empty:
                lines += ["_No rows - the source table is not populated yet._", ""]
            else:
                lines += [df.head(15).to_markdown(index=False), ""]
    finally:
        conn.close()

    (REPORTS_DIR / "sql_results.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {OUT_DIR}/ and reports/sql_results.md")
    if empty:
        print(
            "\nempty results (expected - these read tables other members fill):\n  "
            + ", ".join(empty)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
