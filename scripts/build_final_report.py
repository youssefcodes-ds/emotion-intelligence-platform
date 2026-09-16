"""Assemble the final technical report from the pieces already in reports/.

This does NOT invent content. It stitches together what already exists
(EDA report, data dictionary, SQL results, classical ML report) under the
19-section structure required by the project spec, and leaves an explicit
TODO marker for every section nobody has written yet, so the gaps are
visible rather than silently skipped.

Run from the project root:

    python scripts/build_final_report.py

Output: reports/FINAL_REPORT.md
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

REPORTS = Path("reports")
OUT = REPORTS / "FINAL_REPORT.md"


def read_or_todo(path: Path, owner: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        f"> **TODO ({owner}):** `{path}` does not exist yet. "
        f"This section will be empty in the submitted report until it is written."
    )


def section(number: int, title: str, body: str) -> str:
    return f"\n## {number}. {title}\n\n{body}\n"


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    parts: list[str] = [
        "# Emotion Intelligence Platform — Final Technical Report",
        "",
        f"Generated {now}. Sections marked TODO are assembled automatically "
        "and were empty at generation time — fill them in and rerun this "
        "script, or edit reports/FINAL_REPORT.md directly afterward.",
    ]

    parts.append(section(
        1, "Executive Summary",
        "> **TODO (whole team):** 1-2 paragraphs — what the platform does, "
        "the headline result, and the one thing worth remembering from it."
    ))

    parts.append(section(
        2, "Problem Definition and Business Context",
        "Support teams receive more unstructured text than they can read, "
        "and a binary positive/negative label is too coarse to route or "
        "prioritise a message. This system classifies each message into "
        "one of six emotions and aggregates the results into support "
        "intelligence.\n\n"
        "Emotion labels are not clinical or mental-health assessments."
    ))

    parts.append(section(
        3, "Dataset Description and Data Governance",
        read_or_todo(REPORTS / "data_dictionary.md", "Member 1")
    ))

    parts.append(section(
        4, "System Architecture",
        "See `reports/architecture.md` for the full diagram.\n\n"
        + read_or_todo(REPORTS / "architecture.md", "assembled")
    ))

    parts.append(section(
        5, "SQL / Data Pipeline Design",
        read_or_todo(REPORTS / "sql_results.md", "Member 1")
    ))

    parts.append(section(
        6, "EDA and Statistical Analysis",
        read_or_todo(REPORTS / "eda_report.md", "Member 1")
    ))

    parts.append(section(
        7, "Feature Engineering",
        "> **TODO (Member 2 / Member 3):** describe TF-IDF n-gram "
        "configurations and the TextVectorization setup used for the deep "
        "learning models (max_tokens, sequence_length)."
    ))

    parts.append(section(
        8, "Machine Learning Experiments",
        read_or_todo(REPORTS / "classical_ml" / "REPORT.md", "Member 2")
    ))

    parts.append(section(
        9, "Deep Learning Experiments",
        "> **TODO (Member 3):** embedding-MLP and LSTM architecture, "
        "training curves, and comparison against the classical baseline. "
        "Metrics are in `models/training_report.json` if a classical-style "
        "report has not been generated for the deep models yet."
    ))

    parts.append(section(
        10, "Model Evaluation and Error Analysis",
        "> **TODO (Member 2 / Member 3):** confusion matrices and the "
        "specific misclassified examples worth discussing. "
        "`reports/sql_results/q11_most_confused_pairs.csv` will populate "
        "this automatically once predictions are logged to the database."
    ))

    parts.append(section(
        11, "Explainability / Responsible AI",
        "Feature importance for the classical models is in "
        "`reports/classical_ml/` (per-class CSVs and threshold heatmaps).\n\n"
        "> **TODO:** explainability for the deep learning models "
        "(e.g. attention weights or LIME/SHAP) if planned."
    ))

    parts.append(section(
        12, "Deployment Architecture",
        "Streamlit app (`app/app.py`) loading classical models "
        "(`.joblib`) and deep learning models (`.h5` + `vocabulary.pkl`) "
        "from `models/`. See `reports/architecture.md`.\n\n"
        "> **TODO:** Docker deployment notes — see `DOCKER.md`."
    ))

    parts.append(section(
        13, "MLOps and Monitoring",
        "> **TODO:** prediction logging status. Once the app writes to "
        "the `predictions` table, `reports/sql_results/q14_inference_"
        "latency_by_model.csv` and the model comparison queries populate "
        "automatically via `python -m src.data.run_queries`."
    ))

    parts.append(section(
        14, "Advanced AI Integration",
        "> **TODO (Member 3 / Member 4):** RAG assistant, if built. "
        "`rag/` is currently empty."
    ))

    parts.append(section(
        15, "Testing and Security",
        "Unit tests for the data-cleaning pipeline: `tests/test_text_"
        "cleaning.py` (17 tests, all passing).\n\n"
        "> **TODO:** model and UI test coverage — see "
        "`tests/test_model_smoke.py` and `tests/test_app_smoke.py` if "
        "added."
    ))

    parts.append(section(
        16, "Limitations",
        "- English-only, short social-media style text; performance on "
        "long-form or non-English messages is untested.\n"
        "- Six emotions is a coarse taxonomy; real messages often carry "
        "mixed or absent emotion.\n"
        "- Labels are crowd-derived, not clinical.\n"
        "- Class imbalance (~9:1 largest to smallest class) means rare "
        "classes are harder to predict reliably; see the EDA report."
    ))

    parts.append(section(
        17, "Future Work",
        "- Migrate from SQLite to a server database if the platform moves "
        "toward concurrent production use.\n"
        "- Expand the RAG assistant / topic modelling into the app UI.\n"
        "- Explore the `merged_training.pkl` extended dataset for more "
        "training data on rare classes."
    ))

    parts.append(section(18, "Conclusion", "> **TODO (whole team).**"))

    parts.append(section(
        19, "References",
        "Saravia et al., *CARER: Contextualized Affect Representations for "
        "Emotion Recognition*, EMNLP 2018.\n\n"
        "Dataset: https://huggingface.co/datasets/dair-ai/emotion"
    ))

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")

    todo_count = "\n".join(parts).count("TODO")
    print(f"{todo_count} TODO markers remaining — these are the open gaps.")


if __name__ == "__main__":
    main()
