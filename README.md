# Customer Sentiment & Emotion Intelligence Platform

An end-to-end NLP platform that classifies customer text into six emotion classes,
compares classical NLP against deep learning, and exposes the results through a
Streamlit application.

> Graduation project — TechTrek Advanced Data Science & AI track, Project 9.

---

## 1. Team

| # | Member | Responsibility |
|---|--------|----------------|
| 1 | _TODO: name_ | Data Engineering, SQL, EDA, repository |
| 2 | _TODO: name_ | Classical Machine Learning (TF-IDF + LR / SVM) |
| 3 | _TODO: name_ | Deep Learning, Topic Modeling, Explainability |
| 4 | _TODO: name_ | Deployment, MLOps, Streamlit, Documentation |

Team number: _TODO_ · Track: AI · Submission file: `Team_XX_AI.zip`

---

## 2. Problem

Support teams receive large volumes of unstructured customer text. Reading it
manually does not scale, and a single "positive / negative" label is too coarse to
route or prioritise a message. This project classifies each message into one of six
emotions, aggregates the results into support intelligence (volumes, trends, topic
breakdowns), and serves predictions through a usable interface.

The system is a decision-support prototype. Emotion labels are **not** clinical or
mental-health assessments.

## 3. Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — English
text classification dataset with six emotion classes:
`sadness, joy, love, anger, fear, surprise`.

Official splits: 16,000 train / 2,000 validation / 2,000 test (`split` configuration).

Raw and processed data are **not committed** to this repository. Run the ingestion
script below to reproduce them locally. See `reports/data_dictionary.md` for the
schema of every generated file.

## 4. Repository structure

```
.
├── data/                 # raw / interim / processed  (git-ignored, metadata only)
├── notebooks/            # EDA and experiments
├── src/
│   ├── data/             # loading, validation, cleaning
│   ├── features/         # feature engineering
│   ├── models/           # training + inference
│   ├── evaluation/       # metrics + error analysis
│   └── utils/
├── sql/                  # schema + analytical queries
├── rag/                  # retrieval-augmented assistant
├── app/                  # Streamlit application
├── tests/                # unit / integration tests
├── models/               # versioned model artifacts
├── reports/              # figures, data dictionary, final report
├── requirements.txt
├── .env.example
└── README.md
```

## 5. Setup

```bash
git clone <REPO_URL>
cd emotion-intelligence-platform

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

## 6. Reproducing the pipeline

```bash
python -m src.data.ingest        # download raw dataset -> data/raw/
python -m src.data.build         # clean + featurise  -> data/processed/
python -m src.data.load_sql      # build SQLite warehouse -> data/emotion.db
```

_TODO (members 2–4): add training, app and Docker commands here._

## 7. Working agreement

- Each member owns their own folders. Do not edit another member's files without
  telling them.
- Never commit `data/`, `models/*.pkl`, `.env`, or notebook checkpoints.
- Commit messages: `area: what changed` (e.g. `sql: add window-function queries`).

## 8. Licence and attribution

The dataset is used under its original terms. Cite: Saravia et al., *CARER:
Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018.
