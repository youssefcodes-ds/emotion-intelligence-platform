# System Architecture

```mermaid
flowchart TB
    subgraph Data["Data Layer"]
        A[dair-ai/emotion<br/>Hugging Face] -->|ingest.py| B[data/raw/*.csv]
        B -->|build.py<br/>clean + dedupe + leakage removal| C[data/processed/*.csv]
        C -->|load_sql.py| D[(SQLite<br/>data/emotion.db)]
    end

    subgraph Models["Modelling Layer"]
        C --> E[Classical ML<br/>TF-IDF + LogReg / SVM]
        C --> F[Deep Learning<br/>Embedding MLP + LSTM]
        E --> G[models/*.joblib]
        F --> H[models/*.h5 + vocabulary.pkl]
    end

    subgraph App["Application Layer"]
        G --> I[Streamlit App<br/>app/app.py]
        H --> I
        I -->|writes| J[predictions table]
        J --> D
    end

    subgraph Reports["Reporting Layer"]
        D -->|run_queries.py| K[reports/sql_results/]
        C -->|eda.py| L[reports/eda_report.md<br/>+ figures]
        E -->|report.py| M[reports/classical_ml/REPORT.md]
    end

    style Data fill:#1a3a5c,color:#fff
    style Models fill:#4a2f5c,color:#fff
    style App fill:#2f5c3a,color:#fff
    style Reports fill:#5c4a2f,color:#fff
```

## Layer summary

| Layer | Owner | Key artifact |
|---|---|---|
| Data | Member 1 | `data/emotion.db`, `data/processed/*.csv` |
| Classical ML | Member 2 | `models/*.joblib`, `reports/classical_ml/REPORT.md` |
| Deep learning | Member 3 | `models/*.h5`, `models/vocabulary.pkl` |
| Application | Member 4 | `app/app.py`, Docker |

## Data flow

1. Raw text is downloaded once and cleaned into leakage-safe train/validation/test splits.
2. The same processed splits feed both the classical and deep-learning tracks, so results are comparable.
3. Trained models are loaded by the Streamlit app at runtime.
4. Every prediction made through the app is written back into the `predictions`
   table in the shared SQLite database, which is what the analytical SQL
   queries (model comparison, per-class recall, confusion pairs) read from.
