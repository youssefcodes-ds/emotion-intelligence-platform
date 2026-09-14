# data/

Nothing in here is committed to git except this file and the `.gitkeep` markers.

| Folder | Contents | Produced by |
|---|---|---|
| `raw/` | Untouched dataset dumps from Hugging Face | `python -m src.data.ingest` |
| `interim/` | Cleaned text, not yet featurised | `python -m src.data.build` |
| `processed/` | Final train / validation / test splits used by all models | `python -m src.data.build` |

Run the ingestion and build commands from the project root to regenerate everything.
See `reports/data_dictionary.md` for column-level definitions.
