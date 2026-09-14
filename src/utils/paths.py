"""Project-wide paths.

Every script imports from here so that no module ever hard-codes a relative
path. Works identically on Windows, macOS and Linux.
"""

from pathlib import Path

# .../src/utils/paths.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = RAW_DIR / "_manual"

SQL_DIR = PROJECT_ROOT / "sql"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "emotion.db"

RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
