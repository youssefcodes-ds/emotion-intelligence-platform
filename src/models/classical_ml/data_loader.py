"""Data loading utilities for classical ML pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import ID2LABEL
from src.utils.paths import PROCESSED_DIR


def load_data(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load text and labels for a given dataset split.

    Parameters
    ----------
    split : str
        'train', 'validation', or 'test'

    Returns
    -------
    texts : np.ndarray
        Array of cleaned text strings
    labels : np.ndarray
        Array of integer labels (0-5)
    """
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.build` first."
        )

    df = pd.read_csv(path, encoding="utf-8")
    texts = df["text_clean"].values
    labels = df["label"].values

    return texts, labels


def print_class_distribution(labels: np.ndarray, split: str) -> dict:
    """Print and return class distribution statistics.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels
    split : str
        Name of the split (for printing)

    Returns
    -------
    distribution : dict
        Counts and percentages per class
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    dist = {}
    print(f"\n{split.upper()} CLASS DISTRIBUTION:")
    print("-" * 60)

    for label_id, count in zip(unique, counts):
        pct = 100 * count / total
        label_name = ID2LABEL.get(label_id, f"class_{label_id}")
        print(f"  {label_name:<12} {count:>6,} ({pct:>5.2f}%)")
        dist[label_name] = {"count": int(count), "percentage": round(pct, 2)}

    print("-" * 60)
    print(f"  Total       {total:>20,}")

    return dist
