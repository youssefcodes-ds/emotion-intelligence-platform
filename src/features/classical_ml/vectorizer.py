"""TF-IDF vectorization module for classical ML.

Handles text to TF-IDF feature transformation with n-gram tuning.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ---- Configuration ----
TFIDF_PARAMS = {
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.95,
    "lowercase": True,
    "strip_accents": "unicode",
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": True,
}

NGRAM_CONFIGS = [
    {"name": "unigrams", "ngram_range": (1, 1)},
    {"name": "bigrams", "ngram_range": (1, 2)},
    {"name": "trigrams", "ngram_range": (1, 3)},
]


# ---- Functions ----
def create_vectorizer(ngram_range: tuple) -> TfidfVectorizer:
    """Create a TfidfVectorizer with specified n-gram range.

    Parameters
    ----------
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction

    Returns
    -------
    vectorizer : TfidfVectorizer
        Configured but unfitted vectorizer
    """
    params = TFIDF_PARAMS.copy()
    params["ngram_range"] = ngram_range
    return TfidfVectorizer(**params)


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: np.ndarray) -> tuple[TfidfVectorizer, int]:
    """Fit the TF-IDF vectorizer on training texts.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer
    texts : np.ndarray
        Training texts

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    n_features : int
        Number of features created
    """
    vectorizer.fit(texts)
    n_features = len(vectorizer.get_feature_names_out())
    return vectorizer, n_features


def transform_texts(vectorizer: TfidfVectorizer, texts: np.ndarray):
    """Transform texts using a fitted vectorizer.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    texts : np.ndarray
        Texts to transform

    Returns
    -------
    X : sparse matrix
        TF-IDF feature matrix
    """
    return vectorizer.transform(texts)


def get_feature_names(vectorizer: TfidfVectorizer) -> np.ndarray:
    """Get feature names from the vectorizer.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer

    Returns
    -------
    feature_names : np.ndarray
        Array of feature (n-gram) names
    """
    return vectorizer.get_feature_names_out()
