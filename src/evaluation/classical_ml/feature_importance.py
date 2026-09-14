"""Feature importance and analysis module for classical ML.

Extract and analyze feature importance from trained models.
Provides insights into which n-grams most strongly indicate each emotion class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_feature_importance_logreg(
    model: Any, feature_names: np.ndarray, label_names: list[str], top_k: int = 15
) -> dict:
    """Extract feature importance from Logistic Regression coefficients.

    For each class, identify the n-grams with highest absolute coefficients
    (features with most positive and negative weights).

    Parameters
    ----------
    model : LogisticRegression or ManualCalibratedModel
        Trained logistic regression model
    feature_names : np.ndarray
        Array of feature (n-gram) names from vectorizer
    label_names : list[str]
        List of emotion class names
    top_k : int
        Number of top features to return per class

    Returns
    -------
    importance : dict
        Nested dict: class_name -> {'positive': [...], 'negative': [...]}
        Each contains (feature, coefficient, rank) tuples
    """
    # Get base model if wrapped
    base_model = model.base_model if hasattr(model, "base_model") else model

    importance = {}

    # Coefficients shape: (n_classes, n_features)
    coefficients = base_model.coef_

    for class_idx, class_name in enumerate(label_names):
        class_coefs = coefficients[class_idx]

        # Get top positive features (increase probability of this class)
        positive_idx = np.argsort(class_coefs)[-top_k:][::-1]
        positive_features = [
            (feature_names[i], float(class_coefs[i]), rank)
            for rank, i in enumerate(positive_idx, 1)
        ]

        # Get top negative features (decrease probability of this class)
        negative_idx = np.argsort(class_coefs)[:top_k]
        negative_features = [
            (feature_names[i], float(class_coefs[i]), rank)
            for rank, i in enumerate(negative_idx, 1)
        ]

        importance[class_name] = {
            "positive": positive_features,
            "negative": negative_features,
        }

    return importance


def get_feature_importance_svm(
    model: Any, feature_names: np.ndarray, label_names: list[str], top_k: int = 15
) -> dict:
    """Extract feature importance from Linear SVM coefficients.

    Coefficients represent the importance of each feature for each class
    in the decision function.

    Parameters
    ----------
    model : LinearSVC or ManualCalibratedModel
        Trained SVM model
    feature_names : np.ndarray
        Array of feature (n-gram) names from vectorizer
    label_names : list[str]
        List of emotion class names
    top_k : int
        Number of top features to return per class

    Returns
    -------
    importance : dict
        Nested dict: class_name -> {'positive': [...], 'negative': [...]}
    """
    # Get base model if wrapped
    base_model = model.base_model if hasattr(model, "base_model") else model

    importance = {}

    # Coefficients shape: (n_classes, n_features)
    coefficients = base_model.coef_

    for class_idx, class_name in enumerate(label_names):
        class_coefs = coefficients[class_idx]

        # Get top positive features
        positive_idx = np.argsort(class_coefs)[-top_k:][::-1]
        positive_features = [
            (feature_names[i], float(class_coefs[i]), rank)
            for rank, i in enumerate(positive_idx, 1)
        ]

        # Get top negative features
        negative_idx = np.argsort(class_coefs)[:top_k]
        negative_features = [
            (feature_names[i], float(class_coefs[i]), rank)
            for rank, i in enumerate(negative_idx, 1)
        ]

        importance[class_name] = {
            "positive": positive_features,
            "negative": negative_features,
        }

    return importance


def print_feature_importance(importance: dict, model_name: str) -> None:
    """Print feature importance in a readable format.

    Parameters
    ----------
    importance : dict
        Output from get_feature_importance_*
    model_name : str
        Name of the model (for display)
    """
    print(f"\n{'=' * 80}")
    print(f"FEATURE IMPORTANCE: {model_name}")
    print(f"{'=' * 80}")

    for class_name, features in importance.items():
        print(f"\n{class_name.upper()}")
        print("-" * 80)

        print("  Most predictive features (↑ increases probability):")
        for feature, coef, rank in features["positive"]:
            print(f"    {rank:2d}. {feature:<40s} coef={coef:+.4f}")

        print("\n  Least predictive features (↓ decreases probability):")
        for feature, coef, rank in features["negative"][:5]:  # Show only top 5
            print(f"    {rank:2d}. {feature:<40s} coef={coef:+.4f}")


def export_feature_importance_csv(
    importance: dict, model_name: str, ngram_name: str, output_dir
) -> None:
    """Export feature importance to CSV for further analysis.

    Parameters
    ----------
    importance : dict
        Output from get_feature_importance_*
    model_name : str
        Name of the model ('LogisticRegression' or 'LinearSVC')
    ngram_name : str
        N-gram configuration name
    output_dir : Path
        Directory to save CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name, features in importance.items():
        rows = []

        # Add positive features
        for feature, coef, rank in features["positive"]:
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coefficient": coef,
                    "direction": "positive",
                    "rank": rank,
                }
            )

        # Add negative features
        for feature, coef, rank in features["negative"]:
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coefficient": coef,
                    "direction": "negative",
                    "rank": rank,
                }
            )

        df = pd.DataFrame(rows)
        filename = f"feature_importance_{model_name}_{ngram_name}_{class_name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
