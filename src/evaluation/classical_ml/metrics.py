"""Model evaluation and metrics module for classical ML.

Comprehensive evaluation including accuracy, precision, recall, F1,
confusion matrices, and per-class analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_model(
    model: Any, X, y, label_names: list[str], model_name: str = "Model", set_name: str = "test"
) -> dict:
    """Evaluate a trained model on a dataset.

    Computes accuracy, per-class metrics (precision, recall, F1),
    macro-averaged metrics, and confusion matrix.

    Parameters
    ----------
    model : estimator
        Fitted model with predict method
    X : feature matrix
        Features to evaluate on
    y : np.ndarray
        True labels
    label_names : list[str]
        List of emotion class names
    model_name : str
        Name of model for display
    set_name : str
        Name of dataset ('test', 'validation', etc.)

    Returns
    -------
    metrics : dict
        Comprehensive evaluation metrics
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average=None, labels=range(len(label_names)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=range(len(label_names)))

    # Build metrics dictionary
    metrics = {
        "model": model_name,
        "dataset": set_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }

    for i, label_name in enumerate(label_names):
        metrics["per_class"][label_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return metrics


def print_evaluation_report(
    model: Any, X, y, label_names: list[str], model_name: str = "Model", set_name: str = "test"
) -> str:
    """Print detailed evaluation report using scikit-learn's classification_report.

    Parameters
    ----------
    model : estimator
        Fitted model
    X : feature matrix
        Features to evaluate on
    y : np.ndarray
        True labels
    label_names : list[str]
        List of emotion class names
    model_name : str
        Name of model for display
    set_name : str
        Name of dataset

    Returns
    -------
    report_str : str
        Formatted classification report
    """
    y_pred = model.predict(X)

    report_str = classification_report(
        y, y_pred, target_names=label_names, digits=4, zero_division=0
    )

    return report_str


def print_metrics_summary(metrics: dict, model_name: str = "") -> None:
    """Print metrics in a readable format.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model
    model_name : str
        Name of model for display
    """
    if model_name:
        print(f"\n{model_name.upper()}")
    print("-" * 80)
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print("\n  Per-class breakdown:")

    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"    {class_name:<12} P={class_metrics['precision']:.4f}  "
            f"R={class_metrics['recall']:.4f}  F1={class_metrics['f1']:.4f}  "
            f"Support={class_metrics['support']}"
        )


def print_confusion_matrix(metrics: dict, label_names: list[str]) -> None:
    """Print confusion matrix in a readable format.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model (contains confusion_matrix)
    label_names : list[str]
        List of emotion class names
    """
    cm = np.array(metrics["confusion_matrix"])

    print("\n  Confusion Matrix:")
    print("  " + " ".join([f"{name:>8s}" for name in label_names]))

    for i, label in enumerate(label_names):
        row_str = f"{label:>8s} " + " ".join([f"{cm[i, j]:>8d}" for j in range(len(label_names))])
        print(f"  {row_str}")
