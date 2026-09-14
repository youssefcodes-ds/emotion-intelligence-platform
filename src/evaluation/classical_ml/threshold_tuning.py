"""Threshold tuning and optimization module for classical ML.

Optimize decision thresholds to maximize specific metrics
(precision, recall, F1) per emotion class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class ThresholdTuner:
    """Find optimal decision thresholds for multi-class classification.

    For each class, tests different probability thresholds and selects
    the one that maximizes a chosen metric (precision, recall, or F1).
    """

    def __init__(self, metric: str = "f1"):
        """Initialize threshold tuner.

        Parameters
        ----------
        metric : str
            Metric to optimize: 'precision', 'recall', or 'f1'
        """
        if metric not in ["precision", "recall", "f1"]:
            raise ValueError(f"metric must be 'precision', 'recall', or 'f1', got {metric}")
        self.metric = metric
        self.thresholds = {}
        self.scores = {}

    def find_best_thresholds(
        self, model: Any, X_val, y_val, label_names: list[str]
    ) -> dict:
        """Find optimal thresholds for each class using validation data.

        For each class, uses one-vs-rest approach:
        - Binary problem: is this class or not?
        - Tests thresholds from 0.1 to 0.9
        - Records the threshold that maximizes the chosen metric

        Parameters
        ----------
        model : estimator with predict_proba
            Fitted model with calibrated probabilities
        X_val : feature matrix
            Validation features
        y_val : np.ndarray
            Validation labels
        label_names : list[str]
            List of emotion class names

        Returns
        -------
        thresholds : dict
            class_name -> optimal_threshold
        """
        proba = model.predict_proba(X_val)
        n_classes = len(label_names)

        for class_idx, class_name in enumerate(label_names):
            # Binary problem: this class vs. others
            y_binary = (y_val == class_idx).astype(int)
            class_proba = proba[:, class_idx]

            best_threshold = 0.5
            best_score = -1.0
            scores_for_class = {}

            # Test different thresholds
            for threshold in np.arange(0.1, 1.0, 0.05):
                y_pred_binary = (class_proba >= threshold).astype(int)

                if self.metric == "precision":
                    score = precision_score(y_binary, y_pred_binary, zero_division=0)
                elif self.metric == "recall":
                    score = recall_score(y_binary, y_pred_binary, zero_division=0)
                else:  # f1
                    score = f1_score(y_binary, y_pred_binary, zero_division=0)

                scores_for_class[round(threshold, 2)] = round(score, 4)

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            self.thresholds[class_name] = round(best_threshold, 2)
            self.scores[class_name] = {
                "best_threshold": round(best_threshold, 2),
                "best_score": round(best_score, 4),
                "metric": self.metric,
                "all_scores": scores_for_class,
            }

        return self.thresholds

    def apply_thresholds(
        self, model: Any, X, label_names: list[str], default_threshold: float = 0.5
    ) -> np.ndarray:
        """Apply tuned thresholds to make predictions.

        For multi-class, uses argmax of probabilities above class-specific thresholds.
        If no class exceeds its threshold, uses argmax overall.

        Parameters
        ----------
        model : estimator with predict_proba
            Fitted model
        X : feature matrix
            Data to predict on
        label_names : list[str]
            List of emotion class names
        default_threshold : float
            Fallback threshold if none specified for a class

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels
        """
        proba = model.predict_proba(X)
        n_samples = proba.shape[0]
        predictions = []

        for i in range(n_samples):
            sample_proba = proba[i]

            # Apply class-specific thresholds
            thresholded_proba = np.zeros_like(sample_proba)
            for class_idx, class_name in enumerate(label_names):
                threshold = self.thresholds.get(class_name, default_threshold)
                if sample_proba[class_idx] >= threshold:
                    thresholded_proba[class_idx] = sample_proba[class_idx]

            # If any class exceeded threshold, pick the highest
            if thresholded_proba.max() > 0:
                pred = np.argmax(thresholded_proba)
            else:
                # Otherwise use argmax of all probabilities
                pred = np.argmax(sample_proba)

            predictions.append(pred)

        return np.array(predictions)

    def print_summary(self) -> None:
        """Print threshold tuning results summary."""
        print(f"\n{'=' * 80}")
        print(f"THRESHOLD TUNING RESULTS (optimizing for: {self.metric.upper()})")
        print(f"{'=' * 80}")

        for class_name, score_info in self.scores.items():
            print(f"\n{class_name.upper()}")
            print("-" * 80)
            print(
                f"  Optimal threshold: {score_info['best_threshold']:.2f}  "
                f"({self.metric}={score_info['best_score']:.4f})"
            )

            print(f"  All tested thresholds:")
            for threshold, score in sorted(score_info["all_scores"].items()):
                marker = " ← best" if threshold == score_info["best_threshold"] else ""
                print(f"    {threshold:.2f}: {score:.4f}{marker}")
