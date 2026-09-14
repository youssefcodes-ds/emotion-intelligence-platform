"""Model calibration module for classical ML.

Implements Platt scaling calibration for probability estimates.
Ensures calibrated probabilities sum to 1 and are suitable for threshold tuning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ManualCalibratedModel:
    """Wrapper that calibrates predictions using validation set.
    
    Fits per-class calibrators (Platt scaling or isotonic regression) on
    validation data, then applies them during prediction.
    
    Advantages:
    - No data leakage (validation set only for calibration, not training)
    - Works with both LogisticRegression and LinearSVC
    - Produces proper probability estimates for threshold tuning
    """

    def __init__(self, model: Any, X_val, y_val, method: str = "sigmoid"):
        """Initialize calibrated model.

        Parameters
        ----------
        model : fitted estimator
            Base model already trained on training data
        X_val : feature matrix
            Validation data for calibration
        y_val : np.ndarray
            Validation labels
        method : str
            'sigmoid' (Platt scaling) or 'isotonic' regression
        """
        self.base_model = model
        self.method = method
        self.classes_ = np.unique(y_val)
        self.n_classes = len(self.classes_)

        # Get uncalibrated probabilities on validation set
        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val)
        else:
            # For SVM: convert decision function to probability-like scores
            decision_val = model.decision_function(X_val)
            proba_val = self._sigmoid(decision_val)

        # Train per-class calibrators
        self.calibrators = {}

        if method == "sigmoid":
            # Use Logistic Regression for Platt scaling
            for i, class_id in enumerate(self.classes_):
                y_binary = (y_val == class_id).astype(int)
                calib = LogisticRegression(max_iter=1000)
                calib.fit(proba_val[:, i].reshape(-1, 1), y_binary)
                self.calibrators[i] = calib
        else:  # isotonic
            # Use isotonic regression for more flexible calibration
            for i, class_id in enumerate(self.classes_):
                y_binary = (y_val == class_id).astype(int)
                calib = IsotonicRegression(out_of_bounds="clip")
                calib.fit(proba_val[:, i], y_binary)
                self.calibrators[i] = calib

    @staticmethod
    def _sigmoid(x):
        """Apply sigmoid function to normalize values."""
        return expit(x)

    def predict(self, X):
        """Predict class labels (same as base model).

        Parameters
        ----------
        X : feature matrix
            Data to predict on

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels
        """
        return self.base_model.predict(X)

    def predict_proba(self, X):
        """Predict calibrated probabilities.

        Parameters
        ----------
        X : feature matrix
            Data to predict on

        Returns
        -------
        proba : np.ndarray
            Calibrated probability estimates (sum to 1 per sample)
        """
        # Get base probabilities
        if hasattr(self.base_model, "predict_proba"):
            proba = self.base_model.predict_proba(X)
        else:
            # For SVM
            decision = self.base_model.decision_function(X)
            proba = self._sigmoid(decision)

        # Apply per-class calibration
        calibrated_proba = np.zeros_like(proba)
        for i in range(self.n_classes):
            calibrated_proba[:, i] = self.calibrators[i].predict(
                proba[:, i].reshape(-1, 1)
            ).ravel()

        # Normalize to sum to 1
        calibrated_proba = calibrated_proba / (
            calibrated_proba.sum(axis=1, keepdims=True) + 1e-10
        )
        return calibrated_proba
