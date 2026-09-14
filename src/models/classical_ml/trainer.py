"""Model training module for classical ML.

Handles training of Logistic Regression and Linear SVM classifiers
with balanced class weights for imbalanced emotion classification.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# ---- Configuration ----
LOGREG_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
}

SVM_PARAMS = {
    "class_weight": "balanced",
    "random_state": 42,
    "dual": False,
    "max_iter": 2000,
    "loss": "squared_hinge",
}


# ---- Training Functions ----
def train_logistic_regression(X, y) -> LogisticRegression:
    """Train Logistic Regression with balanced class weights.

    Parameters
    ----------
    X : sparse matrix or ndarray
        Feature matrix (TF-IDF)
    y : np.ndarray
        Labels

    Returns
    -------
    model : LogisticRegression
        Fitted model
    """
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X, y)
    return model


def train_linear_svm(X, y) -> LinearSVC:
    """Train Linear SVM with balanced class weights.

    Parameters
    ----------
    X : sparse matrix or ndarray
        Feature matrix (TF-IDF)
    y : np.ndarray
        Labels

    Returns
    -------
    model : LinearSVC
        Fitted model
    """
    model = LinearSVC(**SVM_PARAMS)
    model.fit(X, y)
    return model
