"""Classical ML orchestration - Main entry point for training and evaluation.

This script orchestrates the complete classical ML pipeline:
1. Load and display class distribution
2. TF-IDF vectorization with n-gram tuning
3. Train Logistic Regression and Linear SVM
4. Calibrate models
5. Evaluate on test set
6. Extract feature importance
7. Tune decision thresholds
8. Save all models and reports

Run from project root:
    python -m src.models.classical_ml.run
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.schema import LABEL_NAMES
from src.evaluation.classical_ml import (
    ThresholdTuner,
    evaluate_model,
    export_feature_importance_csv,
    get_feature_importance_logreg,
    get_feature_importance_svm,
    print_confusion_matrix,
    print_evaluation_report,
    print_feature_importance,
    print_metrics_summary,
)
from src.features.classical_ml import (
    NGRAM_CONFIGS,
    create_vectorizer,
    fit_vectorizer,
    get_feature_names,
    transform_texts,
)
from src.models.classical_ml import (
    ManualCalibratedModel,
    load_data,
    print_class_distribution,
    train_linear_svm,
    train_logistic_regression,
)
from src.utils.paths import MODELS_DIR, ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)


def evaluate_with_thresholds(
    model, X_test, y_test, label_names, thresholds_dict, model_name: str = "Model"
) -> dict:
    """Evaluate model using tuned thresholds.

    Parameters
    ----------
    model : estimator with predict_proba
        Fitted and calibrated model
    X_test : feature matrix
        Test features
    y_test : np.ndarray
        True test labels
    label_names : list[str]
        List of emotion class names
    thresholds_dict : dict
        Dictionary of class_name -> threshold
    model_name : str
        Name of model for reporting

    Returns
    -------
    metrics : dict
        Evaluation metrics with tuned thresholds
    """
    proba = model.predict_proba(X_test)
    n_samples = proba.shape[0]
    predictions = []

    # Apply thresholds
    for i in range(n_samples):
        sample_proba = proba[i]
        thresholded_proba = np.zeros_like(sample_proba)

        for class_idx, class_name in enumerate(label_names):
            threshold = thresholds_dict.get(class_name, 0.5)
            if sample_proba[class_idx] >= threshold:
                thresholded_proba[class_idx] = sample_proba[class_idx]

        # If any class exceeded threshold, pick the highest
        if thresholded_proba.max() > 0:
            pred = np.argmax(thresholded_proba)
        else:
            # Otherwise use argmax of all probabilities
            pred = np.argmax(sample_proba)

        predictions.append(pred)

    y_pred = np.array(predictions)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": model_name,
        "dataset": "test",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {},
    }

    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_names)), zero_division=0
    )

    for i, label_name in enumerate(label_names):
        metrics["per_class"][label_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return metrics


def train_pipeline_single_ngram(
    ngram_config: dict,
    X_train_text,
    y_train,
    X_val_text,
    y_val,
    X_test_text,
    y_test,
    all_results: dict,
) -> None:
    """Run training pipeline for a single n-gram configuration.

    Parameters
    ----------
    ngram_config : dict
        N-gram configuration with 'name' and 'ngram_range'
    X_train_text : np.ndarray
        Training texts
    y_train : np.ndarray
        Training labels
    X_val_text : np.ndarray
        Validation texts
    y_val : np.ndarray
        Validation labels
    X_test_text : np.ndarray
        Test texts
    y_test : np.ndarray
        Test labels
    all_results : dict
        Accumulator for results across all configurations
    """
    ngram_name = ngram_config["name"]
    ngram_range = ngram_config["ngram_range"]

    print("\n" + "=" * 80)
    print(f"N-GRAM CONFIG: {ngram_name.upper()} {ngram_range}")
    print("=" * 80)

    # ---- Step 1: TF-IDF Vectorization ----
    print("\n[STEP 1/7] TF-IDF VECTORIZATION")
    print("-" * 80)
    vectorizer = create_vectorizer(ngram_range)
    vectorizer, n_features = fit_vectorizer(vectorizer, X_train_text)
    print(f"✓ Vectorizer fitted with {n_features:,} features")

    X_train = transform_texts(vectorizer, X_train_text)
    X_val = transform_texts(vectorizer, X_val_text)
    X_test = transform_texts(vectorizer, X_test_text)
    print(
        f"✓ Vectorized shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # ---- Step 2: Train Logistic Regression ----
    print("\n[STEP 2/7] TRAINING LOGISTIC REGRESSION (class_weight='balanced')")
    print("-" * 80)
    logreg = train_logistic_regression(X_train, y_train)
    print(f"✓ Logistic Regression trained")

    # ---- Step 3: Train Linear SVM ----
    print("\n[STEP 3/7] TRAINING LINEAR SVM (class_weight='balanced')")
    print("-" * 80)
    svm = train_linear_svm(X_train, y_train)
    print(f"✓ Linear SVM trained")

    # ---- Step 4: Calibrate models ----
    print("\n[STEP 4/7] CALIBRATING MODELS (Platt scaling on validation set)")
    print("-" * 80)
    calibrated_logreg = ManualCalibratedModel(logreg, X_val, y_val, method="sigmoid")
    calibrated_svm = ManualCalibratedModel(svm, X_val, y_val, method="sigmoid")
    print(f"✓ Both models calibrated using Platt scaling")

    # ---- Step 5: Evaluate on test set (default thresholds) ----
    print("\n[STEP 5/7] EVALUATING ON TEST SET (default thresholds=0.5)")
    print("-" * 80)

    print("\n  Logistic Regression (Calibrated):")
    logreg_metrics = evaluate_model(
        calibrated_logreg, X_test, y_test, LABEL_NAMES, "LogisticRegression", "test"
    )
    print_metrics_summary(logreg_metrics)
    logreg_report = print_evaluation_report(
        calibrated_logreg, X_test, y_test, LABEL_NAMES
    )
    print(logreg_report)
    print_confusion_matrix(logreg_metrics, LABEL_NAMES)

    print("\n  Linear SVM (Calibrated):")
    svm_metrics = evaluate_model(
        calibrated_svm, X_test, y_test, LABEL_NAMES, "LinearSVC", "test"
    )
    print_metrics_summary(svm_metrics)
    svm_report = print_evaluation_report(calibrated_svm, X_test, y_test, LABEL_NAMES)
    print(svm_report)
    print_confusion_matrix(svm_metrics, LABEL_NAMES)

    # ---- Step 6: Feature Importance ----
    print("\n[STEP 6/7] EXTRACTING FEATURE IMPORTANCE")
    print("-" * 80)

    feature_names_array = get_feature_names(vectorizer)

    logreg_importance = get_feature_importance_logreg(
        calibrated_logreg, feature_names_array, LABEL_NAMES, top_k=15
    )
    print_feature_importance(logreg_importance, "Logistic Regression")

    svm_importance = get_feature_importance_svm(
        calibrated_svm, feature_names_array, LABEL_NAMES, top_k=15
    )
    print_feature_importance(svm_importance, "Linear SVM")

    # Export feature importance to CSV
    feature_importance_dir = MODELS_DIR / "classical_ml" / "feature_importance"
    export_feature_importance_csv(
        logreg_importance, "logreg", ngram_name, feature_importance_dir
    )
    export_feature_importance_csv(
        svm_importance, "svm", ngram_name, feature_importance_dir
    )
    print(f"✓ Feature importance exported to {feature_importance_dir}")

    # ---- Step 7: Threshold Tuning ----
    print("\n[STEP 7/7] TUNING DECISION THRESHOLDS")
    print("-" * 80)

    print("\n  Optimizing thresholds for F1-score (Logistic Regression):")
    logreg_tuner = ThresholdTuner(metric="f1")
    logreg_tuner.find_best_thresholds(calibrated_logreg, X_val, y_val, LABEL_NAMES)
    logreg_tuner.print_summary()

    # Evaluate with tuned thresholds
    logreg_metrics_tuned = evaluate_with_thresholds(
        calibrated_logreg,
        X_test,
        y_test,
        LABEL_NAMES,
        logreg_tuner.thresholds,
        "LogisticRegression (Tuned)",
    )
    print("\n  Results with tuned thresholds:")
    print_metrics_summary(logreg_metrics_tuned)

    print("\n  Optimizing thresholds for F1-score (Linear SVM):")
    svm_tuner = ThresholdTuner(metric="f1")
    svm_tuner.find_best_thresholds(calibrated_svm, X_val, y_val, LABEL_NAMES)
    svm_tuner.print_summary()

    # Evaluate with tuned thresholds
    svm_metrics_tuned = evaluate_with_thresholds(
        calibrated_svm,
        X_test,
        y_test,
        LABEL_NAMES,
        svm_tuner.thresholds,
        "LinearSVC (Tuned)",
    )
    print("\n  Results with tuned thresholds:")
    print_metrics_summary(svm_metrics_tuned)

    # ---- Save Models ----
    print("\n[SAVING] Persisting models, vectorizer, and tuners")
    print("-" * 80)

    models_dir = MODELS_DIR / "classical_ml"
    models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = models_dir / f"tfidf_vectorizer_{ngram_name}.joblib"
    logreg_path = models_dir / f"logreg_{ngram_name}.joblib"
    svm_path = models_dir / f"svm_{ngram_name}.joblib"
    cal_logreg_path = models_dir / f"calibrated_logreg_{ngram_name}.joblib"
    cal_svm_path = models_dir / f"calibrated_svm_{ngram_name}.joblib"
    logreg_tuner_path = models_dir / f"logreg_tuner_{ngram_name}.joblib"
    svm_tuner_path = models_dir / f"svm_tuner_{ngram_name}.joblib"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(logreg, logreg_path)
    joblib.dump(svm, svm_path)
    joblib.dump(calibrated_logreg, cal_logreg_path)
    joblib.dump(calibrated_svm, cal_svm_path)
    joblib.dump(logreg_tuner, logreg_tuner_path)
    joblib.dump(svm_tuner, svm_tuner_path)

    print(f"✓ Saved {ngram_name} models:")
    print(f"  - tfidf_vectorizer_{ngram_name}.joblib")
    print(f"  - logreg_{ngram_name}.joblib (+ calibrated)")
    print(f"  - svm_{ngram_name}.joblib (+ calibrated)")
    print(f"  - logreg_tuner_{ngram_name}.joblib")
    print(f"  - svm_tuner_{ngram_name}.joblib")

    # ---- Store Results ----
    config_result = {
        "ngram_name": ngram_name,
        "ngram_range": ngram_range,
        "n_features": int(n_features),
        "models": {
            "logistic_regression": {
                "default_thresholds": logreg_metrics,
                "tuned_thresholds": logreg_metrics_tuned,
                "feature_importance": logreg_importance,
                "thresholds": logreg_tuner.thresholds,
            },
            "linear_svm": {
                "default_thresholds": svm_metrics,
                "tuned_thresholds": svm_metrics_tuned,
                "feature_importance": svm_importance,
                "thresholds": svm_tuner.thresholds,
            },
        },
    }
    all_results["configurations"].append(config_result)


def main() -> int:
    """Run the complete classical ML pipeline."""
    ensure_dirs()

    print("=" * 80)
    print("CLASSICAL ML PIPELINE: TF-IDF + LogisticRegression + LinearSVM")
    print("=" * 80)

    # ---- Load Data ----
    print("\n[STEP 0/7] LOADING DATA")
    print("-" * 80)
    X_train_text, y_train = load_data("train")
    X_val_text, y_val = load_data("validation")
    X_test_text, y_test = load_data("test")

    train_dist = print_class_distribution(y_train, "train")
    val_dist = print_class_distribution(y_val, "validation")
    test_dist = print_class_distribution(y_test, "test")

    # Initialize results accumulator
    all_results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_distribution": {
            "train": train_dist,
            "validation": val_dist,
            "test": test_dist,
        },
        "configurations": [],
    }

    # ---- Run pipeline for each n-gram config ----
    for ngram_config in NGRAM_CONFIGS:
        train_pipeline_single_ngram(
            ngram_config,
            X_train_text,
            y_train,
            X_val_text,
            y_val,
            X_test_text,
            y_test,
            all_results,
        )

    # ---- Save comprehensive report ----
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    report_path = MODELS_DIR / "classical_ml" / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Full training report saved to:")
    print(f"  {report_path}")

    # Print summary table
    print("\n[SUMMARY TABLE]")
    print("-" * 80)
    print(
        f"{'N-gram':<12} {'Model':<18} {'Accuracy':<12} {'Macro F1':<12} {'Best Threshold':<15}"
    )
    print("-" * 80)

    for config in all_results["configurations"]:
        ngram_name = config["ngram_name"]

        # LogisticRegression
        logreg_acc = config["models"]["logistic_regression"]["tuned_thresholds"][
            "accuracy"
        ]
        logreg_f1 = config["models"]["logistic_regression"]["tuned_thresholds"]["macro_f1"]
        logreg_threshold = config["models"]["logistic_regression"]["thresholds"].get(
            "joy", 0.5
        )

        print(
            f"{ngram_name:<12} {'LogisticReg':<18} {logreg_acc:<12.4f} {logreg_f1:<12.4f} {logreg_threshold:<15.2f}"
        )

        # LinearSVC
        svm_acc = config["models"]["linear_svm"]["tuned_thresholds"]["accuracy"]
        svm_f1 = config["models"]["linear_svm"]["tuned_thresholds"]["macro_f1"]
        svm_threshold = config["models"]["linear_svm"]["thresholds"].get("joy", 0.5)

        print(
            f"{'':<12} {'LinearSVM':<18} {svm_acc:<12.4f} {svm_f1:<12.4f} {svm_threshold:<15.2f}"
        )

    print("\n" + "=" * 80)
    print("Models saved to: models/classical_ml/")
    print("Features saved to: models/classical_ml/feature_importance/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
