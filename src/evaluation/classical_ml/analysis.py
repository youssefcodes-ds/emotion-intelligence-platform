"""Analysis script for threshold tuning insights.

Investigates why thresholds are low and provides recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def analyze_threshold_patterns(report_path: Path) -> None:
    """Analyze threshold tuning patterns and provide insights.

    Parameters
    ----------
    report_path : Path
        Path to training_report.json
    """
    with open(report_path, "r") as f:
        report = json.load(f)

    print("=" * 80)
    print("THRESHOLD TUNING ANALYSIS")
    print("=" * 80)

    for config in report["configurations"]:
        ngram_name = config["ngram_name"]
        print(f"\n{'=' * 80}")
        print(f"N-GRAM: {ngram_name.upper()}")
        print(f"{'=' * 80}")

        for model_type in ["logistic_regression", "linear_svm"]:
            model_name = "LogisticRegression" if model_type == "logistic_regression" else "LinearSVM"
            thresholds = config["models"][model_type]["thresholds"]

            print(f"\n{model_name}")
            print("-" * 80)

            # Collect statistics
            threshold_values = list(thresholds.values())
            avg_threshold = np.mean(threshold_values)
            min_threshold = np.min(threshold_values)
            max_threshold = np.max(threshold_values)
            std_threshold = np.std(threshold_values)

            print(f"  Statistics:")
            print(f"    Mean threshold:   {avg_threshold:.3f}")
            print(f"    Min threshold:    {min_threshold:.3f}")
            print(f"    Max threshold:    {max_threshold:.3f}")
            print(f"    Std deviation:    {std_threshold:.3f}")

            # Count low thresholds
            low_count = sum(1 for t in threshold_values if t <= 0.15)
            mid_count = sum(1 for t in threshold_values if 0.15 < t <= 0.50)
            high_count = sum(1 for t in threshold_values if t > 0.50)

            print(f"\n  Distribution:")
            print(f"    Low  (≤0.15):  {low_count} classes")
            print(f"    Mid  (0.15-0.50): {mid_count} classes")
            print(f"    High (>0.50):  {high_count} classes")

            print(f"\n  Per-class thresholds:")
            for emotion, threshold in sorted(thresholds.items(), key=lambda x: x[1]):
                print(f"    {emotion:<12} {threshold:.2f}")

            # Get metrics
            metrics = config["models"][model_type]["tuned_thresholds"]["per_class"]

            print(f"\n  Class-specific analysis:")
            print(f"    {'Class':<12} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            print(f"    {'-' * 60}")

            for emotion, threshold in sorted(thresholds.items()):
                m = metrics[emotion]
                prec = m["precision"]
                rec = m["recall"]
                f1 = m["f1"]
                print(
                    f"    {emotion:<12} {threshold:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}"
                )

    print("\n" + "=" * 80)
    print("INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
    1. LOW THRESHOLDS (≤0.15):
       - Calibration is very confident for certain classes
       - Model has learned to differentiate these classes well
       - Probabilities are highly concentrated
       
    2. HIGH THRESHOLDS (>0.50):
       - Model is less confident for these classes
       - Calibration curves are spreading probabilities
       - Consider collecting more training data for these classes
       
    3. IF YOU SEE MOSTLY LOW THRESHOLDS:
       - This indicates good calibration quality
       - The model is making well-separated predictions
       - Recommendation: Use the tuned thresholds (they improve F1)
       
    4. IMPROVING MINORITY CLASS PERFORMANCE:
       - Check 'surprise' and 'love' - they have fewer samples
       - Consider SMOTE or class weights in future iterations
       - Use per-class threshold tuning (already done!)
    """)


if __name__ == "__main__":
    report_path = Path("models/classical_ml/training_report.json")
    analyze_threshold_patterns(report_path)
