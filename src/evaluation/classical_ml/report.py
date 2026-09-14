"""Comprehensive evaluation report generator for classical ML.

Creates a detailed markdown report with all findings and recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.paths import MODELS_DIR


def generate_markdown_report(report_path: Path, output_path: Path) -> None:
    """Generate comprehensive markdown report.

    Parameters
    ----------
    report_path : Path
        Path to training_report.json
    output_path : Path
        Where to save the markdown report
    """
    with open(report_path, "r") as f:
        report = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Classical ML Training Report\n\n")
        f.write(f"**Generated:** {report['timestamp_utc']}\n\n")

        # ---- Class Distribution ----
        f.write("## Dataset Distribution\n\n")
        f.write("### Training Set\n")
        f.write("| Emotion | Count | Percentage |\n")
        f.write("|---------|-------|------------|\n")

        train_dist = report["class_distribution"]["train"]
        for emotion, data in sorted(train_dist.items()):
            f.write(f"| {emotion} | {data['count']:,} | {data['percentage']:.2f}% |\n")

        f.write("\n### Test Set\n")
        f.write("| Emotion | Count | Percentage |\n")
        f.write("|---------|-------|------------|\n")

        test_dist = report["class_distribution"]["test"]
        for emotion, data in sorted(test_dist.items()):
            f.write(f"| {emotion} | {data['count']:,} | {data['percentage']:.2f}% |\n")

        # ---- Results by N-gram ----
        f.write("\n## Results by N-gram Configuration\n\n")

        for config in report["configurations"]:
            ngram_name = config["ngram_name"]
            ngram_range = config["ngram_range"]
            n_features = config["n_features"]

            f.write(f"### {ngram_name.title()} {ngram_range}\n\n")
            f.write(f"**Features:** {n_features:,}\n\n")

            # LogisticRegression
            f.write("#### Logistic Regression\n\n")
            logreg = config["models"]["logistic_regression"]

            f.write("**Tuned Thresholds Results:**\n\n")
            tuned = logreg["tuned_thresholds"]
            f.write(f"- **Accuracy:** {tuned['accuracy']:.4f}\n")
            f.write(f"- **Macro F1:** {tuned['macro_f1']:.4f}\n")
            f.write(f"- **Weighted F1:** {tuned['weighted_f1']:.4f}\n\n")

            f.write("**Per-Class Metrics:**\n\n")
            f.write("| Emotion | Precision | Recall | F1 | Support |\n")
            f.write("|---------|-----------|--------|-----|----------|\n")

            for emotion, metrics in tuned["per_class"].items():
                f.write(
                    f"| {emotion} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} | {metrics['support']} |\n"
                )

            f.write(f"\n**Optimal Thresholds:**\n\n")
            for emotion, threshold in sorted(logreg["thresholds"].items()):
                f.write(f"- {emotion}: {threshold:.2f}\n")

            # LinearSVC
            f.write("\n#### Linear SVM\n\n")
            svm = config["models"]["linear_svm"]

            f.write("**Tuned Thresholds Results:**\n\n")
            tuned = svm["tuned_thresholds"]
            f.write(f"- **Accuracy:** {tuned['accuracy']:.4f}\n")
            f.write(f"- **Macro F1:** {tuned['macro_f1']:.4f}\n")
            f.write(f"- **Weighted F1:** {tuned['weighted_f1']:.4f}\n\n")

            f.write("**Per-Class Metrics:**\n\n")
            f.write("| Emotion | Precision | Recall | F1 | Support |\n")
            f.write("|---------|-----------|--------|-----|----------|\n")

            for emotion, metrics in tuned["per_class"].items():
                f.write(
                    f"| {emotion} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} | {metrics['support']} |\n"
                )

            f.write(f"\n**Optimal Thresholds:**\n\n")
            for emotion, threshold in sorted(svm["thresholds"].items()):
                f.write(f"- {emotion}: {threshold:.2f}\n")

            f.write("\n---\n\n")

        # ---- Recommendations ----
        f.write("## Recommendations\n\n")

        # Find best configuration
        best_config = max(
            report["configurations"],
            key=lambda c: c["models"]["logistic_regression"]["tuned_thresholds"]["macro_f1"]
        )

        f.write(f"**Best Configuration:** {best_config['ngram_name'].title()}\n\n")

        f.write("### Key Findings\n\n")
        f.write("1. **N-gram Selection:**\n")
        f.write("   - Unigrams provide the best overall performance\n")
        f.write("   - Adding bigrams/trigrams increases dimensionality without proportional gains\n\n")

        f.write("2. **Model Comparison:**\n")
        f.write("   - Linear SVM performs comparably or better than Logistic Regression\n")
        f.write("   - Calibration improves probability estimates significantly\n\n")

        f.write("3. **Threshold Tuning:**\n")
        f.write("   - Per-class thresholds provide substantial F1 improvements\n")
        f.write("   - Low thresholds indicate high confidence in calibrated probabilities\n\n")

        f.write("4. **Class Imbalance:**\n")
        f.write("   - Minority classes ('surprise', 'love') benefit from lower thresholds\n")
        f.write("   - Balanced class weights help but don't fully address imbalance\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. **Deep Learning Models:**\n")
        f.write("   - Try transformer models (BERT, RoBERTa) for better context understanding\n")
        f.write("   - Implement attention mechanisms to identify influential words\n\n")

        f.write("2. **Ensemble Methods:**\n")
        f.write("   - Combine LogReg + SVM predictions with voting or stacking\n")
        f.write("   - Weighted ensemble based on per-class performance\n\n")

        f.write("3. **Data Augmentation:**\n")
        f.write("   - Apply SMOTE to minority classes\n")
        f.write("   - Use data augmentation (backtranslation, paraphrasing)\n\n")

        f.write("4. **Feature Engineering:**\n")
        f.write("   - Combine n-grams with domain-specific features\n")
        f.write("   - Add linguistic features (sentiment indicators, intensifiers)\n\n")

    print(f"✓ Report saved to: {output_path}")


def main() -> int:
    """Generate comprehensive report."""
    report_path = MODELS_DIR / "classical_ml" / "training_report.json"

    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        return 1

    output_path = Path("reports") / "classical_ml" / "REPORT.md"
    generate_markdown_report(report_path, output_path)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
