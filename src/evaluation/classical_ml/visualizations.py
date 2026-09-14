"""Visualization and analysis tools for classical ML results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_training_report(report_path: Path) -> dict:
    """Load the JSON training report.

    Parameters
    ----------
    report_path : Path
        Path to training_report.json

    Returns
    -------
    report : dict
        Full training report
    """
    with open(report_path, "r") as f:
        return json.load(f)


def plot_accuracy_comparison(report: dict, output_dir: Path) -> None:
    """Plot accuracy comparison across configurations and models.

    Parameters
    ----------
    report : dict
        Training report
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = report["configurations"]
    ngram_names = [c["ngram_name"] for c in configs]

    logreg_default = [c["models"]["logistic_regression"]["default_thresholds"]["accuracy"] for c in configs]
    logreg_tuned = [c["models"]["logistic_regression"]["tuned_thresholds"]["accuracy"] for c in configs]
    svm_default = [c["models"]["linear_svm"]["default_thresholds"]["accuracy"] for c in configs]
    svm_tuned = [c["models"]["linear_svm"]["tuned_thresholds"]["accuracy"] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ngram_names))
    width = 0.2

    ax.bar(x - 1.5 * width, logreg_default, width, label="LogReg (Default)", alpha=0.8)
    ax.bar(x - 0.5 * width, logreg_tuned, width, label="LogReg (Tuned)", alpha=0.8)
    ax.bar(x + 0.5 * width, svm_default, width, label="SVM (Default)", alpha=0.8)
    ax.bar(x + 1.5 * width, svm_tuned, width, label="SVM (Tuned)", alpha=0.8)

    ax.set_xlabel("N-gram Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy Comparison: Default vs. Tuned Thresholds", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ngram_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: accuracy_comparison.png")


def plot_f1_comparison(report: dict, output_dir: Path) -> None:
    """Plot F1-score comparison across configurations and models.

    Parameters
    ----------
    report : dict
        Training report
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = report["configurations"]
    ngram_names = [c["ngram_name"] for c in configs]

    logreg_default = [c["models"]["logistic_regression"]["default_thresholds"]["macro_f1"] for c in configs]
    logreg_tuned = [c["models"]["logistic_regression"]["tuned_thresholds"]["macro_f1"] for c in configs]
    svm_default = [c["models"]["linear_svm"]["default_thresholds"]["macro_f1"] for c in configs]
    svm_tuned = [c["models"]["linear_svm"]["tuned_thresholds"]["macro_f1"] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ngram_names))
    width = 0.2

    ax.bar(x - 1.5 * width, logreg_default, width, label="LogReg (Default)", alpha=0.8)
    ax.bar(x - 0.5 * width, logreg_tuned, width, label="LogReg (Tuned)", alpha=0.8)
    ax.bar(x + 0.5 * width, svm_default, width, label="SVM (Default)", alpha=0.8)
    ax.bar(x + 1.5 * width, svm_tuned, width, label="SVM (Tuned)", alpha=0.8)

    ax.set_xlabel("N-gram Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Macro F1-Score Comparison: Default vs. Tuned Thresholds", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ngram_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: f1_comparison.png")


def plot_per_class_metrics(report: dict, output_dir: Path, ngram_name: str = "unigrams", model_type: str = "logistic_regression") -> None:
    """Plot per-class metrics (precision, recall, F1) for a specific configuration.

    Parameters
    ----------
    report : dict
        Training report
    output_dir : Path
        Directory to save plots
    ngram_name : str
        Which n-gram config to plot
    model_type : str
        'logistic_regression' or 'linear_svm'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find config
    config = None
    for c in report["configurations"]:
        if c["ngram_name"] == ngram_name:
            config = c
            break

    if config is None:
        print(f"Config {ngram_name} not found")
        return

    metrics = config["models"][model_type]["tuned_thresholds"]["per_class"]

    classes = list(metrics.keys())
    precision = [metrics[c]["precision"] for c in classes]
    recall = [metrics[c]["recall"] for c in classes]
    f1 = [metrics[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Emotion Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_type.replace('_', ' ').title()} - {ngram_name.title()} (Tuned Thresholds)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    filename = f"per_class_metrics_{model_type}_{ngram_name}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {filename}")


def plot_threshold_heatmap(report: dict, output_dir: Path, ngram_name: str = "unigrams", model_type: str = "logistic_regression") -> None:
    """Plot heatmap of thresholds across emotions.

    Parameters
    ----------
    report : dict
        Training report
    output_dir : Path
        Directory to save plots
    ngram_name : str
        Which n-gram config to plot
    model_type : str
        'logistic_regression' or 'linear_svm'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find config
    config = None
    for c in report["configurations"]:
        if c["ngram_name"] == ngram_name:
            config = c
            break

    if config is None:
        print(f"Config {ngram_name} not found")
        return

    thresholds = config["models"][model_type]["thresholds"]
    emotions = list(thresholds.keys())
    threshold_values = [thresholds[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 2))

    im = ax.imshow([threshold_values], cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_yticks([0])
    ax.set_yticklabels(["Threshold"])
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_title(f"Optimal Thresholds - {model_type.replace('_', ' ').title()} - {ngram_name.title()}", fontsize=14, fontweight="bold")

    # Add text annotations
    for i, (emotion, threshold) in enumerate(zip(emotions, threshold_values)):
        text = ax.text(i, 0, f"{threshold:.2f}", ha="center", va="center", color="black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Threshold Value")
    plt.tight_layout()
    filename = f"threshold_heatmap_{model_type}_{ngram_name}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {filename}")


def create_summary_table(report: dict, output_dir: Path) -> None:
    """Create and save a summary comparison table.

    Parameters
    ----------
    report : dict
        Training report
    output_dir : Path
        Directory to save table
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in report["configurations"]:
        ngram = config["ngram_name"]

        for model_type in ["logistic_regression", "linear_svm"]:
            model_name = "LogisticRegression" if model_type == "logistic_regression" else "LinearSVM"
            metrics = config["models"][model_type]["tuned_thresholds"]

            rows.append({
                "N-gram": ngram,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Best Threshold (joy)": config["models"][model_type]["thresholds"].get("joy", 0.5),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Accuracy", "Macro F1"], ascending=False)

    # Save as CSV
    csv_path = output_dir / "summary_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: summary_comparison.csv")

    # Save as formatted HTML
    html_path = output_dir / "summary_comparison.html"
    df_html = df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    with open(html_path, "w") as f:
        f.write(f"<h2>Classical ML Model Comparison</h2>\n{df_html}")
    print(f"✓ Saved: summary_comparison.html")

    # Print to console
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (sorted by Accuracy)")
    print("=" * 100)
    print(df.to_string(index=False))


def main(report_path: Path, output_dir: Path) -> None:
    """Generate all visualizations and analysis.

    Parameters
    ----------
    report_path : Path
        Path to training_report.json
    output_dir : Path
        Directory to save outputs
    """
    print("=" * 80)
    print("GENERATING VISUALIZATIONS AND ANALYSIS")
    print("=" * 80)

    report = load_training_report(report_path)

    # Overall comparisons
    plot_accuracy_comparison(report, output_dir)
    plot_f1_comparison(report, output_dir)

    # Per-config detailed analysis
    for config in report["configurations"]:
        ngram_name = config["ngram_name"]

        for model_type in ["logistic_regression", "linear_svm"]:
            plot_per_class_metrics(report, output_dir, ngram_name, model_type)
            plot_threshold_heatmap(report, output_dir, ngram_name, model_type)

    # Summary table
    create_summary_table(report, output_dir)

    print("\n✓ All visualizations generated!")


if __name__ == "__main__":
    from pathlib import Path

    report_path = Path("models/classical_ml/training_report.json")
    output_dir = Path("reports/classical_ml")

    main(report_path, output_dir)
