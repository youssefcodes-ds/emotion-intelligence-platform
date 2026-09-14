"""Post-training analysis runner.

Generates visualizations and insights from training results.

Run from project root:
    python -m src.evaluation.classical_ml.analyze
"""

import sys
from pathlib import Path

from src.evaluation.classical_ml.analysis import analyze_threshold_patterns
from src.evaluation.classical_ml.visualizations import main as generate_visualizations
from src.utils.paths import MODELS_DIR


def main() -> int:
    """Generate all post-training analysis and visualizations."""
    report_path = MODELS_DIR / "classical_ml" / "training_report.json"

    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        print("Run training first: python -m src.models.classical_ml.run")
        return 1

    output_dir = Path("reports") / "classical_ml"

    print("=" * 80)
    print("POST-TRAINING ANALYSIS")
    print("=" * 80)

    # Generate visualizations
    print("\n[1/2] Generating visualizations...")
    generate_visualizations(report_path, output_dir)

    # Analyze threshold patterns
    print("\n[2/2] Analyzing threshold tuning patterns...")
    analyze_threshold_patterns(report_path)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
