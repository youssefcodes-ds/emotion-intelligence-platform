"""Complete workflow runner for classical ML pipeline."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a Python command and return exit code.

    Parameters
    ----------
    cmd : list
        Command to run (as list for subprocess)
    description : str
        Description for user

    Returns
    -------
    exit_code : int
        Exit code from command
    """
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)

    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode


def main() -> int:
    """Run complete classical ML workflow: train → analyze → report."""
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║               CLASSICAL ML COMPLETE WORKFLOW                               ║
    ║    TF-IDF + Logistic Regression + Linear SVM with Threshold Tuning        ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Training
    print("\n[STEP 1/3] Training models...")
    exit_code = run_command(
        [sys.executable, "-m", "src.models.classical_ml.run"],
        "TRAINING MODELS"
    )
    if exit_code != 0:
        print("❌ Training failed!")
        return 1

    # Step 2: Analysis
    print("\n[STEP 2/3] Analyzing results...")
    exit_code = run_command(
        [sys.executable, "-m", "src.evaluation.classical_ml.analyze"],
        "ANALYZING RESULTS"
    )
    if exit_code != 0:
        print("⚠️  Analysis failed, but continuing...")

    # Step 3: Report
    print("\n[STEP 3/3] Generating report...")
    exit_code = run_command(
        [sys.executable, "-m", "src.evaluation.classical_ml.report"],
        "GENERATING REPORT"
    )
    if exit_code != 0:
        print("⚠️  Report generation failed, but training succeeded...")

    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                         ✓ WORKFLOW COMPLETE!                              ║
    ╚════════════════════════════════════════════════════════════════════════════╝

    📊 OUTPUTS:
       Models:        models/classical_ml/*.joblib
       Report:        reports/classical_ml/REPORT.md
       Visualizations: reports/classical_ml/*.png
       Analysis:      reports/classical_ml/summary_comparison.csv

    📖 NEXT STEPS:
       1. Review reports/classical_ml/REPORT.md for detailed findings
       2. Check visualizations for model comparisons
       3. Examine feature_importance CSVs to understand model decisions
       4. Start deep learning models with insights from classical ML
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
