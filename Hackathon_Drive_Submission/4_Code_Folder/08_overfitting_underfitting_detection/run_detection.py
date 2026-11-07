"""Step 8 CLI Runner: Overfitting / Underfitting Detection and Rectification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_curve_analyzer import load_history, plot_learning_curves
from fit_detector import detect_fit_status
from rectification_advisor import get_rectification_plan
import pipeline_config as cfg


def run_step_8(history_path: Path | None = None, model_name: str = "baseline_cnn") -> dict:
    if history_path is None:
        history_path = cfg.RESULTS_DIR / "initial_training" / model_name / "training_history.json"
        if not history_path.is_file():
            # Check for alternative histories in results
            cand = cfg.RESULTS_DIR / "efficientnet_b0" / "training_history.json"
            if cand.is_file():
                history_path = cand
                model_name = "efficientnet_b0"

    print(f"\n{'=' * 70}")
    print(f"[STEP 8] OVERFITTING / UNDERFITTING DETECTION & RECTIFICATION")
    print(f"Target history: {history_path}")
    print(f"{'=' * 70}")

    df = load_history(history_path)
    output_plots_dir = cfg.RESULTS_DIR / "learning_curves"
    plots = plot_learning_curves(df, output_plots_dir, model_name=model_name)

    diagnosis = detect_fit_status(df)
    rectification = get_rectification_plan(diagnosis)

    print(f"\nDiagnosis Result: [{diagnosis['diagnosis']}]")
    print(f"  Final Train Accuracy: {diagnosis['final_train_accuracy']*100:.2f}%")
    print(f"  Final Val Accuracy  : {diagnosis['final_val_accuracy']*100:.2f}%")
    print(f"  Generalization Gap  : {diagnosis['generalization_gap']*100:.2f}%")
    print("\nEvidence & Signals:")
    for sig in diagnosis["signals"]:
        print(f"  - {sig}")

    print("\nRectification Strategy:")
    for tech in rectification["recommended_techniques"]:
        print(f"  * {tech}")

    full_report = {
        "model_name": model_name,
        "history_file": str(history_path),
        "diagnosis": diagnosis,
        "rectification_plan": rectification,
        "generated_plots": plots,
    }

    report_json = cfg.RESULTS_DIR / "step08_fit_diagnosis_report.json"
    report_md = cfg.RESULTS_DIR / "step08_fit_diagnosis_report.md"

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Step 8: Overfitting / Underfitting Diagnosis & Rectification Report\n\n")
        f.write(f"**Model Analyzed**: `{model_name}`  \n")
        f.write(f"**Diagnosis Verdict**: `{diagnosis['diagnosis']}`  \n\n")
        f.write(f"## Quantitative Metrics\n")
        f.write(f"- Final Train Accuracy: `{diagnosis['final_train_accuracy']*100:.2f}%`\n")
        f.write(f"- Final Val Accuracy: `{diagnosis['final_val_accuracy']*100:.2f}%`\n")
        f.write(f"- Generalization Gap: `{diagnosis['generalization_gap']*100:.2f}%`\n")
        f.write(f"- Train Loss: `{diagnosis['final_train_loss']:.4f}`, Val Loss: `{diagnosis['final_val_loss']:.4f}`\n\n")
        f.write("## Rectification Plan for Step 9\n")
        for tech in rectification["recommended_techniques"]:
            f.write(f"- **{tech}**\n")
        f.write("\n### Hyperparameter Adjustments:\n")
        for k, v in rectification["hyperparameter_adjustments"].items():
            f.write(f"- `{k}`: {v}\n")

    print(f"\n[STEP 8 COMPLETED] Reports saved to:")
    print(f"  JSON: {report_json}")
    print(f"  MD  : {report_md}")
    print(f"{'=' * 70}\n")
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="baseline_cnn")
    args = parser.parse_args()

    run_step_8(args.history, args.model_name)


if __name__ == "__main__":
    main()
