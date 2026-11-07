"""Step 9: Comparative Analysis - Pre-Correction vs. Post-Correction."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def compare_models(
    baseline_history_path: Path,
    corrected_history_path: Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data = json.loads(Path(baseline_history_path).read_text(encoding="utf-8"))
    base_df = pd.DataFrame(base_data.get("history", []))

    corr_data = json.loads(Path(corrected_history_path).read_text(encoding="utf-8"))
    corr_df = pd.DataFrame(corr_data.get("history", []))

    base_best_val_acc = float(base_df["val_accuracy"].max()) if not base_df.empty else 0.0
    corr_best_val_acc = float(corr_df["val_accuracy"].max()) if not corr_df.empty else 0.0
    acc_delta = corr_best_val_acc - base_best_val_acc

    base_final_val_loss = float(base_df["val_loss"].iloc[-1]) if not base_df.empty else 0.0
    corr_final_val_loss = float(corr_df["val_loss"].iloc[-1]) if not corr_df.empty else 0.0
    loss_delta = corr_final_val_loss - base_final_val_loss

    # Plot comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["Validation Accuracy (%)", "Validation Loss"]
    base_vals = [base_best_val_acc * 100, base_final_val_loss]
    corr_vals = [corr_best_val_acc * 100, corr_final_val_loss]

    x = [0, 1]
    width = 0.35
    ax.bar([p - width/2 for p in x], base_vals, width, label="Initial Baseline", color="#d62728", alpha=0.85)
    ax.bar([p + width/2 for p in x], corr_vals, width, label="After Corrections (Step 9)", color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_title("Initial Baseline vs. Corrected Retrained Model", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()

    plot_path = output_dir / "baseline_vs_corrected_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "baseline_best_val_acc": round(base_best_val_acc, 4),
        "corrected_best_val_acc": round(corr_best_val_acc, 4),
        "accuracy_improvement": round(acc_delta, 4),
        "baseline_final_val_loss": round(base_final_val_loss, 4),
        "corrected_final_val_loss": round(corr_final_val_loss, 4),
        "loss_reduction": round(-loss_delta, 4),
        "comparison_plot": str(plot_path),
    }
