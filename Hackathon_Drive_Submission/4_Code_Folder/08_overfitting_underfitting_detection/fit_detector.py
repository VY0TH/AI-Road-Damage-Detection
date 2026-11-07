"""Step 8: Overfitting / Underfitting Algorithmic Detector."""
from __future__ import annotations

import pandas as pd


def detect_fit_status(
    df: pd.DataFrame,
    gap_threshold: float = 0.10,  # 10% accuracy divergence
    min_train_acc_threshold: float = 0.65,  # underfitting if train acc < 65%
) -> dict:
    if len(df) < 1:
        return {"status": "INSUFFICIENT_DATA", "reason": "No training epochs recorded."}

    last_row = df.iloc[-1]
    final_train_acc = float(last_row.get("train_accuracy", 0.0))
    final_val_acc = float(last_row.get("val_accuracy", 0.0))
    final_train_loss = float(last_row.get("train_loss", 0.0))
    final_val_loss = float(last_row.get("val_loss", 0.0))

    generalization_gap = final_train_acc - final_val_acc
    best_val_loss = float(df["val_loss"].min()) if "val_loss" in df.columns else final_val_loss

    # Check for loss divergence in multi-epoch runs
    loss_diverging = False
    if len(df) >= 3 and "val_loss" in df.columns:
        recent_val_losses = list(df["val_loss"].tail(3))
        loss_diverging = recent_val_losses[2] > recent_val_losses[1] > recent_val_losses[0]

    # Diagnosis Rules
    diagnosis = "NORMAL_FIT"
    confidence = "HIGH"
    signals = []

    if final_train_acc < min_train_acc_threshold and final_val_acc < min_train_acc_threshold:
        diagnosis = "UNDERFITTING"
        signals.append(f"Model capacity/learning is insufficient: Training accuracy ({final_train_acc*100:.1f}%) is below the {min_train_acc_threshold*100:.0f}% benchmark.")
        signals.append(f"Both training loss ({final_train_loss:.3f}) and validation loss ({final_val_loss:.3f}) remain high.")

    elif generalization_gap >= gap_threshold or loss_diverging:
        diagnosis = "OVERFITTING"
        if generalization_gap >= gap_threshold:
            signals.append(f"Substantial generalization gap ({generalization_gap*100:.1f}%) between train ({final_train_acc*100:.1f}%) and validation ({final_val_acc*100:.1f}%).")
        if loss_diverging:
            signals.append(f"Validation loss is increasing across recent epochs while training loss continues decreasing.")
        if final_val_loss > best_val_loss * 1.15:
            signals.append(f"Final validation loss ({final_val_loss:.3f}) exceeded minimum recorded loss ({best_val_loss:.3f}) by >15%.")

    else:
        diagnosis = "HEALTHY_FIT"
        signals.append(f"Training and validation metrics track closely with a modest generalization gap ({generalization_gap*100:.1f}%).")

    return {
        "diagnosis": diagnosis,
        "generalization_gap": round(generalization_gap, 4),
        "final_train_accuracy": round(final_train_acc, 4),
        "final_val_accuracy": round(final_val_acc, 4),
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "loss_diverging": loss_diverging,
        "signals": signals,
    }
