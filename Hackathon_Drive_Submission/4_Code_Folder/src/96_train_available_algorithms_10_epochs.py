"""Train each available dashboard model for 10 epochs into new checkpoints.

The GAN hybrid is excluded because its generated samples are unapproved and its
augmented classifier checkpoint does not exist. Existing checkpoints are never
overwritten. The real grouped split is used for detector and crop-classifier
training; CNN + Spatial LSTM keeps its existing video-mask dataset.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 10
RUN_ROOT = ROOT / "results" / "retraining_10_epochs_20261014"


def run(label: str, script: str, env: dict[str, str], args: list[str] | None = None) -> None:
    command = [sys.executable, str(ROOT / "src" / script), *(args or [])]
    print(f"\n{'=' * 72}\n{label}: {' '.join(command)}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite training run directory: {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)

    env = os.environ.copy()
    env["ROAD_DAMAGE_EPOCHS"] = str(EPOCHS)

    # Fine-tune a fresh detector run on the leakage-grouped dataset.
    run(
        "YOLO12n detector",
        "94_train_yolo_high_recall.py",
        env,
        ["--epochs", str(EPOCHS), "--imgsz", "640", "--batch", "2", "--device", "cpu", "--name", "yolo12n_grouped_10ep_20261014"],
    )

    # ResNet creates crops directly from grouped train/validation images and labels.
    resnet_dir = RUN_ROOT / "resnet50_grouped"
    resnet_env = env.copy()
    resnet_env["ROAD_DAMAGE_OUTPUT_DIR"] = str(resnet_dir)
    run("ResNet50 crop classifier", "82_train_resnet50.py", resnet_env)

    # Train EfficientNet and ViT on the same real grouped crops.
    grouped_crops = resnet_dir / "crops"
    efficientnet_env = env.copy()
    efficientnet_env["ROAD_DAMAGE_OUTPUT_DIR"] = str(RUN_ROOT / "efficientnet_b0_grouped")
    efficientnet_env["ROAD_DAMAGE_DATA_DIR"] = str(grouped_crops)
    run("EfficientNetB0 crop classifier", "39_train_efficientnet.py", efficientnet_env)

    vit_env = env.copy()
    vit_env["ROAD_DAMAGE_OUTPUT_DIR"] = str(RUN_ROOT / "vit_grouped")
    vit_env["ROAD_DAMAGE_CROP_DIR"] = str(grouped_crops)
    run("ViT-B/16 crop classifier", "84_train_vit.py", vit_env)

    # Preserve the current segmenter checkpoint and train a separate 10-epoch run.
    cnn_env = env.copy()
    cnn_env["ROAD_DAMAGE_OUTPUT_DIR"] = str(RUN_ROOT / "cnn_lstm_v3")
    run("CNN + Spatial LSTM segmenter", "58_train_cnn_lstm_segmentation.py", cnn_env)

    print(f"\nAll requested trainable runs completed. New outputs: {RUN_ROOT}", flush=True)
    print("GAN hybrid was intentionally not trained: its samples remain unapproved and its classifier checkpoint is absent.", flush=True)


if __name__ == "__main__":
    main()
