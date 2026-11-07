
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import cv2
import numpy as np
import torch


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

CHECKPOINT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "best_cnn_lstm_spatial_v3.pt"
)
MODEL_SCRIPT = ROOT / "src" / "algorithms" / "cnn_lstm" / "58_train_cnn_lstm_segmentation.py"

OUTPUT_DIR = ROOT / "results" / "cnn_lstm" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8
THRESHOLDS = [0.30, 0.50, 0.70, 0.90]

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():
    spec = spec_from_file_location("training_module", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model script: {MODEL_SCRIPT}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.CNNLSTMSegmenter().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded:", CHECKPOINT.name)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Saved validation Dice:", checkpoint.get("val_dice", "unknown"))
    print("Saved inference threshold:",
          checkpoint.get("inference_threshold", "unknown"))

    return model


# =========================================================
# SELECT VIDEO
# =========================================================

def choose_video():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilename(
            title="Select the road video used in your demo",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None


# =========================================================
# PREPROCESS — SAME AS DEMO
# =========================================================

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )


# =========================================================
# PREDICT PROBABILITIES
# =========================================================

def predict_probabilities(model, frames):
    array = np.stack(frames).astype(np.float32) / 255.0

    x = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
    x = x.to(DEVICE)

    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.sigmoid(logits)[0, :, 0]

    return probabilities.cpu().numpy()


# =========================================================
# BUILD DIAGNOSTIC IMAGE
# =========================================================

def make_panel(image, title, panel_width=480, panel_height=360):
    image = cv2.resize(
        image,
        (panel_width, panel_height),
        interpolation=cv2.INTER_NEAREST
        if len(image.shape) == 2 else cv2.INTER_AREA,
    )

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(image, (0, 0), (panel_width, 36), (25, 25, 25), -1)
    cv2.putText(
        image, title, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return image


def save_diagnostic(original, probability, frame_number, video_stem):
    height, width = original.shape[:2]

    # Probability heatmap: blue/green/yellow/red indicate increasing values.
    probability_large = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    )
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(probability_large, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )

    panels = [
        make_panel(original, "Original frame"),
        make_panel(heatmap, "Probability heatmap"),
    ]

    # Compare multiple thresholds on the exact same prediction.
    for threshold in THRESHOLDS:
        mask = probability >= threshold
        coverage = 100.0 * mask.mean()

        print(
            f"Threshold {threshold:.2f}: "
            f"{coverage:.2f}% of pixels predicted as damage"
        )

        mask_image = np.uint8(mask) * 255
        panels.append(
            make_panel(
                mask_image,
                f"Mask threshold {threshold:.2f} | {coverage:.1f}%"
            )
        )

    # Overlay at the current demo threshold (0.70).
    mask = probability >= 0.70
    mask_large = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    overlay = original.copy()
    red = np.zeros_like(original)
    red[:] = (0, 0, 255)
    blended = cv2.addWeighted(original, 0.45, red, 0.55, 0)
    overlay[mask_large] = blended[mask_large]

    panels.append(make_panel(overlay, "Overlay at threshold 0.70"))

    # Two columns, multiple rows.
    panel_width = panels[0].shape[1]
    panel_height = panels[0].shape[0]
    columns = 2
    rows = (len(panels) + columns - 1) // columns

    canvas = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )

    for index, panel in enumerate(panels):
        row, col = divmod(index, columns)
        canvas[
            row * panel_height:(row + 1) * panel_height,
            col * panel_width:(col + 1) * panel_width,
        ] = panel

    output_path = OUTPUT_DIR / (
        f"{video_stem}_diagnostic_frame_{frame_number:05d}.jpg"
    )
    cv2.imwrite(str(output_path), canvas)

    print("\nDiagnostic image saved to:")
    print(output_path)


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("CNN + SPATIAL LSTM — PREDICTION DIAGNOSTIC")
    print("=" * 60)

    model = load_model()
    video_path = choose_video()

    if not video_path or not video_path.is_file():
        print("No valid video selected.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    window_size = max(1, round(fps * 2))
    sample_positions = np.linspace(
        0, window_size - 1, SEQUENCE_LENGTH
    ).round().astype(int)

    frames = []
    originals = []

    try:
        # Inspect the first two-second window.
        for _ in range(window_size):
            ok, frame = cap.read()
            if not ok:
                break
            originals.append(frame)

        if not originals:
            raise RuntimeError("No frames could be read from this video.")

        sampled = [
            preprocess_frame(originals[min(pos, len(originals) - 1)])
            for pos in sample_positions
        ]

        probabilities = predict_probabilities(model, sampled)

        # Choose the middle sampled prediction for inspection.
        selected_index = len(sampled) // 2
        original_index = min(
            int(sample_positions[selected_index]),
            len(originals) - 1,
        )

        probability = probabilities[selected_index]

        print(f"\nVideo: {video_path.name}")
        print(f"Inspecting frame: {original_index}")
        print(
            "\nProbability statistics "
            "(before applying any threshold):"
        )
        print(f"Minimum:  {probability.min():.4f}")
        print(f"Mean:     {probability.mean():.4f}")
        print(f"Median:   {np.median(probability):.4f}")
        print(f"Maximum:  {probability.max():.4f}")

        save_diagnostic(
            originals[original_index],
            probability,
            original_index,
            video_path.stem,
        )

    finally:
        cap.release()


if __name__ == "__main__":
    main()
