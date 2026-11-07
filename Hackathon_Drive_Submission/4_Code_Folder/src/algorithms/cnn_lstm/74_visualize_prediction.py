
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import cv2
import numpy as np
import torch
from tkinter import Tk, filedialog

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


def load_model():
    spec = spec_from_file_location("training_module", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {MODEL_SCRIPT}")

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

    print("Model loaded:", CHECKPOINT.name)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Saved validation Dice:", checkpoint.get("val_dice", "unknown"))

    return model


def choose_video():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilename(
            title="Select your road video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def predict_probabilities(model, frames):
    array = np.stack(frames).astype(np.float32) / 255.0
    x = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)

    with torch.inference_mode():
        logits = model(x.to(DEVICE))
        probabilities = torch.sigmoid(logits)[0, :, 0]

    return probabilities.cpu().numpy()


def add_title(image, title):
    height, width = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = cv2.resize(image, (480, 300))
    cv2.rectangle(image, (0, 0), (480, 35), (25, 25, 25), -1)
    cv2.putText(
        image, title, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return image


def save_diagnostic(original, probability, video_stem):
    h, w = original.shape[:2]

    prob_large = cv2.resize(probability, (w, h))
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(prob_large, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )

    panels = [
        add_title(original.copy(), "Original frame"),
        add_title(heatmap, "Probability heatmap"),
    ]

    print("\nPredicted coverage at different thresholds:")

    for threshold in THRESHOLDS:
        mask = probability >= threshold
        coverage = float(mask.mean() * 100)

        print(f"Threshold {threshold:.2f}: {coverage:.2f}%")

        panels.append(
            add_title(
                np.uint8(mask) * 255,
                f"Mask threshold {threshold:.2f} ({coverage:.1f}%)",
            )
        )

    mask = probability >= 0.70
    mask_large = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    overlay = original.copy()
    red = np.zeros_like(original)
    red[:] = (0, 0, 255)
    blended = cv2.addWeighted(original, 0.45, red, 0.55, 0)
    overlay[mask_large] = blended[mask_large]

    panels.append(add_title(overlay, "Overlay at threshold 0.70"))

    # Arrange panels in two columns
    panel_h, panel_w = panels[0].shape[:2]
    rows = (len(panels) + 1) // 2

    canvas = np.zeros((rows * panel_h, 2 * panel_w, 3), dtype=np.uint8)

    for i, panel in enumerate(panels):
        row, col = divmod(i, 2)
        canvas[
            row * panel_h:(row + 1) * panel_h,
            col * panel_w:(col + 1) * panel_w,
        ] = panel

    output = OUTPUT_DIR / f"{video_stem}_diagnostic.jpg"
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not save diagnostic image: {output}")

    print("\nDiagnostic image saved to:")
    print(output)


def main():
    print("=" * 55)
    print("CNN + SPATIAL LSTM DIAGNOSTIC")
    print("=" * 55)

    model = load_model()
    video = choose_video()

    if not video or not video.is_file():
        print("No valid video selected.")
        return

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    window_size = max(1, round(fps * 2))
    positions = np.linspace(
        0, window_size - 1, SEQUENCE_LENGTH
    ).round().astype(int)

    frames = []

    try:
        for _ in range(window_size):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("Could not read frames from the video.")

    sampled = [
        preprocess_frame(frames[min(int(pos), len(frames) - 1)])
        for pos in positions
    ]

    probabilities = predict_probabilities(model, sampled)

    index = SEQUENCE_LENGTH // 2
    original = frames[min(int(positions[index]), len(frames) - 1)]
    probability = probabilities[index]

    print("\nProbability statistics:")
    print(f"Minimum: {probability.min():.4f}")
    print(f"Mean:    {probability.mean():.4f}")
    print(f"Median:  {np.median(probability):.4f}")
    print(f"Maximum: {probability.max():.4f}")

    save_diagnostic(original, probability, video.stem)


if __name__ == "__main__":
    main()
