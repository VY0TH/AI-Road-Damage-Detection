
from pathlib import Path
import csv
import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_CHECKPOINT = (
    ROOT / "results" / "training" / "yolo12n_improved_30_epochs"
    / "weights" / "best.pt"
)
CLASSIFIER_CHECKPOINT = (
    ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"
)

OUTPUT_DIR = ROOT / "results" / "road_video"
FRAME_DIR = OUTPUT_DIR / "annotated_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
IMAGE_SIZE = 224
FRAME_INTERVAL_SECONDS = 2
CONFIDENCE = 0.001

CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_classifier():
    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    names = checkpoint.get("class_names", CLASS_NAMES)
    if list(names) != CLASS_NAMES:
        raise ValueError(f"Unexpected classifier class order: {names}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, list(names)


def enhance_crop(crop_bgr):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(l)
    return cv2.cvtColor(
        cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB
    )


def main():
    video_path = input(
        "Enter the full path of your road video: "
    ).strip().strip('"')

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = YOLO(str(YOLO_CHECKPOINT))
    classifier, class_names = load_classifier()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_step = max(1, round(fps * FRAME_INTERVAL_SECONDS))
    rows = []
    frame_index = 0
    processed_frames = 0

    csv_path = OUTPUT_DIR / "video_detections.csv"

    try:
        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            columns = [
                "video", "frame", "timestamp_seconds",
                "x1", "y1", "x2", "y2",
                "yolo_class", "yolo_confidence",
                "efficientnet_class", "efficientnet_confidence",
            ]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                current_frame = frame_index
                frame_index += 1

                if current_frame % frame_step != 0:
                    continue

                timestamp = current_frame / fps
                result = detector.predict(
                    source=frame,
                    imgsz=416,
                    conf=CONFIDENCE,
                    iou=0.45,
                    max_det=50,
                    device=DEVICE,
                    verbose=False,
                )[0]

                annotated = frame.copy()

                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = (
                            box.xyxy[0].cpu().numpy().astype(int)
                        )

                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        x2 = max(0, min(x2, frame.shape[1]))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        y2 = max(0, min(y2, frame.shape[0]))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        crop_rgb = enhance_crop(crop)
                        pil_image = Image.fromarray(crop_rgb)
                        tensor = TRANSFORM(pil_image).unsqueeze(0)

                        with torch.inference_mode():
                            probabilities = torch.softmax(
                                classifier(tensor), dim=1
                            )[0]
                            cls_id = int(probabilities.argmax())
                            cls_conf = float(probabilities[cls_id])

                        detector_id = int(box.cls[0].item())
                        detector_name = str(result.names[detector_id])
                        detector_conf = float(box.conf[0].item())
                        classifier_name = class_names[cls_id]

                        label = (
                            f"{classifier_name} {cls_conf:.2f}"
                        )
                        cv2.rectangle(
                            annotated, (x1, y1), (x2, y2),
                            (0, 255, 0), 2
                        )
                        cv2.putText(
                            annotated, label,
                            (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA
                        )

                        row = {
                            "video": Path(video_path).name,
                            "frame": current_frame,
                            "timestamp_seconds": round(timestamp, 3),
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2,
                            "yolo_class": detector_name,
                            "yolo_confidence": round(detector_conf, 5),
                            "efficientnet_class": classifier_name,
                            "efficientnet_confidence": round(cls_conf, 5),
                        }
                        writer.writerow(row)
                        rows.append(row)

                frame_name = f"frame_{current_frame:07d}.jpg"
                cv2.imwrite(
                    str(FRAME_DIR / frame_name), annotated
                )
                processed_frames += 1

                if processed_frames % 10 == 0:
                    print(
                        f"Processed {processed_frames} sampled frames..."
                    )

    finally:
        cap.release()

    print("\nVideo processing complete.")
    print("Sampled frames:", processed_frames)
    print("Detections recorded:", len(rows))
    print("Annotated frames:", FRAME_DIR)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
