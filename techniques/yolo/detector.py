"""YOLO12n Road Damage Detection & Localization Engine."""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class YOLODetector:
    def __init__(self, checkpoint_path: Path | None = None, conf_threshold: float = 0.25):
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            # Fallback to local weights
            checkpoint_path = PROJECT_ROOT / "yolo12n.pt"

        self.model = YOLO(str(checkpoint_path))
        self.conf_threshold = conf_threshold

    def detect(self, image: Image.Image | np.ndarray) -> list[dict]:
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().item())
                cls_id = int(boxes.cls[i].cpu().item())
                cls_name = r.names.get(cls_id, str(cls_id))
                detections.append({
                    "box": [round(coord, 1) for coord in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
        return detections
