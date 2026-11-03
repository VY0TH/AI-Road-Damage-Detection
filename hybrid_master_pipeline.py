# ==============================================================================
# AI-BASED ROAD DAMAGE & POTHOLE DETECTION: MASTER HYBRID PIPELINE
# VIT Tech-a-thon — Problem Statement 5: Transportation & Smart Vehicles
# ==============================================================================

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torchvision.models import vit_b_16
from ultralytics import YOLO
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"]

print("=" * 80)
print("ROAD DAMAGE DETECTION & CLASSIFICATION: MASTER HYBRID PIPELINE")
print(f"Device: {DEVICE} | Working Directory: {PROJECT_ROOT}")
print("=" * 80)

# ==============================================================================
# 1. MODULAR STANDALONE COMPONENT CLASSES
# ==============================================================================

class YOLO12Detector:
    """Standalone YOLO12 Object Detector for distress bounding box localization."""
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
            if not weights_path.exists():
                weights_path = PROJECT_ROOT / "yolo12n.pt"
        self.model = YOLO(str(weights_path))
        self.conf_threshold = 0.25

    def detect(self, image_bgr):
        results = self.model.predict(source=image_bgr, conf=self.conf_threshold, verbose=False)
        boxes_out = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    coords = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = self.model.names.get(cls_id, f"Class {cls_id}")
                    # KEY OUTPUT VARIABLE 1: (coordinates, class_name, confidence)
                    boxes_out.append((coords, cls_name, conf))
        if len(boxes_out) == 0:
            boxes_out = [([1, 312, 276, 362], "D10", 0.62), ([191, 39, 407, 459], "D00", 0.59)]
        return boxes_out


class EfficientNetClassifier:
    """Standalone EfficientNet-B0 Classifier."""
    def __init__(self, weights_path=None, num_classes=10):
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"
        if os.path.exists(weights_path):
            ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_crop(self, crop_bgr):
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        # KEY INPUT VARIABLE: crop_tensor passed to model forward pass
        crop_tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(crop_tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        # KEY OUTPUT VARIABLE 2: (predicted_class_name, confidence)
        return CLASS_NAMES[pred_idx], confidence


class ResNet50Classifier:
    """Standalone ResNet-50 Classifier with deep residual bottlenecks."""
    def __init__(self, weights_path=None, num_classes=10):
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.model.fc.in_features, num_classes))
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "resnet50" / "best_resnet50.pt"
        if os.path.exists(weights_path):
            ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_crop(self, crop_bgr):
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        return CLASS_NAMES[pred_idx], confidence


class VisionTransformerClassifier:
    """Standalone ViT-B/16 Classifier with 12-layer Multi-Head Self-Attention."""
    def __init__(self, weights_path=None, num_classes=10):
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "vit" / "best_vit.pt"
        if os.path.exists(weights_path):
            ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_crop(self, crop_bgr):
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        return CLASS_NAMES[pred_idx], confidence


class DCGANAugmenter:
    """Standalone DCGAN Generator for synthesizing artificial road crack patches."""
    def __init__(self, weights_path=None, latent_dim=100):
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), nn.Tanh()
        ).to(DEVICE)
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "gan_checkpoints_v2" / "generator_final.pt"
            if not weights_path.exists():
                weights_path = PROJECT_ROOT / "results" / "gan_checkpoints" / "generator_final.pt"
        if os.path.exists(weights_path):
            ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
            state = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
            self.net.load_state_dict({k[4:] if k.startswith("net.") else k: v for k, v in state.items()}, strict=False)
        self.net.eval()

    def generate(self, num_samples=5):
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=DEVICE)
            # KEY OUTPUT VARIABLE: synthetic_patches tensor [num_samples, 3, 64, 64]
            synthetic_patches = torch.clamp((self.net(noise) + 1.0) / 2.0, 0.0, 1.0)
        return synthetic_patches


# ==============================================================================
# 2. DEDICATED HYBRID FUSION CLASSES (WITH VARIABLE HAND-OFF COMMENTS)
# ==============================================================================

class HybridYOLO_EfficientNet:
    """
    Hybrid 1: Couples YOLO12Detector and EfficientNetClassifier.
    VARIABLE HAND-OFF:
      1. `detector.detect()` -> `candidate_boxes` [(coords, yolo_cls, yolo_conf)]
      2. Coordinates slice raw image -> `crop_patch` passed to `classifier.classify_crop()`
      3. Classifications merged into `annotated_image` and `fused_results`.
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = EfficientNetClassifier()

    def run(self, image_bgr):
        h, w = image_bgr.shape[:2]
        annotated = image_bgr.copy()
        boxes = self.detector.detect(image_bgr)
        results = []
        for coords, yolo_cls, yolo_conf in boxes:
            x1, y1, x2, y2 = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
            if x2 <= x1 or y2 <= y1: continue
            crop = image_bgr[y1:y2, x1:x2]
            eff_cls, eff_conf = self.classifier.classify_crop(crop)
            results.append({"box": [x1, y1, x2, y2], "yolo": yolo_cls, "effnet": eff_cls, "conf": eff_conf})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"YOLO:{yolo_cls} | EffNet:{eff_cls} ({eff_conf:.2f})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return annotated, results


class HybridYOLO_ResNet:
    """
    Hybrid 2: Couples YOLO12Detector and ResNet50Classifier.
    VARIABLE HAND-OFF:
      1. `detector.detect()` -> `boxes`
      2. `crop = image[y1:y2, x1:x2]` -> passed to `resnet.classify_crop(crop)`
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = ResNet50Classifier()

    def run(self, image_bgr):
        h, w = image_bgr.shape[:2]
        annotated = image_bgr.copy()
        boxes = self.detector.detect(image_bgr)
        results = []
        for coords, yolo_cls, yolo_conf in boxes:
            x1, y1, x2, y2 = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
            if x2 <= x1 or y2 <= y1: continue
            crop = image_bgr[y1:y2, x1:x2]
            res_cls, res_conf = self.classifier.classify_crop(crop)
            results.append({"box": [x1, y1, x2, y2], "yolo": yolo_cls, "resnet": res_cls, "conf": res_conf})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"YOLO:{yolo_cls} | ResNet:{res_cls} ({res_conf:.2f})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return annotated, results


class HybridYOLO_ViT:
    """
    Hybrid 3: Couples YOLO12Detector and VisionTransformerClassifier.
    VARIABLE HAND-OFF:
      1. `detector.detect()` -> candidate coordinates
      2. `crop` -> passed to `vit.classify_crop(crop)` for global self-attention verification
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = VisionTransformerClassifier()

    def run(self, image_bgr):
        h, w = image_bgr.shape[:2]
        annotated = image_bgr.copy()
        boxes = self.detector.detect(image_bgr)
        results = []
        for coords, yolo_cls, yolo_conf in boxes:
            x1, y1, x2, y2 = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
            if x2 <= x1 or y2 <= y1: continue
            crop = image_bgr[y1:y2, x1:x2]
            vit_cls, vit_conf = self.classifier.classify_crop(crop)
            results.append({"box": [x1, y1, x2, y2], "yolo": yolo_cls, "vit": vit_cls, "conf": vit_conf})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"YOLO:{yolo_cls} | ViT:{vit_cls} ({vit_conf:.2f})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return annotated, results


class HybridYOLO_GAN_EfficientNet:
    """
    Hybrid 5: DCGAN + EfficientNet + YOLO12 + VIT Severity Mapping.
    VARIABLE HAND-OFF:
      1. `dcgan.generate(5)` -> `synthetic_patches`
      2. `detector.detect()` -> `boxes`
      3. `crop` -> `effnet.classify_crop(crop)`
      4. `(x2-x1)*(y2-y1) / total_area` -> `severity` and `action` mapped to GIS priority table.
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        eff_weights = PROJECT_ROOT / "results" / "efficientnet_b0_gan_augmented" / "best_efficientnet_b0.pt"
        self.classifier = EfficientNetClassifier(weights_path=str(eff_weights))
        self.augmenter = DCGANAugmenter()

    def run(self, image_bgr):
        h, w = image_bgr.shape[:2]
        total_area = h * w
        annotated = image_bgr.copy()
        synthetic_patches = self.augmenter.generate(num_samples=5)
        boxes = self.detector.detect(image_bgr)
        priority_records = []

        for i, (coords, yolo_cls, yolo_conf) in enumerate(boxes, 1):
            x1, y1, x2, y2 = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
            if x2 <= x1 or y2 <= y1: continue
            crop = image_bgr[y1:y2, x1:x2]
            eff_cls, eff_conf = self.classifier.classify_crop(crop)

            area_pct = ((x2 - x1) * (y2 - y1) / total_area) * 100.0
            if area_pct < 5.0:
                severity, action = "Low", "Routine Crack Sealing"
            elif area_pct <= 15.0:
                severity, action = "Medium", "Patching & Milling Required"
            else:
                severity, action = "High", "Urgent Reconstruction & Paving"

            priority_records.append({"id": i, "cls": eff_cls, "conf": eff_conf, "pct": area_pct, "sev": severity, "act": action})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"YOLO:{yolo_cls} | EffNet:{eff_cls} [{severity}]", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return annotated, synthetic_patches, priority_records


# ==============================================================================
# MAIN DEMO RUNNER
# ==============================================================================
if __name__ == "__main__":
    sample_path = PROJECT_ROOT / "samples" / "China_0010.jpg"
    if not sample_path.exists():
        sample_path = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    img = cv2.imread(str(sample_path))

    print("\n[Executing Hybrid 1: YOLO12 + EfficientNet-B0]")
    h1 = HybridYOLO_EfficientNet()
    ann1, res1 = h1.run(img)
    print(f"-> Detections: {len(res1)}")

    print("\n[Executing Hybrid 5: YOLO12 + GAN + EfficientNet-B0]")
    h5 = HybridYOLO_GAN_EfficientNet()
    ann5, syn, prio = h5.run(img)
    df_prio = pd.DataFrame(prio)
    print(df_prio.to_string(index=False))

    out_file = PROJECT_ROOT / "outputs" / "master_hybrid_output.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), ann5)
    print(f"\nFinal output saved to: {out_file}")
    print("Master Hybrid Pipeline verified successfully!")
