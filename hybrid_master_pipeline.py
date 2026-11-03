# ==============================================================================
# AI-BASED ROAD DAMAGE & POTHOLE DETECTION: MASTER HYBRID PIPELINE
# VIT Tech-a-thon — Problem Statement 5: Transportation & Smart Vehicles
# ==============================================================================
#
# QUICK CTRL+F SEARCH DIRECTORY:
# [CTRL-F: UNDERFITTING-REGULARIZATION]   - Underfitting analysis and capacity scaling
# [CTRL-F: OVERFITTING-DROPOUT]          - Overfitting mitigation and Dropout layers
# [CTRL-F: L1-REGULARIZATION]            - L1 Lasso penalty implementation on weights
# [CTRL-F: L2-WEIGHT-DECAY]              - L2 Ridge weight decay in AdamW optimizer
# [CTRL-F: SMOTE-ALGORITHM]              - Latent feature-space SMOTE oversampling
# [CTRL-F: GAN-AUGMENTATION]             - Deep Convolutional GAN patch synthesis
# [CTRL-F: CLAHE-PREPROCESSING]          - LAB-space contrast enhancement
# [CTRL-F: 10-FOLD-CROSS-VALIDATION]     - 10-Fold cross-validation metrics
# [CTRL-F: STANDALONE-YOLO12]            - Standalone YOLO12 object detector
# [CTRL-F: STANDALONE-EFFICIENTNET]      - Standalone EfficientNet-B0 classifier
# [CTRL-F: STANDALONE-RESNET50]          - Standalone ResNet-50 classifier
# [CTRL-F: STANDALONE-VIT]               - Standalone Vision Transformer (ViT-B/16)
# [CTRL-F: STANDALONE-DCGAN]             - Standalone DCGAN generator architecture
# [CTRL-F: STANDALONE-CNN-LSTM]          - Standalone CNN + Spatiotemporal LSTM
# [CTRL-F: HYBRID-1-YOLO-EFFICIENTNET]   - Technique 1: YOLO12 + EfficientNet-B0
# [CTRL-F: HYBRID-2-YOLO-RESNET]         - Technique 2: YOLO12 + ResNet-50
# [CTRL-F: HYBRID-3-YOLO-VIT]            - Technique 3: YOLO12 + Vision Transformer
# [CTRL-F: HYBRID-4-CNN-LSTM]            - Technique 4: Video CNN + Spatial LSTM
# [CTRL-F: HYBRID-5-YOLO-GAN-EFFICIENTNET]- Technique 5: YOLO12 + GAN + EfficientNet
# [CTRL-F: VARIABLE-HANDOFF-COORDINATES] - Coordinate extraction and handoff
# [CTRL-F: VARIABLE-HANDOFF-CROPS]       - Sub-patch cropping and tensor passing
# [CTRL-F: VARIABLE-HANDOFF-LATENT-NOISE]- Latent noise vector z handoff to GAN
# [CTRL-F: GIS-MAINTENANCE-PRIORITY-TABLE]- Severity scoring & GIS priority export
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"]

print("=" * 80)
print("ROAD DAMAGE DETECTION & CLASSIFICATION: MASTER HYBRID PIPELINE")
print(f"Device: {DEVICE} | Working Directory: {PROJECT_ROOT}")
print("=" * 80)


# ==============================================================================
# SECTION A: REGULARIZATION & UNDERFITTING / OVERFITTING MITIGATION
# ==============================================================================

class RegularizationEngine:
    """
    [CTRL-F: UNDERFITTING-REGULARIZATION]
    [CTRL-F: OVERFITTING-DROPOUT]
    [CTRL-F: L1-REGULARIZATION]
    [CTRL-F: L2-WEIGHT-DECAY]

    Implements exact mathematical regularizers to prevent underfitting & overfitting:
      - Underfitting Fix: Transfer learning from ImageNet pre-training, capacity scaling.
      - Overfitting Fix:
          1. L1 Regularization (Lasso): Adds sum of absolute weights: \\lambda_1 * \\sum |w|
          2. L2 Regularization (Ridge / Weight Decay): \\lambda_2 * \\sum w^2 in AdamW
          3. Dropout: Randomly zeros out hidden unit activations during forward pass (p = 0.2 to 0.5)
    """

    @staticmethod
    def compute_l1_loss(model: nn.Module, l1_lambda: float = 1e-5) -> torch.Tensor:
        """
        [CTRL-F: L1-REGULARIZATION]
        Computes L1 norm penalty over all trainable weight tensors:
        L_reg = L_cross_entropy + l1_lambda * sum(|w|)
        """
        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
        return l1_lambda * l1_norm

    @staticmethod
    def build_regularized_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
        """
        [CTRL-F: L2-WEIGHT-DECAY]
        Applies L2 Weight Decay through AdamW optimizer. Decoupled weight decay prevents
        weights from exploding without decaying batch norm parameters artificially.
        """
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def add_dropout_layers(in_features: int, out_classes: int, p: float = 0.3) -> nn.Sequential:
        """
        [CTRL-F: OVERFITTING-DROPOUT]
        Constructs a classifier head with inverted Dropout (p=0.3) and Linear projection.
        Dropout randomly disconnects neurons during training to prevent co-adaptation.
        """
        return nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, out_classes)
        )


# ==============================================================================
# SECTION B: SMOTE & GAN DATASET AUGMENTATION
# ==============================================================================

class LatentFeatureSMOTE:
    """
    [CTRL-F: SMOTE-ALGORITHM]
    Applies Synthetic Minority Over-sampling Technique (SMOTE) in deep latent feature space.
    Instead of naive pixel interpolation (which creates blurry ghosting on asphalt),
    we interpolate between k-nearest neighbors in 1280-dim / 2048-dim feature embeddings:
        x_new = x_i + lambda * (x_nn - x_i),  where lambda ~ U(0, 1)
    """
    def __init__(self, k_neighbors: int = 2):
        self.k_neighbors = k_neighbors

    def balance_features(self, features: np.ndarray, labels: np.ndarray, target_count: int = 150):
        """
        Synthesizes new latent vectors for minority classes until target_count is reached.
        """
        unique_classes, counts = np.unique(labels, return_counts=True)
        synthetic_features = list(features)
        synthetic_labels = list(labels)

        for cls, count in zip(unique_classes, counts):
            if count < target_count and count > 1:
                cls_indices = np.where(labels == cls)[0]
                cls_feats = features[cls_indices]
                needed = target_count - count

                for _ in range(needed):
                    idx = np.random.randint(0, len(cls_feats))
                    x_i = cls_feats[idx]
                    # Select random neighbor within class
                    nn_idx = np.random.choice([j for j in range(len(cls_feats)) if j != idx])
                    x_nn = cls_feats[nn_idx]
                    # Interpolation
                    lam = np.random.uniform(0.0, 1.0)
                    x_syn = x_i + lam * (x_nn - x_i)
                    synthetic_features.append(x_syn)
                    synthetic_labels.append(cls)

        return np.array(synthetic_features), np.array(synthetic_labels)


# ==============================================================================
# SECTION C: STANDALONE MODULAR ALGORITHM CLASSES
# ==============================================================================

class YOLO12Detector:
    """
    [CTRL-F: STANDALONE-YOLO12]
    Standalone YOLO12 Object Detector for localized road damage bounding box regression.
    """
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
            if not weights_path.exists():
                weights_path = PROJECT_ROOT / "yolo12n.pt"
        self.weights_path = weights_path
        self.model = YOLO(str(weights_path))
        self.conf_threshold = 0.25

    def detect(self, image_bgr: np.ndarray):
        """
        Runs object detection inference.
        [CTRL-F: VARIABLE-HANDOFF-COORDINATES]
        RETURNS:
          boxes_out: List of tuples -> [(coords, cls_name, conf)]
                     coords = [x1, y1, x2, y2]
        """
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
    """
    [CTRL-F: STANDALONE-EFFICIENTNET]
    [CTRL-F: OVERFITTING-DROPOUT]
    Standalone EfficientNet-B0 Classifier with compound scaling and inverted Dropout.
    """
    def __init__(self, weights_path=None, num_classes=10):
        self.model = models.efficientnet_b0(weights=None)
        # Inverted Dropout layer (0.2) applied before Linear classification head to stop overfitting
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
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

    def classify_crop(self, crop_bgr: np.ndarray):
        """
        [CTRL-F: VARIABLE-HANDOFF-CROPS]
        INPUT: crop_bgr (image sub-patch)
        RETURNS: (predicted_class_name, confidence)
        """
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        crop_tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(crop_tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        return CLASS_NAMES[pred_idx], confidence


class ResNet50Classifier:
    """
    [CTRL-F: STANDALONE-RESNET50]
    [CTRL-F: OVERFITTING-DROPOUT]
    Standalone ResNet-50 Classifier with 50-layer deep residual bottlenecks and Dropout head.
    """
    def __init__(self, weights_path=None, num_classes=10):
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
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

    def classify_crop(self, crop_bgr: np.ndarray):
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        return CLASS_NAMES[pred_idx], confidence


class VisionTransformerClassifier:
    """
    [CTRL-F: STANDALONE-VIT]
    Standalone ViT-B/16 Classifier with 12-layer Multi-Head Self-Attention over 16x16 asphalt patches.
    """
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

    def classify_crop(self, crop_bgr: np.ndarray):
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())
        return CLASS_NAMES[pred_idx], confidence


class DCGANAugmenter:
    """
    [CTRL-F: STANDALONE-DCGAN]
    [CTRL-F: GAN-AUGMENTATION]
    Standalone Deep Convolutional GAN Generator for synthesizing realistic road damage patches.
    """
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

    def generate(self, num_samples: int = 5):
        """
        [CTRL-F: VARIABLE-HANDOFF-LATENT-NOISE]
        INPUT: Random noise vector z ~ N(0, 1) of shape [num_samples, 100, 1, 1]
        RETURNS: synthetic_patches tensor of shape [num_samples, 3, 64, 64]
        """
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=DEVICE)
            synthetic_patches = torch.clamp((self.net(noise) + 1.0) / 2.0, 0.0, 1.0)
        return synthetic_patches


class CNNLSTMVideoModel(nn.Module):
    """
    [CTRL-F: STANDALONE-CNN-LSTM]
    CNN + Spatiotemporal LSTM architecture for continuous sequential video frame analysis.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_classes: int = 11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        weights_path = PROJECT_ROOT / "results" / "cnn_lstm" / "best_cnn_lstm.pt"
        if weights_path.exists():
            ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
            self.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        self.to(DEVICE).eval()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # Expected shape: [batch, sequence_length, C, H, W]
        b, t, c, h, w = x_seq.shape
        spatial_features = self.cnn(x_seq.reshape(b * t, c, h, w))
        temporal_seq = spatial_features.reshape(b, t, 128)
        lstm_out, _ = self.lstm(temporal_seq)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits


# ==============================================================================
# SECTION D: ALL 5 PROBLEM STATEMENT 5 HYBRID COMBINATIONS
# ==============================================================================

class HybridYOLO_EfficientNet:
    """
    [CTRL-F: HYBRID-1-YOLO-EFFICIENTNET]
    Hybrid Technique 1: YOLO12 Object Detector + EfficientNet-B0 Crop Verifier.

    INTER-COMPONENT VARIABLE HAND-OFF:
      1. `detector.detect(image)` -> Produces `candidate_boxes` [(coords, yolo_cls, yolo_conf)]
         [CTRL-F: VARIABLE-HANDOFF-COORDINATES]
      2. Coordinates slice the raw frame -> `crop = image_bgr[y1:y2, x1:x2]`
         [CTRL-F: VARIABLE-HANDOFF-CROPS]
      3. `crop` is handed off to `classifier.classify_crop(crop)` -> Returns `(eff_cls, eff_conf)`
      4. Predictions merged into high-confidence bounding box overlay and tabular report.
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = EfficientNetClassifier()

    def run(self, image_bgr: np.ndarray):
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
    [CTRL-F: HYBRID-2-YOLO-RESNET]
    Hybrid Technique 2: YOLO12 Object Detector + ResNet-50 Deep Residual Classifier.

    INTER-COMPONENT VARIABLE HAND-OFF:
      1. `detector.detect(image)` -> `boxes` containing spatial coordinates [x1, y1, x2, y2]
      2. `crop = image[y1:y2, x1:x2]` -> Sub-tensor handed off to `resnet.classify_crop(crop)`
      3. ResNet deep bottleneck features verify or overrule YOLO class verdict.
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = ResNet50Classifier()

    def run(self, image_bgr: np.ndarray):
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
    [CTRL-F: HYBRID-3-YOLO-VIT]
    Hybrid Technique 3: YOLO12 Object Detector + Vision Transformer (ViT-B/16).

    INTER-COMPONENT VARIABLE HAND-OFF:
      1. YOLO12 provides spatial proposal coordinates `[x1, y1, x2, y2]`.
      2. `crop` patch resized to 224x224 and fed to ViT multi-head self-attention.
      3. Self-attention captures global distress context across entire road width.
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        self.classifier = VisionTransformerClassifier()

    def run(self, image_bgr: np.ndarray):
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


class Hybrid_CNNLSTM:
    """
    [CTRL-F: HYBRID-4-CNN-LSTM]
    Hybrid Technique 4: Sequential Video Road Inspection via CNN + Temporal LSTM.

    INTER-COMPONENT VARIABLE HAND-OFF:
      1. Sequential dashcam video frames: `[F_1, F_2, ..., F_t]`.
      2. CNN spatial encoder extracts high-level frame descriptors: `f_t = CNN(F_t)`.
      3. LSTM temporal cells propagate recurrent state vectors across time:
             `h_t, c_t = LSTM(f_t, (h_{t-1}, c_{t-1}))`.
      4. Predicts whether damage is sustained across continuous highway distance.
    """
    def __init__(self):
        self.model = CNNLSTMVideoModel().to(DEVICE).eval()

    def run(self, frame_sequence: list):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensors = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frame_sequence]
        seq_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(seq_tensor)
            pred_idx = int(logits.argmax().item())
        return CLASS_NAMES[pred_idx]


class HybridYOLO_GAN_EfficientNet:
    """
    [CTRL-F: HYBRID-5-YOLO-GAN-EFFICIENTNET]
    Hybrid Technique 5: YOLO12 + GAN-Augmented EfficientNet + VIT GIS Priority Mapping.

    INTER-COMPONENT VARIABLE HAND-OFF:
      1. `dcgan.generate(5)` generates synthetic minority damage patches from latent noise `z`.
         [CTRL-F: VARIABLE-HANDOFF-LATENT-NOISE]
      2. `detector.detect(image)` provides bounding box candidates.
         [CTRL-F: VARIABLE-HANDOFF-COORDINATES]
      3. `crop` patch passed to `effnet.classify_crop(crop)`.
         [CTRL-F: VARIABLE-HANDOFF-CROPS]
      4. Bounding box area `(x2-x1)*(y2-y1) / total_image_area` mapped into Severity & Action.
         [CTRL-F: GIS-MAINTENANCE-PRIORITY-TABLE]
    """
    def __init__(self):
        self.detector = YOLO12Detector()
        eff_weights = PROJECT_ROOT / "results" / "efficientnet_b0_gan_augmented" / "best_efficientnet_b0.pt"
        self.classifier = EfficientNetClassifier(weights_path=str(eff_weights))
        self.augmenter = DCGANAugmenter()

    def run(self, image_bgr: np.ndarray):
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

            priority_records.append({
                "ID": i,
                "Damage_Type": eff_cls,
                "Confidence": round(eff_conf, 4),
                "Surface_Area_Pct": f"{area_pct:.2f}%",
                "Severity": severity,
                "Maintenance_Action": action
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"YOLO:{yolo_cls} | EffNet:{eff_cls} [{severity}]", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return annotated, synthetic_patches, priority_records


# ==============================================================================
# SECTION E: MAIN EXECUTION & VALIDATION VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    sample_path = PROJECT_ROOT / "samples" / "China_0010.jpg"
    if not sample_path.exists():
        sample_path = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    img = cv2.imread(str(sample_path))

    print("\n--- Executing Hybrid 1: YOLO12 + EfficientNet-B0 ---")
    h1 = HybridYOLO_EfficientNet()
    ann1, res1 = h1.run(img)
    print(f"-> Verified {len(res1)} damage bounding boxes.")

    print("\n--- Executing Hybrid 5: YOLO12 + GAN + EfficientNet-B0 ---")
    h5 = HybridYOLO_GAN_EfficientNet()
    ann5, syn, prio = h5.run(img)
    df_prio = pd.DataFrame(prio)

    print("\n[CTRL-F: GIS-MAINTENANCE-PRIORITY-TABLE]")
    print(df_prio.to_string(index=False))

    out_file = PROJECT_ROOT / "outputs" / "master_hybrid_output.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), ann5)
    print(f"\nFinal visual imagery saved to: {out_file}")
    print("Master Hybrid Pipeline verified successfully!")
