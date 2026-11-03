# Complete Reference Guide: 5 Hybrid Techniques for AI Road Damage & Pothole Detection
**VIT Tech-a-thon — Problem Statement 5: Transportation & Smart Vehicles**

---

## Executive Architectural Summary

| # | Technique | Input Specification | Standalone Component 1 | Standalone Component 2 | Combined Hybrid Output |
|---|---|---|---|---|---|
| **1** | **YOLO12 + EfficientNet** | RGB image ($512 \times 512$ / $640 \times 640$) | YOLO12 detector (`yolo12n.pt`) | EfficientNet-B0 ($224 \times 224$ crops) | `results/combined_yolo_efficientnet/China_0010_combined_result.jpg` |
| **2** | **YOLO12 + ResNet-50** | RGB image ($512 \times 512$ / $640 \times 640$) | YOLO12 detector (`yolo12n.pt`) | ResNet-50 50-layer Bottlenecks | `results/combined_yolo_resnet/combined_prediction.jpg` |
| **3** | **YOLO12 + Vision Transformer** | RGB image ($512 \times 512$ / $640 \times 640$) | YOLO12 detector (`yolo12n.pt`) | ViT-B/16 Multi-Head Self-Attention | `results/combined_yolo_vit/combined_prediction.jpg` |
| **4** | **CNN + LSTM** | 5-Frame video tensor `[1, 5, 3, 224, 224]` | 4-Stage CNN feature extractor | 2-Layer Temporal LSTM Recurrence | `outputs/cnn_lstm_prediction.jpg` |
| **5** | **YOLO12 + GAN + EfficientNet** | Road image + 100-D Gaussian noise $z$ | YOLO12 detector (`yolo12n.pt`) | DCGAN Transposed-Conv Generator | `outputs/final_combined_result.jpg` + GIS Priority Table |

---

## 1. Hybrid Technique 1: YOLO12 + EfficientNet-B0

### A. Input Specification
* **Source**: `samples/China_0010.jpg` (or live road dashcam frame).
* **Format**: BGR / RGB 3-channel color image.
* **Dimensions**: $512 \times 512$ pixels (or arbitrary resolution up to $4000 \times 3000$).
* **Data Types**: `uint8` image array $\to$ normalized `float32` PyTorch tensor.

### B. Standalone YOLO12 Code (Stage 1: Localization)
```python
import cv2
from pathlib import Path
from ultralytics import YOLO

# 1. Load YOLO12 Detector
yolo_model = YOLO("results/training/yolo12n_30_epochs/weights/best.pt")

# 2. Input Image
image_path = "samples/China_0010.jpg"
image_bgr = cv2.imread(image_path)

# 3. Detect Road Distress Bounding Boxes
results = yolo_model.predict(source=image_bgr, conf=0.25, imgsz=640, verbose=False)

# 4. Extract and Display Bounding Boxes
for result in results:
    for box in result.boxes:
        coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        print(f"YOLO Detection: Class {cls_id} | Conf: {conf:.4f} | Box: {coords}")
```

### C. Standalone EfficientNet-B0 Code (Stage 2: Classification)
```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Instantiate Architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)  # 10 distress classes

# 2. Load Weights
checkpoint = torch.load("results/efficientnet_b0/best_efficientnet_b0.pt", map_location="cpu")
state = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state, strict=False)
model.eval()

# 3. Preprocessing Transform (ImageNet Normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Forward Pass on Input Crop
crop_pil = Image.open("samples/China_0010.jpg").crop((100, 100, 300, 300))
input_tensor = preprocess(crop_pil).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    pred_conf = float(probs[pred_class].item())

classes = ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"]
print(f"EfficientNet-B0 Verdict: {classes[pred_class]} (Confidence: {pred_conf:.4f})")
```

### D. Combined Fusion Code (YOLO12 + EfficientNet-B0)
```python
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# Initialize both models
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("results/training/yolo12n_30_epochs/weights/best.pt")

effnet = models.efficientnet_b0(weights=None)
effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 10)
checkpoint = torch.load("results/efficientnet_b0/best_efficientnet_b0.pt", map_location=device)
effnet.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
effnet.to(device).eval()

classes = ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Read input image
raw_bgr = cv2.imread("samples/China_0010.jpg")
annotated = raw_bgr.copy()
h, w = raw_bgr.shape[:2]

# Step 1: YOLO Localization
yolo_results = yolo.predict(source=raw_bgr, conf=0.25, verbose=False)

for res in yolo_results:
    if res.boxes is None or len(res.boxes) == 0:
        continue
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Step 2: The Meeting Point (Crop Damage Region)
        crop = raw_bgr[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = transform(crop_pil).unsqueeze(0).to(device)

        # Step 3: EfficientNet Classification
        with torch.no_grad():
            probs = torch.softmax(effnet(crop_tensor), dim=1)[0]
            eff_cls = classes[int(probs.argmax().item())]
            eff_conf = float(probs.max().item())

        yolo_cls = yolo.names.get(int(b.cls[0]), "Distress")
        yolo_conf = float(b.conf[0])

        # Step 4: Visual Dual Annotation
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"YOLO:{yolo_cls} | EffNet:{eff_cls} ({eff_conf:.2f})"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

cv2.imwrite("results/combined_yolo_efficientnet/China_0010_combined_result.jpg", annotated)
```

### E. Output Specification & Destination
* **Destination Directory**: `results/combined_yolo_efficientnet/`
* **Output Image**: `China_0010_combined_result.jpg`
* **Visual Presentation**: High-contrast green bounding boxes around cracks/potholes with dual-verdict banner overlays.

---

## 2. Hybrid Technique 2: YOLO12 + ResNet-50

### A. Input Specification
* **Source**: `samples/China_0010.jpg` ($512 \times 512$ RGB).
* **Crop Shape**: Dynamically extracted bounding boxes $(w_{crop} \times h_{crop} \times 3) \to 224 \times 224$.

### B. Standalone ResNet-50 Architecture Code
```python
import torch
import torch.nn as nn
from torchvision import models

def build_resnet50_classifier(num_classes=10):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    # Custom classification head for RDD2022 categories
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model
```

### C. Combined Hybrid Code (YOLO12 + ResNet-50)
* **Code Script**: `yolo_resnet_combined.py`
* **Pipeline Mechanism**:
  1. YOLO predicts bounding box proposals around longitudinal cracks (`D00`), transverse cracks (`D10`), alligator cracks (`D20`), and potholes (`D40`).
  2. The 50-layer deep bottleneck blocks in ResNet-50 utilize residual connections $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ to extract fine-grained asphalt texture representations.
  3. Bounding box coordinates and dual-model classification verdicts are drawn onto the output image.

### D. Output Specification & Destination
* **Destination Directory**: `results/combined_yolo_resnet/`
* **Output Image**: `combined_prediction.jpg`

---

## 3. Hybrid Technique 3: YOLO12 + Vision Transformer (ViT-B/16)

### A. Input Specification
* **Source**: `samples/China_0010.jpg`.
* **Patch Partition**: $224 \times 224$ normalized crop divided into $196$ non-overlapping $16 \times 16$ patches.

### B. Standalone Vision Transformer Architecture Code
```python
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

def build_vit_classifier(num_classes=10):
    model = vit_b_16(weights=None)
    # Replace standard ImageNet head with road damage head
    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)
    return model
```

### C. Combined Hybrid Code (YOLO12 + ViT-B/16)
* **Code Script**: `yolo_vit_combined.py`
* **Pipeline Mechanism**:
  1. YOLO12 detects candidates with high sensitivity.
  2. ViT projects $16 \times 16$ pixel patches into 768-D embeddings, adds learnable 1D position embeddings, and processes through 12 Multi-Head Self-Attention (MHSA) transformer encoder blocks:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  3. The global attention mechanism distinguishes genuine crack patterns from road shadow boundaries.

### D. Output Specification & Destination
* **Destination Directory**: `results/combined_yolo_vit/`
* **Output Image**: `combined_prediction.jpg`

---

## 4. Hybrid Technique 4: CNN + LSTM (Spatiotemporal Video Sequence Tracking)

### A. Input Specification
* **Source**: Road inspection dashcam video (`0081_segmentation_smoke.mp4`) or 5-frame sequence buffer.
* **Shape**: $5\text{D Tensor } [B=1, T=5, C=3, H=224, W=224]$.

### B. Standalone CNN Feature Extractor + LSTM Architecture Code
```python
import torch
import torch.nn as nn

class RoadDamageCNNLSTM(nn.Module):
    def __init__(self, num_classes=11, hidden_dim=128):
        super().__init__()
        # 1. Spatial CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # 2. Temporal Recurrent Engine (LSTM)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        # 3. Final Classification Head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        spatial_features = self.cnn(x_reshaped)  # [b*t, 128]
        temporal_seq = spatial_features.view(b, t, 128)  # [b, t, 128]

        lstm_out, _ = self.lstm(temporal_seq)  # [b, t, 128]
        final_prediction = self.fc(lstm_out[:, -1, :])  # Take final time step
        return final_prediction
```

### C. Output Specification & Destination
* **Destination Directory**: `outputs/`
* **Output Image**: `cnn_lstm_prediction.jpg`

---

## 5. Hybrid Technique 5: YOLO12 + GAN + EfficientNet (with Severity & GIS Priority Map)

### A. Input Specification
1. **Full Road Frame**: `samples/China_0010.jpg` ($512 \times 512 \times 3$).
2. **Latent Latent Vector**: $z \sim \mathcal{N}(0, I) \in \mathbb{R}^{5 \times 100 \times 1 \times 1}$ (100-D Gaussian random noise).

### B. Standalone DCGAN Generator Architecture Code
```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
```

### C. Severity Classification & GIS Priority Formula
$$\text{Damage Area Ratio } = \frac{(x_2 - x_1) \cdot (y_2 - y_1)}{W_{\text{image}} \cdot H_{\text{image}}} \times 100\%$$

* **Low Severity (< 5%)**: Surface hair cracks $\to$ **Routine Crack Sealing**
* **Medium Severity (5% – 15%)**: Moderate transverse/longitudinal cracks $\to$ **Patching & Milling Required**
* **High Severity (> 15%)**: Severe alligator cracking / deep potholes $\to$ **Urgent Reconstruction & Paving**

### D. Output Specification & Destination
* **Destination Directory**: `outputs/`
* **Generated Image Files**:
  * `outputs/final_combined_result.jpg`: Annotated detection and severity overlay image.
  * `outputs/combined_gan_1.jpg` to `outputs/combined_gan_5.jpg`: 5 synthetic road damage patches generated by the DCGAN.
  * `outputs/yolo12_prediction.jpg`: Standalone YOLO12 prediction.
* **Console GIS Road Maintenance Priority Table**:
  ```text
  ======================================================================
        VIT PROBLEM STATEMENT 5: ROAD MAINTENANCE PRIORITY REPORT      
  ======================================================================
  ID   | Damage   | Conf   | Area %   | Severity | Action                      
  ----------------------------------------------------------------------
  1    | D10      | 0.58   |  5.25%   | Medium   | Patching & Milling Required 
  2    | D00      | 0.50   | 34.61%   | High     | Urgent Reconstruction & Paving
  3    | D00      | 0.25   | 27.75%   | High     | Urgent Reconstruction & Paving
  4    | D10      | 0.87   |  2.27%   | Low      | Routine Crack Sealing       
  5    | D10      | 0.50   |  7.02%   | Medium   | Patching & Milling Required 
  6    | D20      | 0.27   | 29.88%   | High     | Urgent Reconstruction & Paving
  7    | D20      | 0.38   | 19.07%   | High     | Urgent Reconstruction & Paving
  8    | D10      | 0.64   |  3.62%   | Low      | Routine Crack Sealing       
  9    | D20      | 0.44   | 10.01%   | Medium   | Patching & Milling Required 
  ======================================================================
  ```
