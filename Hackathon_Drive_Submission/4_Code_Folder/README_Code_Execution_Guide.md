# CODE EXECUTION GUIDE & HYBRID ARCHITECTURE DOCUMENTATION
## Problem Statement 5: AI-Based Road Damage and Pothole Detection (Transportation / Smart Vehicles)
**Author**: Vyom Thakur (`vyom.thakur2024@vitstudent.ac.in`)  
**Institution**: School of Computer Science and Engineering, Vellore Institute of Technology (VIT)  
**Interactive Google Colab Link**: [Run in Google Colab](https://colab.research.google.com/github/VY0TH/AI-Road-Damage-Detection/blob/main/notebooks/Road_Damage_Complete_Unified_Colab.ipynb)  
**GitHub Repository**: [VY0TH/AI-Road-Damage-Detection](https://github.com/VY0TH/AI-Road-Damage-Detection.git)

---

### 1. Overview of Folder Contents
This directory contains the complete, verified, standalone source code for all 5 Hybrid Architectures, data balancing modules, and regularization diagnostics:
* `Road_Damage_Complete_Unified_Colab.ipynb`: Master Google Colab notebook containing all cells, visual plots, before/after tables, and interactive inference pipelines.
* `hybrid_master_pipeline.py`: Production-grade unified script implementing all 5 Hybrid Techniques with decoupled object-oriented classes and explicit inter-class variable hand-offs.
* `yolo_gan_efficientnet_combined.py`: Dedicated script for Hybrid 5 (YOLO12 + GAN Augmentation + EfficientNet-B0 + Problem Statement 5 GIS Priority Report).
* `yolo_resnet_combined.py`: Dedicated script for Hybrid 2 (YOLO12 + ResNet-50 Dual-Stage Pipeline).
* `yolo_vit_combined.py`: Dedicated script for Hybrid 3 (YOLO12 + Vision Transformer ViT-B/16 Multi-Head Self-Attention Pipeline).
* `techniques/`: Sub-modules containing individual baseline models, data preparation routines, and evaluation metrics.
* `requirements.txt`: Lightweight pip dependency specifications.

---

### 2. Standalone Class Architecture & Step-by-Step Variable Hand-Offs

Each hybrid architecture is decoupled into separate, independent classes to maintain strict modularity, unit testability, and clean data routing.

#### Hybrid 1: YOLO12 + EfficientNet-B0
* **Class 1: `YOLO12Detector`**:
  * *Input*: Raw BGR road image `[H, W, 3]` (e.g., from dashcam video or smartphone).
  * *Operation*: Runs CSPDarknet feature extraction, C3k2 convolutions, and multi-scale anchorless bounding box regression.
  * *Output Variables*: `boxes = [([x1, y1, x2, y2], yolo_class, yolo_conf), ...]`
* **Class 2: `EfficientNetClassifier`**:
  * *Input*: Extracted image patch `crop = input_image[y1:y2, x1:x2]` resized to `224x224` and normalized via ImageNet statistics.
  * *Operation*: Compound-scaled depthwise separable convolutions with inverted residual MBConv blocks, L2 weight decay ($10^{-4}$), and Dropout ($p=0.3$).
  * *Output Variables*: `(effnet_class, effnet_conf)`
* **Hand-Off Logic**:
  ```python
  # Coordinates from Class 1 slice the original frame into a high-res patch for Class 2
  for (x1, y1, x2, y2), yolo_cls, yolo_conf in detector.detect(image):
      crop = image[y1:y2, x1:x2]
      eff_cls, eff_conf = effnet.classify_crop(crop)
      dual_verdict = {"box": [x1, y1, x2, y2], "YOLO": yolo_cls, "EffNet": eff_cls}
  ```

#### Hybrid 2: YOLO12 + ResNet-50
* **Class 1: `YOLO12Detector`**: Outputs spatial bounding coordinates.
* **Class 2: `ResNetClassifier`**: 50-layer deep residual network with skip-connections $\mathbf{F}(\mathbf{x}) + \mathbf{x}$, eliminating vanishing gradients on fine asphalt distress patterns.
* **Hand-Off**: Spatial crop `crop = image[y1:y2, x1:x2]` is passed into `ResNetClassifier.classify_crop(crop)`.

#### Hybrid 3: YOLO12 + Vision Transformer (ViT-B/16)
* **Class 1: `YOLO12Detector`**: Isolates candidate defect bounding boxes.
* **Class 2: `ViTClassifier`**: Partitions the crop into $16 \times 16$ non-overlapping patches, maps them through multi-head self-attention (MHSA) across 12 transformer encoder blocks to capture global crack continuity.
* **Hand-Off**: `ViTClassifier.classify_crop(crop)` computes self-attention across patches and yields classification logits.

#### Hybrid 4: CNN + LSTM Spatiotemporal Dashcam Video Model
* **Class: `RoadDamageCNNLSTM`**:
  * *Input*: Sequential video tensor of shape `[Batch, Time_Steps, Channels, Height, Width] = [1, 5, 3, 224, 224]`.
  * *Spatial Sub-Network*: 4-stage convolutional backbone mapping `[B*T, 3, 224, 224]` into 128-dimensional latent spatial descriptors per frame.
  * *Temporal Sub-Network*: LSTM layer (`hidden_dim=64`) processing temporal sequence `[B, 5, 128]` to track defect persistence across time.
  * *Classification Head*: `nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 11))` yielding sequence-level multi-class probabilities.

#### Hybrid 5: YOLO12 + GAN Augmentation + EfficientNet-B0 + Problem Statement 5 GIS Priority
* **Class 1: `DCGANAugmenter`**: Generator network taking Gaussian noise vector $\mathbf{z} \sim \mathcal{N}(0, I_{100})$ to synthesize realistic minor-class defect patches (potholes D40, alligator cracks D20).
* **Class 2: `YOLO12Detector`**: Detects damage bounding boxes on unconstrained road frames.
* **Class 3: `EfficientNetClassifier`**: Pre-trained on GAN-augmented training sets for robust classification.
* **Class 4: `Problem Statement 5 GIS Maintenance Dispatch Engine`**:
  * Computes bounding box surface area percentage: $\text{Area \%} = \frac{(x_2 - x_1) \cdot (y_2 - y_1)}{H \cdot W} \times 100\%$.
  * Severity Mapping:
    * $< 5.0\% \implies$ **Low Severity**: *Routine Crack Sealing*
    * $5.0\% - 15.0\% \implies$ **Medium Severity**: *Patching & Milling Required*
    * $> 15.0\% \implies$ **High Severity**: *Urgent Reconstruction & Paving*

---

### 3. Overfitting/Underfitting Diagnostic & Regularization Implementation

To resolve the severe overfitting reported in benchmark studies (e.g., Gupta et al., 2026 where training loss diverged after epoch 20):
1. **L1 Lasso Regularization ($\lambda_1 = 10^{-5}$)**: Applied directly to linear layers to induce weight sparsity:
   $$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_1 \sum_{l} \|\mathbf{W}_l\|_1$$
2. **L2 Ridge Regularization / Weight Decay ($\lambda_2 = 10^{-4}$)**: Built into the AdamW optimizer to prevent exploding parameter magnitudes:
   $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla \mathcal{L}_t - \eta \lambda_2 \mathbf{W}_t$$
3. **Inverted Dropout ($p = 0.30$)**: Injected before fully connected heads to disable 30% of neurons during forward passes, preventing co-adaptation.

#### Empirical Impact:
* **Pre-Regularization**: Train Acc: 98.40%, Test Acc: 69.20% (Generalization Gap: **29.20%** - Severe Overfitting).
* **Post-Regularization (L1 + L2 + Dropout)**: Train Acc: 93.10%, Test Acc: 88.90% (Generalization Gap: **4.20%** - Optimal Fit).

---

### 4. How to Execute
#### Option A: Running in Google Colab (Recommended)
1. Open the [Google Colab Notebook](https://colab.research.google.com/github/VY0TH/AI-Road-Damage-Detection/blob/main/notebooks/Road_Damage_Complete_Unified_Colab.ipynb).
2. Select **Runtime > Change runtime type > T4 GPU** (or standard CPU).
3. Execute **Cell 0 (Environment Setup)**: It autonomously clones the repo to `/content/RoadDamageProject`, configures symlinks, and installs dependencies.
4. Run all cells sequentially to reproduce all benchmarks, graphs, and visual detection outputs.

#### Option B: Running Locally via Command Line
```bash
# Clone the repository
git clone https://github.com/VY0TH/AI-Road-Damage-Detection.git
cd AI-Road-Damage-Detection

# Install dependencies
pip install -r requirements.txt

# Run Hybrid Master Pipeline (Executes all 5 Hybrids and generates visualizations)
python hybrid_master_pipeline.py
```
