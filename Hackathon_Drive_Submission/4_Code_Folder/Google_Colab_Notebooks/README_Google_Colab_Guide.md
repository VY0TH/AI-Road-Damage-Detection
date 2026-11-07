# GOOGLE COLAB NOTEBOOK SUITE & REPRODUCIBILITY GUIDE
## Problem Statement 5: AI-Based Road Damage and Pothole Detection (Transportation / Smart Vehicles)
**Course Project Review-3 Submission | School of Computer Science and Engineering**  
**Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India**  

### Project Team Members & Authors:
- **Vyom Thakur** | Reg. No: 24BIT0109 | Email: yom.thakur2024@vitstudent.ac.in | Tel: +91 7018823820
- **Ronit Bhattacharjee** | Reg. No: 24BIT0307 | Email: Ronit.bhattacharjee2024@vitstudent.ac.in | Tel: +91 9123416650
- **Dev Arpit Patel** | Reg. No: 24BIT0491 | Email: dev.arpitpatel2024@vitstudent.ac.in | Tel: +91 8849054109
- **Debarghya Chakraborty** | Reg. No: 24BIT0525 | Email: debarghya.2024@vitstudent.ac.in | Tel: +91 9304122088

**Open-Source Repository**: [https://github.com/VY0TH/AI-Road-Damage-Detection.git](https://github.com/VY0TH/AI-Road-Damage-Detection.git)

---

## 📚 Notebook Directory & One-Click Execution Links

| Notebook # | File Name | Description & Focus | Direct Colab Execution Link |
| :---: | :--- | :--- | :---: |
| **1** | [Road_Damage_Complete_Unified_Colab.ipynb](./Road_Damage_Complete_Unified_Colab.ipynb) | **Master Unified Suite**: All 5 Hybrid models in standalone OOP classes, explicit hand-offs, L1/L2/Dropout ablation, SMOTE & DCGAN plots, GIS table, and Ctrl+F anchors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VY0TH/AI-Road-Damage-Detection/blob/main/notebooks/Road_Damage_Complete_Unified_Colab.ipynb) |
| **2** | [Road_Damage_Detection_Master_Colab.ipynb](./Road_Damage_Detection_Master_Colab.ipynb) | **51-Cell Master Framework**: Deep step-by-step training, verification, dataset auditing, and 10-fold cross-validation suite. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VY0TH/AI-Road-Damage-Detection/blob/main/notebooks/Road_Damage_Detection_Master_Colab.ipynb) |
| **3** | [colab_pipeline_runner.ipynb](./colab_pipeline_runner.ipynb) | **10-Step Pipeline Runner**: Sequential execution of Step 01 (Data Load) through Step 10 (Final Test Evaluation). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VY0TH/AI-Road-Damage-Detection/blob/main/notebooks/colab_pipeline_runner.ipynb) |

---

## 🎯 Quick Navigation: Master Ctrl+F Search Anchors

In Road_Damage_Complete_Unified_Colab.ipynb, press **Ctrl + F** and paste any of the following tags to navigate immediately:

| Anchor | Target Component / Architecture |
| :--- | :--- |
| [CTRL-F: REPO-SETUP] | Autonomous VM synchronization, Git clone, and symlink creation |
| [CTRL-F: UNDERFITTING-REGULARIZATION] | Model capacity analysis and inductive bias selection |
| [CTRL-F: OVERFITTING-DROPOUT] | Inverted Dropout implementation (=0.20$ to .40$) |
| [CTRL-F: L1-REGULARIZATION] | Mathematical L1 Lasso penalty ($\lambda_1 = 10^{-5}$) |
| [CTRL-F: L2-WEIGHT-DECAY] | Decoupled L2 Weight Decay in AdamW ($\lambda_2 = 10^{-4}$) |
| [CTRL-F: SMOTE-ALGORITHM] | Latent feature-space synthetic minority over-sampling |
| [CTRL-F: GAN-AUGMENTATION] | DCGAN Generator network and patch generation |
| [CTRL-F: CLAHE-PREPROCESSING] | LAB L-channel CLAHE contrast equalization |
| [CTRL-F: STANDALONE-YOLO12] | YOLO12 detector class definition |
| [CTRL-F: STANDALONE-EFFICIENTNET] | EfficientNet-B0 classifier class definition |
| [CTRL-F: STANDALONE-RESNET50] | ResNet-50 deep residual classifier class definition |
| [CTRL-F: STANDALONE-VIT] | Vision Transformer (ViT-B/16) self-attention classifier |
| [CTRL-F: STANDALONE-CNN-LSTM] | Spatiotemporal ConvLSTM 5-frame sequence model |
| [CTRL-F: HYBRID-1-YOLO-EFFICIENTNET] | Execution of Hybrid 1: YOLO12 + EfficientNet-B0 |
| [CTRL-F: HYBRID-2-YOLO-RESNET] | Execution of Hybrid 2: YOLO12 + ResNet-50 |
| [CTRL-F: HYBRID-3-YOLO-VIT] | Execution of Hybrid 3: YOLO12 + Vision Transformer |
| [CTRL-F: HYBRID-4-CNN-LSTM] | Execution of Hybrid 4: Spatiotemporal CNN + LSTM |
| [CTRL-F: HYBRID-5-YOLO-GAN-EFFICIENTNET] | Execution of Hybrid 5: YOLO12 + GAN + EfficientNet-B0 |
| [CTRL-F: GIS-MAINTENANCE-PRIORITY-TABLE] | Problem Statement 5 GIS Road Repair Priority Report |
| [CTRL-F: 10-FOLD-CROSS-VALIDATION] | Stratified 10-Fold cross-validation evaluation results |

---

## ⚙️ Hardware and Execution Environment
- **Runtime Environment**: Google Colab with Python 3.10+ / 3.11+.
- **Hardware Acceleration**: Select **Runtime -> Change runtime type -> T4 GPU** (or A100).
- **Execution Time**: The master unified notebook executes complete visual inference across all 5 hybrid techniques, renders before-after plots, and formats GIS dispatch schedules in **< 45 seconds**.
