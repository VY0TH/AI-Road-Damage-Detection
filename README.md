# Road Damage Image Classification & Detection Pipeline

An end-to-end computer vision and deep learning project for road damage detection and classification. The project is systematically arranged into a 10-step methodological pipeline with dedicated technique folders.

---

## 🚀 10-Step Pipeline Sequence

```
Step 01: Load Image Dataset
   ↓
Step 02: Data Preprocessing
   ↓
Step 03: Data Leakage Checks (Target leakage • duplicate leakage • suspicious features)
   ↓
Step 04: CLAHE (when contrast enhancement is useful)
   ↓
Step 05: 1. Train / Validation / Test Split (70/15/15)
         2. 10-Fold Cross-Validation
   ↓
Step 06: GAN Augmentation (on TRAINING data only)
   ↓
Step 07: Initial Model Training
   ↓
Step 08: Overfitting / Underfitting Detection and Rectification
   ↓
Step 09: Apply Correction Technique and Retrain
   ↓
Step 10: Final Test Evaluation
```

---

## 📂 Project Structure

```text
RoadDamageProject/
│
├── 01_load_image_dataset/               # Step 1: Loading, verification & stats
│   ├── dataset_loader.py                # PyTorch Dataset and DataLoader abstractions
│   ├── inspect_dataset.py               # Dataset summary, integrity check & class counts
│   └── run_load_dataset.py              # CLI launcher
│
├── 02_data_preprocessing/               # Step 2: Resizing, padding, RGB & crop extraction
│   ├── image_preprocessor.py            # Lanczos resizing with aspect-ratio letterboxing
│   ├── crop_extractor.py                # Bounding-box crop extraction from VOC/YOLO
│   └── run_preprocessing.py             # CLI launcher
│
├── 03_data_leakage_checks/              # Step 3: Leakage & shortcut learning audits
│   ├── check_target_leakage.py          # Metadata/EXIF & featurization isolation checks
│   ├── check_duplicate_leakage.py       # SHA-256 exact & dHash perceptual near-duplicates
│   ├── check_suspicious_features.py     # Domain prefix bias & shortcut feature audits
│   └── run_leakage_checks.py            # Master leakage auditor
│
├── 04_clahe_enhancement/                # Step 4: Contrast enhancement on LAB luminance
│   ├── clahe_processor.py               # LAB color space L-channel CLAHE
│   ├── contrast_evaluator.py            # RMS contrast & entropy decision metric
│   └── run_clahe.py                     # CLI launcher with side-by-side comparisons
│
├── 05_splitting_and_cross_validation/   # Step 5: Partitioning and CV
│   ├── train_val_test_split/            # 70/15/15 stratified, group-aware splitting
│   │   ├── stratified_group_splitter.py
│   │   └── run_split.py
│   ├── ten_fold_cross_validation/       # Stratified, group-aware 10-fold CV
│   │   ├── fold_generator.py
│   │   ├── cv_evaluator.py
│   │   └── run_cv.py
│   └── run_splitting_pipeline.py        # Combined Step 5 runner
│
├── 06_gan_augmentation/                 # Step 6: cGAN on TRAINING DATA ONLY
│   ├── gan_model.py                     # Conditional DCGAN architecture with embeddings
│   ├── train_gan.py                     # Trains cGAN strictly on training crops
│   ├── generate_gan_samples.py          # Generates minority damage class samples
│   ├── inject_to_training.py            # Injects to training only + validation isolation audit
│   └── run_gan_pipeline.py              # CLI launcher
│
├── 07_initial_model_training/           # Step 7: Initial baseline training
│   ├── models/                          # BaselineCNN, EfficientNet-B0, ResNet-50, ViT-B/16
│   ├── trainer.py                       # PyTorch training engine with detailed epoch logging
│   └── run_initial_training.py          # CLI launcher
│
├── 08_overfitting_underfitting_detection/ # Step 8: Diagnostics & rectification advisor
│   ├── learning_curve_analyzer.py       # Loss and accuracy curve plotter
│   ├── fit_detector.py                  # Quantitative overfitting/underfitting detector
│   ├── rectification_advisor.py         # Automated actionable remediation strategy
│   └── run_detection.py                 # CLI launcher
│
├── 09_correction_and_retraining/        # Step 9: Engineering fixes & retraining
│   ├── corrections/                     # Loss reweighting, label smoothing, schedulers
│   ├── retrainer.py                     # Retraining engine with corrections incorporated
│   ├── compare_before_after.py          # Side-by-side baseline vs. corrected comparison
│   └── run_correction_retraining.py     # CLI launcher
│
├── 10_final_test_evaluation/            # Step 10: Independent evaluation on held-out test set
│   ├── test_evaluator.py                # Accuracy, Macro/Weighted F1, latency evaluator
│   ├── confusion_matrix_plotter.py      # Heatmap confusion matrix generator
│   ├── generate_final_report.py         # Markdown and JSON report generator
│   └── run_final_evaluation.py          # CLI launcher
│
├── techniques/                          # Dedicated Technique Modules
│   ├── efficientnet/                    # Compound-scaled CNN classifier
│   ├── resnet50/                        # Deep residual network classifier
│   ├── vit/                             # Vision Transformer (ViT-B/16)
│   ├── yolo/                            # YOLO12n object localization & detection
│   ├── gan/                             # Conditional GAN damage generator
│   ├── cnn_lstm/                        # Spatial ConvLSTM video segmentation
│   └── clahe/                           # Adaptive histogram equalization
│
├── run_full_pipeline.py                 # Master end-to-end pipeline runner
├── pipeline_config.py                   # Central configuration & hyperparameters
├── dataset/                             # Raw and processed datasets
└── results/                             # Checkpoints, reports, plots, logs
```

---

## ⚡ Execution

### Running the Entire Sequence End-to-End
Execute all 10 steps sequentially with one command:
```powershell
python run_full_pipeline.py --all
```

### Running Individual Steps
```powershell
python run_full_pipeline.py --step 1   # Load Image Dataset
python run_full_pipeline.py --step 2   # Data Preprocessing
python run_full_pipeline.py --step 3   # Data Leakage Checks
python run_full_pipeline.py --step 4   # CLAHE Enhancement
python run_full_pipeline.py --step 5   # Dataset Splitting & 10-Fold CV
python run_full_pipeline.py --step 6   # GAN Augmentation (Train only)
python run_full_pipeline.py --step 7   # Initial Model Training
python run_full_pipeline.py --step 8   # Overfitting / Underfitting Detection
python run_full_pipeline.py --step 9   # Apply Corrections & Retrain
python run_full_pipeline.py --step 10  # Final Test Evaluation
```

---

## 🖥️ Launch Interactive Dashboard
```powershell
streamlit run src/89_road_damage_app.py
```
