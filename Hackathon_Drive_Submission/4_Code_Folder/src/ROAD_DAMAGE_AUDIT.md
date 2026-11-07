# Road Damage Project Audit and Pothole Diagnosis

> **Current status (2026-10-14):** The latest leakage-grouped validation supersedes the earlier single-image success claims below. The available detector does **not** reliably detect potholes. See “Final grouped-split validation” at the end of this report for current metrics and smoke-test failures.

Audit date: 2026-10-14. Repository root: `C:/Users/Vyom Thakur/Desktop/AI hackathon/RoadDamageProject`.

## Phase 1: Repository inventory

### Main implementation groups

- `01`â€“`37`: dataset inspection, XML/YOLO conversion, split/leakage checks, annotation review, and class-balance diagnostics.
- `38`â€“`44`, `76`, `81`â€“`84`: crop preparation, EfficientNetB0/ResNet50/ViT training, and hybrid inference scripts. `85_run_yolo_vit.py` is absent; ViT inference is implemented in `89_road_damage_app.py` and `44_hybrid_yolo_vit.py`.
- `45`, `55`â€“`67`, `70`, `80`: CNN/Spatial-LSTM training, video/mask preparation, segmentation evaluation, and video demos.
- `46`â€“`48`, `52`, `39`: GAN and conditional-GAN training, sample generation, and opt-in EfficientNet augmentation.
- `49`â€“`51`, `77`: YOLO training and evaluation.
- `68`â€“`69`, `72`â€“`75`, `86`, `88`: reporting, experimental severity, coverage, visualization, activations, and learning curves.
- `78`â€“`79`: SQLite storage utility and legacy dashboard.
- `89_road_damage_app.py`: unified Streamlit interface for image/video analysis, segmentation, model comparison, history, and CSV downloads.
- `90_debug_pothole_detection.py` (added): repeatable multi-threshold YOLO diagnostics with optional ground-truth overlay.

### Active data and class mappings

The active data file is `dataset/subsets/multicountry/RDD2022/yolo/data.yaml`. Its dataset path is now relative to the repository root; run training from the repository root. The class map is:

| ID | Name | Meaning relevant here |
| ---: | --- | --- |
| 0 | D00 | Longitudinal crack |
| 1 | D01 | Longitudinal crack |
| 2 | D10 | Transverse crack |
| 3 | D11 | Transverse crack |
| 4 | D20 | Alligator cracking |
| 5 | D40 | Pothole |
| 6 | D43 | Road damage class |
| 7 | D44 | Road damage class |
| 8 | D50 | Road damage class |
| 9 | D0w0 | Road damage class |

The class meanings above are the projectâ€™s active label names; for inference the app compares class **names** from each checkpoint, never assumes that class index 5 means pothole for a different model.

Current split audit: train has 960 images and 960 label files; validation has 240 images and 240 label files. Every image has a paired label and every label has a paired image. There are no malformed lines, out-of-range IDs, or invalid normalized boxes in either split. The label counts are:

| Class | Train objects | Validation objects |
| --- | ---: | ---: |
| D00 | 691 | 161 |
| D01 | 8 | 3 |
| D10 | 340 | 64 |
| D11 | 1 | 1 |
| D20 | 176 | 39 |
| D40 (pothole) | 121 | 31 |
| D43 | 14 | 2 |
| D44 | 50 | 17 |
| D50 | 32 | 7 |
| D0w0 | 1 | 0 |

There are 402 empty train labels and 98 empty validation labels. The corresponding XML exists for all of them. XML inspection found 319 train and 82 validation images with no XML objects. The other 83 train and 16 validation XML files contain only `Repair` objects, which the current ten-class converter excludes. These are negative examples for the selected ten-class taxonomy, not missing XML annotations; whether `Repair` should be an explicit target class is a separate dataset-design decision. Exact byte-identical image duplicates across these two splits: 0. Near-duplicate detection was not run.

`src/17_convert_multicountry_to_yolo.py` is the converter used for this dataset. It has an absolute Windows project path and excludes XML objects named `Repair`; the YAML path is now portable, but that old conversion script itself remains machine-specific.

### Model files and actual availability

| Pipeline / role | Checkpoint path | Status |
| --- | --- | --- |
| Dashboard YOLO12n detector | `results/training/yolo12n_30_epochs/weights/best.pt` | Present; selected because its recorded overall validation mAP50 is the best among the checked candidates. |
| Alternate YOLO12n | `results/yolo12n_road_damage/weights/best.pt` | Present; weaker recorded metric and failed the pothole-image checks below. |
| Alternate improved YOLO12n | `results/training/yolo12n_improved_30_epochs/weights/best.pt` | Present; weaker recorded metric and failed the pothole-image checks below. |
| EfficientNetB0 classifier | `results/efficientnet_b0/best_efficientnet_b0.pt` | Present; metadata class order `D00,D01,D0w0,D10,D11,D20,D40,D43,D44,D50`; 224 px input. |
| ResNet50 classifier | `results/resnet50_road_damage/best_resnet50.pt` | Present; metadata class order `D00,D01,D10,D11,D20,D40,D43,D44,D50,D0w0`; 224 px input. |
| ViT-B/16 classifier | `results/vit/best_vit.pt` | Present; metadata class order `D00,D01,D0w0,D10,D11,D20,D40,D43,D44,D50`; 224 px input. |
| CNN + Spatial LSTM segmenter | `results/cnn_lstm/segmentation/best_cnn_lstm_spatial_v3.pt` | Present; experimental binary video-mask model, not a damage-class detector. |
| Conditional GAN generator | `results/conditional_gan_selected_checkpoints/conditional_generator_final.pt` | Present. |
| GAN-augmented EfficientNet classifier | `results/efficientnet_b0_gan_augmented/best_efficientnet_b0.pt` | **Absent**; fifth pipeline is unavailable until trained and evaluated. |

All three YOLO checkpoints loaded and reported the ten names in the dataset configuration. Classifier checkpoints use the same class-name set, with differing integer orders; dashboard inference now verifies set equality and maps predictions by the names stored in each checkpoint. Crop preparation (`src/12_create_classification_crops.py`) applies LAB-luminance CLAHE before writing crops. The unified dashboard applies the same transform to inference crops.

### Existing logs, records, and outputs

- `results/training/yolo12n_30_epochs/results.csv`: best recorded epoch is 29, precision `0.43709`, recall `0.15106`, mAP50 `0.10298`, mAP50â€“95 `0.04885`. Epoch 30 is lower. The checkpoint path selected above is from this run.
- `results/training/yolo12n_improved_30_epochs/results.csv`: 8 logged epochs; final logged precision `0.38148`, recall `0.03324`, mAP50 `0.01474`, mAP50â€“95 `0.00480`.
- `results/yolo12n_road_damage/results.csv`: only one logged epoch; precision `0.65026`, recall `0.06514`, mAP50 `0.04124`, mAP50â€“95 `0.01568`.
- Same-split CPU re-evaluation of all three available YOLO checkpoints (240 validation images, 325 objects, `imgsz=640`, `conf=0.001`, `iou=0.7`) is saved in `results/detector_candidate_comparison_20261014/comparison.json`. The selected 30-epoch checkpoint performed best overall (P `0.4367`, R `0.1514`, F1 `0.2249`, mAP50 `0.1030`, mAP50-95 `0.0489`); the 8-epoch improved run scored mAP50 `0.0147`, and the one-epoch `yolo12n_road_damage` run scored mAP50 `0.0181`. Selected model D40 recall was `0.0635`, versus `0.0` for the improved run and `0.0323` for `yolo12n_road_damage`. None is adequate as a high-recall pothole detector.
- Other training/evaluation assets exist under `results/efficientnet_b0`, `results/resnet50_road_damage`, `results/vit`, `results/yolo30_full_validation`, and `results/cnn_lstm`.
- SQLite database: `results/road_damage.db`; existing table `detections` has legacy YOLO/EfficientNet fields and optional latitude/longitude. New dashboard analyses are appended to separate `dashboard_runs` and `dashboard_detections` tables; existing rows/schema are preserved. Annotated images and videos are saved under `results/dashboard_outputs/`.
- `results/pothole_debug/` contains annotated diagnostics and text logs created by the reproducible script. GAN-generated samples exist under `results/gan_individual_samples`; their visual quality has not been approved in this audit.
- Dependencies observed in the current environment: Python 3.14.3, PyTorch 2.14.1 CPU, torchvision 0.29.1 CPU, Ultralytics 8.4.174, Streamlit 1.65.0, OpenCV 5.0.0, pandas 3.0.3, scikit-learn 1.9.1, Pillow 12.3.0, PyYAML available. No CUDA device is available. `requirements.txt` lists application dependencies; install a machine-specific PyTorch build separately.

## Phase 2: Pothole failure reproduction and diagnosis

### Confirmed causes

1. The dashboardâ€™s low-confidence pothole rescue treated `D20` as pothole. In this active dataset, `D20` is alligator cracking and `D40` is pothole. This caused irrelevant low-confidence proposals to bypass the normal cutoff.
2. The dashboard rendered only the crop classifierâ€™s class, hiding the detectorâ€™s class. It therefore obscured disagreements between the localization model and crop model.
3. Dashboard crop inference had omitted CLAHE, although `12_create_classification_crops.py` applies LAB-luminance CLAHE to the crop files used to train all three classifiers. Inference has now been aligned with that training preprocessing.
4. The best available detector is weak. Its historical validation mAP50 is only `0.10298`; it misses a confirmed pothole even on the requested training-image example. Tiling and filtering can improve localization and presentation but cannot make these weights reliable.
5. The prior â€œtoo many boxesâ€ and â€œone large boxâ€ symptoms are consistent with low-threshold false proposals, broad detector boxes, and duplicate handling. The app now uses overlapping 640 px tiles for large frames and suppresses repeated boxes while allowing multiple tight boxes inside a broad proposal.

### Reproduced tests

Diagnostic image: `dataset/subsets/multicountry/RDD2022/yolo/images/train/China_0059.jpg`. Its paired label contains one ground-truth `D40` box: `5 0.613281 0.041016 0.312500 0.078125`. This is a training image, so it is diagnostic only and not a generalization result.

For the dashboard detector, direct full-image inference at `imgsz=640` returned no correct D40 box at any of the tested cutoffs. At `conf=0.01` it returned a D10 box at confidence `0.1315`, a D40 box at `0.0242` at the wrong location, and a D10 box at `0.0106`. At 0.05 and 0.10 only the D10 remained; at 0.25 and 0.40 there were no detections. Raising inference size to 960/1280 did not recover a D40 box.

The validation image `dataset/subsets/multicountry/RDD2022/yolo/images/val/China_0000.jpg` has two labeled D40 boxes. Direct YOLO inference from the selected checkpoint at confidence 0.01 produced three D40 proposals: two overlap the two different ground-truth potholes, while another duplicates one of those areas. Higher cutoffs returned fewer D40 proposals. The alternate improved checkpoint returned no D40 at confidence >=0.05; the `yolo12n_road_damage` checkpoint returned no detections at >=0.01 on this image.

After the dashboard fix, the real ViT hybrid path on this validation image produced two individual D40 boxes. Both overlap distinct labeled potholes at IoU >=0.5; the crop classifier agreed D40 on both. Annotated dashboard output: `results/pothole_debug/dashboard_China_0000_vit_conf_0.10_final.jpg`. This is a one-image smoke check, not an accuracy estimate.

### Changes made

- `src/89_road_damage_app.py`: uses name D40 for pothole filtering; verifies detector/classifier class-name sets; applies the same LAB-luminance CLAHE as training; keeps detector and classifier labels/scores separate; rejects hybrid proposals covering more than 20% of a tile unless the classifier agrees with the detector at confidence at least 0.75; uses overlapping 640 px tiles for large images; suppresses duplicates without suppressing distinct potholes; reports configured checkpoints and timing. The default detection threshold is 0.10.
- `src/90_debug_pothole_detection.py`: validates image/checkpoint paths, derives D40's numeric label from the active YAML, reports checkpoint names, inference time and all boxes at 0.01/0.05/0.10/0.25/0.40, and saves annotated images with optional ground-truth overlays.
- `dataset/subsets/multicountry/RDD2022/yolo/data.yaml`: changed the machine-specific absolute dataset path to a repository-relative path.
- `requirements.txt`: added PyYAML for the reproducible label-map diagnostic script.
- Checkpoint-specific hybrid smoke checks on the same validation image: EfficientNet produced 3 detections (2 YOLO D40, 1 D20/crop D43 disagreement); ResNet produced 2 YOLO D40 (one crop-classified D50); ViT produced 2 D40 with agreement. Annotated outputs are in `results/pothole_debug/pipelines/`.
- NMS smoke checks: one broad box enclosing two separated boxes retained the two smaller boxes; two overlapping proposals around one region collapsed to one.
- `py_compile` passed for the modified app and diagnostic script. Streamlit `AppTest` ran the app with zero exceptions. Ultralytics resolved the new relative dataset path to existing 960-image train and 240-image validation directories.

The dashboard map stays empty when no real GPS values exist. New detections are stored separately from the legacy table.

## Final grouped-split validation (2026-10-14)

### Dataset integrity and split leakage

`src/93_validate_yolo_dataset.py` checked the original YOLO split. It found 960 train images / 1,434 objects and 240 validation images / 325 objects, with no missing labels, malformed annotations, invalid class IDs, or corrupt images. The original split had no byte-identical cross-split images, but image hashing surfaced four near-duplicate groups (dHash distance <=4) across train and validation. Manual contact-sheet review confirmed matching scene/content candidates. This means the original validation score was at risk of leakage.

The original dataset was left untouched. `src/95_make_grouped_yolo_split.py` created `dataset/subsets/multicountry/RDD2022/yolo_grouped/`, moving the three affected validation images and labels into train. The resulting copy has 963 train images / 1,437 objects and 237 validation images / 322 objects. A second audit found zero exact or near-duplicate cross-split candidates, missing labels, malformed boxes, or corrupt images. D40 remains ID 5 in both the dataset YAML and detector checkpoint. Reports: `results/dataset_audit/yolo_integrity_report.json`, `results/dataset_audit/grouped_yolo_integrity_report.json`, and `dataset/subsets/multicountry/RDD2022/yolo_grouped/grouped_split_report.json`.

### Same-split checkpoint comparison

`src/92_evaluate_detector_candidates.py` evaluated all three existing YOLO checkpoints with the same CPU settings (imgsz 640, conf .001, IoU .7) on the grouped real validation set. No synthetic images were used. Results in `results/detector_candidate_comparison_grouped_20261014/comparison.json`:

| Checkpoint | Precision | Recall | F1 | mAP50 | mAP50-95 | D40 recall | D40 AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolo12n_30_epochs` (dashboard selection) | .4367 | .1444 | .2170 | .0954 | .0447 | 0.0000 at operating point | .0485 |
| `yolo12n_improved_30_epochs` | .3854 | .0332 | .0612 | .0146 | .0047 | .0000 | .0032 |
| `yolo12n_road_damage` | .7036 | .0176 | .0343 | .0170 | .0062 | .0189 | .0276 |

The selected checkpoint remains the best aggregate candidate, but grouped validation shows no D40 true positives at its selected operating threshold. Nonzero D40 AP reflects low-confidence ranking and does not mean the dashboard's thresholded inference succeeds. This model is not a reliable pothole detector.

### End-to-end pothole smoke test and diagnosis

The held-out grouped validation image `dataset/subsets/multicountry/RDD2022/yolo/images/val/China_0085.jpg` has three D40 labels: `[330,198,345,221]`, `[305,342,331,391]`, and `[168,405,189,449]` in source-image pixels. The dashboard's EfficientNet, ResNet50, and ViT hybrid paths retained **zero** D40 matches at the configured confidence threshold. The zero-detection image case correctly produced a blank annotated result and a clear empty-state message. Report and generated outputs: `results/pothole_debug/dashboard_smoke_20261014_192857/report.json` and sibling files.

Direct YOLO diagnostics on the same image found one low-confidence D40 proposal at confidence .0154 with box `[314,341,334,392]`, overlapping one labeled pothole only partially (IoU about .47). At confidence .05 and above, no D40 proposal remained. The crop classifiers predicted D00 for this proposal (EfficientNet .884, ResNet .862, ViT .917), so the app's crop verification also rejects it. This is a real model failure; the dashboard must not manufacture or force a pothole result. The false broad boxes are low-quality detector proposals (often D10), not separate potholes.

### Learning-curve review and outstanding training

`src/91_analyze_learning_curves.py` inventories available histories and generates plots/report under `results/overfitting_analysis/`. The five-epoch ViT history shows an accuracy gap and rising validation loss late in training, consistent with overfit signs. EfficientNet does not show a strong gap in its available history. The older ResNet history is short; the road-specific ResNet history has only three epochs. The selected YOLO validation mAP peaked around epoch 27 and fell by epoch 30. The improved YOLO run has only eight logged epochs; CNN+LSTM history is absent, so those histories cannot support a full learning-curve conclusion.

`src/94_train_yolo_high_recall.py` is an explicit, non-overwriting next training entry point using the grouped split, 960 px images, and a new output directory. It was not run because this environment has CPU-only PyTorch. No checkpoint was changed. After training on CUDA, evaluate only on the grouped real validation split and require a successful end-to-end D40 smoke test before describing the dashboard as pothole-capable. Do not train the GAN path: its generated samples remain unapproved and its classifier checkpoint is absent.

### Current verification commands and results

Run from repository root in PowerShell. These commands reproduce audits and evaluations without training:

```powershell
$py = "$env:LOCALAPPDATA\Python\bin\python.exe"
& $py -m py_compile src/79_dashboard.py src/89_road_damage_app.py src/90_debug_pothole_detection.py src/91_smoke_dashboard.py src/91_analyze_learning_curves.py src/92_evaluate_detector_candidates.py src/93_validate_yolo_dataset.py src/94_train_yolo_high_recall.py src/95_make_grouped_yolo_split.py
& $py src/93_validate_yolo_dataset.py --data-yaml dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml
& $py src/92_evaluate_detector_candidates.py --data-yaml dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml --batch 4 --device cpu
& $py src/91_smoke_dashboard.py --image dataset/subsets/multicountry/RDD2022/yolo/images/val/China_0085.jpg --data-yaml dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml
& $py -m streamlit run src/89_road_damage_app.py
```

Compilation, grouped dataset integrity audit, same-split candidate evaluation, dashboard zero-detection handling, and Streamlit AppTest are verified. The three end-to-end pothole paths **failed** the grouped `China_0085` smoke test, as documented above. CUDA training, successful real-image pothole detection, GAN sample approval/training, classifier-pipeline accuracy benchmarking, CNN+LSTM V3 mask evaluation, and long-video stress testing remain outstanding. The previous `China_0000` smoke success is superseded because that image was identified as a near-duplicate split candidate and removed from the grouped validation set.

No weights, labels, source images, or prior metrics were overwritten. No training was run. The current PowerShell policy blocked direct script invocation; invoking PowerShell with `-ExecutionPolicy Bypass` reached the script's approval guard. Use this process-scoped launch form once samples are approved.

## Follow-up audit and current verification â€” 2026-10-14

This section supersedes earlier preprocessing notes and records the latest dashboard code and test output. Current paths were rechecked: the selected YOLO, EfficientNet, ResNet50, ViT, and CNN + Spatial LSTM V3 checkpoints exist; `vit_road_damage/best_vit_road_damage.pt` and the GAN-augmented EfficientNet checkpoint do not. The conditional GAN generator exists. The database currently has 11 `detections` records and zero valid coordinate pairs. Existing data was not changed.

### Pothole diagnosis and outputs

The reproducible test image is `dataset/subsets/multicountry/RDD2022/yolo/images/val/China_0000.jpg`; its paired validation label has two D40 objects. Dataset YAML and active YOLO checkpoint both name D40 as ID 5. Classifier checkpoints differ in integer ordering (ResNet has D40 at index 5; EfficientNet and ViT at index 6), but they share the same name set. The dashboard verifies that set and maps predictions by the checkpoint's class names.

The YOLO detector provides the box coordinates; crop classifiers only label those crops and cannot enlarge or reposition a box. Tile-local coordinates are offset back into source-image coordinates once. On this validation example, output boxes matched the two separate ground-truth potholes at IoU 0.64â€“0.80 for EfficientNet and ResNet pipelines, and 0.68â€“0.80 for ViT. This indicates the two boxes in this smoke test are localized, but does not validate the detector generally. The detector remains weak overall. A fresh CPU validation on all 240 real validation images (325 labeled objects) measured precision 0.4367, recall 0.1514, F1 0.2249, mAP50 0.1030, and mAP50-95 0.0489. For D40, precision at the model's selected operating point was 1.0 but recall was only 0.0635; D40 mAP50 was 0.1194. The high point precision with very low recall means it misses most labeled potholes. Per-class report and plots: `results/pothole_debug/full_validation_20261014/yolo12n_30_epochs_report/validation_metrics.json`. No retraining was done. A broad box in the earlier screenshot cannot be attributed to an exact raw prediction because the original uploaded image and raw prediction file for that screenshot are not present in the repository. Code inspection establishes that such a box originates from YOLO localization; a crop classifier disagreement is now annotated explicitly as `det <class> <score> | cls <class> <score>`.

Current smoke run: `results/pothole_debug/dashboard_smoke_20261014_172243/report.json`. It records input, label, detector checkpoint, D40 mapping, coordinates, detector and classifier confidences, IoUs, timing, and outputs. The original/annotated images and the zero-detection test image are alongside the report in that run directory. At confidence 0.10:

| Pipeline | D40 detector confidence / crop class (confidence) | Box `[x1,y1,x2,y2]` | Distinct GT boxes matched |
| --- | --- | --- | ---: |
| YOLO12n + EfficientNetB0 | 0.285 / D40 (0.672); 0.103 / D50 (0.970) | `[201,382,235,420]`; `[201,241,243,266]` | 2 |
| YOLO12n + ResNet50 | 0.285 / D50 (0.689); 0.103 / D50 (0.918) | `[201,382,235,420]`; `[201,241,243,266]` | 2 |
| YOLO12n + ViT-B/16 | 0.285 / D40 (0.928); 0.143 / D40 (0.937) | `[201,382,235,420]`; `[197,217,245,267]` | 2 |

The ResNet and one EfficientNet crop disagree with the detector label; the UI reports both labels and their separate scores. Raw YOLO inference on this same clean-road image also produced three large D20 proposals at confidence 0.1088?0.1402; one box `[321,0,512,291]` covered 21% of the 512?512 frame despite no D20 ground truth. The hybrid classifier labeled that broad crop D43 at 0.616, so the new area-and-agreement safeguard rejects it. Broad proposals are retained only when the crop classifier strongly agrees with the detector. This is evidence for the specific observed false boxes, not a general box-size accuracy claim. The smoke test itself is one image, not an end-to-end accuracy result. Run it with `python src/91_smoke_dashboard.py` from the repository root. It tests a valid white image at confidence 0.90 as a zero-detection case. Optional video smoke: `python src/91_smoke_dashboard.py --video "dataset/videos/Pothole Videos/pothole_video/test/rgb/0081.mp4"`.

Observed CPU image inference times in this run: EfficientNet 0.478 s, ResNet50 0.494 s, and ViT 1.418 s. The 2-second segmentation clip took 0.669 s to process; the encoded output is 8 frames at 3.996 FPS. These are smoke-run timings, not throughput benchmarks.

### Preprocessing correction

`src/12_create_classification_crops.py` applies CLAHE to the LAB luminance channel (clip limit 2.0, tile grid 8Ã—8) before saving both train and validation crops. The dashboard had omitted this preprocessing at inference. `src/89_road_damage_app.py` now applies the same transform before EfficientNet, ResNet, or ViT classification and verifies detector/classifier class-name sets before loading. Before/after demonstration from a labeled real D40 crop: `results/pothole_debug/clahe_demo_20261014/D40_crop_before_clahe.jpg` and `D40_crop_after_clahe.jpg`.

### CNN + Spatial LSTM and video

The dashboard's configured V3 checkpoint completed a held-out video *inference smoke test* on `dataset/videos/Pothole Videos/pothole_video/test/rgb/0081.mp4` (48 frames, 2.0 seconds): 8 output mask frames were produced in about 0.67 seconds. The output opens with OpenCV and contains 8 readable frames at 3.996 output FPS: `results/pothole_debug/dashboard_smoke_20261014_172243/0081_segmentation_smoke.mp4`. This is execution verification, not a Dice/IoU evaluation. `results/cnn_lstm/segmentation/final_test_report.txt` reports V2 metrics on a selected 10-video test subset (mean per-video Dice 0.6958, IoU 0.5588); it does not evaluate the dashboard's V3 checkpoint. V3's separate external demo coverage reached about 95% on several frames, so coverage severity remains experimental and unreliable outside the test set.

### Supporting artifacts and implementation status

Crop manifest audit: 1,434 train crops from 558 source images and 325 validation crops from 142 source images; no source image occurs in both splits. Current YOLO split counts and annotation audit are recorded above. Exact image duplicates across train/validation were previously checked at zero; near-duplicate detection and corrupt-image scan remain unrun. YOLO data config has no separate object-detection test split. The segmentation dataset has 40 train, 10 validation, and 10 test videos according to its methodology file.

The GAN sample generator produced 10 images per class at 64 by 64. A visual spot-check of `results/gan_individual_samples/D40/sample_01.png` looked like low-detail texture rather than a clearly identifiable pothole; generated sample quality and labels therefore are not approved for training. The GAN classifier is absent and remains excluded from inference. `train_gan_hybrid.ps1` refuses to train until `results/gan_individual_samples/TRAINING_APPROVED.txt` exists and refuses to overwrite the classifier checkpoint. GAN training was not run.

Available classifier logs are crop-classification results, not detector or end-to-end results. The EfficientNet report records validation crop accuracy 0.8308 and macro-F1 0.5288 over 325 crops; the ViT report records 0.8554 accuracy and 0.5684 macro-F1 over the same 325 crop examples. ResNet's training history records 0.8615 validation crop accuracy at epoch 3, without a matching per-class report in its output directory. These values do not measure whether YOLO finds each damage instance.

| Item | Status | Source / checkpoint | Test and evidence | Remaining work |
| --- | --- | --- | --- | --- |
| YOLO12n + EfficientNetB0 | Implemented but not validated | `89_road_damage_app.py`; selected YOLO + `results/efficientnet_b0/best_efficientnet_b0.pt` | One validation image; 2 D40 boxes matched; crop classifier disagreed on one. `dashboard_smoke_20261014_172243/report.json` | Full end-to-end held-out evaluation; weak detector recall and crop disagreements need improvement. |
| YOLO12n + ResNet50 | Implemented but not validated | `89_road_damage_app.py`; selected YOLO + `results/resnet50_road_damage/best_resnet50.pt` | One validation image; 2 D40 boxes localized, both crops classified D50. Same smoke report. | Correct class behavior and evaluate the full pipeline; crop validation history is not detection accuracy. |
| YOLO12n + ViT | Implemented but not validated | `89_road_damage_app.py`; selected YOLO + `results/vit/best_vit.pt` | One validation image; 2 D40 boxes matched and crop classifier agreed. Same smoke report. | Full end-to-end evaluation and class-wise error analysis. |
| CNN + Spatial LSTM | Partially implemented | `89_road_damage_app.py`; `results/cnn_lstm/segmentation/best_cnn_lstm_spatial_v3.pt` | Held-out 2-second video inference smoke; output video validated. Existing test metrics target V2, not V3. | Evaluate V3 against held-out masks; review severe external-video overprediction; test longer videos and memory limits. |
| YOLO12n + GAN-augmented EfficientNetB0 | Unavailable | Generator exists; augmented classifier checkpoint absent; training command `train_gan_hybrid.ps1` | Selector and load guard tested; generator sample visual spot-check poor. | Improve and verify sample quality/labels, train only on real + approved synthetic train samples, compare on real validation, then enable. |
| CLAHE | Implemented and tested | `12_create_classification_crops.py`, `89_road_damage_app.py` | Same LAB luminance transform verified at training/inference; before/after crop artifacts above. | None for current three crop pipelines; do not claim CLAHE for other scripts unless traced. |
| Conventional data augmentation | Implemented but not validated | `39_train_efficientnet.py`, `41_train_resnet.py`, `43_train_vit.py` | Source confirms stochastic transforms are in train loaders and validation uses resize/normalize only. | Compare augmentation choices on held-out data; do not infer benefit from code presence. |
| GAN generation and training-only augmentation | Partially implemented | `46`â€“`48`, `52`, `39_train_efficientnet.py`, `train_gan_hybrid.ps1` | Generator checkpoint and 10/class generated samples exist; loader adds GAN samples only to train; validation loads real crop folders. | Sample quality/label validation, train/evaluate augmented classifier; keep GAN disabled until then. |
| Severity estimation | Partially implemented | `72_estimate_severity.py`; dashboard segmentation coverage thresholds | V3 demo CSV contains predicted coverage and labels its basis; an external clip had about 95% predicted coverage. | Treat as experimental pixel coverage only; no engineering severity calibration. |
| Image inference | Implemented and tested | `89_road_damage_app.py` | Three hybrid image smoke tests, side-by-side image result UI, and zero-detection case; smoke report above. | Larger held-out benchmark; app tests do not establish accuracy. |
| Video inference | Partially implemented | `89_road_damage_app.py`, `80_process_road_video.py` | CNN+LSTM V3 smoke produced valid 8-frame MP4; hybrid video paths not tested in this run. | Test each advertised video model, long-upload memory behavior, FPS/timestamps, and CSV. |
| SQLite history | Implemented and tested | `78_store_detections.py`, `89_road_damage_app.py`; `results/road_damage.db` | New run/detection tables are written and shown in history; isolated DB test retained the legacy row, stored a zero-detection run, and inserted one detection. Outputs are stored in `results/dashboard_outputs/`. | Confirm retention/cleanup policy for stored video outputs; add user-supplied location entry with provenance if needed. |
| GPS/GIS | Implemented but not validated | `79_dashboard.py`, `89_road_damage_app.py`; legacy DB fields and optional user-supplied coordinates | Existing records with valid lat/lon and dashboard-run locations marked `user supplied` are mapped; coordinates are range-checked. No coordinates are inferred. | Exercise map rendering with a known coordinate and test the empty state. |
| Evaluation and model comparison | Partially implemented | `49`â€“`51`, classifier report/history files, `62` | Full real validation results for the selected YOLO checkpoint and crop-classifier reports exist; V2 segmentation report is separate; metrics are not comparable across tasks. | Evaluate all alternative detector checkpoints on the same split; add full end-to-end classifier-pipeline metrics and display `Not evaluated` for missing metrics. |
| Model architecture and activations | Implemented and tested | torchvision EfficientNetB0, ResNet50, ViT-B/16; YOLO12n; `58_train_cnn_lstm_segmentation.py` | Runtime inspection: YOLO/EfficientNet SiLU, ResNet ReLU, ViT GELU; CNN/LSTM source uses ReLU conv blocks and sigmoid mask output. | Document exact checkpoint/config versions and test architecture compatibility after any change. |
| Streamlit dashboard integration | Implemented and smoke-tested | `89_road_damage_app.py`; legacy DB viewer `79_dashboard.py` | AppTest has no exceptions; GAN unavailable state renders; model options exclude GAN; smoke report above. | Keep `79` as legacy database-only view or consolidate its entry point; add UI upload-driven integration tests and metrics overview. |

### Updated verification commands

Latest rerun on 2026-10-14: `src/91_smoke_dashboard.py` passed for all three hybrid image pipelines, the zero-detection image, and the held-out segmentation video. Full output is `results/pothole_debug/dashboard_smoke_20261014_173342/report.json`. On validation image `China_0000.jpg`, the configured YOLO checkpoint detected two separate D40 boxes that matched two distinct ground-truth potholes; EfficientNet and ViT classified both as D40, while ResNet classified both as D50. This is a one-image smoke test, not a general accuracy result. The video smoke wrote 8 frames in 0.743 seconds; V3 segmentation metrics remain unevaluated. `AppTest` passed with zero exceptions, and a temporary SQLite test confirmed that appending dashboard history preserves a pre-existing legacy row and records both a zero-detection run and a detection.

Run from repository root in PowerShell:

```powershell
$py = "C:\Users\Vyom Thakur\AppData\Local\Python\bin\python.exe"
& $py -m py_compile src/79_dashboard.py src/89_road_damage_app.py src/90_debug_pothole_detection.py src/91_smoke_dashboard.py
& $py src/90_debug_pothole_detection.py --find-pothole-image --split val --checkpoint results/training/yolo12n_30_epochs/weights/best.pt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\train_gan_hybrid.ps1
& $py src/91_smoke_dashboard.py --video "dataset/videos/Pothole Videos/pothole_video/test/rgb/0081.mp4"
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run src/89_road_damage_app.py
```

At the time of this follow-up audit, no training or checkpoint replacement had been performed. Later 10-epoch runs are recorded in the source organization and retraining update below. Alternative detector comparison, end-to-end classifier-pipeline evaluation, GAN training/comparison, live GIS map validation, and long-video stress tests remain incomplete.

## Source organization and 10-epoch retraining update (2026-10-15)

Algorithm implementation scripts now live in `src/algorithms/{yolo,efficientnet,resnet50,vit,gan,cnn_lstm,evaluation}/`. The numbered `src/*.py` paths are compatibility launchers, so existing commands continue to work. `src/algorithms/README.md` describes the layout. All Python files compile after the move.

Four dashboard model components completed 10 epochs each on CPU. Outputs are isolated under `results/retraining_10_epochs_20261014/`; current dashboard checkpoints were not overwritten or automatically replaced.

| Model | Training result | Output |
|---|---|---|
| YOLO12n detector | 10 epochs; grouped validation P .435, R .0257, mAP50 .0196, mAP50-95 .00729. D40 recall 0 at its operating point; D40 mAP50 .000215. Not suitable to select as the pothole detector. | `results/training/yolo12n_grouped_10ep_20261014/weights/best.pt` |
| ResNet50 crop classifier | 10 epochs; best grouped validation crop accuracy .8602 at epoch 9. Crop accuracy is not end-to-end detection accuracy. | `results/retraining_10_epochs_20261014/resnet50_grouped/best_resnet50.pt` and `training_history.csv` |
| EfficientNetB0 crop classifier | 10 epochs; best grouped validation crop accuracy .8354 at epoch 9. GAN augmentation was off; zero synthetic samples used. | `results/retraining_10_epochs_20261014/efficientnet_b0_grouped/best_efficientnet_b0.pt` and `training_history.json` |
| ViT-B/16 crop classifier | Recovered from the best saved epoch-2 checkpoint and trained through epoch 10; best validation crop accuracy .8323 at resumed epoch 4. | `results/retraining_10_epochs_20261014/vit_grouped_resume/best_vit_road_damage.pt` and `training_history.csv` |
| CNN + Spatial LSTM V3 segmenter | 10 epochs; best grouped-sequence validation Dice .5150 at epoch 8 (IoU .3468). | `results/retraining_10_epochs_20261014/cnn_lstm_v3/best_cnn_lstm_spatial_v3.pt` |

The GAN hybrid was **not** retrained: generated samples remain unapproved and the augmented classifier checkpoint is missing. The 10-epoch YOLO run did not solve pothole detection, so the dashboard selection remains unchanged. New model checkpoints need their own full validation and smoke tests before use in the app.

## Pipeline status

| Pipeline | Status |
| --- | --- |
| YOLO12n + EfficientNetB0 | CPU smoke on `China_0000.jpg`: 3 outputs, including 2 D40 detector boxes; one additional D20 proposal was classified D43. |
| YOLO12n + ResNet50 | CPU smoke on `China_0000.jpg`: 2 D40 detector boxes; ResNet classified one as D50, so disagreement is shown explicitly. |
| YOLO12n + ViT-B/16 | CPU smoke on `China_0000.jpg`: 2 separate D40 detector boxes, both classified D40 and both matching a distinct D40 label at IoU >=0.5. |
| CNN + Spatial LSTM | Checkpoint and video path exist; previously returned masks for held-out validation sequences. Experimental segmentation; external-video reliability remains unverified. |
| YOLO12n + GAN + EfficientNetB0 | Not available: augmented classifier checkpoint is absent. Do not select or report this as a trained model. |

## Reproduction and validation commands

Run from repository root in PowerShell:

```powershell
$py = "C:\Users\Vyom Thakur\AppData\Local\Python\bin\python.exe"
& $py -m py_compile src/89_road_damage_app.py src/90_debug_pothole_detection.py
& $py src/90_debug_pothole_detection.py dataset/subsets/multicountry/RDD2022/yolo/images/train/China_0059.jpg --checkpoint results/training/yolo12n_30_epochs/weights/best.pt
& $py src/90_debug_pothole_detection.py --find-pothole-image --split val --checkpoint results/training/yolo12n_30_epochs/weights/best.pt
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run src/89_road_damage_app.py
```

The diagnostic script saves threshold outputs to `results/pothole_debug/` and prints each output path. For an alternate checkpoint, pass it with `--checkpoint`.

## Next training recommendation (not run)

Inference settings alone cannot meet the â€œall potholesâ€ goal with the current detector. Before training, review the 83/16 `Repair` XML cases and retain them as negatives only if that taxonomy decision is intentional. Then train a fresh checkpoint in a new output directory, starting from `yolo12n.pt`, with the verified current YAML, `imgsz=960`, moderate batch size adjusted to available RAM, `patience=15`, fixed seed, and validation each epoch. Compare class-specific D40 precision/recall and mAP50â€“95 plus overall metrics against the existing three checkpoints on the same real validation split. Do not overwrite any current checkpoint. The CPU training scripts are very slow on this Intel CPU; use a CUDA-capable machine for this run. No new weights or claimed improvements exist yet.




## Dashboard false-positive filtering update (2026-10-15)

The latest dashboard screenshot showed weak D40 rescue boxes and broad non-pothole detections where detector and crop classifier disagreed. `src/89_road_damage_app.py` now applies stricter review to weak D40 proposals: detector confidence must be at least 0.03, classifier D40 confidence at least 0.90, and the candidate must be at least 18 pixels on each side. Non-D40 detector/classifier class disagreements are filtered. Detector-localized D40 boxes remain visible even when a crop classifier disagrees, since prior smoke evidence showed crop classifiers mislabeled true pothole boxes.

After the change, the validation smoke at `China_0000.jpg` retained two individual boxes matching the two distinct D40 annotations for each of EfficientNet, ResNet, and ViT; the blank image still returned zero detections. Report and annotated outputs: `results/pothole_debug/dashboard_smoke_20261015_231033/report.json`. The uploaded screenshot itself was not available as a local image for exact rerun, so this verifies the revised behavior on the project's labeled validation example, not that specific road image.
