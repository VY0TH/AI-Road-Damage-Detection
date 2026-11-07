# Algorithm source folders

The implementation files are grouped by algorithm. Existing numbered `src/*.py` entry points are tiny compatibility launchers; the source of truth lives in these folders.

| Folder | Contents |
|---|---|
| `yolo/` | Detector training, validation, candidate evaluation, and diagnostics |
| `efficientnet/` | EfficientNet crop preparation, training, and hybrid inference |
| `resnet50/` | ResNet50 training and hybrid inference |
| `vit/` | ViT training and hybrid inference |
| `gan/` | GAN training and sample generation |
| `cnn_lstm/` | CNN + Spatial LSTM training, video segmentation, and evaluation |
| `evaluation/` | Cross-model reports, learning curves, and activation visualizations |

Examples using the organized source paths:

```powershell
python src/algorithms/yolo/94_train_yolo_high_recall.py --epochs 10 --device cpu
python src/algorithms/efficientnet/39_train_efficientnet.py
python src/algorithms/resnet50/82_train_resnet50.py
python src/algorithms/vit/84_train_vit.py
python src/algorithms/cnn_lstm/58_train_cnn_lstm_segmentation.py
python src/algorithms/efficientnet/hybrid_yolo_gan_efficientnet.py path/to/road_image.jpg
```

`hybrid_yolo_gan_efficientnet.py` is the standalone image inference entry point
for YOLO plus the GAN-augmented EfficientNetB0 classifier. It also accepts a
directory of images and writes annotated images and per-image CSV results under
`results/hybrid_yolo_gan_efficientnet/` by default.

The previous `src/<number>_<name>.py` paths remain as launch shims for compatibility. The grouped implementation scripts are the source of truth.

## Evaluation and cross-validation

The historical fold runner is disabled because its starting checkpoint leaked fold images. See `run_clean_cv.py` and `src/algorithms/yolo/97_prepare_clean_cv.py` for the reviewed-group guarded workflow.
