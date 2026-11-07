# Step 1: Load Image Dataset

## Overview
This step is the foundational entry point of the image classification pipeline:
1. Discovers and inspects dataset folder structures (supports class-named folders `<split>/<class>/<image>` and CSV manifests).
2. Verifies image file headers and verifies against corrupt/broken image files.
3. Computes dataset summary statistics: total images, class frequencies, min/max/mean resolutions, and file format distribution.
4. Provides standard PyTorch `Dataset` (`ImageFolderDataset`, `ManifestDataset`) and `DataLoader` abstractions.

## Files
- `dataset_loader.py`: Reusable PyTorch `Dataset` and `DataLoader` classes.
- `inspect_dataset.py`: Comprehensive dataset integrity and distribution checker.
- `run_load_dataset.py`: Standalone CLI execution script.

## Usage
Run directly from the project root:
```powershell
python 01_load_image_dataset/run_load_dataset.py
```
Outputs are saved to `results/step01_dataset_summary.json`.
