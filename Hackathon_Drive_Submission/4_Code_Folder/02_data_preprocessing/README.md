# Step 2: Data Preprocessing

## Overview
This step standardizes all images before downstream modeling:
1. **Aspect Ratio Preservation & Letterboxing**: Scales images using high-quality Lanczos resampling while padding to prevent geometric distortion of cracks and potholes.
2. **Channel Standardization**: Ensures every image is 3-channel RGB.
3. **Crop Extraction**: If starting from full-frame scene images with annotations (VOC XML or YOLO), extracts labeled bounding boxes with configurable margins.
4. **Validation & Filtering**: Automatically detects and drops zero-byte or unreadable files, emitting a structured `preprocessing_manifest.csv`.

## Files
- `image_preprocessor.py`: Standardization and letterbox padding engine.
- `crop_extractor.py`: Bounding-box crop extractor for road damage objects.
- `run_preprocessing.py`: CLI execution script.

## Usage
```powershell
python 02_data_preprocessing/run_preprocessing.py
```
Outputs are saved to `dataset/processed/preprocessed_standardized/`.
