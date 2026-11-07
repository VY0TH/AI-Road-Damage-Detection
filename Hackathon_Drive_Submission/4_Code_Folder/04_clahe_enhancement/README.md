# Step 4: Contrast-Limited Adaptive Histogram Equalization (CLAHE)

## Overview
Contrast enhancement is specifically applied when road surface imagery suffers from:
1. Low illumination, overcast skies, or dense tree shadows obscuring asphalt cracks.
2. Low natural dynamic range (low RMS contrast < 0.20).

### Key Rules
- **LAB Color Space Processing**: CLAHE is strictly applied to the L (luminance) channel only, leaving A and B chrominance channels unaltered to avoid artificial color distortion.
- **Selective / Adaptive Application**: Contrast enhancement is evaluated per image. Images with already high dynamic range are preserved to prevent amplifying background camera sensor noise or pebble speckles.

## Files
- `clahe_processor.py`: Core OpenCV LAB CLAHE transform.
- `contrast_evaluator.py`: Quantitative contrast evaluator (RMS contrast, Michelson, Shannon entropy).
- `run_clahe.py`: Standalone CLI runner with before-and-after side-by-side comparison generator.

## Usage
```powershell
python 04_clahe_enhancement/run_clahe.py
```
Outputs visual comparisons to `results/clahe_comparisons/` and report to `results/step04_clahe_contrast_report.json`.
