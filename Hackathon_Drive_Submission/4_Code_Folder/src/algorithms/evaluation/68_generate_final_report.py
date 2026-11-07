
from pathlib import Path
import csv

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
OUT_DIR = ROOT / "results" / "cnn_lstm" / "segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

report_path = OUT_DIR / "final_test_report.txt"
csv_path = OUT_DIR / "final_test_results.csv"

results = [
    ("0081.npz", 0.5237, 0.3547, 9.02, 7.75),
    ("0082.npz", 0.8987, 0.8160, 22.90, 25.12),
    ("0083.npz", 0.4820, 0.3175, 2.78, 7.16),
    ("0084.npz", 0.6573, 0.4895, 16.47, 19.24),
    ("0085.npz", 0.6923, 0.5294, 32.91, 25.46),
    ("0086.npz", 0.7066, 0.5464, 17.19, 18.29),
    ("0087.npz", 0.8518, 0.7419, 29.33, 26.45),
    ("0088.npz", 0.8089, 0.6791, 29.97, 35.02),
    ("0089.npz", 0.4191, 0.2651, 12.79, 17.85),
    ("0090.npz", 0.9178, 0.8480, 26.84, 25.21),
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video", "dice", "iou",
        "ground_truth_foreground_pct",
        "predicted_foreground_pct",
    ])
    writer.writerows(results)

report = """ROAD DAMAGE SEGMENTATION
FINAL TEST REPORT

Model: CNN + Spatial LSTM V2
Checkpoint: best_cnn_lstm_spatial_v2.pt
Test videos: 10
Threshold: 0.5

OVERALL TEST METRICS
Mean per-video Dice: 0.6958
Mean per-video IoU: 0.5588
Global pixel Dice: 0.7514
Global pixel IoU: 0.6018
Pixel accuracy: 0.8986
Ground-truth foreground: 20.02%
Predicted foreground: 20.76%

IMPORTANT LIMITATIONS
- Evaluation uses a selected subset of 10 test videos.
- Results may not generalize to other roads or environments.
- This model performs binary damage segmentation, not damage-type classification.
- Pixel accuracy should not be interpreted alone.
- Test metrics are reported at the fixed threshold of 0.5.
"""

report_path.write_text(report, encoding="utf-8")

print("Final report saved:", report_path)
print("Per-video CSV saved:", csv_path)
