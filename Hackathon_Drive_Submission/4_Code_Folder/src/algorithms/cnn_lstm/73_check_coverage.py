
from pathlib import Path
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
CSV_PATH = (
    ROOT / "results" / "cnn_lstm" / "demo"
    / "19409613-uhd_3840_2160_30fps_coverage_v3.csv"
)

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

col = "predicted_damage_coverage_pct"

if col not in df.columns:
    raise ValueError(f"Column '{col}' not found.")

coverage = pd.to_numeric(df[col], errors="coerce").dropna()

print("\nCoverage statistics:")
print(coverage.describe().to_string())

print("\nPercentage of frames at different coverage levels:")
for threshold in [5, 15, 30, 50, 70, 90]:
    pct = (coverage >= threshold).mean() * 100
    print(f"Coverage >= {threshold:2d}%: {pct:.2f}% of frames")
