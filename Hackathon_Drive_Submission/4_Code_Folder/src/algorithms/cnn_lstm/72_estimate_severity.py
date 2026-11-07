
from pathlib import Path
import pandas as pd
import numpy as np

# Project folders
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DEMO_DIR = ROOT / "results" / "cnn_lstm" / "demo"

# Severity thresholds based on predicted damaged-area coverage
LOW_MAX = 5.0
MEDIUM_MAX = 15.0

# Smooth predictions over nearby frames
SMOOTHING_WINDOW = 5


def find_coverage_column(df):
    """Find the column containing predicted damage coverage."""
    candidates = [
        "damage_coverage_percent",
        "coverage_percent",
        "predicted_coverage_percent",
        "damage_coverage",
        "coverage",
        "predicted_coverage",
    ]

    # Prefer known column names
    for name in candidates:
        if name in df.columns:
            return name

    # Otherwise search by column name
    for col in df.columns:
        name = col.lower()
        if "coverage" in name or ("damage" in name and "percent" in name):
            return col

    raise ValueError(
        "Could not identify the coverage column. "
        f"Available columns: {list(df.columns)}"
    )


def classify_severity(coverage):
    if coverage < LOW_MAX:
        return "Low"
    elif coverage < MEDIUM_MAX:
        return "Medium"
    return "High"


def main():
    if not DEMO_DIR.exists():
        raise FileNotFoundError(f"Demo folder not found: {DEMO_DIR}")

    # Find generated CNN + LSTM coverage files
    files = sorted(DEMO_DIR.glob("*_coverage_v3.csv"))

    if not files:
        files = sorted(DEMO_DIR.glob("*coverage*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No coverage CSV found in {DEMO_DIR}\n"
            "Run src/70_standalone_video_demo.py first."
        )

    print("\nAvailable prediction files:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file.name}")

    choice = input(
        "\nEnter file number (press Enter for the latest listed file): "
    ).strip()

    if choice:
        if not choice.isdigit() or not 1 <= int(choice) <= len(files):
            raise ValueError("Invalid file number.")
        source = files[int(choice) - 1]
    else:
        source = max(files, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("The selected CSV contains no rows.")

    coverage_col = find_coverage_column(df)
    coverage = pd.to_numeric(df[coverage_col], errors="coerce")

    if coverage.isna().all():
        raise ValueError(f"Column '{coverage_col}' has no numeric values.")

    # Convert fractions (0.0–1.0) to percentages if necessary
    valid = coverage.dropna()
    if not valid.empty and valid.max() <= 1.0:
        coverage = coverage * 100.0

    df["raw_coverage_percent"] = coverage.clip(0, 100)

    # Smooth isolated spikes across neighbouring frames
    df["smoothed_coverage_percent"] = (
        df["raw_coverage_percent"]
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )

    df["estimated_severity"] = (
        df["smoothed_coverage_percent"]
        .apply(classify_severity)
    )

    df["severity_basis"] = (
        "Predicted damaged-area coverage; not physical damage depth"
    )

    output = source.with_name(
        source.stem.replace("_coverage", "_severity") + ".csv"
    )
    df.to_csv(output, index=False)

    print("\n========== SEVERITY ESTIMATION ==========")
    print(f"Input file:  {source.name}")
    print(f"Output file: {output.name}")
    print(f"Coverage column: {coverage_col}")
    print("\nSeverity thresholds:")
    print(f"Low:    below {LOW_MAX}%")
    print(f"Medium: {LOW_MAX}% to below {MEDIUM_MAX}%")
    print(f"High:   {MEDIUM_MAX}% or above")

    print("\nFrame counts by estimated severity:")
    print(df["estimated_severity"].value_counts().to_string())

    print(
        "\nMean smoothed coverage: "
        f"{df['smoothed_coverage_percent'].mean():.2f}%"
    )
    print("\nNOTE: These labels are experimental area-based estimates.")
    print("Review false-positive masks before using them for decisions.")


if __name__ == "__main__":
    main()
