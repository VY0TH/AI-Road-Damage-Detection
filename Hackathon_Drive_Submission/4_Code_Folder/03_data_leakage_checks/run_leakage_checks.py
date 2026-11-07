"""Step 3 CLI Runner: Comprehensive Data Leakage Audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_target_leakage import audit_target_leakage
from check_duplicate_leakage import audit_duplicate_leakage
from check_suspicious_features import audit_suspicious_features
import pipeline_config as cfg


def run_step_3(dataset_dir: Path | None = None) -> dict:
    target_dir = dataset_dir or cfg.PROCESSED_DATA_DIR
    print(f"\n{'=' * 70}")
    print(f"[STEP 3] RUNNING COMPREHENSIVE DATA LEAKAGE AUDIT")
    print(f"Target dataset: {target_dir}")
    print(f"{'=' * 70}")

    print("\n1. Checking Target Leakage (Metadata, EXIF, Featurization)...")
    target_results = audit_target_leakage(target_dir)
    print(f"   Target Leakage: {'PASS' if target_results['target_leakage_passed'] else 'FAIL'}")
    if target_results["details"]:
        print(f"   Warnings: {target_results['details']}")

    print("\n2. Checking Duplicate Leakage (Exact SHA-256 & Perceptual dHash)...")
    dup_results = audit_duplicate_leakage(target_dir)
    print(f"   Exact Cross-Split Duplicates: {dup_results['exact_cross_split_duplicates_count']}")
    print(f"   Near-Duplicates Detected    : {dup_results['near_duplicates_sample_count']}")
    print(f"   Duplicate Leakage Status    : {'PASS' if dup_results['duplicate_leakage_passed'] else 'FLAGGED'}")

    print("\n3. Checking Suspicious Features & Shortcut Correlations...")
    suspicious_results = audit_suspicious_features(target_dir)
    print(f"   Filename Overlap Across Splits: {len(suspicious_results['filename_overlap_across_splits'])}")
    print(f"   Suspicious Domain Shortcuts   : {len(suspicious_results['suspicious_prefix_correlations'])}")
    if suspicious_results["suspicious_prefix_correlations"]:
        for item in suspicious_results["suspicious_prefix_correlations"]:
            print(f"   -> [ALERT] {item['warning']}")

    full_report = {
        "dataset_root": str(target_dir),
        "target_leakage": target_results,
        "duplicate_leakage": dup_results,
        "suspicious_features": suspicious_results,
        "overall_verdict": (
            "PASSED" if (target_results["target_leakage_passed"] and dup_results["exact_cross_split_duplicates_count"] == 0)
            else "REQUIRES_ATTENTION"
        ),
    }

    # Save JSON and Markdown report
    output_json = cfg.RESULTS_DIR / "step03_leakage_audit_report.json"
    output_md = cfg.RESULTS_DIR / "step03_leakage_audit_report.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Step 3: Data Leakage Audit Report\n\n")
        f.write(f"**Target Dataset**: `{target_dir}`  \n")
        f.write(f"**Overall Verdict**: `{full_report['overall_verdict']}`  \n\n")
        f.write("## 1. Target Leakage Checks\n")
        f.write(f"- Status: {'PASS' if target_results['target_leakage_passed'] else 'FAIL'}\n")
        f.write(f"- Metadata / EXIF Leaks: {target_results['exif_metadata_leaks_found']}\n\n")
        f.write("## 2. Duplicate Leakage Checks\n")
        f.write(f"- Exact Cross-Split Duplicates: {dup_results['exact_cross_split_duplicates_count']}\n")
        f.write(f"- Perceptual Near-Duplicates: {dup_results['near_duplicates_sample_count']}\n\n")
        f.write("## 3. Suspicious Features & Domain Bias\n")
        f.write(f"- Cross-split Filename Collisions: {len(suspicious_results['filename_overlap_across_splits'])}\n")
        f.write(f"- Domain Prefix Shortcut Alerts: {len(suspicious_results['suspicious_prefix_correlations'])}\n")
        for item in suspicious_results["suspicious_prefix_correlations"]:
            f.write(f"  - ⚠️ {item['warning']}\n")

    print(f"\n{'=' * 70}")
    print(f"[STEP 3 COMPLETED] Overall Verdict: {full_report['overall_verdict']}")
    print(f"Reports saved to:")
    print(f"  JSON: {output_json}")
    print(f"  MD  : {output_md}")
    print(f"{'=' * 70}\n")
    return full_report


if __name__ == "__main__":
    run_step_3()
