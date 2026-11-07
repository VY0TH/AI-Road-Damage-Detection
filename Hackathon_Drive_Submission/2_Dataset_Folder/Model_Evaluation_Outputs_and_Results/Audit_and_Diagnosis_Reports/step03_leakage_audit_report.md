# Step 3: Data Leakage Audit Report

**Target Dataset**: `C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject\dataset\processed\efficientnet_crops`  
**Overall Verdict**: `PASSED`  

## 1. Target Leakage Checks
- Status: PASS
- Metadata / EXIF Leaks: 0

## 2. Duplicate Leakage Checks
- Exact Cross-Split Duplicates: 0
- Perceptual Near-Duplicates: 1

## 3. Suspicious Features & Domain Bias
- Cross-split Filename Collisions: 0
- Domain Prefix Shortcut Alerts: 2
  - ⚠️ Class D01 is overwhelmingly (>90%) sourced from 'India', risking shortcut learning.
  - ⚠️ Class D50 is overwhelmingly (>90%) sourced from 'Japan', risking shortcut learning.
