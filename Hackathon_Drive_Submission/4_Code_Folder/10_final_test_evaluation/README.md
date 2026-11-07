# Step 10: Final Test Evaluation

## Overview
Executes independent, rigorous evaluation on the held-out, untouched test split (100% real images):
1. **Performance Metrics**:
   - Multi-class Confusion Matrix (raw counts and row-normalized percentages).
   - Per-class Precision, Recall, F1-Score, and Support across all 10 road damage classes.
   - Macro Average F1 and Weighted Average F1.
   - Real-world inference latency (ms per crop).
2. **Methodological Verification Checklist**:
   - Asserts zero duplicate leakage and 100% real imagery.

## Files
- `test_evaluator.py`: Comprehensive test evaluation engine.
- `confusion_matrix_plotter.py`: Labeled heatmap generator.
- `generate_final_report.py`: Assembles the final Markdown report with tables and embedded figures.
- `run_final_evaluation.py`: Standalone CLI runner.

## Usage
```powershell
python 10_final_test_evaluation/run_final_evaluation.py --model-name baseline_cnn
```
Outputs are written to `results/final_evaluation/`.
