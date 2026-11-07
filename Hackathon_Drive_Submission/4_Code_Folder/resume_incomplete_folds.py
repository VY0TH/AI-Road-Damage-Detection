"""Unsafe legacy resume runner disabled.

It used YOLO(last.pt).train(resume=True), which reuses saved task/data/epoch
arguments and previously launched an unrelated COCO 100-epoch job. Historical
road-damage fold runs are also contaminated by initialization leakage.
Use run_clean_cv.py only after configuring clean folds.
"""
raise SystemExit(
    "Resume runner disabled: saved args may target a different dataset/task. "
    "Do not resume old fold checkpoints; use the reviewed clean CV workflow."
)
