"""Legacy CV runner disabled because it initialized from leaky road-damage weights.
Use run_clean_cv.py after manually reviewing duplicate groups and constructing
clean fold configs. See results/project_audit.md.
"""
raise SystemExit(
    "Legacy CV is disabled: its original weights were trained on all source fold images. "
    "Prepare reviewed folds and use run_clean_cv.py with generic pretrained weights."
)
