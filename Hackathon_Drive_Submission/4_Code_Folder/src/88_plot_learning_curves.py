"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "evaluation" / "88_plot_learning_curves.py"), run_name="__main__")
