"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "gan" / "47_train_conditional_gan.py"), run_name="__main__")
