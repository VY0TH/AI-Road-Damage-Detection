"""Single entry point for the project's initial model training scripts.

Each model has its own architecture, data format, and training loop, so this
runner launches the existing implementation in an isolated Python process.
Choose one model or ``all`` to run them in sequence.

Examples from the project root::

    python src/initial_model_training.py --list
    python src/initial_model_training.py --model yolo12-baseline
    python src/initial_model_training.py --model efficientnet --epochs 10
    python src/initial_model_training.py --model all

The YOLO12 baseline script is currently configured as a 3-epoch CPU smoke run.
The ``--epochs`` option applies to EfficientNet only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
SRC_ROOT = Path(__file__).resolve().parent

TRAINING_SCRIPTS = {
    "yolo-small": SRC_ROOT / "11_train_yolo_small.py",
    "yolo12-baseline": SRC_ROOT / "algorithms" / "yolo" / "22_train_yolo12_baseline.py",
    "efficientnet": SRC_ROOT / "algorithms" / "efficientnet" / "39_train_efficientnet.py",
    "resnet50": SRC_ROOT / "algorithms" / "resnet50" / "41_train_resnet.py",
    "vit": SRC_ROOT / "algorithms" / "vit" / "43_train_vit.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*TRAINING_SCRIPTS, "all"))
    parser.add_argument("--epochs", type=int, help="Set EfficientNet epochs (default is defined by its script)")
    parser.add_argument("--list", action="store_true", help="List available initial training workflows")
    args = parser.parse_args()

    if args.list:
        print("Available initial training workflows:")
        for name, path in TRAINING_SCRIPTS.items():
            print(f"  {name:18} {path.relative_to(SRC_ROOT)}")
        return 0
    if args.model is None:
        parser.error("choose --model or use --list")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    models = list(TRAINING_SCRIPTS) if args.model == "all" else [args.model]
    for name in models:
        script = TRAINING_SCRIPTS[name]
        if not script.is_file():
            raise FileNotFoundError(f"Training script not found: {script}")

        environment = os.environ.copy()
        if name == "efficientnet" and args.epochs is not None:
            environment["ROAD_DAMAGE_EPOCHS"] = str(args.epochs)

        print(f"\n{'=' * 72}\nStarting {name}: {script}\n{'=' * 72}", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        print(f"Finished {name}.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
