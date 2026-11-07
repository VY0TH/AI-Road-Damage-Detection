"""Step 1 CLI Runner: Load and Validate Image Dataset."""
import sys
from pathlib import Path

# Add project root and step dir to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import ImageFolderDataset, create_dataloader
from inspect_dataset import inspect_directory, print_report
import pipeline_config as cfg


def run_step_1(dataset_dir: Path | None = None) -> dict:
    target_dir = dataset_dir or cfg.PROCESSED_DATA_DIR
    print(f"\n[STEP 1] Loading and Inspecting Image Dataset from: {target_dir}")
    report = inspect_directory(target_dir)
    print_report(report)

    output_json = cfg.RESULTS_DIR / "step01_dataset_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Test loading batch via PyTorch DataLoader
    train_dir = target_dir / "train"
    if train_dir.is_dir():
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        ds = ImageFolderDataset(train_dir, cfg.CLASS_NAMES, transform=transform)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)
        for images, labels in loader:
            print(f"[STEP 1 Verification] Successfully loaded batch: images shape={images.shape}, labels={labels.tolist()}")
            break

    print(f"[STEP 1 COMPLETED] Dataset report saved to: {output_json}\n")
    return report


if __name__ == "__main__":
    run_step_1()
