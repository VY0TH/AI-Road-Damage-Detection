
from pathlib import Path
from collections import defaultdict

ROOT = Path("dataset/subsets/multicountry/RDD2022/yolo")
SPLITS = ("train", "val")

CLASS_NAMES = {
    0: "D00", 1: "D01", 2: "D10", 3: "D11",
    4: "D20", 5: "D40", 6: "D43", 7: "D44",
    8: "D50", 9: "D0w0"
}

RARE_CLASSES = {1, 3, 6, 9}
MAX_EXAMPLES = 10


def main():
    examples = defaultdict(list)
    object_counts = defaultdict(int)
    empty_examples = []

    for split in SPLITS:
        label_dir = ROOT / "labels" / split
        image_dir = ROOT / "images" / split

        if not label_dir.exists() or not image_dir.exists():
            print(f"ERROR: Missing folders for {split}")
            print("Expected:", label_dir, "and", image_dir)
            return

        labels = sorted(label_dir.glob("*.txt"))
        print(f"\nScanning {split}: {len(labels)} label files")

        for label_file in labels:
            lines = [
                line.strip()
                for line in label_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            if not lines:
                if len(empty_examples) < 20:
                    empty_examples.append(
                        (split, label_file.stem)
                    )
                continue

            for line in lines:
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    continue

                object_counts[(split, class_id)] += 1

                if class_id in RARE_CLASSES:
                    if len(examples[(split, class_id)]) < MAX_EXAMPLES:
                        examples[(split, class_id)].append(
                            (label_file.name, line)
                        )

    print("\n" + "=" * 65)
    print("RARE-CLASS REPORT")
    print("=" * 65)

    for class_id in sorted(RARE_CLASSES):
        print(f"\n{CLASS_NAMES[class_id]} (ID {class_id})")

        for split in SPLITS:
            count = object_counts[(split, class_id)]
            print(f"  {split}: {count} objects")

            for filename, annotation in examples[(split, class_id)]:
                print(f"    {filename}: {annotation}")

    print("\nSample images with empty labels:")
    for split, stem in empty_examples:
        print(f"  {split}: {stem}")

    print("\nReport complete. No files were modified.")


if __name__ == "__main__":
    main()
