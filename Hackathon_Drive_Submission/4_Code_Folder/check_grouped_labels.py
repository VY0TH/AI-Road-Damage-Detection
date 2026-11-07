from pathlib import Path
import yaml

yaml_path = Path("dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml")
cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
root = Path(cfg["path"])
if not root.is_absolute():
    root = Path.cwd() / root

for split in ("train", "val"):
    label_dir = root / "labels" / split
    bad = []
    total_boxes = 0

    for f in label_dir.glob("*.txt"):
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                cls = int(parts[0])
                box = [float(x) for x in parts[1:]]
                valid = (
                    len(parts) == 5
                    and 0 <= cls < int(cfg["nc"])
                    and all(0 <= x <= 1 for x in box)
                    and box[2] > 0 and box[3] > 0
                )
                if not valid:
                    bad.append(f"{f}:{line_no}: {line}")
                else:
                    total_boxes += 1
            except (ValueError, IndexError):
                bad.append(f"{f}:{line_no}: {line}")

    print(f"\n{split}: valid boxes={total_boxes}, invalid rows={len(bad)}")
    for item in bad[:20]:
        print("  ", item)
