import cv2
from pathlib import Path

ROOT = Path(
    r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"
)

image_path = (
    ROOT / "dataset/subsets/multicountry/RDD2022/yolo"
    / "images/val/United_States_0120.jpg"
)

label_path = (
    ROOT / "dataset/subsets/multicountry/RDD2022/yolo"
    / "labels/val/United_States_0120.txt"
)

output_path = (
    ROOT / "results/ground_truth_United_States_0120.jpg"
)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

height, width = image.shape[:2]

with open(label_path, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    class_id, xc, yc, bw, bh = map(float, line.split())

    x1 = int((xc - bw / 2) * width)
    y1 = int((yc - bh / 2) * height)
    x2 = int((xc + bw / 2) * width)
    y2 = int((yc + bh / 2) * height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, f"GT class {int(class_id)}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), image)

print(f"Ground-truth image saved: {output_path}")