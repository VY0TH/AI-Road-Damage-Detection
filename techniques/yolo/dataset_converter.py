"""YOLO Dataset Conversion and YAML Configuration Generator."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

import pipeline_config as cfg


def convert_voc_to_yolo_label(xml_path: Path, img_width: int, img_height: int) -> list[str]:
    lines = []
    tree = ET.parse(xml_path)
    for obj in tree.getroot().findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in cfg.CLASS_TO_IDX:
            continue
        cls_id = cfg.CLASS_TO_IDX[name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin") or 0)
        ymin = float(bndbox.findtext("ymin") or 0)
        xmax = float(bndbox.findtext("xmax") or 0)
        ymax = float(bndbox.findtext("ymax") or 0)

        # Convert to YOLO normalized format (x_center, y_center, width, height)
        bw = xmax - xmin
        bh = ymax - ymin
        x_center = (xmin + bw / 2.0) / img_width
        y_center = (ymin + bh / 2.0) / img_height
        w_norm = bw / img_width
        h_norm = bh / img_height

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return lines


def generate_data_yaml(output_yaml_path: Path, base_dir: Path, train_rel: str = "images/train", val_rel: str = "images/val"):
    data = {
        "path": str(base_dir.resolve().as_posix()),
        "train": train_rel,
        "val": val_rel,
        "nc": len(cfg.CLASS_NAMES),
        "names": cfg.CLASS_NAMES,
    }
    output_yaml_path = Path(output_yaml_path)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return output_yaml_path
