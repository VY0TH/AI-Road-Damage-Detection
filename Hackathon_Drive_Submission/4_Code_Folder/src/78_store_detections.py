
from pathlib import Path
import csv
import sqlite3
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "hybrid_yolo_efficientnet"
DB_PATH = ROOT / "results" / "road_damage.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_database(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT NOT NULL,
            detection_index INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            yolo_class TEXT,
            yolo_confidence REAL,
            efficientnet_class TEXT,
            efficientnet_confidence REAL,
            classes_agree INTEGER,
            latitude REAL,
            longitude REAL,
            severity TEXT DEFAULT 'Unassessed',
            source_csv TEXT NOT NULL,
            UNIQUE(source_csv, detection_index)
        )
    """)
    conn.commit()


def import_csv(conn, csv_path):
    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "image", "detection_index",
            "x1", "y1", "x2", "y2",
            "yolo_class", "yolo_confidence",
            "efficientnet_class", "efficientnet_confidence",
            "classes_agree",
        }

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            print(f"Skipped unexpected CSV format: {csv_path.name}")
            return 0

        rows = list(reader)

    timestamp = datetime.now(timezone.utc).isoformat()
    inserted = 0

    for row in rows:
        conn.execute("""
            INSERT OR IGNORE INTO detections (
                image, detection_index, timestamp_utc,
                x1, y1, x2, y2,
                yolo_class, yolo_confidence,
                efficientnet_class, efficientnet_confidence,
                classes_agree, source_csv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["image"],
            int(row["detection_index"]),
            timestamp,
            int(row["x1"]),
            int(row["y1"]),
            int(row["x2"]),
            int(row["y2"]),
            row["yolo_class"],
            float(row["yolo_confidence"]),
            row["efficientnet_class"],
            float(row["efficientnet_confidence"]),
            int(row["classes_agree"].strip().lower() in ("true", "1", "yes")),
            csv_path.name,
        ))
        inserted += conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    print(f"{csv_path.name}: imported {inserted} new records")
    return inserted


def main():
    csv_files = sorted(INPUT_DIR.glob("*_hybrid.csv"))

    if not csv_files:
        print(f"No hybrid CSV files found in: {INPUT_DIR}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        create_database(conn)

        for csv_path in csv_files:
            import_csv(conn, csv_path)

        total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

    print(f"\nDatabase: {DB_PATH}")
    print(f"Total stored detections: {total}")


if __name__ == "__main__":
    main()
