
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "results" / "road_damage.db"
OUTPUT_DIR = ROOT / "results" / "hybrid_yolo_efficientnet"

st.set_page_config(
    page_title="Road Damage Monitoring",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stAlert"] { color: #222b25; background: #fff4d6; border-color: #d7a72f; }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Road Damage Monitoring System")
st.caption(
    "YOLO12n detection • EfficientNetB0 classification • "
    "Local database"
)

st.info(
    "For image and video uploads, model selection, and recorded model metrics, "
    "run the unified app from the project root: "
    r'& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run src/89_road_damage_app.py'
)

if not DB_PATH.exists():
    st.error("Database not found. Run src/78_store_detections.py first.")
    st.stop()

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        "SELECT * FROM detections ORDER BY id DESC",
        conn,
    )

if df.empty:
    st.warning(
        "No detections have been stored yet. "
        "Run the hybrid inference and import its CSV output."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
classes = sorted(
    set(df["efficientnet_class"].dropna().astype(str))
)
selected_classes = st.sidebar.multiselect(
    "EfficientNetB0 class",
    classes,
    default=classes,
)

filtered = df[
    df["efficientnet_class"].isin(selected_classes)
].copy()

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Stored detections", len(filtered))
c2.metric(
    "Images represented",
    filtered["image"].nunique(),
)
c3.metric(
    "Class agreement",
    f"{100 * filtered['classes_agree'].mean():.1f}%"
    if len(filtered) else "N/A",
)

st.subheader("Detection records")
columns = [
    "image",
    "yolo_class",
    "yolo_confidence",
    "efficientnet_class",
    "efficientnet_confidence",
    "classes_agree",
    "timestamp_utc",
]
st.dataframe(
    filtered[columns],
    width="stretch",
    hide_index=True,
)

st.subheader("Damage class distribution")
counts = (
    filtered["efficientnet_class"]
    .value_counts()
    .rename_axis("Class")
    .rename("Detections")
)
st.bar_chart(counts)

st.subheader("Annotated images")
if filtered.empty:
    st.info("No records match the selected filters.")
else:
    image_names = sorted(filtered["image"].dropna().unique())
    chosen_image = st.selectbox("Choose an image", image_names)

    image_path = OUTPUT_DIR / f"{Path(chosen_image).stem}_hybrid.jpg"
    if image_path.exists():
        st.image(
            str(image_path),
            caption=f"Hybrid prediction: {chosen_image}",
            width="stretch",
        )
    else:
        st.info(f"Annotated image not found: {image_path.name}")

st.subheader("GPS / map")
if {"latitude", "longitude"}.issubset(filtered.columns):
    gps = filtered.dropna(subset=["latitude", "longitude"])
    if not gps.empty:
        st.map(
            gps.rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )[["lat", "lon"]]
        )
    else:
        st.info(
            "No GPS coordinates are recorded yet. "
            "Map display will activate when coordinates are available."
        )
else:
    st.info("GPS fields are not available in the database.")
