import io
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from incubator_pipeline import DEFAULT_WEIGHTS_PATH, IncubatorDisplayReader

st.set_page_config(
    page_title="Incubator Display Dashboard",
    layout="wide",
    page_icon="🧑‍⚕️",
)


def load_image(file) -> np.ndarray:
    if isinstance(file, (str, Path)):
        image = cv2.imread(str(file))
        if image is None:
            raise FileNotFoundError(file)
        return image
    data = np.asarray(bytearray(file.read()), dtype="uint8")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image")
    return image


@st.cache_resource(show_spinner=False)
def load_reader(weights_path: str) -> IncubatorDisplayReader:
    return IncubatorDisplayReader(weights_path=weights_path)


st.sidebar.header("Model configuration")
weights_path = st.sidebar.text_input(
    "YOLO weights path",
    value=str(Path(DEFAULT_WEIGHTS_PATH)),
)
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)

reader = load_reader(weights_path)
reader.detector.conf_threshold = conf_threshold

st.title("Neonatal Incubator Display Reader")

st.markdown(
    "Upload incubator monitor photos to detect regions, extract vitals via OCR, and log telemetry for review."
)

col_left, col_right = st.columns([3, 2])

with col_left:
    uploaded = st.file_uploader("Upload photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    dataset_dir = st.text_input("Optional: directory of images for batch processing", "dataset")
    run_batch = st.button("Process directory", type="secondary")

    session_records = st.session_state.setdefault("records", [])

    def process_image(image_array: np.ndarray, label: str) -> Dict[str, object]:
        readings = reader.read(image_array)
        annotated = reader.annotate_image(image_array)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"{label} (annotated)", use_column_width=True)
        data = {
            "image": label,
        }
        for key, reading in readings.items():
            data[key] = reading.value
            data[f"{key}_det_conf"] = reading.detection_confidence
            data[f"{key}_ocr_conf"] = reading.ocr_confidence
        session_records.append(data)
        return data

    if uploaded:
        st.subheader("Uploaded images")
        for file in uploaded:
            image_array = load_image(file)
            process_image(image_array, file.name)

    if run_batch:
        path = Path(dataset_dir)
        if not path.exists():
            st.error(f"Directory not found: {path}")
        else:
            st.subheader(f"Batch processing: {path}")
            for image_path in sorted(path.glob("*.jpg")):
                process_image(load_image(image_path), image_path.name)

with col_right:
    st.subheader("Telemetry log")
    if st.session_state["records"]:
        df = pd.DataFrame(st.session_state["records"])
        st.dataframe(df, use_container_width=True)
        numeric_cols = [col for col in df.columns if col.endswith("_ocr_conf") or col.endswith("_det_conf")]
        if numeric_cols:
            st.markdown("### Confidence distributions")
            melted = df.melt(value_vars=numeric_cols, var_name="metric", value_name="confidence")
            fig = px.box(melted, x="metric", y="confidence", points="all")
            st.plotly_chart(fig, use_container_width=True)

        export_csv = st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="incubator_readings.csv",
            mime="text/csv",
        )
    else:
        st.info("No readings yet. Upload an image to get started.")

st.sidebar.divider()
st.sidebar.write("Tip: annotate at least 200 varied images for robust detector training. Use Label Studio for bounding boxes.")
