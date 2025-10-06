import io
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

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
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.15, 0.05)

reader = load_reader(weights_path)
reader.conf_threshold = conf_threshold

# Video processor class for webcam
class IncubatorVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.reader = None
        self.conf_threshold = 0.25
        self.latest_readings = {}
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.reader is not None:
            try:
                # Run detection and OCR
                self.latest_readings = self.reader.read(img, conf=self.conf_threshold)
                # Annotate the frame
                annotated = self.reader.annotate_image(img, conf=self.conf_threshold)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as e:
                print(f"Error processing frame: {e}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Neonatal Incubator Display Reader")

st.markdown(
    "Upload incubator monitor photos or use live webcam to detect regions, extract vitals via OCR, and log telemetry for review."
)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Upload Images", "🎥 Live Webcam", "📁 Batch Processing"])

session_records = st.session_state.setdefault("records", [])

def process_image(image_array: np.ndarray, label: str) -> Dict[str, object]:
    readings = reader.read(image_array, conf=conf_threshold)
    annotated = reader.annotate_image(image_array, conf=conf_threshold)
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

# Tab 1: Upload Images
with tab1:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        uploaded = st.file_uploader("Upload photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded:
            st.subheader("Uploaded images")
            for file in uploaded:
                image_array = load_image(file)
                process_image(image_array, file.name)
    
    with col_right:
        st.subheader("Telemetry log")
        if st.session_state["records"]:
            df = pd.DataFrame(st.session_state["records"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No readings yet. Upload an image to get started.")

# Tab 2: Live Webcam
with tab2:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Live Webcam Detection")
        st.markdown("**Instructions:** Click 'START' to begin webcam detection. The app will detect and extract readings in real-time.")
        
        # Initialize video processor
        ctx = webrtc_streamer(
            key="incubator-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=IncubatorVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.reader = reader
            ctx.video_processor.conf_threshold = conf_threshold
        
        # Capture frame button
        capture_frame = st.button("📸 Capture Current Frame", type="primary")
        
        if capture_frame and ctx.video_processor:
            if ctx.video_processor.latest_readings:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                data = {
                    "image": f"webcam_capture_{timestamp}",
                }
                for key, reading in ctx.video_processor.latest_readings.items():
                    data[key] = reading.value
                    data[f"{key}_det_conf"] = reading.detection_confidence
                    data[f"{key}_ocr_conf"] = reading.ocr_confidence
                session_records.append(data)
                st.success(f"✅ Frame captured at {timestamp}")
            else:
                st.warning("No readings detected in current frame")
    
    with col_right:
        st.subheader("Live Readings")
        
        if ctx.video_processor and ctx.video_processor.latest_readings:
            readings_df = []
            for key, reading in ctx.video_processor.latest_readings.items():
                readings_df.append({
                    "Parameter": key.replace("_", " ").title(),
                    "Value": reading.value if reading.value else "N/A",
                    "Detection Conf": f"{reading.detection_confidence:.2f}",
                    "OCR Conf": f"{reading.ocr_confidence:.2f}" if reading.ocr_confidence else "N/A"
                })
            
            if readings_df:
                st.dataframe(pd.DataFrame(readings_df), use_container_width=True, hide_index=True)
        else:
            st.info("Start webcam to see live readings")
        
        st.subheader("Captured Frames Log")
        if st.session_state["records"]:
            webcam_records = [r for r in st.session_state["records"] if "webcam_capture" in r.get("image", "")]
            if webcam_records:
                df = pd.DataFrame(webcam_records)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No frames captured yet")
        else:
            st.info("No frames captured yet")

# Tab 3: Batch Processing
with tab3:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        dataset_dir = st.text_input("Directory of images for batch processing", "dataset")
        run_batch = st.button("Process directory", type="secondary")
        
        if run_batch:
            path = Path(dataset_dir)
            if not path.exists():
                st.error(f"Directory not found: {path}")
            else:
                st.subheader(f"Batch processing: {path}")
                progress_bar = st.progress(0)
                image_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")))
                
                for idx, image_path in enumerate(image_files):
                    process_image(load_image(image_path), image_path.name)
                    progress_bar.progress((idx + 1) / len(image_files))
                
                st.success(f"✅ Processed {len(image_files)} images")
    
    with col_right:
        st.subheader("Batch Results Summary")
        if st.session_state["records"]:
            df = pd.DataFrame(st.session_state["records"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Process a directory to see results")

# Overall telemetry section (below tabs)
st.divider()
st.header("📊 All Telemetry Data")

if st.session_state["records"]:
    df = pd.DataFrame(st.session_state["records"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.metric("Total Records", len(df))
        
        # Download button
        export_csv = st.download_button(
            label="📥 Download All Data (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="incubator_readings.csv",
            mime="text/csv",
        )
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state["records"] = []
            st.rerun()
    
    # Confidence distributions
    numeric_cols = [col for col in df.columns if col.endswith("_ocr_conf") or col.endswith("_det_conf")]
    if numeric_cols:
        st.markdown("### Confidence Distributions")
        melted = df.melt(value_vars=numeric_cols, var_name="metric", value_name="confidence")
        fig = px.box(melted, x="metric", y="confidence", points="all")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No readings yet. Upload an image, use webcam, or process a directory to get started.")

st.sidebar.divider()
st.sidebar.write("💡 **Tip:** Annotate at least 200 varied images for robust detector training. Use Label Studio for bounding boxes.")
st.sidebar.write("📹 **Webcam:** Requires HTTPS or localhost. If webcam doesn't work, use ngrok or run locally.")
