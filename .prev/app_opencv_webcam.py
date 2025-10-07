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


def get_available_cameras(max_cameras=10):
    """Detect available cameras on the system."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


st.sidebar.header("Model configuration")
weights_path = st.sidebar.text_input(
    "YOLO weights path",
    value=str(Path(DEFAULT_WEIGHTS_PATH)),
)
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.header("📹 Camera Settings")

# Camera detection
with st.sidebar:
    if st.button("🔍 Scan for Cameras", use_container_width=True):
        st.rerun()

available_cameras = get_available_cameras()
if available_cameras:
    st.sidebar.success(f"✅ Found {len(available_cameras)} camera(s)")
    
    # Create descriptive labels for cameras
    camera_options = []
    camera_descriptions = {}
    for idx in available_cameras:
        # Try to get camera info
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            desc = f"Camera {idx} ({width}x{height})"
            camera_options.append(idx)
            camera_descriptions[idx] = desc
            cap.release()
        else:
            camera_options.append(idx)
            camera_descriptions[idx] = f"Camera {idx}"
    
    selected_camera = st.sidebar.selectbox(
        "Select Camera",
        options=camera_options,
        format_func=lambda x: camera_descriptions[x],
        key="camera_selector"
    )
else:
    st.sidebar.warning("⚠️ No cameras detected")
    st.sidebar.info("💡 Click 'Scan for Cameras' to refresh or check your camera connections.")
    selected_camera = 0

# Camera resolution settings
st.sidebar.subheader("Resolution")
resolution_options = {
    "640x480 (VGA)": (640, 480),
    "1280x720 (HD)": (1280, 720),
    "1920x1080 (Full HD)": (1920, 1080),
    "Auto": None
}
selected_resolution = st.sidebar.selectbox(
    "Camera Resolution",
    options=list(resolution_options.keys()),
    index=0
)

# Frame rate control
st.sidebar.subheader("Performance")
frame_delay = st.sidebar.slider(
    "Frame Delay (ms)",
    min_value=50,
    max_value=500,
    value=100,
    step=50,
    help="Higher values reduce processing load but make video less smooth"
)

reader = load_reader(weights_path)
reader.conf_threshold = conf_threshold

st.title("Neonatal Incubator Display Reader")

st.markdown(
    "Upload incubator monitor photos or use live webcam to detect regions, extract vitals via OCR, and log telemetry for review."
)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Upload Images", "🎥 Live Webcam (OpenCV)", "📁 Batch Processing"])

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

# Tab 2: Live Webcam (OpenCV)
with tab2:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Live Webcam Detection (OpenCV)")
        st.markdown("""
        **Instructions:**
        1. Click 'Start Webcam' to enable your camera
        2. Position the incubator display in front of the camera
        3. Click 'Capture Frame' to save a reading
        4. Click 'Stop Webcam' when done
        """)
        
        # Initialize session state for webcam
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            start_webcam = st.button("▶️ Start Webcam", type="primary", use_container_width=True)
        
        with col_btn2:
            stop_webcam = st.button("⏹️ Stop Webcam", type="secondary", use_container_width=True)
        
        if start_webcam:
            st.session_state.webcam_active = True
        
        if stop_webcam:
            st.session_state.webcam_active = False
        
        # Display selected camera info
        st.info(f"📹 Selected: Camera {selected_camera} | Resolution: {selected_resolution}")
        
        # Webcam feed
        frame_placeholder = st.empty()
        capture_button_placeholder = st.empty()
        
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(selected_camera)
            
            # Set resolution if specified
            if resolution_options[selected_resolution] is not None:
                width, height = resolution_options[selected_resolution]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not cap.isOpened():
                st.error(f"❌ Cannot access Camera {selected_camera}. Please check your camera permissions or try a different camera.")
                st.session_state.webcam_active = False
            else:
                # Get actual resolution
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                st.success(f"✅ Camera {selected_camera} active at {actual_width}x{actual_height}. Adjust position and click 'Capture Frame' to save readings.")
                
                # Continuous frame update
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Run detection on frame
                        try:
                            annotated = reader.annotate_image(frame, conf=conf_threshold)
                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", use_column_width=True)
                            
                            # Store current frame for capture
                            st.session_state.current_frame = frame.copy()
                        except Exception as e:
                            st.error(f"Error processing frame: {e}")
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", use_column_width=True)
                    
                    # Check if stop button was pressed
                    if not st.session_state.webcam_active:
                        break
                    
                    time.sleep(frame_delay / 1000.0)  # Control frame rate
                
                cap.release()
        
        # Capture button (outside the loop)
        if 'current_frame' in st.session_state:
            if capture_button_placeholder.button("📸 Capture Current Frame", type="primary", use_container_width=True):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                label = f"webcam_capture_{timestamp}"
                
                try:
                    readings = reader.read(st.session_state.current_frame, conf=conf_threshold)
                    
                    data = {
                        "image": label,
                    }
                    for key, reading in readings.items():
                        data[key] = reading.value
                        data[f"{key}_det_conf"] = reading.detection_confidence
                        data[f"{key}_ocr_conf"] = reading.ocr_confidence
                    
                    session_records.append(data)
                    st.success(f"✅ Frame captured at {timestamp}")
                except Exception as e:
                    st.error(f"Error capturing frame: {e}")
    
    with col_right:
        st.subheader("Live Readings")
        
        if 'current_frame' in st.session_state and st.session_state.webcam_active:
            try:
                readings = reader.read(st.session_state.current_frame, conf=conf_threshold)
                
                readings_df = []
                for key, reading in readings.items():
                    readings_df.append({
                        "Parameter": key.replace("_", " ").title(),
                        "Value": reading.value if reading.value else "N/A",
                        "Det Conf": f"{reading.detection_confidence:.2f}",
                        "OCR Conf": f"{reading.ocr_confidence:.2f}" if reading.ocr_confidence else "N/A"
                    })
                
                if readings_df:
                    st.dataframe(pd.DataFrame(readings_df), use_container_width=True, hide_index=True)
                else:
                    st.info("No detections in current frame")
            except:
                st.info("Waiting for frames...")
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
st.sidebar.subheader("ℹ️ Tips & Info")
st.sidebar.write("💡 **Training:** Annotate at least 200 varied images for robust detector training. Use Label Studio for bounding boxes.")
st.sidebar.write("📹 **Camera:** Uses OpenCV to access your local camera. Make sure camera permissions are enabled.")
st.sidebar.write("🔄 **Switch Camera:** Use the camera selector above if you have multiple cameras (webcam, external USB camera, etc.).")
st.sidebar.write("📐 **Resolution:** Higher resolutions provide better quality but may be slower to process.")
