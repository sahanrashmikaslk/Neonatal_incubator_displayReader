import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from incubator_pipeline import DEFAULT_WEIGHTS_PATH, IncubatorDisplayReader

# Import postprocessing for validation
try:
    from incubator_pipeline.postprocessing import (
        VALUE_RANGES,
        apply_postprocessing,
        format_display_value,
        get_validation_status_emoji
    )
    POSTPROCESSING_AVAILABLE = True
except ImportError:
    POSTPROCESSING_AVAILABLE = False
    VALUE_RANGES = {}
    def apply_postprocessing(readings, use_previous=False, previous_valid=None):
        return readings, {}
    def format_display_value(param, value):
        return str(value) if value else "N/A"
    def get_validation_status_emoji(status):
        return ""

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
ocr_conf_threshold = st.sidebar.slider("OCR confidence filter", 0.0, 0.9, 0.3, 0.05, 
                                        help="Minimum OCR confidence for validation")

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

# Session state for validation tracking
if 'validated_records' not in st.session_state:
    st.session_state.validated_records = []
if 'previous_valid' not in st.session_state:
    st.session_state.previous_valid = {}
if 'latest_validated' not in st.session_state:
    st.session_state.latest_validated = {}

def validate_and_format_readings(readings, ocr_threshold, use_temporal_smoothing=True):
    """Process readings through validation and return formatted results."""
    if not POSTPROCESSING_AVAILABLE:
        return readings, {}
    
    # Convert readings to dict format for postprocessing
    readings_dict = {}
    for key, reading in readings.items():
        if reading.ocr_confidence and reading.ocr_confidence >= ocr_threshold:
            readings_dict[key] = {
                'value': reading.value,
                'detection_confidence': reading.detection_confidence,
                'ocr_confidence': reading.ocr_confidence
            }
    
    # Apply validation with temporal smoothing if enabled
    validated, validation_log = apply_postprocessing(
        readings_dict,
        use_previous_on_invalid=use_temporal_smoothing,
        previous_valid_readings=st.session_state.previous_valid if use_temporal_smoothing else None
    )
    
    # Update previous valid for next frame
    if use_temporal_smoothing:
        for key, val_data in validated.items():
            if val_data.get('status') == 'valid':
                st.session_state.previous_valid[key] = val_data
    
    return validated, validation_log

def process_image(image_array: np.ndarray, label: str) -> Dict[str, object]:
    readings = reader.read(image_array, conf=conf_threshold)
    annotated = reader.annotate_image(image_array, conf=conf_threshold)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=f"{label} (annotated)", use_container_width=True)
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
                            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                            
                            # Store current frame for capture
                            st.session_state.current_frame = frame.copy()
                        except Exception as e:
                            st.error(f"Error processing frame: {e}")
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                    
                    # Check if stop button was pressed
                    if not st.session_state.webcam_active:
                        break
                    
                    time.sleep(frame_delay / 1000.0)  # Control frame rate
                
                cap.release()
        
        # Capture button (outside the loop)
        if 'current_frame' in st.session_state:
            if capture_button_placeholder.button("📸 Capture Current Frame", type="primary", use_container_width=True):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                label = f"webcam_capture_{time.strftime('%Y%m%d_%H%M%S')}"
                
                try:
                    # Use already validated readings from session state
                    validated = st.session_state.latest_validated
                    
                    if validated:
                        data = {
                            "timestamp": timestamp,
                            "image": label,
                        }
                        
                        for key, val_data in validated.items():
                            value = val_data.get('value', 'N/A')
                            data[key] = value
                            data[f"{key}_status"] = val_data.get('status', 'unknown')
                            data[f"{key}_det_conf"] = val_data.get('detection_confidence', 0)
                            data[f"{key}_ocr_conf"] = val_data.get('ocr_confidence', 0)
                        
                        st.session_state.validated_records.append(data)
                        session_records.append(data)
                        st.success(f"✅ Validated frame captured at {timestamp}")
                    else:
                        st.warning("No validated readings to capture")
                except Exception as e:
                    st.error(f"Error capturing frame: {e}")
    
    with col_right:
        st.subheader("Live Validated Readings")
        
        if 'current_frame' in st.session_state and st.session_state.webcam_active:
            try:
                readings = reader.read(st.session_state.current_frame, conf=conf_threshold)
                
                # Apply validation
                validated, validation_log = validate_and_format_readings(
                    readings, 
                    ocr_conf_threshold,
                    use_temporal_smoothing=True
                )
                
                # Store latest validated readings
                st.session_state.latest_validated = validated
                
                # Create validated readings table
                if validated:
                    readings_df = []
                    for key, val_data in validated.items():
                        status = val_data.get('status', 'unknown')
                        emoji = get_validation_status_emoji(status)
                        value = val_data.get('value', 'N/A')
                        formatted_value = format_display_value(key, value) if value != 'N/A' else 'N/A'
                        
                        readings_df.append({
                            "Status": emoji,
                            "Parameter": key.replace("_", " ").title(),
                            "Value": formatted_value,
                            "Det Conf": f"{val_data.get('detection_confidence', 0):.2f}",
                            "OCR Conf": f"{val_data.get('ocr_confidence', 0):.2f}"
                        })
                    
                    df = pd.DataFrame(readings_df)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Show validation details in expander
                    if validation_log:
                        with st.expander("📋 Validation Details"):
                            for param, log in validation_log.items():
                                st.text(f"{param}: {log}")
                else:
                    st.info("No validated readings (check OCR confidence threshold)")
            except Exception as e:
                st.info(f"Waiting for frames... {str(e) if st.session_state.get('debug') else ''}")
        else:
            st.info("Start webcam to see live validated readings")
        
        st.subheader("📊 Captured Validated Data")
        
        if st.session_state.validated_records:
            validated_df = pd.DataFrame(st.session_state.validated_records)
            st.dataframe(validated_df, use_container_width=True, height=300)
            
            # Excel Download button
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    validated_df.to_excel(writer, index=False, sheet_name='Validated Readings')
                    
                    # Add summary sheet
                    summary_data = []
                    for col in validated_df.columns:
                        if col.endswith('_status'):
                            param = col.replace('_status', '')
                            if param in validated_df.columns:
                                valid_count = (validated_df[col] == 'valid').sum()
                                total_count = validated_df[col].notna().sum()
                                summary_data.append({
                                    'Parameter': param,
                                    'Valid': valid_count,
                                    'Total': total_count,
                                    'Valid %': f"{(valid_count/total_count*100):.1f}%" if total_count > 0 else "N/A"
                                })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"incubator_validated_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating Excel: {e}")
        else:
            st.info("No validated frames captured yet. Click 'Capture Current Frame' to start logging data.")

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
        
        # CSV Download button
        export_csv = st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"incubator_readings_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # Excel Download button
        if st.session_state.validated_records:
            try:
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    validated_df = pd.DataFrame(st.session_state.validated_records)
                    validated_df.to_excel(writer, index=False, sheet_name='Validated Readings')
                    
                    # Add a summary sheet if available
                    if not validated_df.empty:
                        summary_data = []
                        for col in validated_df.columns:
                            if col.endswith('_status'):
                                param = col.replace('_status', '')
                                if param in validated_df.columns:
                                    valid_count = (validated_df[col] == 'valid').sum()
                                    total_count = validated_df[col].notna().sum()
                                    summary_data.append({
                                        'Parameter': param,
                                        'Valid': valid_count,
                                        'Total': total_count,
                                        'Valid %': f"{(valid_count/total_count*100):.1f}%" if total_count > 0 else "N/A"
                                    })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"incubator_validated_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Error creating Excel file: {e}")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state["records"] = []
            st.session_state["validated_records"] = []
            st.session_state["previous_valid"] = {}
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
