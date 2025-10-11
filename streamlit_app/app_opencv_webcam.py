import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

# Suppress OpenCV camera errors
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Configure Tesseract OCR
from tesseract_config import configure_tesseract
TESSERACT_AVAILABLE = configure_tesseract()

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

# Display Tesseract status warning if not available
if not TESSERACT_AVAILABLE:
    st.error("⚠️ **Tesseract OCR not found!** Please install Tesseract OCR to use this application. "
             "Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    st.stop()


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
def load_reader(weights_path: str, fast_mode: bool = True, use_half_precision: bool = False, 
                cache_preprocessed: bool = True) -> IncubatorDisplayReader:
    return IncubatorDisplayReader(
        weights_path=weights_path, 
        fast_mode=fast_mode,
        use_half_precision=use_half_precision,
        cache_preprocessed=cache_preprocessed
    )


def get_available_cameras(max_cameras=10):
    """Detect available cameras on the system."""
    available_cameras = []
    for i in range(max_cameras):
        try:
            # Use CAP_DSHOW on Windows to avoid obsensor errors
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Verify the camera actually works
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
            else:
                # Stop scanning after first unavailable camera
                if i > 0 and len(available_cameras) == 0:
                    break
                elif i > max(available_cameras, default=-1) + 2:
                    # Stop if we've gone 2 indices past the last working camera
                    break
        except Exception:
            # Silently skip problematic indices
            continue
    
    return available_cameras


st.sidebar.header("Model configuration")
weights_path = st.sidebar.text_input(
    "YOLO weights path",
    value=str(Path(DEFAULT_WEIGHTS_PATH)),
)
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)

# Validation settings
st.sidebar.subheader("✅ Validation Settings")
ocr_conf_threshold = st.sidebar.slider(
    "OCR confidence filter", 
    0.0, 0.9, 0.3, 0.05,
    help="Minimum OCR confidence for validation. Readings below this threshold are marked as invalid."
)

# Show expected ranges
if POSTPROCESSING_AVAILABLE and VALUE_RANGES:
    with st.sidebar.expander("📊 Expected Value Ranges"):
        for param, range_info in VALUE_RANGES.items():
            if 'min' in range_info and 'max' in range_info:
                st.text(f"{param}: {range_info['min']} - {range_info['max']}")

st.sidebar.divider()

# OCR Mode selection
ocr_mode = st.sidebar.radio(
    "🔍 OCR Mode",
    ["Fast (Real-time)", "Accurate (Batch)"],
    index=0,
    help="Fast: 1 preprocessing strategy, best for live webcam. Accurate: 15 strategies, best for upload/batch processing."
)
fast_mode = (ocr_mode == "Fast (Real-time)")

# Advanced optimizations
with st.sidebar.expander("⚙️ Advanced Optimizations"):
    enable_caching = st.checkbox(
        "Enable ROI Caching",
        value=True,
        help="Cache preprocessed ROIs to avoid redundant processing. Speeds up when display values don't change."
    )
    use_half_precision = st.checkbox(
        "Use Half Precision (FP16)",
        value=False,
        help="Use FP16 for faster YOLO inference. Requires CUDA-capable GPU."
    )

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
        # Try to get camera info using DirectShow
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
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

# Performance settings
st.sidebar.subheader("⚡ Performance Settings")

# Frame skipping for speed
frame_skip = st.sidebar.slider(
    "Process Every N Frames",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
    help="Process every Nth frame. Higher = faster but less responsive. 1=process all frames."
)

# Frame delay
frame_delay = st.sidebar.slider(
    "Frame Delay (ms)",
    min_value=0,
    max_value=500,
    value=50,
    step=50,
    help="Delay between frames. Lower = faster but more CPU usage."
)

# Image resize for faster processing
process_resolution = st.sidebar.select_slider(
    "Processing Resolution",
    options=["Full", "75%", "50%", "25%"],
    value="75%",
    help="Reduce image size before processing for speed. Full = slowest but most accurate."
)

reader = load_reader(weights_path, fast_mode, use_half_precision, enable_caching)
reader.conf_threshold = conf_threshold

st.title("Neonatal Incubator Display Reader")

# Display OCR mode indicator
if fast_mode:
    st.info("🚀 **Fast Mode Active** - Using optimized single-strategy OCR for real-time performance (15x faster)")
else:
    st.info("🎯 **Accurate Mode Active** - Using multi-strategy OCR with 15 combinations for maximum accuracy")

st.markdown(
    "Upload incubator monitor photos or use live webcam to detect regions, extract vitals via OCR, and log telemetry for review."
)

# Validation status legend
if POSTPROCESSING_AVAILABLE:
    with st.expander("📖 Validation Status Legend"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✅ Valid** - Reading is within expected range")
        with col2:
            st.markdown("**🔄 Corrected** - Invalid reading replaced with previous valid value")
        with col3:
            st.markdown("**❌ Invalid** - Reading is out of range or failed validation")

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
        
        # Temporal smoothing option
        use_temporal_smoothing_upload = st.checkbox(
            "Use Temporal Smoothing",
            value=True,
            help="Use previous valid readings when current reading is invalid",
            key="temporal_upload"
        )
        
        if uploaded:
            st.subheader("Uploaded images")
            for file in uploaded:
                image_array = load_image(file)
                
                # Get readings
                readings = reader.read(image_array, conf=conf_threshold)
                annotated = reader.annotate_image(image_array, conf=conf_threshold)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"{file.name} (annotated)", use_container_width=True)
                
                # Apply validation
                validated, validation_log = validate_and_format_readings(
                    readings,
                    ocr_conf_threshold,
                    use_temporal_smoothing=use_temporal_smoothing_upload
                )
                
                # Display validation results
                if validated:
                    st.markdown("**Validated Readings:**")
                    val_cols = st.columns(len(validated))
                    for idx, (key, val_data) in enumerate(validated.items()):
                        with val_cols[idx]:
                            status = val_data.get('status', 'unknown')
                            emoji = get_validation_status_emoji(status)
                            value = val_data.get('value', 'N/A')
                            formatted_value = format_display_value(key, value) if value != 'N/A' else 'N/A'
                            st.metric(
                                f"{emoji} {key.replace('_', ' ').title()}",
                                formatted_value,
                                delta=f"Conf: {val_data.get('ocr_confidence', 0):.2f}"
                            )
                    
                    # Show validation log
                    if validation_log:
                        with st.expander("📋 Validation Details"):
                            for param, log in validation_log.items():
                                st.text(f"{param}: {log}")
                
                # Store validated data
                data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": file.name,
                }
                
                for key, val_data in validated.items():
                    value = val_data.get('value', 'N/A')
                    data[key] = value
                    data[f"{key}_status"] = val_data.get('status', 'unknown')
                    data[f"{key}_det_conf"] = val_data.get('detection_confidence', 0)
                    data[f"{key}_ocr_conf"] = val_data.get('ocr_confidence', 0)
                
                st.session_state.validated_records.append(data)
                session_records.append(data)
                
                st.divider()
    
    with col_right:
        st.subheader("📊 Validated Readings Log")
        if st.session_state.validated_records:
            validated_df = pd.DataFrame(st.session_state.validated_records)
            st.dataframe(validated_df, use_container_width=True, height=400)
            
            # Summary statistics
            st.markdown("**Validation Summary:**")
            for col in validated_df.columns:
                if col.endswith('_status'):
                    param = col.replace('_status', '')
                    if param in validated_df.columns:
                        valid_count = (validated_df[col] == 'valid').sum()
                        total_count = validated_df[col].notna().sum()
                        if total_count > 0:
                            st.metric(
                                f"{param.replace('_', ' ').title()}",
                                f"{valid_count}/{total_count}",
                                delta=f"{(valid_count/total_count*100):.1f}% valid"
                            )
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
            # Use DirectShow backend on Windows to avoid obsensor errors
            cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
            
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
                
                # Initialize frame counter and FPS tracking
                frame_counter = 0
                fps_counter = 0
                fps_start_time = time.time()
                fps_display = st.empty()
                
                # Continuous frame update
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Apply frame skipping
                        frame_counter += 1
                        should_process = (frame_counter % frame_skip == 0)
                        
                        # Resize frame for faster processing if needed
                        process_frame = frame.copy()
                        if process_resolution == "75%":
                            process_frame = cv2.resize(process_frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
                        elif process_resolution == "50%":
                            process_frame = cv2.resize(process_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                        elif process_resolution == "25%":
                            process_frame = cv2.resize(process_frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                        
                        # Only process every Nth frame
                        if should_process:
                            try:
                                annotated = reader.annotate_image(process_frame, conf=conf_threshold)
                                
                                # Resize back to original size for display if needed
                                if process_resolution != "Full":
                                    annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]), 
                                                          interpolation=cv2.INTER_LINEAR)
                                
                                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                                
                                # Store current frame for capture
                                st.session_state.current_frame = frame.copy()
                            except Exception as e:
                                st.error(f"Error processing frame: {e}")
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                        else:
                            # Show raw frame without processing
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                        
                        # Update FPS counter
                        fps_counter += 1
                        if fps_counter >= 30:  # Update every 30 frames
                            elapsed = time.time() - fps_start_time
                            current_fps = fps_counter / elapsed if elapsed > 0 else 0
                            fps_display.metric("⚡ FPS", f"{current_fps:.1f}")
                            fps_counter = 0
                            fps_start_time = time.time()
                    
                    # Check if stop button was pressed
                    if not st.session_state.webcam_active:
                        break
                    
                    # Control frame rate with delay
                    if frame_delay > 0:
                        time.sleep(frame_delay / 1000.0)
                
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
        
        # Batch processing options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_temporal_smoothing_batch = st.checkbox(
                "Use Temporal Smoothing",
                value=True,
                help="Use previous valid readings when current reading is invalid",
                key="temporal_batch"
            )
        with col_opt2:
            show_annotations = st.checkbox(
                "Show Annotated Images",
                value=False,
                help="Display annotated images during batch processing (slower)"
            )
        
        run_batch = st.button("🚀 Process Directory", type="primary", use_container_width=True)
        
        if run_batch:
            path = Path(dataset_dir)
            if not path.exists():
                st.error(f"Directory not found: {path}")
            else:
                st.subheader(f"Batch processing: {path}")
                progress_bar = st.progress(0)
                status_text = st.empty()
                image_placeholder = st.empty() if show_annotations else None
                
                image_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")))
                
                batch_start_time = time.time()
                
                for idx, image_path in enumerate(image_files):
                    status_text.text(f"Processing {idx + 1}/{len(image_files)}: {image_path.name}")
                    
                    # Load and process image
                    image_array = load_image(image_path)
                    readings = reader.read(image_array, conf=conf_threshold)
                    
                    # Apply validation
                    validated, validation_log = validate_and_format_readings(
                        readings,
                        ocr_conf_threshold,
                        use_temporal_smoothing=use_temporal_smoothing_batch
                    )
                    
                    # Show annotated image if requested
                    if show_annotations and image_placeholder:
                        annotated = reader.annotate_image(image_array, conf=conf_threshold)
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        image_placeholder.image(rgb, caption=f"{image_path.name}", use_container_width=True)
                    
                    # Store validated data
                    data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": image_path.name,
                    }
                    
                    for key, val_data in validated.items():
                        value = val_data.get('value', 'N/A')
                        data[key] = value
                        data[f"{key}_status"] = val_data.get('status', 'unknown')
                        data[f"{key}_det_conf"] = val_data.get('detection_confidence', 0)
                        data[f"{key}_ocr_conf"] = val_data.get('ocr_confidence', 0)
                    
                    st.session_state.validated_records.append(data)
                    session_records.append(data)
                    
                    progress_bar.progress((idx + 1) / len(image_files))
                
                batch_time = time.time() - batch_start_time
                status_text.empty()
                
                st.success(f"✅ Processed {len(image_files)} images in {batch_time:.2f}s ({len(image_files)/batch_time:.2f} images/sec)")
    
    with col_right:
        st.subheader("📊 Batch Validation Summary")
        
        if st.session_state.validated_records:
            validated_df = pd.DataFrame(st.session_state.validated_records)
            
            # Summary statistics
            st.markdown("**Validation Summary:**")
            summary_data = []
            for col in validated_df.columns:
                if col.endswith('_status'):
                    param = col.replace('_status', '')
                    if param in validated_df.columns:
                        valid_count = (validated_df[col] == 'valid').sum()
                        invalid_count = (validated_df[col] == 'invalid').sum()
                        corrected_count = (validated_df[col] == 'corrected').sum()
                        total_count = validated_df[col].notna().sum()
                        
                        if total_count > 0:
                            summary_data.append({
                                'Parameter': param.replace('_', ' ').title(),
                                'Valid': valid_count,
                                'Invalid': invalid_count,
                                'Corrected': corrected_count,
                                'Total': total_count,
                                'Valid %': f"{(valid_count/total_count*100):.1f}%"
                            })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Show detailed results
            with st.expander("📋 Detailed Results"):
                st.dataframe(validated_df, use_container_width=True, height=400)
            
            # Download validated batch results
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    validated_df.to_excel(writer, index=False, sheet_name='Batch Results')
                    
                    if summary_data:
                        summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Batch Results (Excel)",
                    data=excel_data,
                    file_name=f"batch_validated_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating Excel: {e}")
        else:
            st.info("Process a directory to see validation summary")

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
