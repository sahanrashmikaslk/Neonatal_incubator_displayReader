# Incubator Display Reader - Webcam Support

This directory contains enhanced versions of the Streamlit app with live webcam detection support.

## Files

- **`app.py`**: Original version with streamlit-webrtc (for deployment with WebRTC)
- **`app_opencv_webcam.py`**: OpenCV-based webcam version (recommended for local use)
- **`incubator_pipeline.py`**: Core detection and OCR pipeline

## Installation

### Required Packages

```bash
pip install -r requirements.txt
```

### Optional: For WebRTC version

```bash
pip install streamlit-webrtc av
```

## Running the App

### Option 1: OpenCV Webcam (Recommended for Local Use)

```bash
cd streamlit_app
streamlit run app_opencv_webcam.py
```

**Features:**

- ✅ Works locally without WebRTC
- ✅ Simple start/stop webcam controls
- ✅ Real-time detection overlay
- ✅ Capture frames to save readings
- ✅ Live readings display

**Requirements:**

- Local webcam access
- Camera permissions enabled

### Option 2: WebRTC Webcam (For Production Deployment)

```bash
cd streamlit_app
streamlit run app.py
```

**Features:**

- ✅ Works in production with HTTPS
- ✅ Browser-based webcam streaming
- ✅ Better for deployed apps

**Requirements:**

- HTTPS connection (or localhost)
- May require STUN/TURN servers for remote access

## Usage Instructions

### 1. Upload Images Tab

- Upload one or multiple images
- View annotated results with detected values
- Readings are logged automatically

### 2. Live Webcam Tab (OpenCV Version)

1. Click **"Start Webcam"** to enable camera
2. Position incubator display in frame
3. Watch real-time detection and OCR
4. Click **"Capture Frame"** to save current readings
5. Click **"Stop Webcam"** when finished

### 3. Batch Processing Tab

- Enter directory path containing images
- Click **"Process directory"** to run batch detection
- Progress bar shows processing status

### 4. Telemetry Data Section

- View all captured readings in table format
- Download data as CSV
- View confidence distributions
- Clear all data when needed

## Configuration

### Model Settings (Sidebar)

- **YOLO weights path**: Path to trained model (default: `../models/incubator_yolov8n.pt`)
- **Detection confidence**: Threshold for detection (0.1-0.9, default: 0.25)

## Detected Parameters

The system detects and reads:

- Heart Rate Value
- Humidity Value
- Skin Temperature Value
- SpO2 Value

## Troubleshooting

### Webcam Not Working (OpenCV Version)

1. Check camera permissions in system settings
2. Ensure no other app is using the camera
3. Try different camera index (edit code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`)

### Webcam Not Working (WebRTC Version)

1. Ensure you're using HTTPS or localhost
2. Check browser camera permissions
3. Try a different browser (Chrome/Firefox recommended)

### Low Detection Accuracy

1. Adjust confidence threshold in sidebar
2. Ensure good lighting conditions
3. Position display clearly in frame
4. Check that model weights are loaded correctly

### Performance Issues

1. Lower the frame rate by increasing `time.sleep()` value
2. Process every Nth frame instead of all frames
3. Reduce image resolution before detection

## Development Notes

### Adding New Parameters

Edit `incubator_pipeline.py`:

```python
CLASS_NAMES = [
    'heart_rate_value',
    'humidity_value',
    'skin_temp_value',
    'spo2_value',
    # Add new classes here
]
```

### Customizing Detection

Adjust confidence threshold:

```python
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
```

### Frame Processing

Modify the annotation style in `incubator_pipeline.py`:

- Colors: `BOX_COLOR_NUMERIC`, `BOX_COLOR_OTHER`
- Font: `FONT`, `FONT_SCALE`, `FONT_THICKNESS`
- Box thickness: `BOX_THICKNESS`

## License

This project is part of the Neonatal Incubator Display Reader system.
