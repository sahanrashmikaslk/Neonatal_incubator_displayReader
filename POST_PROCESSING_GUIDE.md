# Post-Processing Module Integration Guide

This guide explains how to integrate the post-processing validation logic into your Streamlit app.

## 📁 Files Created

1. **`incubator_pipeline/postprocessing.py`** - Reusable post-processing module
2. **`streamlit_app/app_example.py`** - Example Streamlit app with live video support

## 🚀 Quick Start

### 1. Import the Module in Your Streamlit App

```python
from incubator_pipeline.postprocessing import (
    apply_postprocessing,
    format_display_value,
    get_validation_status_emoji,
    VALUE_RANGES,
    NUMERIC_CLASSES
)
```

### 2. Initialize Previous Values in Session State

```python
import streamlit as st

# Initialize session state for temporal smoothing
if 'previous_valid' not in st.session_state:
    st.session_state.previous_valid = {}
```

### 3. Process Each Frame/Image

#### For Single Images (no temporal smoothing):

```python
# Get raw OCR predictions
raw_predictions = run_ocr_on_detection(image, detector)

# Apply post-processing WITHOUT previous value smoothing
result = apply_postprocessing(
    raw_predictions,
    min_confidence=0.5,
    previous_valid=None,
    use_previous_on_invalid=False  # Important: False for single images
)

corrected = result['corrected_predictions']
validation_log = result['validation_log']
```

#### For Live Video Streams (with temporal smoothing):

```python
# Get raw OCR predictions
raw_predictions = run_ocr_on_detection(frame, detector)

# Apply post-processing WITH previous value smoothing
result = apply_postprocessing(
    raw_predictions,
    min_confidence=0.5,
    previous_valid=st.session_state.previous_valid,  # Pass previous state
    use_previous_on_invalid=True  # Important: True for live video
)

corrected = result['corrected_predictions']
validation_log = result['validation_log']

# Update stored previous values for next frame
st.session_state.previous_valid = result['previous_valid']
```

### 4. Display Results

```python
for class_name in NUMERIC_CLASSES:
    config = VALUE_RANGES[class_name]

    if class_name in corrected:
        corr_data = corrected[class_name]

        # Format value with unit
        display_value = format_display_value(class_name, corr_data)
        st.write(f"{config['description']}: {display_value}")

        # Show source
        source = corr_data.get('source')
        if source == 'previous_valid':
            st.caption("Using previous valid value")
        elif source == 'ocr_corrected':
            st.caption(f"Corrected from: {corr_data['raw_value']}")
    else:
        st.write(f"{config['description']}: N/A")
```

## 🎯 Key Features

### Medical Range Validation

The module validates readings against realistic medical ranges:

- **Heart Rate**: 60-220 bpm (integer only)
- **Humidity**: 30-95% (integer only)
- **Skin Temperature**: 32.0-39.0°C (1 decimal place)
- **SpO2**: 70-100% (integer only)

### Automatic Corrections

- **Decimal correction**: Fixes missing decimals (e.g., `356` → `35.6`)
- **Integer enforcement**: Removes decimals from integer-only values (e.g., `145.5` → `145`)
- **Confidence filtering**: Rejects low-confidence OCR readings

### Temporal Smoothing (Live Video Only)

- Holds previous valid values when current reading is invalid
- Prevents flickering/jumping values in live streams
- Can be enabled/disabled via `use_previous_on_invalid` parameter

## 📊 Understanding the Output

### Corrected Predictions

```python
corrected_predictions = {
    'heart_rate_value': {
        'value': '145',              # Corrected value
        'raw_value': '145.5',        # Original OCR value
        'source': 'ocr_corrected',   # or 'ocr_direct' or 'previous_valid'
        'det_conf': 0.92,            # YOLO detection confidence
        'ocr_conf': 0.88,            # OCR confidence
        'bbox': [x1, y1, x2, y2]     # Bounding box
    }
}
```

### Validation Log

```python
validation_log = {
    'heart_rate_value': {
        'status': 'valid',           # or 'invalid' or 'not_detected'
        'issues': ['Removed decimal from integer value: 145.5 → 145'],
        'used_previous': False       # True if previous value was used
    }
}
```

### Source Types

- **`ocr_direct`**: Raw OCR value was valid and used directly
- **`ocr_corrected`**: OCR value was corrected (decimal fix, etc.)
- **`previous_valid`**: Current reading invalid, using previous valid value (live video only)

## 🔧 Configuration Options

### Adjust Confidence Thresholds

```python
result = apply_postprocessing(
    predictions,
    min_confidence=0.7,  # Increase for stricter validation
    use_previous_on_invalid=True
)
```

### Customize Medical Ranges

Edit `VALUE_RANGES` in `postprocessing.py`:

```python
VALUE_RANGES = {
    'heart_rate_value': {
        'min': 50,           # Adjust minimum
        'max': 250,          # Adjust maximum
        'decimals': 0,       # Expected decimal places
        'unit': 'bpm',
        'description': 'Heart Rate',
        'integer_only': True  # Enforce integer
    },
    # ... other parameters
}
```

## 🎬 Live Video Integration

For real-time video streams using `streamlit-webrtc`:

```python
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class IncubatorProcessor(VideoProcessorBase):
    def __init__(self):
        self.previous_valid = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run detection + OCR
        raw_predictions = run_ocr_on_detection(img, detector, reader)

        # Apply post-processing with smoothing
        result = apply_postprocessing(
            raw_predictions,
            min_confidence=0.5,
            previous_valid=self.previous_valid,
            use_previous_on_invalid=True  # Enable smoothing for video
        )

        # Update state
        self.previous_valid = result['previous_valid']

        # Draw annotations on frame
        annotated = draw_predictions(img, result['corrected_predictions'])

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="incubator",
    video_processor_factory=IncubatorProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
```

## ⚠️ Important Notes

1. **Single Images**: Set `use_previous_on_invalid=False` to avoid showing stale data
2. **Live Video**: Set `use_previous_on_invalid=True` for temporal stability
3. **Session State**: Always store `previous_valid` in `st.session_state` for persistence
4. **Reset Option**: Provide a button to reset `previous_valid` when needed

## 🧪 Testing

Test the module in your notebook:

```python
from incubator_pipeline.postprocessing import apply_postprocessing

test_predictions = {
    'heart_rate_value': {'value': '145.5', 'ocr_conf': 0.9, 'det_conf': 0.92}
}

result = apply_postprocessing(test_predictions, min_confidence=0.5)
print(result['corrected_predictions'])
# Output: {'heart_rate_value': {'value': '145', ...}}
```

## 📝 Migration Checklist

- [ ] Copy `incubator_pipeline/postprocessing.py` to your project
- [ ] Import functions in your Streamlit app
- [ ] Initialize `st.session_state.previous_valid = {}`
- [ ] Update frame processing to use `apply_postprocessing()`
- [ ] Set `use_previous_on_invalid=True` for live video
- [ ] Set `use_previous_on_invalid=False` for single images
- [ ] Add UI controls for confidence thresholds
- [ ] Add "Reset Previous Values" button
- [ ] Test with sample images/videos
- [ ] Deploy and monitor

## 🆘 Troubleshooting

**Problem**: Values keep showing as "N/A"

- Check OCR confidence threshold (try lowering to 0.3)
- Verify detection confidence is appropriate (try 0.20)
- Check if readings are within valid ranges

**Problem**: Values flickering in live video

- Enable temporal smoothing: `use_previous_on_invalid=True`
- Ensure `previous_valid` is persisted in session state

**Problem**: Stale values in single image analysis

- Disable temporal smoothing: `use_previous_on_invalid=False`
- Don't pass `previous_valid` parameter

## 📚 Next Steps

1. Review `app_example.py` for complete implementation
2. Customize `VALUE_RANGES` for your specific use case
3. Add alerting when values go out of range
4. Implement data logging for historical analysis
5. Add export functionality for reports

---

Need help? Check the example implementation in `streamlit_app/app_example.py`
