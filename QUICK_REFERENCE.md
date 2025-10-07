# Post-Processing Module - Quick Reference

## 📦 What's in the Module?

The `incubator_pipeline/postprocessing.py` module contains all validation and correction logic for OCR readings.

## ⚡ Quick Integration (2 minutes)

### 1. Import in Your Streamlit App

```python
from incubator_pipeline.postprocessing import apply_postprocessing, NUMERIC_CLASSES
import streamlit as st

# Initialize session state
if 'previous_valid' not in st.session_state:
    st.session_state.previous_valid = {}
```

### 2. Process Frames

```python
# After running YOLO + OCR to get raw_predictions:

# FOR LIVE VIDEO (with smoothing):
result = apply_postprocessing(
    raw_predictions,
    min_confidence=0.5,
    previous_valid=st.session_state.previous_valid,
    use_previous_on_invalid=True  # ⭐ Enables temporal smoothing
)
st.session_state.previous_valid = result['previous_valid']  # ⭐ Update state

# FOR SINGLE IMAGES (no smoothing):
result = apply_postprocessing(
    raw_predictions,
    min_confidence=0.5,
    use_previous_on_invalid=False  # ⭐ No temporal smoothing
)

# Use the corrected values:
corrected = result['corrected_predictions']
```

### 3. Display Results

```python
for class_name in NUMERIC_CLASSES:
    if class_name in corrected:
        value = corrected[class_name]['value']
        source = corrected[class_name]['source']
        st.write(f"{class_name}: {value} (from {source})")
    else:
        st.write(f"{class_name}: N/A")
```

## 🎯 Key Differences

| Feature                    | Single Image    | Live Video                           |
| -------------------------- | --------------- | ------------------------------------ |
| `use_previous_on_invalid`  | ❌ `False`      | ✅ `True`                            |
| `previous_valid` parameter | ❌ `None`       | ✅ `st.session_state.previous_valid` |
| Update state               | ❌ No           | ✅ Yes                               |
| Invalid values             | Show as N/A     | Use previous valid value             |
| Use case                   | Static analysis | Real-time monitoring                 |

## 📊 What It Does

### Validates & Corrects:

- ✅ Range validation (60-220 bpm for heart rate, etc.)
- ✅ Decimal correction (`356` → `35.6` for temperature)
- ✅ Integer enforcement (`145.5` → `145` for heart rate)
- ✅ Confidence filtering (rejects low confidence readings)

### For Live Video Only:

- ✅ Temporal smoothing (holds previous valid value when current is invalid)
- ✅ Prevents flickering values
- ✅ Maintains reading stability

## 🚨 Common Mistakes

### ❌ DON'T: Use previous values for single images

```python
# WRONG for single image:
result = apply_postprocessing(
    raw_predictions,
    previous_valid=st.session_state.previous_valid,  # ❌
    use_previous_on_invalid=True  # ❌
)
```

### ✅ DO: Use previous values only for live video

```python
# CORRECT for live video:
result = apply_postprocessing(
    raw_predictions,
    previous_valid=st.session_state.previous_valid,  # ✅
    use_previous_on_invalid=True  # ✅
)
st.session_state.previous_valid = result['previous_valid']  # ✅ Update
```

## 🔄 Reset Previous Values

Add a button to reset stored values:

```python
if st.button("Reset Previous Values"):
    st.session_state.previous_valid = {}
    st.success("Reset!")
```

## 📝 Complete Example

```python
import streamlit as st
from incubator_pipeline.postprocessing import (
    apply_postprocessing,
    format_display_value,
    VALUE_RANGES,
    NUMERIC_CLASSES
)

# Initialize
if 'previous_valid' not in st.session_state:
    st.session_state.previous_valid = {}

# Settings
use_smoothing = st.checkbox("Enable Smoothing (Live Video)", value=True)
min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

# Process frame
raw_predictions = run_ocr_on_detection(frame, detector, reader)

result = apply_postprocessing(
    raw_predictions,
    min_confidence=min_conf,
    previous_valid=st.session_state.previous_valid if use_smoothing else None,
    use_previous_on_invalid=use_smoothing
)

if use_smoothing:
    st.session_state.previous_valid = result['previous_valid']

# Display
for class_name in NUMERIC_CLASSES:
    config = VALUE_RANGES[class_name]
    st.subheader(config['description'])

    if class_name in result['corrected_predictions']:
        corr_data = result['corrected_predictions'][class_name]
        value_str = format_display_value(class_name, corr_data)
        st.metric(label="", value=value_str)

        if corr_data['source'] == 'previous_valid':
            st.caption("🔄 Using previous valid value")
    else:
        st.metric(label="", value="N/A")
```

## 📚 Full Documentation

- **Complete guide**: `POST_PROCESSING_GUIDE.md`
- **Example app**: `streamlit_app/app_example.py`
- **Module source**: `incubator_pipeline/postprocessing.py`

---

**Remember**: The `.pt` file only contains YOLO weights. All validation logic must be added to your Streamlit app using this module! 🎯
