# Camera Switcher Feature Guide

## Overview

The enhanced Streamlit app now includes a **camera switcher** that allows you to select from multiple connected cameras and adjust settings for optimal performance.

## 🎥 Camera Features

### 1. **Camera Selection**

- Automatically detects all connected cameras
- Shows camera index and native resolution
- Easy dropdown selector to switch between cameras
- **Scan for Cameras** button to refresh camera list

### 2. **Resolution Control**

Choose from preset resolutions:

- **640x480 (VGA)** - Fast processing, lower quality
- **1280x720 (HD)** - Balanced performance (recommended)
- **1920x1080 (Full HD)** - Best quality, slower processing
- **Auto** - Uses camera's default resolution

### 3. **Performance Control**

- **Frame Delay Slider**: Adjust processing speed (50-500ms)
  - 50ms: Smooth video, high CPU usage
  - 100ms: Balanced (default)
  - 500ms: Lower CPU usage, choppy video

## 📋 How to Use

### Basic Camera Switching

1. **Open the app** (running at http://localhost:8503)

2. **Check sidebar** for "📹 Camera Settings"

3. **View detected cameras:**

   ```
   ✅ Found 2 camera(s)

   Select Camera:
   ├─ Camera 0 (1280x720)  ← Laptop webcam
   └─ Camera 1 (1920x1080) ← USB camera
   ```

4. **Select your desired camera** from dropdown

5. **Choose resolution** (optional)

6. **Go to "Live Webcam" tab**

7. **Click "Start Webcam"** - Selected camera will activate

### Camera Not Detected?

1. **Click "🔍 Scan for Cameras"** button
2. Check physical connections
3. Close other apps using camera (Teams, Zoom, Skype)
4. Check Windows Privacy Settings:
   - Settings → Privacy → Camera
   - Enable camera access for desktop apps

### Switching Cameras While Active

To switch cameras:

1. Click "⏹️ Stop Webcam"
2. Select different camera in sidebar
3. Click "▶️ Start Webcam" again

## 🎯 Use Cases

### Multiple Camera Setup

**Example 1: Dual Camera Monitoring**

```
Camera 0 (Laptop): General room view
Camera 1 (USB):    Close-up of incubator display
```

**Example 2: External High-Quality Camera**

```
Camera 0: Built-in laptop camera (backup)
Camera 1: 1080p USB webcam (primary for readings)
```

### Resolution Selection

**When to use VGA (640x480):**

- Old/slow computer
- Quick testing
- Network streaming

**When to use HD (1280x720):**

- ✅ Recommended default
- Good balance of quality and speed
- Works with most hardware

**When to use Full HD (1920x1080):**

- High-quality camera available
- Need clearer text for OCR
- Powerful computer

## ⚙️ Technical Details

### Camera Detection

```python
def get_available_cameras(max_cameras=10):
    """Detect available cameras on the system."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras
```

### Camera Initialization

```python
cap = cv2.VideoCapture(selected_camera)

# Set custom resolution
if resolution is not None:
    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
```

## 📊 Camera Information Display

The app shows:

- **Selected camera index**: "Camera 0", "Camera 1", etc.
- **Native resolution**: Actual camera resolution
- **Active status**: Real-time resolution when streaming
- **Total cameras found**: Count in sidebar

Example display:

```
📹 Selected: Camera 1 | Resolution: 1280x720 (HD)
✅ Camera 1 active at 1280x720. Adjust position and click 'Capture Frame' to save readings.
```

## 🔧 Troubleshooting

### Problem: Camera not in list

**Solutions:**

1. Unplug and replug USB camera
2. Click "Scan for Cameras"
3. Restart the Streamlit app
4. Check Device Manager (Windows)

### Problem: "Cannot access Camera X"

**Solutions:**

1. Close other apps using the camera
2. Try a different camera from dropdown
3. Check camera permissions
4. Test with Windows Camera app first

### Problem: Video is laggy/slow

**Solutions:**

1. Increase "Frame Delay" slider (sidebar)
2. Select lower resolution (640x480)
3. Adjust confidence threshold (reduce detections)
4. Close unnecessary applications

### Problem: OCR not accurate

**Solutions:**

1. Select higher resolution (1080p)
2. Ensure good lighting
3. Position camera directly in front of display
4. Adjust confidence threshold
5. Use camera with better focus

## 🎬 Example Workflow

### Workflow 1: Compare Two Cameras

1. Start with Camera 0
2. Go to Live Webcam tab
3. Start webcam → Test detection
4. Stop webcam
5. Switch to Camera 1 (sidebar)
6. Start webcam → Compare quality
7. Choose best camera for your setup

### Workflow 2: Optimize Performance

1. Select your camera
2. Start with HD (1280x720)
3. If laggy:
   - Reduce to VGA (640x480)
   - Increase frame delay to 200ms
4. If too fast/smooth not needed:
   - Increase frame delay to save CPU

### Workflow 3: High-Quality Capture

1. Connect external HD camera
2. Select Camera 1 (USB camera)
3. Set resolution: Full HD (1920x1080)
4. Ensure good lighting
5. Capture frames for critical readings

## 📝 Sidebar Settings Summary

```
Model Configuration
├─ YOLO weights path
└─ Detection confidence: 0.25

📹 Camera Settings
├─ 🔍 Scan for Cameras
├─ Select Camera: Camera 1 (1280x720)
├─ Resolution: 1280x720 (HD)
└─ Frame Delay: 100ms

ℹ️ Tips & Info
└─ Usage guidelines
```

## 🚀 Advanced Tips

1. **Label Your Cameras**: Note which camera is which (built-in, USB-A, USB-C)

2. **Test Before Critical Use**: Try all cameras with actual incubator display

3. **Lighting Matters**: Good lighting improves OCR accuracy more than camera quality

4. **Position Stability**: Use tripod or mount for consistent readings

5. **Resolution vs Speed**: Higher isn't always better if it slows down too much

6. **Frame Rate**: 10 FPS (100ms delay) is usually sufficient for static displays

## 📱 Camera Types Supported

✅ **Built-in Laptop Webcam** (Camera 0)
✅ **USB Webcams** (Camera 1, 2, ...)
✅ **External HD Cameras**
✅ **USB-C Cameras**
❌ IP Cameras (requires different implementation)
❌ Network Cameras (requires RTSP support)

## 🎓 Best Practices

1. **Always test new cameras** before production use
2. **Keep camera lenses clean** for clear OCR
3. **Use consistent camera position** for reproducible results
4. **Document your camera setup** (which camera, resolution, position)
5. **Have backup camera** in case primary fails

---

## Quick Reference

| Setting         | Recommended               | Notes                      |
| --------------- | ------------------------- | -------------------------- |
| **Camera**      | External USB if available | Better quality than laptop |
| **Resolution**  | 1280x720 (HD)             | Best balance               |
| **Frame Delay** | 100ms                     | 10 FPS is sufficient       |
| **Confidence**  | 0.25                      | Adjust based on accuracy   |

**Enjoy your multi-camera setup!** 📹✨
