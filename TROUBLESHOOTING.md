# Troubleshooting Guide

## TensorBoard/Protobuf Import Errors

If you encounter errors like:

```
ImportError: cannot import name 'notf' from 'tensorboard.compat'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

### Quick Fix

Run these commands in your terminal:

```powershell
# Uninstall conflicting versions
pip uninstall -y protobuf tensorboard grpcio-status

# Install compatible versions
pip install --user protobuf==3.20.3
pip install --user tensorboard==2.14.0
pip install --user grpcio-status==1.48.2
```

### Alternative: Use the troubleshooting cell in the notebook

1. Open `notebooks/incubator_pipeline.ipynb`
2. Find the cell titled **"Troubleshooting: TensorBoard/Protobuf Issues"**
3. Run that cell
4. **Restart the kernel** (Kernel → Restart Kernel)
5. Re-run all cells from the beginning

### Root Cause

The issue occurs due to:

- Python 3.12+ with newer protobuf (6.x) is incompatible with older TensorBoard
- Ultralytics (YOLOv8) uses TensorBoard for logging during training
- The MessageFactory API changed between protobuf versions

### Preventing Future Issues

Add these environment variables at the start of your notebook:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
```

This has been added to the notebook already.

## Other Common Issues

### CUDA/GPU Not Detected

If PyTorch doesn't detect your GPU:

```powershell
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory During Training

Reduce batch size in the training cell:

```python
detector, train_results = train_detector(data_yaml_path, batch=8)  # Reduce from 16 to 8
```

### EasyOCR Installation Issues

If EasyOCR fails to install:

```powershell
pip install --user torch torchvision
pip install --user easyocr==1.7.1
```

### Module Not Found Errors

Restart the Jupyter kernel:

1. Click **Kernel** → **Restart Kernel**
2. Re-run cells from the top

## Getting Help

If issues persist:

1. Check Python version: `python --version` (3.8-3.11 recommended)
2. Update pip: `python -m pip install --upgrade pip`
3. Create a fresh virtual environment
4. Reinstall all dependencies from `requirements.txt`
