#!/usr/bin/env python3
"""
Convert YOLOv8 model to ONNX format for better Raspberry Pi performance

Usage:
    python convert_model_to_onnx.py
"""

import os
import sys

# Set environment variable to allow loading custom models (PyTorch 2.6+ compatibility)
os.environ['TORCH_WEIGHTS_ONLY'] = '0'

from ultralytics import YOLO
from pathlib import Path

def convert_to_onnx():
    """Convert the trained YOLOv8 model to ONNX format"""
    
    # Model paths
    models_dir = Path(__file__).parent / "models"
    input_model = models_dir / "incubator_yolov8n.pt"
    output_model = models_dir / "incubator_yolov8n.onnx"
    
    if not input_model.exists():
        print(f"❌ Model not found: {input_model}")
        print("Available models:")
        for model_file in models_dir.glob("*.pt"):
            print(f"  - {model_file.name}")
        return False
    
    print(f"🔄 Converting {input_model.name} to ONNX format...")
    print(f"📍 Input:  {input_model}")
    print(f"📍 Output: {output_model}")
    
    try:
        print("Loading model...")
        model = YOLO(str(input_model))
        
        print("Exporting to ONNX format...")
        # Export to ONNX with optimizations for inference
        export_path = model.export(
            format='onnx',
            opset=12,  # Compatible with older ONNX runtimes
            simplify=True,  # Simplify the model for faster inference
            dynamic=False,  # Fixed input size for optimization
            half=False,  # Keep FP32 for better accuracy on CPU
        )
        
        print(f"✅ Model successfully converted!")
        print(f"📦 ONNX model saved to: {export_path}")
        print(f"\n📊 Model info:")
        print(f"   Format: ONNX (optimized for CPU inference)")
        print(f"   Opset: 12 (compatible with most runtimes)")
        print(f"   Precision: FP32 (full precision)")
        print(f"\n🚀 Next steps:")
        print(f"   1. Transfer the .onnx file to your Raspberry Pi")
        print(f"   2. Install onnxruntime: pip install onnxruntime")
        print(f"   3. Use the ONNX model for faster inference")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print(f"\n💡 Try this workaround:")
        print(f"   Open Python and run:")
        print(f"   >>> from ultralytics import YOLO")
        print(f"   >>> model = YOLO('{input_model}')")
        print(f"   >>> model.export(format='onnx')")
        return False

def convert_to_ncnn():
    """Convert the trained YOLOv8 model to NCNN format (alternative for ARM devices)"""
    
    models_dir = Path(__file__).parent / "models"
    input_model = models_dir / "incubator_yolov8n.pt"
    
    print(f"\n🔄 Converting to NCNN format (optimized for ARM/mobile devices)...")
    
    try:
        print("Loading model...")
        model = YOLO(str(input_model))
        
        print("Exporting to NCNN format...")
        # Export to NCNN format
        export_path = model.export(format='ncnn')
        
        print(f"✅ NCNN model successfully created!")
        print(f"📦 Files saved to: {export_path}")
        print(f"\n🚀 NCNN is optimized for:")
        print(f"   - ARM processors (Raspberry Pi)")
        print(f"   - Mobile devices")
        print(f"   - Low-power inference")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during NCNN conversion: {e}")
        print(f"ℹ️  NCNN export may require additional dependencies")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Model Conversion Tool for Raspberry Pi Deployment")
    print("=" * 60)
    
    # Convert to ONNX (recommended)
    success_onnx = convert_to_onnx()
    
    print("\n" + "-" * 60)
    
    # Optionally convert to NCNN
    print("\n❓ Do you also want to convert to NCNN format? (y/n)")
    print("   NCNN may offer better performance on ARM devices")
    choice = input("   Choice: ").strip().lower()
    
    if choice == 'y':
        convert_to_ncnn()
    
    print("\n" + "=" * 60)
    print("✨ Conversion complete!")
    print("=" * 60)
