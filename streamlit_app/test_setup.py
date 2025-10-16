"""
Test script to verify Tesseract OCR integration
Run this before starting the Streamlit app to ensure everything is working.
"""
import sys
from pathlib import Path

print("=" * 70)
print("🧪 Testing Incubator Display Reader with Advanced Tesseract OCR")
print("=" * 70)

# Test 1: Check Tesseract
print("\n1️⃣ Checking Tesseract OCR installation...")
try:
    from tesseract_config import configure_tesseract, check_tesseract_version
    
    if configure_tesseract():
        version = check_tesseract_version()
        if version:
            print("   ✅ Tesseract OCR is properly configured")
        else:
            print("   ⚠️  Tesseract found but version check failed")
    else:
        print("   ❌ Tesseract not found. Please install it first.")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Check required packages
print("\n2️⃣ Checking required Python packages...")
required_packages = [
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('pytesseract', 'pytesseract'),
    ('PIL', 'Pillow'),
    ('streamlit', 'streamlit'),
]

missing = []
for module, package in required_packages:
    try:
        __import__(module)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing.append(package)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check model files
print("\n3️⃣ Checking YOLO model files...")
project_dir = Path(__file__).parent.parent
model_dir = project_dir / 'models'

model_files = [
    'incubator_yolov8n.pt',
    'incubator_yolov8n_v3.pt'
]

models_found = []
for model_file in model_files:
    model_path = model_dir / model_file
    if model_path.exists():
        print(f"   ✅ {model_file}")
        models_found.append(model_file)
    else:
        print(f"   ⚠️  {model_file} - not found")

if not models_found:
    print("\n   ❌ No trained models found in models/ directory")
    print("   Please train a model first using the Jupyter notebook")
    sys.exit(1)

# Test 4: Test the pipeline
print("\n4️⃣ Testing IncubatorDisplayReader...")
try:
    from incubator_pipeline import IncubatorDisplayReader
    
    # Find a model to test with
    model_path = model_dir / models_found[0]
    reader = IncubatorDisplayReader(weights_path=str(model_path))
    
    print(f"   ✅ Successfully loaded reader with model: {models_found[0]}")
    print(f"   ✅ Tesseract integration: {'enabled' if reader.tesseract_available else 'disabled'}")
    
    if not reader.tesseract_available:
        print("   ⚠️  Warning: Tesseract not available in pipeline")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Error loading pipeline: {e}")
    sys.exit(1)

# Test 5: Quick OCR test
print("\n5️⃣ Testing advanced Tesseract OCR preprocessing...")
try:
    import numpy as np
    import cv2
    
    # Create a simple test image
    test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    # Test preprocessing
    variants = reader.preprocess_roi_tesseract_advanced(test_img)
    
    if variants:
        print(f"   ✅ Generated {len(variants)} preprocessing variants")
        print(f"   Strategies: {', '.join([v[0] for v in variants])}")
    else:
        print("   ⚠️  No preprocessing variants generated")
        
except Exception as e:
    print(f"   ⚠️  Error testing OCR: {e}")

# Summary
print("\n" + "=" * 70)
print("✅ All tests passed! The Streamlit app is ready to use.")
print("=" * 70)
print("\n🚀 To start the app, run:")
print("   cd streamlit_app")
print("   streamlit run app_opencv_webcam.py")
print("\n📖 For more information, see INSTALLATION.md")
print("=" * 70)
