"""
Tesseract Configuration Helper
Automatically detects and configures Tesseract OCR path for Windows systems.
"""
import os
import sys
from pathlib import Path
import pytesseract

def find_tesseract():
    """
    Attempts to find Tesseract OCR installation on Windows.
    Returns the path to tesseract.exe if found, None otherwise.
    """
    # Common installation paths
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    
    # Check if tesseract is in PATH
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract found in system PATH")
        return "system_path"
    except:
        pass
    
    # Check common installation paths
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Tesseract found at: {path}")
            return path
    
    return None

def configure_tesseract():
    """
    Configures pytesseract with the correct Tesseract path.
    Call this function before using Tesseract OCR.
    """
    tesseract_path = find_tesseract()
    
    if tesseract_path == "system_path":
        # Already in PATH, no configuration needed
        return True
    elif tesseract_path:
        # Set the path for pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Tesseract configured successfully")
        return True
    else:
        print("❌ Tesseract OCR not found!")
        print("\n📥 Please install Tesseract OCR:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Run the installer")
        print("   3. Add to system PATH or restart this application")
        return False

def check_tesseract_version():
    """Check and display Tesseract version."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"📊 Tesseract version: {version}")
        return version
    except Exception as e:
        print(f"❌ Error checking Tesseract version: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Tesseract Configuration Helper\n")
    print("=" * 60)
    
    if configure_tesseract():
        check_tesseract_version()
        print("\n✅ Tesseract is ready to use!")
    else:
        print("\n⚠️  Please install Tesseract before running the app.")
        sys.exit(1)
