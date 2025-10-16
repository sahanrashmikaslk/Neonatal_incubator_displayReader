"""
Incubator Display Reader Pipeline
Provides the IncubatorDisplayReader class for detection and OCR.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import re
import warnings
import torch
import pytesseract
from PIL import Image
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn import modules as yolo_modules
from ultralytics.nn.modules import Conv
from torch.nn import modules as torch_nn_modules

# Fix torch.load to work with ultralytics weights
if not hasattr(torch, '_original_load_saved'):
    torch._original_load_saved = torch.load
    torch._no_nep50_warning = lambda: lambda func: func

def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch._original_load_saved(*args, **kwargs)

torch.load = _safe_torch_load

# Add safe globals for torch serialization
allowed_globals = {DetectionModel, Conv}
allowed_globals.update({obj for obj in vars(yolo_modules).values() if isinstance(obj, type)})
allowed_globals.update({obj for obj in vars(torch_nn_modules).values() if isinstance(obj, type)})
add_safe_globals(list(allowed_globals))

if not hasattr(np, "_no_nep50_warning"):
    def _no_nep50_warning():
        def decorator(func):
            return func
        return decorator
    np._no_nep50_warning = _no_nep50_warning

# Default paths
PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / 'models'
DEFAULT_WEIGHTS_PATH = MODEL_DIR / 'incubator_yolov8n.pt'

# Class configuration
CLASS_NAMES = [
    'heart_rate_value',
    'humidity_value',
    'skin_temp_value',
    'spo2_value',
]

NUMERIC_CLASSES = [c for c in CLASS_NAMES if c.endswith('value')]

# Annotation colors
TEXT_COLOR = (255, 255, 255)  # white
BOX_COLOR_NUMERIC = (60, 170, 255)  # soft orange for numeric classes
BOX_COLOR_OTHER = (120, 200, 80)  # green for non-numeric
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1


@dataclass
class Reading:
    """Single OCR reading result."""
    value: Optional[str]
    detection_confidence: float
    ocr_confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = None


class IncubatorDisplayReader:
    """Main class for detecting and reading incubator display values."""
    
    def __init__(self, weights_path: str = None, conf_threshold: float = 0.25, fast_mode: bool = True, 
                 use_half_precision: bool = False, cache_preprocessed: bool = True):
        """
        Initialize the reader with YOLO detector and Tesseract OCR.
        
        Args:
            weights_path: Path to YOLO weights file
            conf_threshold: Detection confidence threshold
            fast_mode: If True, use fast OCR (1 strategy). If False, use accurate OCR (15 strategies)
            use_half_precision: Use FP16 for faster inference (requires CUDA)
            cache_preprocessed: Cache preprocessed ROIs to avoid redundant processing
        """
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH
        
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load YOLO detector
        self.detector = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.fast_mode = fast_mode
        self.use_half_precision = use_half_precision
        self.cache_preprocessed = cache_preprocessed
        
        # ROI cache for avoiding redundant preprocessing
        self._roi_cache = {} if cache_preprocessed else None
        self._cache_max_size = 100
        
        # Check if Tesseract is available
        self.tesseract_available = self._check_tesseract()
        if not self.tesseract_available:
            warnings.warn("Tesseract OCR not found. Please install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
    
    def preprocess_roi_tesseract_advanced(self, roi: np.ndarray):
        """
        Advanced multi-strategy preprocessing for LCD/LED displays.
        Returns multiple preprocessing variants to try with Tesseract.
        """
        if roi.size == 0 or roi is None:
            return []
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        variants = []
        
        # Strategy 1: Extreme upscaling + CLAHE + Bilateral + Adaptive Threshold
        try:
            huge = cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            huge = clahe.apply(huge)
            huge = cv2.bilateralFilter(huge, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(huge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            variants.append(('adaptive', thresh))
        except:
            pass
        
        # Strategy 2: Otsu's thresholding
        try:
            huge = cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            huge = clahe.apply(huge)
            blur = cv2.GaussianBlur(huge, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(('otsu', otsu))
        except:
            pass
        
        # Strategy 3: Inverted Otsu
        try:
            huge = cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            huge = clahe.apply(huge)
            blur = cv2.GaussianBlur(huge, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(('inverted', otsu))
        except:
            pass
        
        # Strategy 4: Morphological operations
        try:
            huge = cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            huge = clahe.apply(huge)
            _, binary = cv2.threshold(huge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            variants.append(('dilated', dilated))
        except:
            pass
        
        # Strategy 5: Sharpening
        try:
            huge = cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(huge, -1, kernel_sharpen)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            sharpened = clahe.apply(sharpened)
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(('sharpened', binary))
        except:
            pass
        
        return variants
    
    def _compute_roi_hash(self, roi: np.ndarray) -> str:
        """Compute a hash of the ROI for caching."""
        # Use downsampled ROI for faster hash computation
        small = cv2.resize(roi, (32, 32))
        return hash(small.tobytes())
    
    def extract_value_fast(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Fast OCR extraction using only the best preprocessing strategy.
        Optimized for real-time webcam processing with optional caching.
        """
        if not self.tesseract_available:
            return None, 0.0
        
        if roi is None or roi.size == 0:
            return None, 0.0
        
        # Check cache if enabled
        if self.cache_preprocessed:
            roi_hash = self._compute_roi_hash(roi)
            if roi_hash in self._roi_cache:
                return self._roi_cache[roi_hash]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Single best strategy: 8x upscale + CLAHE + Otsu (optimized)
        try:
            # Use 6x instead of 8x for faster processing (still good quality)
            huge = cv2.resize(gray, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            huge = clahe.apply(huge)
            # Reduce Gaussian blur kernel for speed
            blur = cv2.GaussianBlur(huge, (3, 3), 0)
            _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            pil_image = Image.fromarray(processed)
            
            # Optimized Tesseract config: PSM 7, OEM 3, whitelist
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
            
            # Extract text (single call, no data fetch)
            text = pytesseract.image_to_string(pil_image, config=config)
            text = text.strip().replace(' ', '').replace('\n', '').replace('..', '.').strip('.')
            
            # Get confidence only if we have text
            if text and any(c.isdigit() for c in text):
                data = pytesseract.image_to_data(pil_image, config=config, 
                                                 output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 50
            else:
                avg_confidence = 0
            
            result = (text if text and any(c.isdigit() for c in text) else None, avg_confidence / 100.0)
            
            # Cache result if enabled
            if self.cache_preprocessed and result[0] is not None:
                # Manage cache size
                if len(self._roi_cache) >= self._cache_max_size:
                    # Remove oldest entry (simple FIFO)
                    self._roi_cache.pop(next(iter(self._roi_cache)))
                self._roi_cache[roi_hash] = result
            
            return result
            
        except:
            pass
        
        return None, 0.0
    
    def extract_value(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Extract numeric value using advanced Tesseract OCR with multiple preprocessing strategies.
        Use this for batch processing or when accuracy is more important than speed.
        """
        if not self.tesseract_available:
            return None, 0.0
        
        if roi is None or roi.size == 0:
            return None, 0.0
        
        # Get multiple preprocessing variants
        variants = self.preprocess_roi_tesseract_advanced(roi)
        
        if not variants:
            return None, 0.0
        
        # Try different PSM modes
        psm_modes = [
            ('psm7', r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'),
            ('psm8', r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'),
            ('psm13', r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.'),
        ]
        
        best_result = None
        best_conf = 0.0
        
        # Try all combinations
        for variant_name, processed in variants:
            pil_image = Image.fromarray(processed)
            
            for psm_name, config in psm_modes:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(pil_image, config=config)
                    text = text.strip().replace(' ', '').replace('\n', '')
                    
                    # Get confidence
                    data = pytesseract.image_to_data(pil_image, config=config, 
                                                     output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Clean text
                    if text:
                        text = text.replace('..', '.')
                        text = text.strip('.')
                        
                        # Only consider if we got numeric text
                        if text and any(c.isdigit() for c in text):
                            if avg_confidence > best_conf:
                                best_result = text
                                best_conf = avg_confidence
                except:
                    continue
        
        return best_result, best_conf / 100.0
    
    def read(self, image: np.ndarray, conf: float = None) -> Dict[str, Reading]:
        """
        Detect regions and extract OCR readings from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf: Optional confidence threshold override
            
        Returns:
            Dictionary mapping class names to Reading objects
        """
        if conf is None:
            conf = self.conf_threshold
        
        # Optimized YOLO prediction settings
        results = self.detector.predict(
            source=image, 
            conf=conf, 
            verbose=False,
            half=self.use_half_precision,  # Use FP16 if enabled
            device='cuda' if self.use_half_precision else None,  # Use GPU if half precision
            imgsz=640,  # Standard size for speed
            augment=False,  # Disable augmentation for speed
            agnostic_nms=False,  # Faster NMS
            max_det=10  # Limit max detections for speed
        )
        det = results[0]
        
        outputs = {}
        
        for box, cls, score in zip(
            det.boxes.xyxy.cpu().numpy(),
            det.boxes.cls.cpu().numpy(),
            det.boxes.conf.cpu().numpy()
        ):
            cls = int(cls)
            name = CLASS_NAMES[cls]
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)
            
            roi = image[y1:y2, x1:x2]
            
            if name in NUMERIC_CLASSES and roi.size != 0:
                # Use fast or accurate mode based on initialization
                if self.fast_mode:
                    value, ocr_conf = self.extract_value_fast(roi)
                else:
                    value, ocr_conf = self.extract_value(roi)
                outputs[name] = Reading(
                    value=value,
                    detection_confidence=float(score),
                    ocr_confidence=float(ocr_conf),
                    bbox=box.tolist()
                )
            else:
                outputs[name] = Reading(
                    value=None,
                    detection_confidence=float(score),
                    bbox=box.tolist()
                )
        
        return outputs
    
    def draw_label(self, image: np.ndarray, text: str, anchor: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw label with background on image."""
        x, y = anchor
        text = text if text else ""
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        pad = 6
        y = max(y, text_h + pad)
        top_left = (x, y - text_h - pad)
        bottom_right = (x + text_w + 2 * pad, y + baseline)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
        text_org = (x + pad, y - pad)
        cv2.putText(image, text, text_org, FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    def annotate_image(self, image: np.ndarray, conf: float = None) -> np.ndarray:
        """
        Annotate image with detection boxes and OCR results.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf: Optional confidence threshold override
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        if conf is None:
            conf = self.conf_threshold
        readings = self.read(image, conf=conf)
        
        for name, reading in readings.items():
            if reading.bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, reading.bbox)
            color = BOX_COLOR_NUMERIC if name in NUMERIC_CLASSES else BOX_COLOR_OTHER
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # Draw label
            label_text = f"{name}: {reading.value}" if reading.value else name
            self.draw_label(annotated, label_text, (x1, y1 - 8), color)
        
        return annotated
