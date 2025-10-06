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
import easyocr
import torch
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
FONT_SCALE = 0.7
FONT_THICKNESS = 2


@dataclass
class Reading:
    """Single OCR reading result."""
    value: Optional[str]
    detection_confidence: float
    ocr_confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = None


class IncubatorDisplayReader:
    """Main class for detecting and reading incubator display values."""
    
    def __init__(self, weights_path: str = None, conf_threshold: float = 0.25):
        """
        Initialize the reader with YOLO detector and EasyOCR.
        
        Args:
            weights_path: Path to YOLO weights file
            conf_threshold: Detection confidence threshold
        """
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH
        
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load YOLO detector
        self.detector = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        
        # Initialize EasyOCR reader
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for better OCR accuracy."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.resize(blur, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    def clean_numeric(self, text: str) -> str:
        """Clean OCR text to extract numeric values."""
        cleaned = re.sub(r'[^0-9.%]', '', text)
        cleaned = cleaned.replace('..', '.')
        return cleaned.strip('.')
    
    def extract_value(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """Extract numeric value from ROI using OCR."""
        if roi.size == 0:
            return None, 0.0
        
        processed = self.preprocess_roi(roi)
        results = self.reader.readtext(processed, detail=1)
        
        if not results:
            return None, 0.0
        
        best = max(results, key=lambda x: x[2])
        _, raw_text, confidence = best
        text = self.clean_numeric(raw_text)
        
        return text or None, float(confidence)
    
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
        results = self.detector.predict(source=image, conf=conf, verbose=False)
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
