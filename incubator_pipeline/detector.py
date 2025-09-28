"""YOLO-based display region detector."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import CLASS_NAMES, ID_TO_CLASS_NAME


class DetectionResult(dict):
    """Simple dict subclass to hold detection metadata."""

    bbox: List[float]
    class_id: int
    class_name: str
    confidence: float
    crop: np.ndarray


class DisplayDetector:
    """Wrapper around Ultralytics YOLO for incubator displays."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(self.weights_path)

    @staticmethod
    def _ensure_image(image: str | Path | np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        if isinstance(image, (str, Path)):
            arr = cv2.imread(str(image))
            if arr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return arr, str(image)
        if isinstance(image, np.ndarray):
            return image.copy(), None
        raise TypeError("image must be a file path or numpy array")

    def predict(self, image: str | Path | np.ndarray) -> list[DetectionResult]:
        frame, maybe_path = self._ensure_image(image)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        detections: list[DetectionResult] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        height, width = frame.shape[:2]
        for bbox, class_id, conf in zip(xyxy, class_ids, confidences):
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            crop = frame[y1:y2, x1:x2]
            detections.append(
                DetectionResult(
                    bbox=bbox.tolist(),
                    class_id=int(class_id),
                    class_name=ID_TO_CLASS_NAME.get(int(class_id), CLASS_NAMES[int(class_id)]),
                    confidence=float(conf),
                    crop=crop,
                    image_path=maybe_path,
                )
            )
        return detections

    def predict_batch(self, images: Iterable[str | Path | np.ndarray]) -> list[list[DetectionResult]]:
        return [self.predict(image) for image in images]
