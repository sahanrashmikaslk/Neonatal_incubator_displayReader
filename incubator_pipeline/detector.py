"""YOLO-based display region detector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
from torch.nn import modules as torch_nn_modules
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from ultralytics import utils as yolo_utils
from ultralytics.nn import modules as yolo_modules
from ultralytics.nn.tasks import DetectionModel

from .config import CLASS_NAMES, ID_TO_CLASS_NAME

try:  # pragma: no cover - optional dependency path
    from ultralytics.utils import IterableSimpleNamespace
except ImportError:  # pragma: no cover
    IterableSimpleNamespace = None


def _register_safe_globals() -> None:
    """Allowlist Ultralytics/Torch classes for torch.load when weights_only=True."""
    try:
        allowed = {DetectionModel}
        allowed.update({obj for obj in vars(yolo_modules).values() if isinstance(obj, type)})
        allowed.update({obj for obj in vars(torch_nn_modules).values() if isinstance(obj, type)})
        if IterableSimpleNamespace is not None:
            allowed.add(IterableSimpleNamespace)
        allowed.update({obj for obj in vars(yolo_utils).values() if isinstance(obj, type)})
        add_safe_globals(list(allowed))
    except Exception:  # pragma: no cover - defensive
        pass


_register_safe_globals()


if not hasattr(torch, "_original_load_incubator"):
    torch._original_load_incubator = torch.load


def _incubator_safe_torch_load(*args, **kwargs):
    """Fallback loader using weights_only=False for trusted local checkpoints."""
    kwargs.setdefault("weights_only", False)
    return torch._original_load_incubator(*args, **kwargs)


torch.load = _incubator_safe_torch_load


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

    def predict(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> list[DetectionResult]:
        frame, maybe_path = self._ensure_image(image)
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model.predict(
            source=frame,
            conf=threshold,
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

    def predict_batch(
        self,
        images: Iterable[str | Path | np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> list[list[DetectionResult]]:
        return [self.predict(image, conf_threshold=conf_threshold) for image in images]
