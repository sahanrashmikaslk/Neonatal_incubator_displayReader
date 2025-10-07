"""High level orchestration for incubator display reading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

from .config import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_WEIGHTS_PATH,
    PARAMETERS,
    ParameterSpec,
)
from .detector import DisplayDetector
from .ocr import EasyOCREngine, OcrResult
from .postprocess import assemble_parameters, normalize_numeric


@dataclass
class Reading:
    parameter: str
    value: Optional[str]
    detection_confidence: float
    ocr_confidence: float
    unit: Optional[str]
    bbox: List[float]


class IncubatorDisplayReader:
    """Combines object detection and OCR to extract incubator telemetry."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        conf_threshold: float = DEFAULT_DETECTION_CONFIDENCE,
        device: Optional[str] = None,
        ocr_languages: Optional[List[str]] = None,
        ocr_gpu: Optional[bool] = None,
    ) -> None:
        self.detector = DisplayDetector(weights_path, conf_threshold, device)
        gpu = ocr_gpu if ocr_gpu is not None else torch.cuda.is_available()
        self.ocr = EasyOCREngine(languages=ocr_languages or ["en"], gpu=gpu)

    def _run_detector(self, image: str | Path | np.ndarray):
        return self.detector.predict(image)

    def _attach_ocr(self, detections):
        value_classes = {spec.value for spec in PARAMETERS.values()}
        enriched = []
        for det in detections:
            det_copy = dict(det)
            crop = det_copy.get("crop")
            if crop is not None and det_copy.get("class_name") in value_classes:
                ocr_result: OcrResult = self.ocr.read_text(crop)
                det_copy["ocr"] = {
                    "text": ocr_result.text,
                    "confidence": ocr_result.confidence,
                    "raw_text": ocr_result.raw_text,
                }
                det_copy["text"] = ocr_result.text
                det_copy["ocr_confidence"] = ocr_result.confidence
            enriched.append(det_copy)
        return enriched

    def read(
        self,
        image: str | Path | np.ndarray,
        conf: Optional[float] = None,
    ) -> Dict[str, Reading]:
        detections = self.detector.predict(image, conf_threshold=conf)
        detections = self._attach_ocr(detections)
        grouped = assemble_parameters(detections)
        readings: Dict[str, Reading] = {}
        for parameter, payload in grouped.items():
            value_det = payload.get("value_detection", {})
            ocr = value_det.get("ocr") or {}
            readings[parameter] = Reading(
                parameter=parameter,
                value=ocr.get("text"),
                detection_confidence=float(value_det.get("confidence", 0.0)),
                ocr_confidence=float(ocr.get("confidence", 0.0) or 0.0),
                unit=payload.get("unit"),
                bbox=value_det.get("bbox", []),
            )
        return readings

    def read_to_dataframe(
        self,
        images: Iterable[str | Path | np.ndarray],
        conf: Optional[float] = None,
    ) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        for image in images:
            result = self.read(image, conf=conf)
            flat = {f"{k}": v.value for k, v in result.items()}
            conf = {f"{k}_conf": v.ocr_confidence for k, v in result.items()}
            record = {**flat, **conf, "image": str(image)}
            records.append(record)
        return pd.DataFrame(records)

    def annotate_image(
        self,
        image: str | Path | np.ndarray,
        conf: Optional[float] = None,
        readings: Optional[Dict[str, Reading]] = None,
        color=(0, 255, 0),
    ) -> np.ndarray:
        frame, _ = DisplayDetector._ensure_image(image)
        readings = readings or self.read(frame, conf=conf)
        annotated = frame.copy()
        for reading in readings.values():
            bbox = reading.bbox
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{reading.parameter}: {reading.value or '?'}"
            cv2.putText(
                annotated,
                label,
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    def to_numeric(self, readings: Dict[str, Reading]) -> Dict[str, Optional[float]]:
        return {key: normalize_numeric(value.value) for key, value in readings.items()}
