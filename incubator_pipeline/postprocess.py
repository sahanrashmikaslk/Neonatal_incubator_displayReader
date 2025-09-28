"""Utilities for shaping detector + OCR outputs into structured telemetry."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import PARAMETERS, ParameterSpec

DetectionPayload = Dict[str, object]


def select_best_detection(detections: Iterable[DetectionPayload], class_name: str) -> Optional[DetectionPayload]:
    candidates = [det for det in detections if det.get("class_name") == class_name]
    if not candidates:
        return None
    return max(candidates, key=lambda det: float(det.get("confidence", 0.0)))


def assemble_parameters(detections: List[DetectionPayload]) -> Dict[str, Dict[str, object]]:
    reading: Dict[str, Dict[str, object]] = {}
    for param, spec in PARAMETERS.items():
        value_det = select_best_detection(detections, spec.value)
        label_det = select_best_detection(detections, spec.label) if spec.label else None
        if value_det is None:
            continue
        entry: Dict[str, object] = {
            "value_detection": value_det,
            "label_detection": label_det,
            "unit": spec.unit,
            "parameter": param,
        }
        reading[param] = entry
    return reading


def flatten_readings(readings: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for param, payload in readings.items():
        value_det = payload.get("value_detection", {})
        text = value_det.get("text") or value_det.get("ocr", {}).get("text") if isinstance(value_det.get("ocr"), dict) else None
        if text is not None:
            flat[f"{param}"] = text
        unit = payload.get("unit")
        if unit:
            flat[f"{param}_unit"] = unit
    return flat


def normalize_numeric(text: Optional[str]) -> Optional[float]:
    if text is None or text == "":
        return None
    try:
        return float(text.replace("%", ""))
    except ValueError:
        return None
