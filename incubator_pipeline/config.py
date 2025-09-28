"""Configuration constants for the neonatal incubator display reader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


CLASS_NAMES = [
    "spo2_label",
    "spo2_value",
    "heart_rate_label",
    "heart_rate_value",
    "pi_value",
    "air_temp_label",
    "air_temp_value",
    "skin_temp_label",
    "skin_temp_value",
    "humidity_label",
    "humidity_value",
    "oxygen_label",
    "oxygen_value",
    "warning_banner",
    "waveform_region",
]


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    value: str
    label: str | None = None
    unit: str | None = None


PARAMETERS: Dict[str, ParameterSpec] = {
    "spo2": ParameterSpec(
        key="spo2",
        value="spo2_value",
        label="spo2_label",
        unit="%",
    ),
    "heart_rate": ParameterSpec(
        key="heart_rate",
        value="heart_rate_value",
        label="heart_rate_label",
        unit="bpm",
    ),
    "pi": ParameterSpec(
        key="pi",
        value="pi_value",
        label=None,
    ),
    "air_temperature": ParameterSpec(
        key="air_temperature",
        value="air_temp_value",
        label="air_temp_label",
        unit="°C",
    ),
    "skin_temperature": ParameterSpec(
        key="skin_temperature",
        value="skin_temp_value",
        label="skin_temp_label",
        unit="°C",
    ),
    "humidity": ParameterSpec(
        key="humidity",
        value="humidity_value",
        label="humidity_label",
        unit="%",
    ),
    "oxygen": ParameterSpec(
        key="oxygen",
        value="oxygen_value",
        label="oxygen_label",
        unit="%",
    ),
}


VALUE_CLASS_NAMES = {spec.value for spec in PARAMETERS.values()}
LABEL_CLASS_NAMES = {spec.label for spec in PARAMETERS.values() if spec.label}

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS_NAME = {idx: name for name, idx in CLASS_NAME_TO_ID.items()}

DEFAULT_WEIGHTS_PATH = "models/incubator_yolov8n.pt"
DEFAULT_DETECTION_CONFIDENCE = 0.25
"""
CLASS_NAMES: All YOLO class labels used for detection.
PARAMETERS: Declarative mapping describing which detection classes correspond to each physical reading.
VALUE_CLASS_NAMES / LABEL_CLASS_NAMES: Convenience sets for downstream filtering.
"""
