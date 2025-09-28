"""OCR utilities built on EasyOCR for incubator displays."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import cv2
import easyocr
import numpy as np


_NUMERIC_PATTERN = re.compile(r"[^0-9.%]")


def clean_numeric(text: str) -> str:
    """Sanitize OCR text to retain only digits, percent, and decimal points."""
    cleaned = _NUMERIC_PATTERN.sub("", text.upper())
    cleaned = cleaned.replace("..", ".")
    return cleaned.strip(".")


def enhance_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return resized


@dataclass
class OcrResult:
    text: Optional[str]
    confidence: float
    raw_text: Optional[str]


class EasyOCREngine:
    """Thin wrapper providing cached EasyOCR reader."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: Optional[bool] = None) -> None:
        self.languages = languages or ["en"]
        self._gpu = gpu

    @property
    @lru_cache(maxsize=1)
    def reader(self) -> easyocr.Reader:  # type: ignore[misc]
        return easyocr.Reader(self.languages, gpu=self._gpu)

    def read_text(self, roi: np.ndarray) -> OcrResult:
        if roi.size == 0:
            return OcrResult(text=None, confidence=0.0, raw_text=None)
        enhanced = enhance_roi(roi)
        results = self.reader.readtext(enhanced)
        if not results:
            return OcrResult(text=None, confidence=0.0, raw_text=None)
        # results: list[(bbox, text, confidence)]
        best = max(results, key=lambda item: item[2])
        raw_text = best[1]
        cleaned = clean_numeric(raw_text)
        return OcrResult(text=cleaned or None, confidence=float(best[2]), raw_text=raw_text)
