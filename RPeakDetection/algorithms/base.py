"""Detector interface for R-peak comparison benchmarks."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from RPeakDetection.types import RPeakDetectionResult


class RPeakDetector(Protocol):
    name: str
    is_causal: bool
    uses_prefilter: bool

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        """Detect R-peaks on a single-lead ECG segment."""
