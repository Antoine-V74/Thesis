"""R-peak detection algorithm registry."""
from __future__ import annotations

from typing import Dict, List

from RPeakDetection.algorithms.ampt import AmptDetector
from RPeakDetection.algorithms.adaptive_threshold import AdaptiveThresholdDetector
from RPeakDetection.algorithms.base import RPeakDetector
from RPeakDetection.algorithms.ecg_detectors import (
    ChristovDetector,
    EngzeeDetector,
    HamiltonDetector,
    PanTompkinsDetector,
    TwoAverageDetector,
)

_ALGORITHM_CLASSES = {
    "adaptive_threshold": AdaptiveThresholdDetector,
    "ampt": AmptDetector,
    "hamilton": HamiltonDetector,
    "christov": ChristovDetector,
    "pan_tompkins": PanTompkinsDetector,
    "engzee": EngzeeDetector,
    "two_average": TwoAverageDetector,
}

DEFAULT_ALGORITHMS: List[str] = list(_ALGORITHM_CLASSES.keys())


def get_detector(name: str) -> RPeakDetector:
    key = name.strip().lower()
    if key not in _ALGORITHM_CLASSES:
        raise KeyError(f"Unknown algorithm '{name}'. Choose from: {sorted(_ALGORITHM_CLASSES)}")
    return _ALGORITHM_CLASSES[key]()


def list_algorithms() -> List[str]:
    return sorted(_ALGORITHM_CLASSES.keys())
