"""Wrappers for berndporr/py-ecg-detectors (batch/offline algorithms)."""
from __future__ import annotations

import numpy as np
from ecgdetectors import Detectors

from RPeakDetection.types import RPeakDetectionResult


class _EcgDetectorsWrapper:
    """Batch detector: peaks returned after full-segment processing."""

    method_name: str
    is_causal = False
    uses_prefilter = False

    def __init__(self, method_name: str, notes: str):
        self.method_name = method_name
        self.name = method_name
        self.notes = notes

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        x = np.asarray(raw, dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        detectors = Detectors(float(fs))
        peaks = getattr(detectors, f"{self.method_name}_detector")(x)
        peak_samples = np.asarray(peaks, dtype=int)
        # Batch algorithms emit peaks at end of processing; no separate confirmation step.
        return RPeakDetectionResult(
            algorithm=self.name,
            peak_samples=peak_samples,
            confirmation_samples=peak_samples.copy(),
            confirmation_delays_ms=np.zeros(len(peak_samples), dtype=float),
            polarity="unknown",
            uses_prefilter=False,
            is_causal=False,
            notes=self.notes,
        )


class HamiltonDetector(_EcgDetectorsWrapper):
    def __init__(self) -> None:
        super().__init__(
            "hamilton",
            "Hamilton 2002 open-source ECG analysis (py-ecg-detectors).",
        )


class ChristovDetector(_EcgDetectorsWrapper):
    def __init__(self) -> None:
        super().__init__(
            "christov",
            "Christov 2004 combined adaptive threshold (py-ecg-detectors).",
        )


class PanTompkinsDetector(_EcgDetectorsWrapper):
    def __init__(self) -> None:
        super().__init__(
            "pan_tompkins",
            "Pan-Tompkins 1985 (py-ecg-detectors). Higher latency in real-time use.",
        )


class EngzeeDetector(_EcgDetectorsWrapper):
    def __init__(self) -> None:
        super().__init__(
            "engzee",
            "Engelse-Zeelenberg / Engzee single-scan (py-ecg-detectors).",
        )


class TwoAverageDetector(_EcgDetectorsWrapper):
    def __init__(self) -> None:
        super().__init__(
            "two_average",
            "Elgendi two-average detector (py-ecg-detectors).",
        )
