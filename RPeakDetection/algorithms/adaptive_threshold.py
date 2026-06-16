"""Current Layer 2 causal adaptive threshold detector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from RPeakDetection.types import RPeakDetectionResult

_ROOT = Path(__file__).resolve().parents[2]
_LAYER2 = _ROOT / "Layer2"
if str(_LAYER2) not in sys.path:
    sys.path.insert(0, str(_LAYER2))

from r_peak_detector import detect_r_peaks  # noqa: E402


class AdaptiveThresholdDetector:
    name = "adaptive_threshold"
    is_causal = True
    uses_prefilter = True

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        x = np.asarray(raw, dtype=float)
        out = detect_r_peaks(x, fs)
        return RPeakDetectionResult(
            algorithm=self.name,
            peak_samples=out.peak_samples,
            confirmation_samples=out.confirmation_samples,
            confirmation_delays_ms=out.confirmation_delays_ms.astype(float),
            polarity=out.polarity,
            uses_prefilter=True,
            is_causal=True,
            notes="Layer2 causal bandpass + amplitude/slope adaptive threshold.",
        )
