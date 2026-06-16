"""Shared types for R-peak detection comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RPeakDetectionResult:
    """Uniform output from every detector implementation."""

    algorithm: str
    peak_samples: np.ndarray
    confirmation_samples: np.ndarray
    confirmation_delays_ms: np.ndarray
    polarity: str = "unknown"
    uses_prefilter: bool = False
    is_causal: bool = False
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_peaks(self) -> int:
        return int(len(self.peak_samples))

    def peak_times_s(self, fs: float) -> np.ndarray:
        return self.peak_samples.astype(float) / float(fs)

    def confirmation_times_s(self, fs: float) -> np.ndarray:
        return self.confirmation_samples.astype(float) / float(fs)
