"""Causal AMPT-style detector for mobile/real-time QRS benchmarking.

This implements the core idea of the AccYouRate Modified Pan-Tompkins (AMPT)
baseline: Pan-Tompkins-like preprocessing with a simplified single-threshold
decision rule suitable for streaming use. It is not an exact reproduction of a
vendor implementation; it is a transparent causal baseline for this repository.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.signal import butter, lfilter

from RPeakDetection.types import RPeakDetectionResult


class AmptDetector:
    name = "ampt"
    is_causal = True
    uses_prefilter = True

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        x = np.asarray(raw, dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        if len(x) == 0:
            empty_i = np.asarray([], dtype=int)
            empty_f = np.asarray([], dtype=float)
            return RPeakDetectionResult(
                algorithm=self.name,
                peak_samples=empty_i,
                confirmation_samples=empty_i,
                confirmation_delays_ms=empty_f,
                polarity="absolute",
                uses_prefilter=True,
                is_causal=True,
                notes="Causal AMPT-style simplified Pan-Tompkins baseline.",
            )

        fs = float(fs)
        filt = _causal_bandpass(x, fs)
        slope = _causal_derivative(filt, fs)
        energy = slope * slope
        integrated = _causal_moving_average(energy, max(1, int(round(0.150 * fs))))

        cal_n = min(len(integrated), max(1, int(round(2.0 * fs))))
        init = integrated[:cal_n]
        noise_level = float(np.percentile(init, 50.0))
        signal_level = float(np.percentile(init, 95.0))
        if signal_level <= noise_level:
            signal_level = noise_level + max(float(np.max(init)), 1e-9)

        refractory = max(1, int(round(0.200 * fs)))
        descent_needed = max(1, int(round(0.030 * fs)))
        max_width = max(descent_needed, int(round(0.220 * fs)))
        r_search = max(1, int(round(0.180 * fs)))

        in_peak = False
        peak_start = -1
        peak_idx = -1
        peak_val = -np.inf
        falling_count = 0
        last_emit_idx = -10**12

        peaks: List[int] = []
        confirmations: List[int] = []

        prev_y = integrated[0]
        for i, y in enumerate(integrated):
            threshold = noise_level + 0.25 * max(signal_level - noise_level, 0.0)
            threshold = max(threshold, 1e-12)

            if i < cal_n:
                prev_y = y
                continue

            if i - last_emit_idx < refractory:
                noise_level = _ema(noise_level, y, 0.02)
                prev_y = y
                continue

            if not in_peak:
                if y >= threshold and y >= prev_y:
                    in_peak = True
                    peak_start = i
                    peak_idx = i
                    peak_val = y
                    falling_count = 0
                else:
                    noise_level = _ema(noise_level, y, 0.02)
                prev_y = y
                continue

            if y > peak_val:
                peak_val = y
                peak_idx = i
                falling_count = 0
            elif y < prev_y:
                falling_count += 1

            too_wide = i - peak_start >= max_width
            descended = falling_count >= descent_needed
            if descended or too_wide:
                search_start = max(last_emit_idx + refractory, i - r_search)
                search_end = min(len(filt), i + 1)
                if search_start < search_end:
                    segment = filt[search_start:search_end]
                    r_idx = int(search_start + np.argmax(np.abs(segment)))
                    if r_idx - last_emit_idx >= refractory:
                        peaks.append(r_idx)
                        confirmations.append(i)
                        last_emit_idx = r_idx
                        signal_level = _ema(signal_level, peak_val, 0.125)
                    else:
                        noise_level = _ema(noise_level, peak_val, 0.125)

                in_peak = False
                peak_start = -1
                peak_idx = -1
                peak_val = -np.inf
                falling_count = 0

            prev_y = y

        peak_arr = np.asarray(peaks, dtype=int)
        conf_arr = np.asarray(confirmations, dtype=int)
        delays = (conf_arr - peak_arr) * 1000.0 / fs
        return RPeakDetectionResult(
            algorithm=self.name,
            peak_samples=peak_arr,
            confirmation_samples=conf_arr,
            confirmation_delays_ms=delays.astype(float),
            polarity="absolute",
            uses_prefilter=True,
            is_causal=True,
            notes=(
                "Causal AMPT-style simplified Pan-Tompkins baseline "
                "(Neri et al., Sensors 2023)."
            ),
        )


def _causal_bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(1e-4, 5.0 / nyq)
    high = min(0.499, min(20.0, 0.45 * fs) / nyq)
    b, a = butter(2, [low, high], btype="band")
    return lfilter(b, a, x)


def _causal_derivative(x: np.ndarray, fs: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if len(x) < 5:
        return y
    scale = fs / 8.0
    for i in range(4, len(x)):
        y[i] = (2.0 * x[i] + x[i - 1] - x[i - 3] - 2.0 * x[i - 4]) * scale
    return y


def _causal_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    ring = np.zeros(max(1, win), dtype=float)
    total = 0.0
    pos = 0
    for i, value in enumerate(x):
        total -= ring[pos]
        ring[pos] = value
        total += value
        pos = (pos + 1) % len(ring)
        y[i] = total / float(len(ring))
    return y


def _ema(old: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * float(old) + alpha * float(new)
