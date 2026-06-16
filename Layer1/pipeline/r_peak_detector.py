"""
Fast causal adaptive-threshold R-peak detector.

This detector is designed for deployment-style Layer 1 timing:

    causal filtered ECG -> adaptive amplitude/slope threshold
    -> peak tracking -> early causal confirmation after descent

It does not use oracle annotations, centered windows, filtfilt, or future
samples for the peak decision.  In batch evaluation the first few seconds can
be used as a causal arming/calibration period; after that every emitted peak
has a separate confirmation sample that measures the real-time delay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class FastCausalThresholdConfig:
    """Configuration for the causal adaptive detector."""

    calibration_s: float = 2.0
    polarity: str = "auto"  # "auto", "positive", or "negative"

    threshold_frac: float = 0.35
    noise_update_alpha: float = 0.02
    signal_update_alpha: float = 0.125
    min_threshold: float = 0.03

    slope_ma_ms: float = 12.0
    slope_threshold_frac: float = 0.25
    slope_noise_update_alpha: float = 0.02
    slope_signal_update_alpha: float = 0.125
    min_slope_threshold: float = 1e-6

    detector_refractory_ms: float = 90.0
    min_peak_hold_ms: float = 4.0
    descent_confirm_ms: float = 4.0
    peak_drop_frac: float = 0.18
    max_peak_width_ms: float = 90.0


@dataclass
class FastCausalThresholdResult:
    """Detector output and diagnostics."""

    peak_samples: np.ndarray
    confirmation_samples: np.ndarray
    confirmation_delays_ms: np.ndarray
    thresholds: np.ndarray
    slope_thresholds: np.ndarray
    polarity: str


class FastCausalThresholdDetector:
    """
    Streaming-style adaptive R-peak detector.

    The detector enters a peak when both amplitude and a trailing slope
    envelope exceed adaptive thresholds.  It then tracks the maximum sample
    and emits that maximum as soon as the waveform has clearly descended,
    rather than waiting for a full low-threshold hysteresis exit.
    """

    def __init__(self, cfg: Optional[FastCausalThresholdConfig], fs: float):
        self.cfg = cfg or FastCausalThresholdConfig()
        self.fs = float(fs)

    def process_signal(self, x: np.ndarray) -> FastCausalThresholdResult:
        """Run the causal detector over a 1-D ECG signal."""
        raw = np.asarray(x, dtype=float)
        if len(raw) == 0:
            empty_i = np.array([], dtype=int)
            empty_f = np.array([], dtype=float)
            return FastCausalThresholdResult(
                empty_i, empty_i, empty_f, empty_f, empty_f, "positive",
            )

        polarity = self._select_polarity(raw)
        sig = raw if polarity == "positive" else -raw

        n = len(sig)
        cal_n = min(n, max(1, int(round(self.cfg.calibration_s * self.fs))))
        refractory = max(1, int(round(
            self.cfg.detector_refractory_ms * self.fs / 1000.0)))
        min_hold = max(1, int(round(
            self.cfg.min_peak_hold_ms * self.fs / 1000.0)))
        descent_needed = max(1, int(round(
            self.cfg.descent_confirm_ms * self.fs / 1000.0)))
        max_width = max(min_hold, int(round(
            self.cfg.max_peak_width_ms * self.fs / 1000.0)))

        slope_win = max(1, int(round(
            self.cfg.slope_ma_ms * self.fs / 1000.0)))
        slope_ring = np.zeros(slope_win, dtype=float)
        slope_sum = 0.0
        slope_pos = 0

        init = sig[:cal_n]
        amp_noise, amp_signal = self._initial_levels(init)
        slope_init = self._initial_slope_envelope(init, slope_win)
        slope_noise, slope_signal = self._initial_levels(slope_init)

        in_peak = False
        peak_start = -1
        peak_idx = -1
        peak_val = -np.inf
        falling_count = 0
        last_emit_idx = -10**12

        peaks: List[int] = []
        confirmations: List[int] = []
        thresholds = np.zeros(n, dtype=float)
        slope_thresholds = np.zeros(n, dtype=float)

        prev = sig[0]
        for i, value in enumerate(sig):
            dx = value - prev if i > 0 else 0.0
            dx_abs = abs(dx)
            prev = value

            slope_sum -= slope_ring[slope_pos]
            slope_ring[slope_pos] = dx_abs
            slope_sum += dx_abs
            slope_pos = (slope_pos + 1) % slope_win
            slope_env = slope_sum / float(slope_win)

            amp_thr = self._adaptive_threshold(amp_noise, amp_signal)
            slope_thr = self._adaptive_slope_threshold(slope_noise, slope_signal)
            thresholds[i] = amp_thr
            slope_thresholds[i] = slope_thr

            if i < cal_n:
                continue

            if i - last_emit_idx < refractory:
                if value < amp_thr:
                    amp_noise = self._ema(
                        amp_noise, max(value, 0.0), self.cfg.noise_update_alpha)
                    slope_noise = self._ema(
                        slope_noise, slope_env, self.cfg.slope_noise_update_alpha)
                continue

            # Requiring a rising sample prevents a second entry on the falling
            # limb of the same broad QRS complex when the refractory period is
            # intentionally short for high-heart-rate preparations.
            enters_peak = value >= amp_thr and slope_env >= slope_thr and dx >= 0.0

            if not in_peak:
                if enters_peak:
                    in_peak = True
                    peak_start = i
                    peak_idx = i
                    peak_val = value
                    falling_count = 0
                else:
                    amp_noise = self._ema(
                        amp_noise, max(value, 0.0), self.cfg.noise_update_alpha)
                    slope_noise = self._ema(
                        slope_noise, slope_env, self.cfg.slope_noise_update_alpha)
                continue

            if value > peak_val:
                peak_val = value
                peak_idx = i
                falling_count = 0
            elif i > peak_idx:
                falling_count += 1

            held_long_enough = i - peak_start >= min_hold
            dropped_enough = value <= peak_val * (1.0 - self.cfg.peak_drop_frac)
            descended_enough = falling_count >= descent_needed
            too_wide = i - peak_start >= max_width

            if held_long_enough and (dropped_enough or descended_enough or too_wide):
                peaks.append(int(peak_idx))
                confirmations.append(int(i))
                last_emit_idx = int(peak_idx)
                amp_signal = self._ema(
                    amp_signal, max(peak_val, self.cfg.min_threshold),
                    self.cfg.signal_update_alpha,
                )
                slope_signal = self._ema(
                    slope_signal, max(slope_env, self.cfg.min_slope_threshold),
                    self.cfg.slope_signal_update_alpha,
                )
                in_peak = False
                peak_start = -1
                peak_idx = -1
                peak_val = -np.inf
                falling_count = 0

        peak_arr = np.asarray(peaks, dtype=int)
        conf_arr = np.asarray(confirmations, dtype=int)
        delays = (conf_arr - peak_arr) * 1000.0 / self.fs
        return FastCausalThresholdResult(
            peak_samples=peak_arr,
            confirmation_samples=conf_arr,
            confirmation_delays_ms=delays.astype(float),
            thresholds=thresholds,
            slope_thresholds=slope_thresholds,
            polarity=polarity,
        )

    def _select_polarity(self, x: np.ndarray) -> str:
        if self.cfg.polarity in ("positive", "negative"):
            return self.cfg.polarity
        if self.cfg.polarity != "auto":
            raise ValueError("polarity must be 'auto', 'positive', or 'negative'")
        n = min(len(x), max(1, int(round(self.cfg.calibration_s * self.fs))))
        head = x[:n]
        p95 = float(np.percentile(head, 95.0))
        p05 = float(np.percentile(head, 5.0))
        return "positive" if abs(p95) >= abs(p05) else "negative"

    def _initial_levels(self, segment: np.ndarray) -> tuple[float, float]:
        seg = np.asarray(segment, dtype=float)
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            return self.cfg.min_threshold, 4.0 * self.cfg.min_threshold
        positive = np.maximum(seg, 0.0)
        noise = float(np.percentile(positive, 50.0))
        signal = float(np.percentile(positive, 95.0))
        if signal <= noise:
            signal = noise + self.cfg.min_threshold
        noise = max(noise, 0.25 * self.cfg.min_threshold)
        signal = max(signal, 4.0 * self.cfg.min_threshold)
        return noise, signal

    @staticmethod
    def _initial_slope_envelope(segment: np.ndarray, win: int) -> np.ndarray:
        if len(segment) == 0:
            return np.array([], dtype=float)
        dx = np.abs(np.diff(segment, prepend=segment[0]))
        kernel = np.ones(max(1, win), dtype=float) / float(max(1, win))
        # This is only for arming-period initialization, not runtime emission.
        return np.convolve(dx, kernel, mode="full")[:len(dx)]

    def _adaptive_threshold(self, noise: float, signal: float) -> float:
        thr = noise + self.cfg.threshold_frac * max(signal - noise, 0.0)
        return max(float(thr), self.cfg.min_threshold)

    def _adaptive_slope_threshold(self, noise: float, signal: float) -> float:
        thr = noise + self.cfg.slope_threshold_frac * max(signal - noise, 0.0)
        return max(float(thr), self.cfg.min_slope_threshold)

    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        return (1.0 - alpha) * float(old) + alpha * float(new)


def fast_causal_threshold_detect(
    x: np.ndarray,
    fs: float,
    cfg: Optional[FastCausalThresholdConfig] = None,
) -> FastCausalThresholdResult:
    """Convenience wrapper for one-shot batch evaluation."""
    return FastCausalThresholdDetector(cfg, fs).process_signal(x)


if __name__ == "__main__":
    fs = 360.0
    t = np.arange(int(20 * fs)) / fs
    ecg = 0.02 * np.sin(2.0 * np.pi * 0.3 * t)
    for beat_t in np.arange(0.5, 19.5, 0.8):
        ecg += np.exp(-((t - beat_t) ** 2) / (2 * 0.012 ** 2))
    result = fast_causal_threshold_detect(ecg, fs)
    print("fast_causal_threshold_detector smoke test")
    print(f"  polarity={result.polarity}")
    print(f"  peaks={len(result.peak_samples)}")
    if len(result.confirmation_delays_ms):
        print(f"  mean confirmation delay={np.mean(result.confirmation_delays_ms):.1f} ms")
