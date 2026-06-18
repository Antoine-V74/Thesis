"""Self-contained causal adaptive threshold R-peak detectors."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
from scipy.signal import butter, iirnotch, lfilter

from RPeakDetection.types import RPeakDetectionResult


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
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


LEGACY_CONFIG = AdaptiveThresholdConfig()

# The previous 90 ms refractory produced many duplicate detections on MIT-BIH.
# A longer refractory is closer to classic real-time QRS detector rules while
# still allowing very fast rhythms.
V2_CONFIG = replace(LEGACY_CONFIG, detector_refractory_ms=160.0)


class AdaptiveThresholdDetector:
    name = "adaptive_threshold"
    is_causal = True
    uses_prefilter = True

    def __init__(self, cfg: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.cfg = cfg or LEGACY_CONFIG

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        return _detect_adaptive_threshold(
            raw=np.asarray(raw, dtype=float),
            fs=float(fs),
            cfg=self.cfg,
            algorithm=self.name,
            notes=(
                "Causal bandpass plus adaptive amplitude/slope thresholds "
                "(legacy 90 ms refractory)."
            ),
        )


class AdaptiveThresholdV2Detector(AdaptiveThresholdDetector):
    name = "adaptive_threshold_v2"

    def __init__(self) -> None:
        super().__init__(V2_CONFIG)

    def detect(self, raw: np.ndarray, fs: float) -> RPeakDetectionResult:
        return _detect_adaptive_threshold(
            raw=np.asarray(raw, dtype=float),
            fs=float(fs),
            cfg=self.cfg,
            algorithm=self.name,
            notes=(
                "Causal adaptive amplitude/slope threshold with a 160 ms "
                "physiologic refractory period."
            ),
        )


def _detect_adaptive_threshold(
    raw: np.ndarray,
    fs: float,
    cfg: AdaptiveThresholdConfig,
    algorithm: str,
    notes: str,
) -> RPeakDetectionResult:
    x = np.where(np.isfinite(raw), raw, 0.0)
    if len(x) == 0:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return RPeakDetectionResult(
            algorithm=algorithm,
            peak_samples=empty_i,
            confirmation_samples=empty_i,
            confirmation_delays_ms=empty_f,
            polarity="positive",
            uses_prefilter=True,
            is_causal=True,
            notes=notes,
            meta={"config": cfg.__dict__},
        )

    filt = _causal_filter_ecg(x, fs)
    polarity = _select_polarity(filt, cfg, fs)
    sig = filt if polarity == "positive" else -filt

    n = len(sig)
    cal_n = min(n, max(1, int(round(cfg.calibration_s * fs))))
    refractory = max(1, int(round(cfg.detector_refractory_ms * fs / 1000.0)))
    min_hold = max(1, int(round(cfg.min_peak_hold_ms * fs / 1000.0)))
    descent_needed = max(1, int(round(cfg.descent_confirm_ms * fs / 1000.0)))
    max_width = max(min_hold, int(round(cfg.max_peak_width_ms * fs / 1000.0)))

    slope_win = max(1, int(round(cfg.slope_ma_ms * fs / 1000.0)))
    slope_ring = np.zeros(slope_win, dtype=float)
    slope_sum = 0.0
    slope_pos = 0

    init = sig[:cal_n]
    amp_noise, amp_signal = _initial_levels(init, cfg)
    slope_init = _initial_slope_envelope(init, slope_win)
    slope_noise, slope_signal = _initial_levels(slope_init, cfg)

    in_peak = False
    peak_start = -1
    peak_idx = -1
    peak_val = -np.inf
    falling_count = 0
    last_emit_idx = -10**12

    peaks: List[int] = []
    confirmations: List[int] = []

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

        amp_thr = _adaptive_threshold(amp_noise, amp_signal, cfg)
        slope_thr = _adaptive_slope_threshold(slope_noise, slope_signal, cfg)

        if i < cal_n:
            continue

        if i - last_emit_idx < refractory:
            if value < amp_thr:
                amp_noise = _ema(amp_noise, max(value, 0.0), cfg.noise_update_alpha)
                slope_noise = _ema(slope_noise, slope_env, cfg.slope_noise_update_alpha)
            continue

        enters_peak = value >= amp_thr and slope_env >= slope_thr and dx >= 0.0

        if not in_peak:
            if enters_peak:
                in_peak = True
                peak_start = i
                peak_idx = i
                peak_val = value
                falling_count = 0
            else:
                amp_noise = _ema(amp_noise, max(value, 0.0), cfg.noise_update_alpha)
                slope_noise = _ema(slope_noise, slope_env, cfg.slope_noise_update_alpha)
            continue

        if value > peak_val:
            peak_val = value
            peak_idx = i
            falling_count = 0
        elif i > peak_idx:
            falling_count += 1

        held_long_enough = i - peak_start >= min_hold
        dropped_enough = value <= peak_val * (1.0 - cfg.peak_drop_frac)
        descended_enough = falling_count >= descent_needed
        too_wide = i - peak_start >= max_width

        if held_long_enough and (dropped_enough or descended_enough or too_wide):
            peaks.append(int(peak_idx))
            confirmations.append(int(i))
            last_emit_idx = int(peak_idx)
            amp_signal = _ema(
                amp_signal,
                max(peak_val, cfg.min_threshold),
                cfg.signal_update_alpha,
            )
            slope_signal = _ema(
                slope_signal,
                max(slope_env, cfg.min_slope_threshold),
                cfg.slope_signal_update_alpha,
            )
            in_peak = False
            peak_start = -1
            peak_idx = -1
            peak_val = -np.inf
            falling_count = 0

    peak_arr = np.asarray(peaks, dtype=int)
    conf_arr = np.asarray(confirmations, dtype=int)
    delays = (conf_arr - peak_arr) * 1000.0 / fs
    return RPeakDetectionResult(
        algorithm=algorithm,
        peak_samples=peak_arr,
        confirmation_samples=conf_arr,
        confirmation_delays_ms=delays.astype(float),
        polarity=polarity,
        uses_prefilter=True,
        is_causal=True,
        notes=notes,
        meta={"config": cfg.__dict__},
    )


def _causal_filter_ecg(
    raw: np.ndarray,
    fs: float,
    low: float = 5.0,
    high: float = 20.0,
) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(1e-4, low / nyq)
    hi = min(0.499, min(high, 0.45 * fs) / nyq)
    b, a = butter(4, [lo, hi], btype="band")
    filt = lfilter(b, a, raw)

    for f0, min_fs in ((50.0, 110.0), (60.0, 125.0)):
        if fs <= min_fs or f0 >= 0.5 * fs:
            continue
        bn, an = iirnotch(f0 / nyq, 30.0)
        filt = lfilter(bn, an, filt)
    return filt


def _select_polarity(x: np.ndarray, cfg: AdaptiveThresholdConfig, fs: float) -> str:
    if cfg.polarity in ("positive", "negative"):
        return cfg.polarity
    if cfg.polarity != "auto":
        raise ValueError("polarity must be 'auto', 'positive', or 'negative'")
    n = min(len(x), max(1, int(round(cfg.calibration_s * fs))))
    head = x[:n]
    p95 = float(np.percentile(head, 95.0))
    p05 = float(np.percentile(head, 5.0))
    return "positive" if abs(p95) >= abs(p05) else "negative"


def _initial_levels(
    segment: np.ndarray,
    cfg: AdaptiveThresholdConfig,
) -> tuple[float, float]:
    seg = np.asarray(segment, dtype=float)
    seg = seg[np.isfinite(seg)]
    if len(seg) == 0:
        return cfg.min_threshold, 4.0 * cfg.min_threshold
    positive = np.maximum(seg, 0.0)
    noise = float(np.percentile(positive, 50.0))
    signal = float(np.percentile(positive, 95.0))
    if signal <= noise:
        signal = noise + cfg.min_threshold
    noise = max(noise, 0.25 * cfg.min_threshold)
    signal = max(signal, 4.0 * cfg.min_threshold)
    return noise, signal


def _initial_slope_envelope(segment: np.ndarray, win: int) -> np.ndarray:
    if len(segment) == 0:
        return np.array([], dtype=float)
    dx = np.abs(np.diff(segment, prepend=segment[0]))
    kernel = np.ones(max(1, win), dtype=float) / float(max(1, win))
    return np.convolve(dx, kernel, mode="full")[:len(dx)]


def _adaptive_threshold(noise: float, signal: float, cfg: AdaptiveThresholdConfig) -> float:
    thr = noise + cfg.threshold_frac * max(signal - noise, 0.0)
    return max(float(thr), cfg.min_threshold)


def _adaptive_slope_threshold(
    noise: float,
    signal: float,
    cfg: AdaptiveThresholdConfig,
) -> float:
    thr = noise + cfg.slope_threshold_frac * max(signal - noise, 0.0)
    return max(float(thr), cfg.min_slope_threshold)


def _ema(old: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * float(old) + alpha * float(new)
