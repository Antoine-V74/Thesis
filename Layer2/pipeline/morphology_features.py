"""
Beat-centered morphology features for PVC / aberrant-beat detection.

All features use prefix morph__ (added by full_features).

Designed for beat-synchronous gates: the window is centered on the trigger beat;
focus_peak_s should be the beat position relative to window start (typically
morphology_window_s / 2).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _beat_segment(
    window: np.ndarray,
    fs: float,
    peak_idx: int,
    half_width_s: float = 0.10,
) -> np.ndarray:
    """Extract a fixed-length segment around peak_idx (zero-padded if needed)."""
    half = max(2, int(round(half_width_s * fs)))
    target_len = 2 * half + 1
    n = len(window)
    lo = peak_idx - half
    hi = peak_idx + half + 1
    seg = np.zeros(target_len, dtype=float)
    src_lo = max(0, lo)
    src_hi = min(n, hi)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    if src_hi > src_lo:
        seg[dst_lo:dst_hi] = window[src_lo:src_hi]
    return seg


def _normalize_segment(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=float)
    s = float(np.std(seg))
    if s < 1e-9:
        return seg - float(np.mean(seg))
    return (seg - float(np.mean(seg))) / s


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_segment(a)
    b = _normalize_segment(b)
    if len(a) != len(b) or len(a) < 4:
        return float("nan")
    c = float(np.dot(a, b) / len(a))
    return float(np.clip(c, -1.0, 1.0))


def _qrs_width_ms(seg: np.ndarray, fs: float) -> float:
    """Full-width at half-maximum of |segment| in milliseconds."""
    if len(seg) < 4 or fs <= 0:
        return float("nan")
    env = np.abs(seg)
    peak = float(np.max(env))
    if peak < 1e-9:
        return float("nan")
    thr = 0.5 * peak
    above = env >= thr
    if not np.any(above):
        return float("nan")
    idx = np.where(above)[0]
    return float((idx[-1] - idx[0] + 1) / fs * 1000.0)


def morphology_features(
    window: np.ndarray,
    fs: float,
    r_peaks_s: Optional[np.ndarray] = None,
    focus_peak_s: Optional[float] = None,
    half_width_s: float = 0.10,
) -> Dict[str, float]:
    """
    Morphology features for one beat in a local ECG window.

    Parameters
    ----------
    window : filtered ECG snippet
    fs : Hz
    r_peaks_s : peak times (s) relative to window start
    focus_peak_s : trigger beat time relative to window start; default = window centre
    """
    x = np.asarray(window, dtype=float)
    n = len(x)
    _nan = float("nan")
    empty = {
        "template_corr": _nan,
        "neighbor_corr": _nan,
        "qrs_width_ms": _nan,
        "beat_amp": _nan,
        "amp_vs_median": _nan,
        "post_pre_area_ratio": _nan,
    }
    if n < 8 or not np.isfinite(fs) or fs <= 0:
        return empty

    if focus_peak_s is None:
        focus_peak_s = (n / 2.0) / fs
    focus_idx = int(round(float(focus_peak_s) * fs))
    focus_idx = max(0, min(n - 1, focus_idx))

    seg_focus = _beat_segment(x, fs, focus_idx, half_width_s)
    empty["qrs_width_ms"] = _qrs_width_ms(seg_focus, fs)
    empty["beat_amp"] = float(np.max(np.abs(seg_focus)))

    # Peaks in window
    peak_indices = []
    if r_peaks_s is not None:
        for t in np.asarray(r_peaks_s, dtype=float).ravel():
            if np.isfinite(t):
                pi = int(round(t * fs))
                if 0 <= pi < n:
                    peak_indices.append(pi)
    if not peak_indices:
        peak_indices = [focus_idx]

    segments = [_beat_segment(x, fs, pi, half_width_s) for pi in peak_indices]
    amps = [float(np.max(np.abs(s))) for s in segments]
    med_amp = float(np.median(amps)) if amps else _nan
    empty["amp_vs_median"] = (
        float(empty["beat_amp"] / med_amp) if med_amp > 1e-9 else _nan
    )

    # Template = pointwise median across all detected beats in window
    if len(segments) >= 2:
        stack = np.stack(segments, axis=0)
        template = np.median(stack, axis=0)
        empty["template_corr"] = _pearson_corr(seg_focus, template)
        others = [
            _pearson_corr(seg_focus, s)
            for i, s in enumerate(segments)
            if peak_indices[i] != focus_idx
        ]
        others = [c for c in others if np.isfinite(c)]
        empty["neighbor_corr"] = float(np.median(others)) if others else _nan
    elif len(segments) == 1:
        empty["template_corr"] = 1.0
        empty["neighbor_corr"] = _nan

    # Area asymmetry: energy after peak vs before (wide QRS / aberrant shape)
    half = len(seg_focus) // 2
    pre = float(np.sum(seg_focus[:half] ** 2))
    post = float(np.sum(seg_focus[half:] ** 2))
    empty["post_pre_area_ratio"] = (
        float(post / pre) if pre > 1e-12 else _nan
    )

    return empty
