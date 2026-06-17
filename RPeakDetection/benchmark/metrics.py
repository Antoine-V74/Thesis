"""Detection metrics and timing summaries for R-peak benchmarks."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from RPeakDetection.types import RPeakDetectionResult


def greedy_match(
    ref_s: np.ndarray,
    det_s: np.ndarray,
    tol_s: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    pairs: List[Tuple[float, int, int]] = []
    for i, r in enumerate(ref_s):
        cand = np.where((det_s >= r - tol_s) & (det_s <= r + tol_s))[0]
        for j in cand:
            pairs.append((abs(float(det_s[j] - r)), i, int(j)))
    pairs.sort(key=lambda x: x[0])

    used_ref, used_det = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for dt, i, j in pairs:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        matches.append((i, j, float(det_s[j] - ref_s[i])))

    missed = [i for i in range(len(ref_s)) if i not in used_ref]
    extra = [j for j in range(len(det_s)) if j not in used_det]
    return matches, missed, extra


def summarize_detection(
    n_ref: int,
    n_det: int,
    n_matched: int,
    timing_errors_s: np.ndarray,
    confirmation_timing_errors_s: Optional[np.ndarray],
    duration_s: float,
    processing_ms: float,
    result: RPeakDetectionResult,
    fs: float,
) -> Dict:
    missed = n_ref - n_matched
    extra = n_det - n_matched
    sens = n_matched / n_ref if n_ref else float("nan")
    ppv = n_matched / n_det if n_det else float("nan")
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv > 0 else float("nan")
    hours = max(duration_s / 3600.0, 1e-9)

    timing_ms = timing_errors_s.astype(float) * 1000.0
    abs_timing_ms = np.abs(timing_ms)

    explicit_conf = result.confirmation_delays_ms
    has_explicit_confirmation = (
        result.is_causal
        and len(explicit_conf) > 0
        and float(np.nanmax(explicit_conf)) > 0.0
    )

    mean_confirmation_delay_ms = float("nan")
    median_confirmation_delay_ms = float("nan")
    p95_confirmation_delay_ms = float("nan")
    if has_explicit_confirmation:
        mean_confirmation_delay_ms = float(np.mean(explicit_conf))
        median_confirmation_delay_ms = float(np.median(explicit_conf))
        p95_confirmation_delay_ms = float(np.percentile(explicit_conf, 95))

    # For matched beats: lag relative to annotation (positive = detection after label).
    mean_detection_lag_ms = float(np.mean(timing_ms)) if len(timing_ms) else float("nan")
    median_detection_lag_ms = float(np.median(timing_ms)) if len(timing_ms) else float("nan")
    p95_detection_lag_ms = float(np.percentile(timing_ms, 95)) if len(timing_ms) else float("nan")

    mean_abs_timing_error_ms = float(np.mean(abs_timing_ms)) if len(abs_timing_ms) else float("nan")
    median_abs_timing_error_ms = float(np.median(abs_timing_ms)) if len(abs_timing_ms) else float("nan")
    p95_abs_timing_error_ms = (
        float(np.percentile(abs_timing_ms, 95)) if len(abs_timing_ms) else float("nan")
    )

    conf_timing_ms = np.array([], dtype=float)
    if (
        result.is_causal
        and has_explicit_confirmation
        and confirmation_timing_errors_s is not None
        and len(confirmation_timing_errors_s)
    ):
        conf_timing_ms = confirmation_timing_errors_s.astype(float) * 1000.0

    mean_confirmed_event_lag_ms = (
        float(np.mean(conf_timing_ms)) if len(conf_timing_ms) else float("nan")
    )
    median_confirmed_event_lag_ms = (
        float(np.median(conf_timing_ms)) if len(conf_timing_ms) else float("nan")
    )
    p95_confirmed_event_lag_ms = (
        float(np.percentile(conf_timing_ms, 95)) if len(conf_timing_ms) else float("nan")
    )
    mean_abs_confirmed_event_error_ms = (
        float(np.mean(np.abs(conf_timing_ms))) if len(conf_timing_ms) else float("nan")
    )

    n_samples = max(int(duration_s * float(fs)), 1)
    processing_ms_per_sample = processing_ms / n_samples
    processing_ms_per_second_signal = processing_ms / max(duration_s, 1e-9)

    return {
        "n_annotated_beats": n_ref,
        "n_detected_peaks": n_det,
        "n_matched": n_matched,
        "n_missed": missed,
        "n_extra": extra,
        "sensitivity": round(sens, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "extra_peaks_per_hour": round(extra / hours, 2),
        "has_explicit_confirmation": has_explicit_confirmation,
        "mean_confirmation_delay_ms": round(mean_confirmation_delay_ms, 2),
        "median_confirmation_delay_ms": round(median_confirmation_delay_ms, 2),
        "p95_confirmation_delay_ms": round(p95_confirmation_delay_ms, 2),
        "mean_detection_lag_ms": round(mean_detection_lag_ms, 2),
        "median_detection_lag_ms": round(median_detection_lag_ms, 2),
        "p95_detection_lag_ms": round(p95_detection_lag_ms, 2),
        "mean_confirmed_event_lag_ms": round(mean_confirmed_event_lag_ms, 2),
        "median_confirmed_event_lag_ms": round(median_confirmed_event_lag_ms, 2),
        "p95_confirmed_event_lag_ms": round(p95_confirmed_event_lag_ms, 2),
        "mean_abs_confirmed_event_error_ms": round(mean_abs_confirmed_event_error_ms, 2),
        "mean_abs_timing_error_ms": round(mean_abs_timing_error_ms, 2),
        "median_abs_timing_error_ms": round(median_abs_timing_error_ms, 2),
        "p95_abs_timing_error_ms": round(p95_abs_timing_error_ms, 2),
        "processing_ms": round(processing_ms, 2),
        "processing_ms_per_sample": round(processing_ms_per_sample, 6),
        "processing_ms_per_second_signal": round(processing_ms_per_second_signal, 4),
        "is_causal": result.is_causal,
        "uses_prefilter": result.uses_prefilter,
        "polarity": result.polarity,
    }
