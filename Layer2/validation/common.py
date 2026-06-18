"""
Shared Layer 2 validation helpers for beat-synchronous and cross-dataset runs.

This module intentionally contains no 5-second sliding-window validation loop.
It keeps reusable pieces that were formerly in the old window validator:
filtering, scoring wrappers, feature alignment, and calibrator fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_L2))
sys.path.insert(0, str(_DATA))

from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()

from decision import (  # noqa: E402
    BaselineCalibrator,
    DEFAULT_HARD_RULES,
    FROZEN_COUPLING_THRESHOLD,
    FROZEN_ZSCORE_QUANTILE,
    RRReliabilityConfig,
    check_rr_reliability,
)
from RPeakDetection.algorithms import get_detector  # noqa: E402


CALIBRATION_RECORDS = {"100", "101", "103", "115", "117", "121", "122"}
NORMAL_BEATS = {"N", "L", "R", "e", "j"}
ABNORMAL_BEATS = {"V", "F", "E", "/", "f", "!"}


def _bandpass(x: np.ndarray, fs: float, low: float = 5.0, high: float = 20.0) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)
    nyq = 0.5 * fs
    lo = max(1e-4, low / nyq)
    hi = min(0.499, min(high, 0.45 * fs) / nyq)
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def _notch(x: np.ndarray, fs: float, f0: float, q: float = 30.0) -> np.ndarray:
    if f0 >= fs / 2:
        return x
    b, a = iirnotch(f0 / (0.5 * fs), q)
    return filtfilt(b, a, x)


def apply_filters(raw: np.ndarray, fs: float) -> np.ndarray:
    filt = _bandpass(raw, fs)
    if fs > 110:
        filt = _notch(filt, fs, 50.0)
    if fs > 125:
        filt = _notch(filt, fs, 60.0)
    return filt


DEFAULT_CAUSAL_RPEAK_ALGORITHM = "adaptive_threshold_v2"


def rpeak_detector_peaks(
    raw: np.ndarray,
    fs: float,
    algorithm: str = DEFAULT_CAUSAL_RPEAK_ALGORITHM,
) -> np.ndarray:
    """R-peak timestamps in seconds from the RPeakDetection registry."""
    result = get_detector(algorithm).detect(np.asarray(raw, dtype=float), fs)
    return result.peak_times_s(fs)



def align_features_for_scoring(
    feats: Dict[str, float],
    calibrator: BaselineCalibrator,
) -> Dict[str, float]:
    """Align extracted features to calibrator schema, keeping hard-rule features."""
    aligned = {k: feats.get(k, float("nan")) for k in calibrator.feature_names}
    for k in calibrator.hard_rules:
        aligned[k] = feats.get(k, float("nan"))
    return aligned


def score_one(
    features: Dict[str, float],
    calibrator: Optional[BaselineCalibrator],
) -> Dict:
    """Score one feature dict with the standard decision gate."""
    _nan = float("nan")
    _empty = {"permit": False, "mahalanobis": _nan, "mahalanobis_threshold": _nan,
              "max_zscore": _nan, "zscore_threshold": _nan,
              "signal_mahal_proxy": _nan, "rr_mahal_proxy": _nan,
              "hard_rule_violated": "",
              "top1_feature": "", "top1_zscore": _nan,
              "top2_feature": "", "top2_zscore": _nan,
              "top3_feature": "", "top3_zscore": _nan}

    if calibrator is None:
        return {**_empty, "reason": "not_calibrated"}

    mahal_valid = all(
        np.isfinite(features.get(k, _nan))
        for k in calibrator.mahal_feature_names
    )
    if not mahal_valid:
        return {**_empty,
                "mahalanobis_threshold": calibrator.threshold_mahalanobis,
                "zscore_threshold": calibrator.threshold_max_zscore,
                "reason": "missing_features"}

    d = calibrator.decide(features)
    top = d["top_deviating_features"]

    def _top(i: int, key: str):
        return top[i][key] if i < len(top) else ("" if key == "feature" else _nan)

    return {
        "permit": bool(d["permit"]),
        "mahalanobis": float(d["mahalanobis"]),
        "mahalanobis_threshold": float(d["mahalanobis_threshold"]),
        "signal_mahal_proxy": float(d["signal_mahal_proxy"]),
        "rr_mahal_proxy": float(d["rr_mahal_proxy"]),
        "max_zscore": float(d["max_abs_zscore"]),
        "zscore_threshold": float(d["zscore_threshold"]),
        "reason": str(d["reason"]),
        "hard_rule_violated": str(d.get("hard_rule_violated") or ""),
        "top1_feature": _top(0, "feature"),
        "top1_zscore": float(_top(0, "zscore")),
        "top2_feature": _top(1, "feature"),
        "top2_zscore": float(_top(1, "zscore")),
        "top3_feature": _top(2, "feature"),
        "top3_zscore": float(_top(2, "zscore")),
    }


def score_one_hybrid(
    features: Dict[str, float],
    calibrator: Optional[BaselineCalibrator],
    n_beats_in_window: int,
    n_recent_clean_beats: int,
    reliability_config: Optional[RRReliabilityConfig] = None,
) -> Dict:
    """Score one feature dict with the hybrid reliability-aware gate."""
    _nan = float("nan")
    _empty = {
        "permit": False, "mahalanobis": _nan, "mahalanobis_threshold": _nan,
        "signal_mahal_proxy": _nan, "rr_mahal_proxy": _nan,
        "max_zscore": _nan, "zscore_threshold": _nan,
        "hard_rule_violated": "",
        "current_r_reliable": False, "rr_history_reliable": False,
        "n_recent_clean_beats": n_recent_clean_beats,
        "top1_feature": "", "top1_zscore": _nan,
        "top2_feature": "", "top2_zscore": _nan,
        "top3_feature": "", "top3_zscore": _nan,
    }

    if calibrator is None:
        return {**_empty, "reason": "not_calibrated"}

    mahal_valid = all(
        np.isfinite(features.get(k, _nan))
        for k in calibrator.mahal_feature_names
    )
    if not mahal_valid:
        return {**_empty,
                "mahalanobis_threshold": calibrator.threshold_mahalanobis,
                "zscore_threshold": calibrator.threshold_max_zscore,
                "reason": "missing_features"}

    current_r_reliable, rr_history_reliable = check_rr_reliability(
        features, n_beats_in_window, reliability_config
    )

    d = calibrator.decide_hybrid(
        features,
        current_r_reliable=current_r_reliable,
        rr_history_reliable=rr_history_reliable,
        n_recent_clean_beats=n_recent_clean_beats,
        warm_beats=(reliability_config.warm_beats
                    if reliability_config is not None else 5),
    )
    top = d["top_deviating_features"]

    def _top(i, key):
        return top[i][key] if i < len(top) else ("" if key == "feature" else _nan)

    return {
        "permit": bool(d["permit"]),
        "mahalanobis": float(d["mahalanobis"]),
        "mahalanobis_threshold": float(d["mahalanobis_threshold"]),
        "signal_mahal_proxy": float(d["signal_mahal_proxy"]),
        "rr_mahal_proxy": float(d["rr_mahal_proxy"]),
        "max_zscore": float(d["max_abs_zscore"]),
        "zscore_threshold": float(d["zscore_threshold"]),
        "reason": str(d["reason"]),
        "hard_rule_violated": str(d.get("hard_rule_violated") or ""),
        "current_r_reliable": bool(current_r_reliable),
        "rr_history_reliable": bool(rr_history_reliable),
        "n_recent_clean_beats": int(n_recent_clean_beats),
        "top1_feature": _top(0, "feature"),
        "top1_zscore": float(_top(0, "zscore")),
        "top2_feature": _top(1, "feature"),
        "top2_zscore": float(_top(1, "zscore")),
        "top3_feature": _top(2, "feature"),
        "top3_zscore": float(_top(2, "zscore")),
    }


def _select_finite_features(
    features_list: List[Dict[str, float]],
    min_finite_frac: float = 0.95,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """Return (feature_names, filled_features) with NaN-imputed median."""
    all_keys = list(features_list[0].keys())
    finite_frac = {
        k: float(np.mean([np.isfinite(f.get(k, float("nan"))) for f in features_list]))
        for k in all_keys
    }
    names = [k for k, frac in finite_frac.items() if frac >= min_finite_frac]
    for key in names:
        vals = [f[key] for f in features_list if np.isfinite(f.get(key, float("nan")))]
        med = float(np.median(vals)) if vals else 0.0
        for f in features_list:
            if not np.isfinite(f.get(key, float("nan"))):
                f[key] = med
    return names, features_list


def _fit_calibrator(
    features_list: List[Dict[str, float]],
    feature_names: List[str],
    threshold_quantile: float,
    feature_set: str = "all",
    zscore_quantile: float = FROZEN_ZSCORE_QUANTILE,
    coupling_threshold: float = FROZEN_COUPLING_THRESHOLD,
) -> BaselineCalibrator:
    """Fit a calibrator on a list of healthy feature dicts."""
    hard_rules = {k: list(v) for k, v in DEFAULT_HARD_RULES.items()}
    hard_rules["rr__beat_coupling_ratio"] = [coupling_threshold, None]
    return BaselineCalibrator().fit(
        features_list,
        feature_names=feature_names,
        threshold_quantile=threshold_quantile,
        zscore_quantile=zscore_quantile,
        use_robust=True,
        use_shrinkage=True,
        val_frac=0.3,
        use_default_hard_rules=(feature_set == "all"),
        hard_rules=hard_rules if feature_set == "all" else None,
        feature_set=feature_set,
    )
