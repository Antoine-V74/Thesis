"""
Cross-dataset Layer 1+2 benchmark.

Tests the frozen MIT-BIH-tuned gate on external ECG datasets.

Two modes
---------
zero_shot
    Per-record healthy calibration only.
    No abnormal labels used from the external dataset.
    Hard-rule thresholds and gate structure frozen from MIT-BIH development.
    This is the valid generalization claim for the thesis.

diagnostic_retuned
    Allows per-record Mahalanobis threshold tuning using external abnormal labels.
    Equivalent to running with --abnormal-target-inhibit on the external dataset.
    Reports the feature-space upper bound: "could Layer 2 separate this dataset
    at all if optimally tuned?" Do NOT call this generalization.

Datasets supported
------------------
mitdb   : MIT-BIH Arrhythmia (dev set, included for reference)
nstdb   : MIT-BIH Noise Stress Test (noise robustness)
svdb    : Supraventricular Arrhythmia DB (irregular rhythm / SVT stress)
incartdb: St Petersburg INCART 12-lead (domain / morphology shift)
cudb    : Creighton University VF DB (extreme arrhythmia / VF)
vfdb    : MIT-BIH VF DB (VF episodes)

Usage
-----
    # Zero-shot only (main result)
    .venv\\Scripts\\python Layer2\\validation\\run_cross_dataset_validation.py ^
        --data-dir data ^
        --out-dir Results/cross_dataset ^
        --datasets mitdb nstdb svdb incartdb ^
        --mode zero_shot

    # Both modes
    .venv\\Scripts\\python Layer2\\validation\\run_cross_dataset_validation.py ^
        --data-dir data ^
        --out-dir Results/cross_dataset ^
        --datasets mitdb nstdb svdb incartdb ^
        --mode both

    # Quick smoke test (limit beats per record)
    .venv\\Scripts\\python Layer2\\validation\\run_cross_dataset_validation.py ^
        --data-dir data --out-dir Results/cross_dataset_quick ^
        --datasets nstdb svdb --mode zero_shot --window-limit 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, iirnotch, lfilter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_L2))
sys.path.insert(0, str(_DATA))
from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()

from full_features import full_features
from decision import BaselineCalibrator, FROZEN_COUPLING_THRESHOLD
from stimulation_cadence import ProspectiveCadenceGate
from common import DEFAULT_CAUSAL_RPEAK_ALGORITHM, apply_filters, score_one, score_one_hybrid
from common import _fit_calibrator, _select_finite_features, align_features_for_scoring
from RPeakDetection.algorithms import get_detector

try:
    import wfdb
except ImportError:
    sys.exit("Missing: pip install wfdb")


# ---------------------------------------------------------------------------
# Dataset registry (data/dataset_registry.py)
# ---------------------------------------------------------------------------

from dataset_registry import (  # noqa: E402
    DATASET_ABNORMAL,
    DATASET_ANN_EXT,
    DATASET_CHANNEL,
    DATASET_NORMAL,
    dataset_dir,
    resolve_dataset,
)

# Beats that are irregular but NOT ventricular (SVT, AF, junctional).
# These test false-inhibit risk on non-dangerous arrhythmias.
DATASET_SVT: Dict[str, frozenset] = {
    "mitdb":    frozenset({"A", "a", "J", "S"}),
    "nstdb":    frozenset({"A", "a", "J", "S"}),
    "svdb":     frozenset({"A", "a", "J", "S"}),
    "incartdb": frozenset({"A", "a", "J", "S"}),
    "cudb":     frozenset({"A", "a", "J", "S"}),
    "vfdb":     frozenset({"A", "a", "J", "S"}),
    "ltafdb":   frozenset({"A", "a", "J", "S"}),
}

# NSTDB SNR levels encoded in filename (dB)
NSTDB_SNR: Dict[str, Optional[int]] = {
    "118e_6": -6, "119e_6": -6,
    "118e00": 0,  "119e00": 0,
    "118e06": 6,  "119e06": 6,
    "118e12": 12, "119e12": 12,
    "118e18": 18, "119e18": 18,
    "118e24": 24, "119e24": 24,
}

# Group label for reporting
DATASET_GROUP: Dict[str, str] = {
    "mitdb":    "dev_set",
    "nstdb":    "noise_robustness",
    "svdb":     "svt_irregular",
    "incartdb": "domain_shift",
    "cudb":     "vf_extreme",
    "vfdb":     "vf_extreme",
    "ltafdb":   "af_dominant",
}


# ---------------------------------------------------------------------------
# Beat label helper
# ---------------------------------------------------------------------------

def _causal_bandpass(x: np.ndarray, fs: float, low: float = 5.0, high: float = 20.0) -> np.ndarray:
    x = np.where(np.isfinite(np.asarray(x, dtype=float)), x, 0.0)
    nyq = 0.5 * fs
    lo = max(1e-4, low / nyq)
    hi = min(0.499, min(high, 0.45 * fs) / nyq)
    b, a = butter(2, [lo, hi], btype="band")
    return lfilter(b, a, x)


def _causal_notch(x: np.ndarray, fs: float, f0: float, q: float = 30.0) -> np.ndarray:
    if f0 >= 0.5 * fs:
        return x
    b, a = iirnotch(f0 / (0.5 * fs), q)
    return lfilter(b, a, x)


def fast_causal_r_peaks(raw: np.ndarray, fs: float) -> np.ndarray:
    """Causal-filtered fast detector peaks in seconds (no RR supervisor)."""
    result = get_detector(DEFAULT_CAUSAL_RPEAK_ALGORITHM).detect(np.asarray(raw, dtype=float), fs)
    return result.peak_times_s(fs)


def fast_causal_r_peak_events(
    raw: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return detector peak estimates, confirmation times, and confirmation delays."""
    result = get_detector(DEFAULT_CAUSAL_RPEAK_ALGORITHM).detect(np.asarray(raw, dtype=float), fs)
    peak_s = result.peak_times_s(fs)
    confirmation_s = result.confirmation_times_s(fs)
    delay_ms = np.asarray(result.confirmation_delays_ms, dtype=float)
    return peak_s, confirmation_s, delay_ms


def fast_same_beat_ok(
    beat_time_s: float,
    peaks_all_s: np.ndarray,
    min_rr_s: float = 0.20,
) -> Tuple[bool, str, float]:
    """
    Minimal same-beat veto available immediately at the current R trigger.

    Full morphology is deliberately excluded here.  This check only uses the
    current trigger time and previous R peaks, so it can protect the current
    stimulation opportunity without waiting for the QRS tail.
    """
    prev_peaks = np.asarray(peaks_all_s, dtype=float)
    prev_peaks = prev_peaks[prev_peaks < (beat_time_s - 0.003)]
    if len(prev_peaks) == 0:
        return False, "no_previous_r_peak", float("nan")

    current_rr_s = float(beat_time_s - prev_peaks[-1])
    if current_rr_s < min_rr_s:
        return False, "same_beat_rr_too_short", float("nan")

    if len(prev_peaks) >= 4:
        recent_rrs = np.diff(prev_peaks[-min(20, len(prev_peaks)):])
        valid_rrs = recent_rrs[(recent_rrs > min_rr_s) & (recent_rrs < 3.0)]
        if len(valid_rrs) >= 3:
            median_rr = float(np.median(valid_rrs))
            if median_rr > min_rr_s:
                coupling = current_rr_s / median_rr
                if coupling < FROZEN_COUPLING_THRESHOLD:
                    return False, "same_beat_coupling_veto", float(coupling)
                return True, "same_beat_ok", float(coupling)

    return True, "same_beat_ok_no_coupling", float("nan")


def context_peaks_with_candidate(
    context_peaks_s: np.ndarray,
    candidate_s: float,
    replace_tolerance_s: float = 0.120,
) -> np.ndarray:
    """
    Use fast-causal peaks as RR context, but align the current scored beat to
    the high-PPV stimulation candidate.

    If a fast-causal trigger lies close to the adaptive candidate, replace it
    with the candidate timestamp.  Keeping both would create a false short RR.
    """
    peaks = np.asarray(context_peaks_s, dtype=float)
    keep = np.abs(peaks - float(candidate_s)) > replace_tolerance_s
    merged = np.sort(np.concatenate([peaks[keep], np.array([float(candidate_s)])]))
    return merged


def _persistent_risk_state_update(
    full_score: Dict[str, object],
    previous_soft_failures: int,
    soft_failure_persistence: int = 2,
) -> Dict[str, object]:
    """
    Convert the completed beat's full Layer 2 result into an ongoing risk state.

    Isolated morphology / distribution outliers are treated as beat-local unless
    they persist. Signal-quality and RR-history hard rules are allowed to carry
    over immediately because they are more likely to contaminate the next beat.
    """
    if bool(full_score.get("permit", False)):
        return {
            "active": False,
            "reason": "state_within_baseline",
            "soft_failures": 0,
            "source": "permit",
        }

    reason = str(full_score.get("reason", ""))
    violated = str(full_score.get("hard_rule_violated", "") or "")
    is_persistent_hard_rule = (
        reason == "hard_rule"
        and (
            violated.startswith("signal__")
            or violated.startswith("rr__rr_count")
            or violated.startswith("rr__short_rr_fraction")
            or violated.startswith("rr__long_rr_fraction")
        )
    )
    if is_persistent_hard_rule:
        return {
            "active": True,
            "reason": f"persistent_{violated}",
            "soft_failures": soft_failure_persistence,
            "source": "persistent_hard_rule",
        }

    soft_failures = previous_soft_failures + 1
    active = soft_failures >= soft_failure_persistence
    return {
        "active": active,
        "reason": (
            f"persistent_soft_layer2_failures_{soft_failures}"
            if active else f"isolated_layer2_failure_{reason}"
        ),
        "soft_failures": soft_failures,
        "source": reason or "layer2_failure",
    }


def causal_layer2_filter(raw: np.ndarray, fs: float) -> np.ndarray:
    """Causal ECG filter used for deployment-style Layer 2 feature extraction."""
    if not _SCIPY_OK:
        return np.asarray(raw, dtype=float)
    filt_causal = _causal_bandpass(raw, fs)
    if fs > 110:
        filt_causal = _causal_notch(filt_causal, fs, 50.0)
    if fs > 125:
        filt_causal = _causal_notch(filt_causal, fs, 60.0)
    return filt_causal

def beat_label(symbol: str, dataset: str) -> str:
    """Map a WFDB beat symbol to healthy / abnormal_v / svt / noise / mixed."""
    normal = DATASET_NORMAL.get(dataset, frozenset())
    abnormal = DATASET_ABNORMAL.get(dataset, frozenset())
    svt = DATASET_SVT.get(dataset, frozenset())
    if symbol in normal:
        return "healthy"
    if symbol in abnormal:
        return "abnormal_v"
    if symbol in svt:
        return "svt"
    return "mixed"


# ---------------------------------------------------------------------------
# Feature extraction (same as run_beat_validation)
# ---------------------------------------------------------------------------

def extract_beat_features(
    filt: np.ndarray,
    fs: float,
    beat_time_s: float,
    peaks_all_s: np.ndarray,
    morphology_window_s: float = 5.0,
    rr_lookback_s: float = 30.0,
    species: str = "human",
    raw: Optional[np.ndarray] = None,
    feature_window_mode: str = "centered",
    post_r_lookahead_s: float = 0.08,
) -> Tuple[Dict[str, float], int]:
    """
    raw : optional unfiltered ECG (same length as filt).
          When provided, the pre-filter HF noise ratio is added to the feature
          dict as signal__raw_hf_noise_ratio and used as the primary SQI gate.
          Values: clean ECG ≈ 0.03-0.08; NSTDB SNR+12≈0.05; SNR+6≈0.16; SNR0≈0.40.
    """
    if feature_window_mode == "centered":
        half = morphology_window_s / 2.0
        start_s = max(0.0, beat_time_s - half)
        end_s = min(len(filt) / fs, beat_time_s + half)
        visible_peak_cutoff_s = end_s
    elif feature_window_mode == "causal":
        start_s = max(0.0, beat_time_s - morphology_window_s)
        end_s = min(len(filt) / fs, beat_time_s + post_r_lookahead_s)
        # In deployment the current trigger and past peaks are known; future
        # peaks inside the short post-R lookahead are not available.
        visible_peak_cutoff_s = beat_time_s
    else:
        raise ValueError("feature_window_mode must be 'centered' or 'causal'")

    start = int(start_s * fs)
    end = max(start + 1, int(end_s * fs))
    w = filt[start:end]

    rr_lb = max(0.0, beat_time_s - rr_lookback_s)
    rr_peaks = peaks_all_s[(peaks_all_s >= rr_lb) & (peaks_all_s <= beat_time_s)]
    rr_rel = rr_peaks - start_s

    win_peaks = peaks_all_s[(peaks_all_s >= start_s) & (peaks_all_s <= visible_peak_cutoff_s)]
    n_beats_win = len(win_peaks)

    focus_peak_s = beat_time_s - start_s
    feats, _ = full_features(
        w, r_peaks_s=rr_rel, fs=fs, species=species,
        compute_spectral_hrv=False, compute_entropy=False,
        focus_peak_s=focus_peak_s,
    )

    # Per-beat coupling ratio (catches isolated PVCs).
    # Exclude peaks within 3 ms of beat_time_s to avoid millisecond-rounding artefacts.
    prev_peaks = peaks_all_s[peaks_all_s < (beat_time_s - 0.003)]
    if len(prev_peaks) >= 4:
        current_rr_s = float(beat_time_s - prev_peaks[-1])
        recent_rrs = np.diff(prev_peaks[-min(20, len(prev_peaks)):])
        valid_rrs = recent_rrs[(recent_rrs > 0.20) & (recent_rrs < 3.0)]
        if len(valid_rrs) >= 3 and current_rr_s > 0.20:
            median_rr = float(np.median(valid_rrs))
            if median_rr > 0.20:
                feats["rr__beat_coupling_ratio"] = current_rr_s / median_rr

    # Raw-signal SQI gate: HF noise fraction BEFORE the bandpass filter.
    # This is the correct SQI metric: on the filtered signal, HF noise is already
    # removed by design, so computing it there is meaningless.
    # signal__raw_hf_noise_ratio = power(>40 Hz) / total_power in unfiltered ECG.
    # Clean MIT-BIH: 0.03-0.08.  NSTDB SNR+12: ~0.05.  SNR+6: ~0.16.  SNR0: ~0.40.
    if raw is not None and len(raw) > start:
        w_raw = raw[start:min(end, len(raw))]
        n_r = len(w_raw)
        if n_r >= 32:
            freqs_r = np.fft.rfftfreq(n_r, d=1.0 / fs)
            psd_r = np.abs(np.fft.rfft(w_raw - w_raw.mean())) ** 2
            total_r = float(np.sum(psd_r)) + 1e-12
            feats["signal__raw_hf_noise_ratio"] = float(
                np.sum(psd_r[freqs_r >= 40.0])
            ) / total_r

    return feats, n_beats_win


# ---------------------------------------------------------------------------
# Classify inhibit reason
# ---------------------------------------------------------------------------

def _classify_inhibit(reason: str) -> str:
    if "hard_rule" in reason:
        return "hard_rule"
    if "mahalanobis" in reason or "zscore" in reason or "signal" in reason:
        return "morphology" if "signal" in reason else "threshold_other"
    if "rr" in reason:
        return "rr"
    return "technical"


# ---------------------------------------------------------------------------
# Single-record evaluation
# ---------------------------------------------------------------------------

def evaluate_record(
    stem: str,
    dataset: str,
    channel: int,
    ann_ext: str,
    mode: str,                      # "zero_shot" or "diagnostic_retuned"
    feature_sets: List[str],
    cal_frac: float,
    threshold_quantile: float,
    morphology_window_s: float,
    rr_lookback_s: float,
    species: str,
    abnormal_target: float,         # only used in diagnostic_retuned
    max_healthy_fi: float,
    window_limit: Optional[int],
    include_adaptive: bool,
    feature_window_mode: str,
    post_r_lookahead_s: float,
    cadence_observation_lookahead_s: float,
    cadence_min_safe_observations: int,
    cadence_require_last_observation_safe: bool,
) -> List[Dict]:
    try:
        rec = wfdb.rdrecord(stem)
        ann = wfdb.rdann(stem, ann_ext)
    except Exception as exc:
        logging.warning(f"  skip {Path(stem).name}: {exc}")
        return []

    n_ch = rec.p_signal.shape[1]
    ch = min(channel, n_ch - 1)
    raw = rec.p_signal[:, ch].astype(float)
    fs = float(rec.fs)
    filt = causal_layer2_filter(raw, fs) if feature_window_mode == "causal" else apply_filters(raw, fs)

    all_peaks = np.array([
        s / fs for s, sym in zip(ann.sample, ann.symbol)
        if sym in (DATASET_NORMAL[dataset] | DATASET_ABNORMAL[dataset])
    ], dtype=float)

    # RPeakDetection trigger streams.
    ad_peaks = np.array([], dtype=float)
    fast_peaks, fast_confirmations, fast_confirmation_delays_ms = (
        fast_causal_r_peak_events(raw, fs)
    )

    # --- Calibration from first cal_frac of healthy annotated beats ---
    healthy_times = sorted(
        s / fs for s, sym in zip(ann.sample, ann.symbol)
        if beat_label(sym, dataset) == "healthy"
    )
    n_cal = max(5, int(len(healthy_times) * cal_frac))
    cal_feats: List[Dict] = []
    for t_beat in healthy_times[:n_cal]:
        feats, _ = extract_beat_features(
            filt, fs, t_beat, all_peaks,
            morphology_window_s, rr_lookback_s, species,
            raw=raw,
            feature_window_mode=feature_window_mode,
            post_r_lookahead_s=post_r_lookahead_s,
        )
        cal_feats.append(feats)

    if len(cal_feats) < 5:
        logging.warning(f"  {Path(stem).name}: <5 healthy calibration beats, skip")
        return []

    fn, cal_feats = _select_finite_features(cal_feats)
    cal_end_t = healthy_times[min(n_cal, len(healthy_times)) - 1]

    # Abnormal features for diagnostic_retuned mode (gathered AFTER cal_end_t)
    abn_feats: List[Dict] = []
    h_val_feats: List[Dict] = []
    if mode == "diagnostic_retuned":
        for s, sym in zip(ann.sample, ann.symbol):
            t_b = s / fs
            if t_b <= cal_end_t:
                continue
            lbl = beat_label(sym, dataset)
            if lbl == "abnormal_v":
                f, _ = extract_beat_features(filt, fs, t_b, all_peaks,
                                             morphology_window_s, rr_lookback_s, species,
                                             raw=raw,
                                             feature_window_mode=feature_window_mode,
                                             post_r_lookahead_s=post_r_lookahead_s)
                abn_feats.append(f)
        for t_b in healthy_times[n_cal:]:
            f, _ = extract_beat_features(filt, fs, t_b, all_peaks,
                                         morphology_window_s, rr_lookback_s, species,
                                         raw=raw,
                                         feature_window_mode=feature_window_mode,
                                         post_r_lookahead_s=post_r_lookahead_s)
            h_val_feats.append(f)

    # Fit calibrators per feature set
    calibrators: Dict[str, BaselineCalibrator] = {}
    for fset in feature_sets:
        cal = _fit_calibrator(cal_feats, fn, threshold_quantile, fset)
        if mode == "diagnostic_retuned" and len(abn_feats) >= 5:
            abn_aligned = [{k: f.get(k, float("nan")) for k in cal.feature_names}
                           for f in abn_feats]
            h_aligned = [{k: f.get(k, float("nan")) for k in cal.feature_names}
                         for f in h_val_feats] if h_val_feats else None
            cal.calibrate_thresholds_for_abnormal_inhibit(
                abn_aligned,
                target_inhibit_rate=abnormal_target,
                healthy_validation_features=h_aligned,
                max_healthy_false_inhibit_rate=max_healthy_fi,
            )
        calibrators[fset] = cal

    # Tolerance for L1 peak → annotation label matching
    tol_s = 0.080

    def _reference_at(
        t_s: float,
        tol_override_s: Optional[float] = None,
    ) -> Tuple[str, str, float, float, bool]:
        local_tol_s = tol_s if tol_override_s is None else tol_override_s
        best_sym, best_lbl, best_dt = "?", "mixed", tol_s + 1
        best_ref_s = float("nan")
        for s, sym in zip(ann.sample, ann.symbol):
            ref_s = s / fs
            dt = abs(ref_s - t_s)
            if dt < best_dt:
                best_dt = dt
                best_ref_s = ref_s
                best_sym = sym
                best_lbl = beat_label(sym, dataset)
        signed_error_ms = (
            float((t_s - best_ref_s) * 1000.0)
            if np.isfinite(best_ref_s) else float("nan")
        )
        abs_error_ms = abs(signed_error_ms) if np.isfinite(signed_error_ms) else float("nan")
        if best_dt > local_tol_s:
            return "?", "mixed", signed_error_ms, abs_error_ms, False
        return best_sym, best_lbl, signed_error_ms, abs_error_ms, True

    def _label_at(t_s: float, tol_override_s: Optional[float] = None) -> Tuple[str, str]:
        sym, lbl, _signed_ms, _abs_ms, _matched = _reference_at(t_s, tol_override_s)
        return sym, lbl

    rows: List[Dict] = []
    rec_name = Path(stem).name

    def _full_layer2_decisions(
        beat_time_s: float,
        lbl: str,
        peaks: np.ndarray,
        *,
        post_r_lookahead_override_s: Optional[float] = None,
    ) -> Tuple[Dict[str, Dict[str, object]], int]:
        """Return full causal/centered Layer 2 decisions by feature set."""
        decisions: Dict[str, Dict[str, object]] = {}
        if lbl == "healthy" and beat_time_s <= cal_end_t:
            return decisions, 0
        lookahead_s = (
            post_r_lookahead_s
            if post_r_lookahead_override_s is None
            else float(post_r_lookahead_override_s)
        )
        feats, n_beats = extract_beat_features(
            filt, fs, beat_time_s, peaks,
            morphology_window_s, rr_lookback_s, species,
            raw=raw,
            feature_window_mode=feature_window_mode,
            post_r_lookahead_s=lookahead_s,
        )
        for fset, cal_obj in calibrators.items():
            aligned = align_features_for_scoring(feats, cal_obj)
            decisions[fset] = score_one(aligned, cal_obj)
        return decisions, n_beats

    def _score(beat_time_s: float, sym: str, lbl: str,
                peaks: np.ndarray, eval_mode: str) -> bool:
        if lbl == "healthy" and beat_time_s <= cal_end_t:
            return False
        decisions, n_beats = _full_layer2_decisions(beat_time_s, lbl, peaks)
        if not decisions:
            return False
        base = {
            "dataset": dataset,
            "group": DATASET_GROUP.get(dataset, "unknown"),
            "record": rec_name,
            "beat_time_s": round(beat_time_s, 3),
            "beat_symbol": sym,
            "label": lbl,
            "n_beats_morph_window": n_beats,
            "benchmark_mode": mode,
        }
        for fset, sc in decisions.items():
            r = {**base, "feature_set": fset, "eval_mode": eval_mode, **sc}
            reason_str = str(r.get("reason", ""))
            r["inhibit_class"] = _classify_inhibit(reason_str)
            r["sqi_inhibit"] = not r["permit"] and any(
                s in reason_str for s in
                ("hf_noise_ratio", "lf_wander_ratio", "raw_hf_noise_ratio")
            )
            rows.append(r)
        return True

    n_scored = 0
    # Oracle: one decision per annotated beat
    for s, sym in zip(ann.sample, ann.symbol):
        lbl = beat_label(sym, dataset)
        if lbl == "mixed":
            continue
        if window_limit and n_scored >= window_limit:
            break
        t_beat = s / fs
        if _score(t_beat, sym, lbl, all_peaks, "oracle"):
            n_scored += 1
        if len(ad_peaks):
            _score(t_beat, sym, lbl, ad_peaks, "rpeak_adaptive_extra_rr_at_beat")

    # Adaptive L1: one decision per accepted L1 trigger
    if len(ad_peaks):
        n_trigger_scored = 0
        for t_beat in ad_peaks:
            if window_limit and n_trigger_scored >= window_limit:
                break
            sym, lbl = _label_at(float(t_beat))
            if _score(float(t_beat), sym, lbl, ad_peaks, "rpeak_adaptive_extra_gated"):
                n_trigger_scored += 1

    # Two-stream therapy architecture:
    #   fast_causal peaks provide continuous RR/coupling context,
    #   adaptive Layer 1 peaks act as high-PPV stimulation candidates.
    if len(ad_peaks) and len(fast_peaks):
        n_trigger_scored = 0
        for t_beat in ad_peaks:
            if window_limit and n_trigger_scored >= window_limit:
                break
            t_float = float(t_beat)
            sym, lbl = _label_at(t_float, tol_override_s=0.120)
            merged_context = context_peaks_with_candidate(fast_peaks, t_float)
            if _score(t_float, sym, lbl, merged_context, "adaptive_candidate_fast_context"):
                n_trigger_scored += 1

    # Fast causal threshold detector: deployment-style causal threshold triggers.
    if len(fast_peaks) and calibrators:
        n_trigger_scored = 0
        for t_beat in fast_peaks:
            if window_limit and n_trigger_scored >= window_limit:
                break
            sym, lbl = _label_at(float(t_beat), tol_override_s=0.120)
            if _score(float(t_beat), sym, lbl, fast_peaks, "fast_causal_gated"):
                n_trigger_scored += 1

    # Stateful deployment mode:
    #   current stimulation decision = fast same-beat veto AND previous beat's full Layer 2 state
    #   current full Layer 2 analysis updates the state consumed by the next trigger.
    if len(fast_peaks) and calibrators:
        previous_full: Dict[str, Dict[str, object]] = {
            fset: {"permit": False, "reason": "no_previous_full_layer2"}
            for fset in calibrators
        }
        n_nextbeat_scored = 0
        for t_beat in fast_peaks:
            if window_limit and n_nextbeat_scored >= window_limit:
                break
            t_float = float(t_beat)
            sym, lbl = _label_at(t_float, tol_override_s=0.120)

            # During calibration, stimulation is disabled; do not report therapy metrics.
            if t_float <= cal_end_t:
                continue

            same_ok, same_reason, coupling_ratio = fast_same_beat_ok(t_float, fast_peaks)
            current_full, n_beats = _full_layer2_decisions(t_float, lbl, fast_peaks)
            if not current_full:
                continue

            base = {
                "dataset": dataset,
                "group": DATASET_GROUP.get(dataset, "unknown"),
                "record": rec_name,
                "beat_time_s": round(t_float, 3),
                "beat_symbol": sym,
                "label": lbl,
                "n_beats_morph_window": n_beats,
                "benchmark_mode": mode,
                "eval_mode": "fast_causal_nextbeat",
                "same_beat_fast_ok": same_ok,
                "same_beat_fast_reason": same_reason,
                "same_beat_coupling_ratio": coupling_ratio,
            }

            for fset, full_sc in current_full.items():
                prev = previous_full.get(
                    fset, {"permit": False, "reason": "no_previous_full_layer2"}
                )
                prev_permit = bool(prev.get("permit", False))
                permit = bool(same_ok and prev_permit)
                if permit:
                    reason = "nextbeat_permit"
                elif not same_ok:
                    reason = same_reason
                else:
                    reason = "previous_full_layer2_inhibit"

                row = {
                    **base,
                    "feature_set": fset,
                    "permit": permit,
                    "inhibit": not permit,
                    "reason": reason,
                    "previous_full_layer2_permit": prev_permit,
                    "previous_full_layer2_reason": prev.get("reason", ""),
                    "current_full_layer2_permit": bool(full_sc.get("permit", False)),
                    "current_full_layer2_reason": full_sc.get("reason", ""),
                    "mahalanobis": full_sc.get("mahalanobis", np.nan),
                    "mahalanobis_threshold": full_sc.get("mahalanobis_threshold", np.nan),
                    "signal_mahal_proxy": full_sc.get("signal_mahal_proxy", np.nan),
                    "rr_mahal_proxy": full_sc.get("rr_mahal_proxy", np.nan),
                    "max_zscore": full_sc.get("max_zscore", full_sc.get("max_abs_zscore", np.nan)),
                    "zscore_threshold": full_sc.get("zscore_threshold", np.nan),
                    "hard_rule_violated": full_sc.get("hard_rule_violated", None),
                }
                row["inhibit_class"] = _classify_inhibit(reason)
                row["sqi_inhibit"] = False
                rows.append(row)

            for fset, full_sc in current_full.items():
                previous_full[fset] = {
                    "permit": bool(full_sc.get("permit", False)),
                    "reason": full_sc.get("reason", ""),
                }

            n_nextbeat_scored += 1

    # Stateful risk supervisor:
    #   current stimulation decision = fast same-beat veto AND no active persistent
    #   Layer 2 risk state. Isolated full Layer 2 failures do not automatically
    #   block the next beat unless they persist.
    if len(fast_peaks):
        risk_state: Dict[str, Dict[str, object]] = {
            fset: {
                "active": True,
                "reason": "no_previous_stateful_layer2",
                "soft_failures": 0,
                "source": "startup",
            }
            for fset in calibrators
        }
        n_stateful_scored = 0
        for t_beat in fast_peaks:
            if window_limit and n_stateful_scored >= window_limit:
                break
            t_float = float(t_beat)
            sym, lbl = _label_at(t_float, tol_override_s=0.120)

            # During calibration, stimulation is disabled; do not report therapy metrics.
            if t_float <= cal_end_t:
                continue

            same_ok, same_reason, coupling_ratio = fast_same_beat_ok(t_float, fast_peaks)
            current_full, n_beats = _full_layer2_decisions(t_float, lbl, fast_peaks)
            if not current_full:
                continue

            base = {
                "dataset": dataset,
                "group": DATASET_GROUP.get(dataset, "unknown"),
                "record": rec_name,
                "beat_time_s": round(t_float, 3),
                "beat_symbol": sym,
                "label": lbl,
                "n_beats_morph_window": n_beats,
                "benchmark_mode": mode,
                "eval_mode": "fast_causal_stateful",
                "same_beat_fast_ok": same_ok,
                "same_beat_fast_reason": same_reason,
                "same_beat_coupling_ratio": coupling_ratio,
            }

            next_risk_state: Dict[str, Dict[str, object]] = {}
            for fset, full_sc in current_full.items():
                state = risk_state.get(
                    fset,
                    {
                        "active": True,
                        "reason": "no_previous_stateful_layer2",
                        "soft_failures": 0,
                        "source": "startup",
                    },
                )
                risk_active = bool(state.get("active", True))
                permit = bool(same_ok and not risk_active)
                if permit:
                    reason = "stateful_permit"
                elif not same_ok:
                    reason = same_reason
                else:
                    reason = "stateful_layer2_risk_inhibit"

                row = {
                    **base,
                    "feature_set": fset,
                    "permit": permit,
                    "inhibit": not permit,
                    "reason": reason,
                    "stateful_layer2_risk_active": risk_active,
                    "stateful_layer2_risk_reason": state.get("reason", ""),
                    "stateful_layer2_soft_failures": state.get("soft_failures", 0),
                    "stateful_layer2_risk_source": state.get("source", ""),
                    "current_full_layer2_permit": bool(full_sc.get("permit", False)),
                    "current_full_layer2_reason": full_sc.get("reason", ""),
                    "mahalanobis": full_sc.get("mahalanobis", np.nan),
                    "mahalanobis_threshold": full_sc.get("mahalanobis_threshold", np.nan),
                    "signal_mahal_proxy": full_sc.get("signal_mahal_proxy", np.nan),
                    "rr_mahal_proxy": full_sc.get("rr_mahal_proxy", np.nan),
                    "max_zscore": full_sc.get("max_zscore", full_sc.get("max_abs_zscore", np.nan)),
                    "zscore_threshold": full_sc.get("zscore_threshold", np.nan),
                    "hard_rule_violated": full_sc.get("hard_rule_violated", None),
                }
                row["inhibit_class"] = _classify_inhibit(reason)
                row["sqi_inhibit"] = False
                rows.append(row)

                next_risk_state[fset] = _persistent_risk_state_update(
                    full_sc,
                    int(state.get("soft_failures", 0) or 0),
                )

            risk_state.update(next_risk_state)
            n_stateful_scored += 1

    # Prospective 1-in-8 stimulation cadence:
    #   beats 1-7 update the Layer 2 safety state, beat 8 is the only
    #   stimulation opportunity. The candidate beat itself is not analyzed
    #   before the trigger decision.
    if len(fast_peaks) and calibrators:
        cadence_gates = {
            fset: ProspectiveCadenceGate(
                cycle_length=8,
                observation_beats=7,
                min_safe_observations=cadence_min_safe_observations,
                require_last_observation_safe=cadence_require_last_observation_safe,
            )
            for fset in calibrators
        }
        cadence_policy = (
            f"min{cadence_min_safe_observations}of7"
            f"_lastsafe{int(cadence_require_last_observation_safe)}"
        )
        n_cadence_scored = 0
        for peak_idx, t_beat in enumerate(fast_peaks):
            if window_limit and n_cadence_scored >= window_limit:
                break
            t_float = float(t_beat)
            sym, lbl, r_lag_ms, r_abs_ms, r_matched = _reference_at(
                t_float, tol_override_s=0.120
            )

            # During calibration, stimulation is disabled and the cadence
            # starts fresh afterwards.
            if t_float <= cal_end_t:
                for gate in cadence_gates.values():
                    gate.reset()
                continue

            any_gate = next(iter(cadence_gates.values()))
            is_candidate = any_gate.next_phase == any_gate.cycle_length

            if is_candidate:
                same_ok, same_reason, coupling_ratio = fast_same_beat_ok(t_float, fast_peaks)
                confirmation_delay_ms = (
                    float(fast_confirmation_delays_ms[peak_idx])
                    if peak_idx < len(fast_confirmation_delays_ms) else float("nan")
                )
                confirmation_time_s = (
                    float(fast_confirmations[peak_idx])
                    if peak_idx < len(fast_confirmations) else float("nan")
                )
                confirmed_lag_ms = (
                    float(r_lag_ms + confirmation_delay_ms)
                    if np.isfinite(r_lag_ms) and np.isfinite(confirmation_delay_ms)
                    else float("nan")
                )
                base = {
                    "dataset": dataset,
                    "group": DATASET_GROUP.get(dataset, "unknown"),
                    "record": rec_name,
                    "beat_time_s": round(t_float, 3),
                    "beat_symbol": sym,
                    "label": lbl,
                    "n_beats_morph_window": np.nan,
                    "benchmark_mode": mode,
                    "eval_mode": "fast_causal_cadence_1of8",
                    "cadence_policy": cadence_policy,
                    "cadence_observation_lookahead_s": cadence_observation_lookahead_s,
                    "same_beat_fast_ok": same_ok,
                    "same_beat_fast_reason": same_reason,
                    "same_beat_coupling_ratio": coupling_ratio,
                    "same_beat_fast_used_for_decision": False,
                    "rpeak_reference_matched": r_matched,
                    "rpeak_detection_lag_ms": round(r_lag_ms, 2),
                    "rpeak_abs_timing_error_ms": round(r_abs_ms, 2),
                    "rpeak_confirmation_delay_ms": round(confirmation_delay_ms, 2),
                    "rpeak_confirmed_event_lag_ms": round(confirmed_lag_ms, 2),
                    "rpeak_confirmation_time_s": round(confirmation_time_s, 3),
                }
                for fset, gate in cadence_gates.items():
                    cadence = gate.step(
                        safety_decision=None,
                        trigger_ok=True,
                        trigger_reason="r_peak_detected",
                    )
                    row = {**base, "feature_set": fset, **cadence}
                    row["inhibit_class"] = _classify_inhibit(str(row.get("reason", "")))
                    row["sqi_inhibit"] = False
                    rows.append(row)
                n_cadence_scored += 1
                continue

            future_peaks = fast_peaks[fast_peaks > (t_float + 0.003)]
            if len(future_peaks):
                max_causal_lookahead_s = max(0.0, float(future_peaks[0] - t_float - 0.003))
            else:
                max_causal_lookahead_s = float(cadence_observation_lookahead_s)
            obs_lookahead_s = min(
                float(cadence_observation_lookahead_s),
                max_causal_lookahead_s,
            )
            current_full, _n_beats = _full_layer2_decisions(
                t_float,
                lbl,
                fast_peaks,
                post_r_lookahead_override_s=obs_lookahead_s,
            )
            for fset, gate in cadence_gates.items():
                gate.step(current_full.get(fset) if current_full else None)

    return rows


# ---------------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------------

def _overall_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, grp, bm, fset, em), sub in df.groupby(
        ["dataset", "group", "benchmark_mode", "feature_set", "eval_mode"]
    ):
        h = sub[sub["label"] == "healthy"]
        ab = sub[sub["label"] == "abnormal_v"]
        svt = sub[sub["label"] == "svt"]
        rows.append({
            "dataset": ds,
            "group": grp,
            "benchmark_mode": bm,
            "feature_set": fset,
            "eval_mode": em,
            "n_healthy": len(h),
            "n_abnormal": len(ab),
            "n_svt": len(svt),
            "healthy_permit": round(h["permit"].mean(), 4) if len(h) else float("nan"),
            "false_inhibit": round(1 - h["permit"].mean(), 4) if len(h) else float("nan"),
            "abnormal_inhibit": round(1 - ab["permit"].mean(), 4) if len(ab) else float("nan"),
            "false_permit": round(ab["permit"].mean(), 4) if len(ab) else float("nan"),
            "svt_inhibit": round(1 - svt["permit"].mean(), 4) if len(svt) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    data_dir: Path,
    out_dir: Path,
    datasets: List[str],
    benchmark_mode: str,            # "zero_shot", "diagnostic_retuned", "both"
    feature_sets: List[str],
    cal_frac: float,
    threshold_quantile: float,
    morphology_window_s: float,
    rr_lookback_s: float,
    species: str,
    abnormal_target: float,
    max_healthy_fi: float,
    window_limit: Optional[int],
    include_adaptive: bool,
    feature_window_mode: str,
    post_r_lookahead_s: float,
    max_records_per_dataset: int,
    cadence_observation_lookahead_s: float,
    cadence_min_safe_observations: int,
    cadence_require_last_observation_safe: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("cross_dataset")

    modes_to_run = (
        ["zero_shot", "diagnostic_retuned"]
        if benchmark_mode == "both"
        else [benchmark_mode]
    )

    all_rows: List[Dict] = []

    for dataset_arg in datasets:
        info = resolve_dataset(dataset_arg)
        dataset = info.folder
        ds_dir = dataset_dir(data_dir, dataset)
        if not ds_dir.is_dir():
            log.warning(f"Dataset directory not found: {ds_dir} — skip")
            continue

        ann_ext = DATASET_ANN_EXT.get(dataset, "atr")
        channel = DATASET_CHANNEL.get(dataset, 0)
        hea_files = sorted(ds_dir.glob("*.hea"))
        if max_records_per_dataset > 0:
            hea_files = hea_files[:max_records_per_dataset]
        log.info(f"[{dataset}] ({info.title}) {len(hea_files)} records, channel={channel}")

        for hea in hea_files:
            stem = str(hea.with_suffix(""))
            t0 = time.time()

            for mode in modes_to_run:
                rows = evaluate_record(
                    stem=stem,
                    dataset=dataset,
                    channel=channel,
                    ann_ext=ann_ext,
                    mode=mode,
                    feature_sets=feature_sets,
                    cal_frac=cal_frac,
                    threshold_quantile=threshold_quantile,
                    morphology_window_s=morphology_window_s,
                    rr_lookback_s=rr_lookback_s,
                    species=species,
                    abnormal_target=abnormal_target,
                    max_healthy_fi=max_healthy_fi,
                    window_limit=window_limit,
                    include_adaptive=include_adaptive,
                    feature_window_mode=feature_window_mode,
                    post_r_lookahead_s=post_r_lookahead_s,
                    cadence_observation_lookahead_s=cadence_observation_lookahead_s,
                    cadence_min_safe_observations=cadence_min_safe_observations,
                    cadence_require_last_observation_safe=cadence_require_last_observation_safe,
                )
                all_rows.extend(rows)

            elapsed = time.time() - t0
            log.info(f"  {hea.stem}: {len(all_rows)} total rows  {elapsed:.1f}s")

    if not all_rows:
        log.warning("No rows collected — check data_dir and datasets.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_beat.csv", index=False)
    log.info(f"per_beat.csv: {len(df)} rows")

    # Overall summary table
    overall = _overall_table(df)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)

    # Cross-dataset comparison (key table for thesis)
    # Focus: oracle + RPeakDetection deployment modes, feature_set=all, zero_shot
    pivot_rows = []
    for (ds, em), sub in overall.groupby(["dataset", "eval_mode"]):
        zs = sub[sub["benchmark_mode"] == "zero_shot"]
        dr = sub[sub["benchmark_mode"] == "diagnostic_retuned"]
        zs_all = zs[zs["feature_set"] == "all"]
        dr_all = dr[dr["feature_set"] == "all"]
        if len(zs_all):
            r = zs_all.iloc[0]
            pivot_rows.append({
                "dataset": ds,
                "eval_mode": em,
                "healthy_permit_zero_shot": r["healthy_permit"],
                "abnormal_inhibit_zero_shot": r["abnormal_inhibit"],
                "false_permit_zero_shot": r["false_permit"],
                "svt_inhibit_zero_shot": r["svt_inhibit"],
            })
            if len(dr_all):
                rd = dr_all.iloc[0]
                pivot_rows[-1]["abnormal_inhibit_retuned"] = rd["abnormal_inhibit"]
                pivot_rows[-1]["healthy_permit_retuned"] = rd["healthy_permit"]

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        pivot_df.to_csv(out_dir / "cross_dataset_matrix.csv", index=False)

    # NSTDB SNR breakdown (if present) — includes SQI inhibit rate per SNR level
    nstdb_df = df[df["dataset"] == "nstdb"].copy()
    if len(nstdb_df):
        nstdb_df["snr_db"] = nstdb_df["record"].map(NSTDB_SNR)
        nstdb_tagged = nstdb_df.assign(
            dataset=nstdb_df["record"].map(
                lambda r: f"nstdb_snr{NSTDB_SNR.get(r, '?'):+d}" if NSTDB_SNR.get(r) is not None else r
            )
        )
        nstdb_summary = _overall_table(nstdb_tagged)

        # Add SQI inhibit rate (fraction of ALL evaluated beats that were SQI-inhibited)
        if "sqi_inhibit" in df.columns:
            sqi_rates = []
            for (ds_tag, bm, fset, em), grp in nstdb_tagged.groupby(
                ["dataset", "benchmark_mode", "feature_set", "eval_mode"]
            ):
                sqi_rate = grp["sqi_inhibit"].mean() if len(grp) else float("nan")
                sqi_rates.append({
                    "dataset": ds_tag, "benchmark_mode": bm,
                    "feature_set": fset, "eval_mode": em,
                    "sqi_inhibit_rate": round(sqi_rate, 4),
                })
            sqi_df = pd.DataFrame(sqi_rates)
            nstdb_summary = nstdb_summary.merge(
                sqi_df,
                on=["dataset", "benchmark_mode", "feature_set", "eval_mode"],
                how="left",
            )

        nstdb_summary.to_csv(out_dir / "nstdb_snr_breakdown.csv", index=False)
        log.info("nstdb_snr_breakdown.csv written")

        # Print key NSTDB SNR table
        show_nstdb = nstdb_summary[
            (nstdb_summary["benchmark_mode"] == "zero_shot")
            & (nstdb_summary["eval_mode"] == "oracle")
            & (nstdb_summary["feature_set"] == "all")
        ].sort_values("dataset")
        cols = ["dataset", "healthy_permit", "abnormal_inhibit", "false_permit"]
        if "sqi_inhibit_rate" in show_nstdb.columns:
            cols.append("sqi_inhibit_rate")
        log.info("\n=== NSTDB by SNR (zero_shot, oracle, all) ===")
        log.info("\n" + show_nstdb[cols].to_string(index=False))

    # Print summary
    log.info("\n=== Cross-dataset summary (zero_shot, oracle, all features) ===")
    show = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "oracle")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, adaptive extra candidate, all features) ===")
    show2 = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "rpeak_adaptive_extra_gated")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show2.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, fast causal RPeakDetection + Layer 2, all features) ===")
    show3 = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "fast_causal_gated")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show3.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, adaptive candidates + fast causal context, all features) ===")
    show3b = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "adaptive_candidate_fast_context")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show3b.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, fast causal next-beat supervisor, all features) ===")
    show4 = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "fast_causal_nextbeat")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show4.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, fast causal persistent-risk supervisor, all features) ===")
    show5 = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "fast_causal_stateful")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show5.to_string(index=False))

    log.info("\n=== Cross-dataset summary (zero_shot, fast causal prospective 1-in-8 cadence, all features) ===")
    show6 = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["eval_mode"] == "fast_causal_cadence_1of8")
        & (overall["feature_set"] == "all")
    ][["dataset", "n_healthy", "n_abnormal", "healthy_permit",
       "abnormal_inhibit", "false_permit", "svt_inhibit"]]
    log.info("\n" + show6.to_string(index=False))

    log.info(f"\nDone -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("Results/layer2/cross_dataset"))
    p.add_argument(
        "--datasets", nargs="+",
        default=["mitdb", "nstdb", "svdb"],
        help="Datasets to evaluate. Available: mitdb nstdb svdb incartdb cudb vfdb",
    )
    p.add_argument(
        "--mode", choices=["zero_shot", "diagnostic_retuned", "both"],
        default="zero_shot",
        help=(
            "zero_shot: frozen MIT-BIH gate, healthy-only calibration (valid generalization). "
            "diagnostic_retuned: allows abnormal-label threshold tuning (upper bound only). "
            "both: run both."
        ),
    )
    p.add_argument("--feature-sets", nargs="+",
                   default=["all", "signal_only"])
    p.add_argument("--cal-frac", type=float, default=0.6)
    p.add_argument("--threshold-quantile", type=float, default=0.999)
    p.add_argument("--morphology-window-s", type=float, default=5.0)
    p.add_argument("--rr-lookback-s", type=float, default=30.0)
    p.add_argument(
        "--feature-window-mode",
        choices=["centered", "causal"],
        default="centered",
        help=(
            "centered uses beat +/- half window (offline upper bound). "
            "causal uses [beat-window, beat+post-r-lookahead]."
        ),
    )
    p.add_argument(
        "--post-r-lookahead-s",
        type=float,
        default=0.08,
        help="Causal Layer 2 post-R samples allowed before permit/inhibit decision.",
    )
    p.add_argument(
        "--cadence-observation-lookahead-s",
        type=float,
        default=0.40,
        help=(
            "Post-R lookahead used for the 7 unstimulated cadence observation beats. "
            "It is capped before the next detected peak. The 8th beat is still "
            "trigger-only and is not analyzed by Layer 2."
        ),
    )
    p.add_argument(
        "--cadence-min-safe-observations",
        type=int,
        default=6,
        help="Minimum safe Layer 2 decisions required among the 7 observation beats.",
    )
    p.add_argument(
        "--cadence-allow-unsafe-last-observation",
        action="store_true",
        help="Allow stimulation even if the 7th observation beat was unsafe.",
    )
    p.add_argument("--species", default="human")
    p.add_argument(
        "--abnormal-target", type=float, default=0.95,
        help="Target abnormal inhibit rate for diagnostic_retuned mode (default 0.95).",
    )
    p.add_argument(
        "--max-healthy-fi", type=float, default=0.18,
        help="Max healthy false-inhibit cap for diagnostic_retuned mode (default 0.18).",
    )
    p.add_argument("--window-limit", type=int, default=None,
                   help="Limit beats evaluated per record (for quick smoke tests).")
    p.add_argument(
        "--max-records-per-dataset",
        type=int,
        default=0,
        help="Limit records per dataset for quick causal sweeps. 0 means all records.",
    )
    p.add_argument("--no-adaptive", action="store_true",
                   help="Skip adaptive Layer 1 (faster).")
    args = p.parse_args(argv)

    run_benchmark(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        datasets=args.datasets,
        benchmark_mode=args.mode,
        feature_sets=args.feature_sets,
        cal_frac=args.cal_frac,
        threshold_quantile=args.threshold_quantile,
        morphology_window_s=args.morphology_window_s,
        rr_lookback_s=args.rr_lookback_s,
        species=args.species,
        abnormal_target=args.abnormal_target,
        max_healthy_fi=args.max_healthy_fi,
        window_limit=args.window_limit,
        include_adaptive=not args.no_adaptive,
        feature_window_mode=args.feature_window_mode,
        post_r_lookahead_s=args.post_r_lookahead_s,
        max_records_per_dataset=args.max_records_per_dataset,
        cadence_observation_lookahead_s=args.cadence_observation_lookahead_s,
        cadence_min_safe_observations=args.cadence_min_safe_observations,
        cadence_require_last_observation_safe=not args.cadence_allow_unsafe_last_observation,
    )


if __name__ == "__main__":
    main()
