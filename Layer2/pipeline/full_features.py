"""
Layer 2 — full feature assembler.

Combines all Layer 2 features into a single flat dict and a metadata dict that
maps each feature name to its group ("signal" or "rr").

Signal-only features (prefix: signal__)
    Computed from the raw ECG window alone. Do not require R-peak locations.
    Can be computed even when the detector fails or produces no peaks.
    Includes:
        - Wavelet features (db4, 4 levels: log-energy + Shannon entropy per band)
        - Sample entropy and approximate entropy
        - Amplitude / energy features: RMS, peak-to-peak, abs P95/P99, line length,
          signal energy, zero-crossing rate

RR-dependent features (prefix: rr__)
    Computed from R-peak timestamps or RR intervals.
    Skipped entirely (not added to the dict) if r_peaks_s is None.
    Includes all features from rhythm_features.rhythm_features().

Separation rationale:
    An inhibit triggered by signal__ features points to electrode/noise problems.
    An inhibit triggered by rr__ features points to rhythm deterioration.
    Logging and scoring them separately supports real-time diagnosis.

Usage
-----
    from full_features import full_features

    feats, groups = full_features(window, fs=360.0, r_peaks_s=peaks, species="human")
    # feats  : Dict[str, float]
    # groups : Dict[str, str]  — maps each key to "signal" or "rr"

    # Signal-only features (safe to compute even if no peaks):
    signal_feats = {k: v for k, v in feats.items() if groups[k] == "signal"}

    # RR-dependent features:
    rr_feats = {k: v for k, v in feats.items() if groups[k] == "rr"}
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from morphology_features import morphology_features
from rhythm_features import rhythm_features
from signal_features import entropy_features, wavelet_features


# ---------------------------------------------------------------------------
# Amplitude / energy features (signal-only, pure NumPy)
# ---------------------------------------------------------------------------

def spectral_sqi_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    """
    FFT-based signal quality index (SQI) features.

    Both features are dimensionless power ratios in [0, 1] and do not require
    R-peak locations.  They serve as explicit SQI gates: when either feature
    exceeds its hard-rule threshold the system inhibits with reason
    "hard_rule:signal__hf_noise_ratio" or "hard_rule:signal__lf_wander_ratio",
    which unambiguously identifies a signal-quality failure rather than an
    ECG-state failure.

    Typical values
    --------------
    Clean ECG (MIT-BIH, INCART, 360/257 Hz):
        hf_noise_ratio  : 0.02 – 0.08
        lf_wander_ratio : 0.08 – 0.25
    NSTDB electrode-motion noise (SNR = 0 dB):
        hf_noise_ratio  : 0.30 – 0.60
        lf_wander_ratio : 0.10 – 0.35
    NSTDB baseline-wander noise:
        hf_noise_ratio  : 0.02 – 0.06
        lf_wander_ratio : 0.55 – 0.85

    Returns (no prefix — prefix added by full_features)
    -------
    hf_noise_ratio  : power in [40 Hz, Nyquist] / total power
    lf_wander_ratio : power in [0, 0.67 Hz] / total power
                      (0.67 Hz ≈ 40 bpm — below any resting heart rate)
    """
    x = np.asarray(window, dtype=float)
    n = len(x)
    if n < 32 or not (np.isfinite(fs) and fs > 0):
        return {"hf_noise_ratio": float("nan"), "lf_wander_ratio": float("nan")}

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    total = float(np.sum(psd)) + 1e-12

    hf = float(np.sum(psd[freqs >= 40.0])) / total
    lf = float(np.sum(psd[freqs < 0.67])) / total

    return {"hf_noise_ratio": hf, "lf_wander_ratio": lf}


def amplitude_features(window: np.ndarray) -> Dict[str, float]:
    """
    Compute amplitude and energy features from a raw ECG window.

    All features are R-peak-independent and can be computed on any signal.

    Parameters
    ----------
    window : 1D numpy array, float, pre-filtered ECG (any amplitude unit).

    Returns
    -------
    Dict with keys (no prefix — prefix is added by full_features):
        rms            : root-mean-square amplitude
        peak_to_peak   : max - min
        abs_p95        : 95th percentile of |window|
        abs_p99        : 99th percentile of |window|
        line_length    : mean absolute successive difference (normalised by window length)
        energy         : sum of squares
        zero_cross_rate: fraction of consecutive samples with opposite signs
    """
    x = np.asarray(window, dtype=float)
    n = len(x)

    if n == 0:
        return {
            "rms": float("nan"),
            "peak_to_peak": float("nan"),
            "abs_p95": float("nan"),
            "abs_p99": float("nan"),
            "line_length": float("nan"),
            "energy": float("nan"),
            "zero_cross_rate": float("nan"),
        }

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak_to_peak = float(np.max(x) - np.min(x))
    abs_x = np.abs(x)
    abs_p95 = float(np.percentile(abs_x, 95))
    abs_p99 = float(np.percentile(abs_x, 99))
    energy = float(np.sum(x ** 2))

    if n >= 2:
        diffs = np.diff(x)
        # Line length normalised by number of samples so windows of different
        # lengths are comparable. Units: [signal unit] per sample.
        line_length = float(np.sum(np.abs(diffs)) / n)
        # Zero-crossing rate: fraction of consecutive pairs where sign changes.
        zero_cross_rate = float(np.sum(np.sign(x[:-1]) != np.sign(x[1:])) / (n - 1))
    else:
        line_length = float("nan")
        zero_cross_rate = float("nan")

    return {
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "abs_p95": abs_p95,
        "abs_p99": abs_p99,
        "line_length": line_length,
        "energy": energy,
        "zero_cross_rate": zero_cross_rate,
    }


# ---------------------------------------------------------------------------
# Full feature assembler
# ---------------------------------------------------------------------------

def full_features(
    window: np.ndarray,
    fs: float,
    r_peaks_s: Optional[np.ndarray] = None,
    species: str = "human",
    wavelet: str = "db4",
    wavelet_level: int = 4,
    compute_spectral_hrv: bool = True,
    compute_entropy: bool = True,
    min_rr_for_spectral: int = 10,
    short_rr_thresh_ms: Optional[float] = None,
    long_rr_thresh_ms: Optional[float] = None,
    focus_peak_s: Optional[float] = None,
    morphology_half_width_s: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Compute the full Layer 2 feature vector for one ECG window.

    Parameters
    ----------
    window : 1D numpy array, float. Pre-filtered ECG, any length.
    fs : sampling rate in Hz. Not used for signal features directly (signal
        features are amplitude-based), but passed for documentation and future
        frequency-normalised features.
    r_peaks_s : R-peak timestamps in seconds, relative to the start of the
        window (i.e. values should be in [0, len(window)/fs]). If None, RR
        features are omitted entirely.
    species : "human", "rat", or "pig". Controls RR thresholds and HRV bands.
    wavelet : PyWavelets wavelet name. Default "db4".
    wavelet_level : DWT decomposition levels. Default 4.
    compute_spectral_hrv : if False, spectral HRV features are skipped (faster).
    compute_entropy : if False, sample entropy and approximate entropy are skipped.
        These are O(n^2) in window length and dominate runtime in batch processing.
        Default True preserves the original full feature set.
    min_rr_for_spectral : minimum number of RR intervals for spectral HRV.
    short_rr_thresh_ms : override short-RR threshold for rr__ features.
    long_rr_thresh_ms : override long-RR threshold for rr__ features.

    Returns
    -------
    features : Dict[str, float]
        Flat dict of all features. Signal features are prefixed "signal__",
        RR features are prefixed "rr__". NaN is used for features that cannot
        be computed (e.g. spectral HRV on a too-short window).
    feature_groups : Dict[str, str]
        Maps each feature name to "signal" or "rr". Same keys as features.

    Notes
    -----
    - Signal features are always present (even with no R-peaks).
    - RR features are absent from the dict if r_peaks_s is None.
    - Downstream code (BaselineCalibrator) must be fitted on the same
      feature subset — do not mix calibrations with and without rr__ features.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite float, got {fs!r}.")

    window = np.asarray(window, dtype=float)

    # ── Signal-only features ────────────────────────────────────────────────
    wave = wavelet_features(window, wavelet=wavelet, level=wavelet_level)
    ent  = entropy_features(window) if compute_entropy else {}
    amp  = amplitude_features(window)
    sqi  = spectral_sqi_features(window, fs)   # dimensionless power ratios

    features: Dict[str, float] = {}
    feature_groups: Dict[str, str] = {}

    for raw_dict in (wave, ent, amp, sqi):
        for k, v in raw_dict.items():
            key = f"signal__{k}"
            features[key] = float(v)
            feature_groups[key] = "signal"

    # ── Beat morphology (needs R-peak timing; focus_peak_s = trigger beat) ─
    if r_peaks_s is not None:
        morph = morphology_features(
            window,
            fs,
            r_peaks_s=r_peaks_s,
            focus_peak_s=focus_peak_s,
            half_width_s=morphology_half_width_s,
        )
        for k, v in morph.items():
            key = f"morph__{k}"
            features[key] = float(v)
            feature_groups[key] = "morph"

    # ── RR-dependent features ───────────────────────────────────────────────
    if r_peaks_s is not None:
        peaks = np.asarray(r_peaks_s, dtype=float).ravel()
        if len(peaks) >= 2:
            rr_dict = rhythm_features(
                r_peaks_s=peaks,
                species=species,
                short_rr_thresh_ms=short_rr_thresh_ms,
                long_rr_thresh_ms=long_rr_thresh_ms,
                compute_spectral=compute_spectral_hrv,
                min_rr_for_spectral=min_rr_for_spectral,
            )
            for k, v in rr_dict.items():
                key = f"rr__{k}"
                features[key] = float(v)
                feature_groups[key] = "rr"
        # If < 2 peaks, we silently skip rr features: caller may have passed a
        # single detected peak or an empty array due to poor signal quality.
        # This is treated as missing information → rr__ features absent from dict.

    return features, feature_groups


# ---------------------------------------------------------------------------
# Convenience: split a feature dict by group
# ---------------------------------------------------------------------------

def split_by_group(
    features: Dict[str, float],
    feature_groups: Dict[str, str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Split a full feature dict into signal-only and RR-dependent sub-dicts.

    Parameters
    ----------
    features : dict as returned by full_features()
    feature_groups : group dict as returned by full_features()

    Returns
    -------
    (signal_features, rr_features) — two dicts, original prefixes preserved.
    """
    signal = {k: v for k, v in features.items() if feature_groups.get(k) == "signal"}
    rr     = {k: v for k, v in features.items() if feature_groups.get(k) == "rr"}
    return signal, rr
