"""
Layer 2 — RR interval and rhythm feature extraction.

Input:
    r_peaks_s : array of R-peak timestamps in seconds (float), OR
    rr_ms     : array of RR intervals in milliseconds (float).
    At least one must be provided. If both are given, rr_ms takes priority.

Output:
    Dict[str, float] — all RR and spectral HRV features, flat.
    Feature names carry no prefix here; prefixing is done by full_features.py.

Species support:
    "human" — HR 40–200 bpm, standard HRV bands (LF 0.04–0.15 Hz, HF 0.15–0.40 Hz)
    "rat"   — HR 250–500 bpm, rodent-adjusted bands (LF 0.20–0.75 Hz, HF 0.75–3.0 Hz)
    "pig"   — HR 60–160 bpm, human-like bands

Spectral HRV:
    Computed by interpolating the RR series onto a regular grid, then applying a
    windowed FFT. Pure NumPy — no scipy dependency.
    If the RR sequence is too short or has too few intervals, spectral features are
    returned as NaN with consistent names (no crash, no silent omission).

References:
    Thireau et al. 2006/2008 (rodent HRV bands)
    Task Force 1996 (human HRV standards)
    Richman & Moorman 2000 (sample entropy - see signal_features.py)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Species configuration
# ---------------------------------------------------------------------------

_SPECIES_CONFIGS: Dict[str, Dict] = {
    "human": {
        "short_rr_thresh_ms": 400.0,   # < 400 ms = short (HR > 150 bpm)
        "long_rr_thresh_ms": 1500.0,   # > 1500 ms = long  (HR < 40 bpm)
        "lf_band": (0.04, 0.15),       # Hz
        "hf_band": (0.15, 0.40),       # Hz
        "resample_hz": 4.0,            # RR interpolation grid rate
        "min_rr_ms": 250.0,            # hard physiological floor (~240 bpm)
        "max_rr_ms": 2500.0,           # hard physiological ceiling (~24 bpm)
        # Rate-zone boundaries (bpm) for ICD/AED-style branching (see below).
        "tachy_bpm": 100.0,            # above normal sinus rest
        "vt_bpm": 180.0,               # ventricular-tachycardia zone entry
        "vf_bpm": 250.0,               # fibrillation-rate zone entry
    },
    "rat": {
        "short_rr_thresh_ms": 100.0,   # < 100 ms = short (HR > 600 bpm)
        "long_rr_thresh_ms": 300.0,    # > 300 ms = long  (HR < 200 bpm)
        "lf_band": (0.20, 0.75),       # Hz — Thireau et al.
        "hf_band": (0.75, 3.0),        # Hz — Thireau et al.
        "resample_hz": 20.0,           # higher rate for faster rhythm
        "min_rr_ms": 80.0,             # ~750 bpm floor
        "max_rr_ms": 500.0,            # ~120 bpm ceiling
        "tachy_bpm": 500.0,
        "vt_bpm": 600.0,
        "vf_bpm": 750.0,
    },
    "pig": {
        "short_rr_thresh_ms": 350.0,
        "long_rr_thresh_ms": 1200.0,
        "lf_band": (0.04, 0.15),
        "hf_band": (0.15, 0.40),
        "resample_hz": 4.0,
        "min_rr_ms": 300.0,
        "max_rr_ms": 1500.0,
        "tachy_bpm": 120.0,
        "vt_bpm": 180.0,
        "vf_bpm": 250.0,
    },
}

# Ordered rate zones, slowest to fastest. Used by classify_rate_zone().
RATE_ZONES = ("brady", "normal", "tachy", "vt_zone", "vf_zone")


def classify_rate_zone(hr_bpm: float, species: str = "human") -> str:
    """
    Map an instantaneous heart rate to an ICD/AED-style rate zone.

    Rate-zone branching is the first discriminator in every implantable
    defibrillator: the aggressiveness of the safety response scales with rate.
    Layer 2 uses the zone to decide whether the onset and stability
    discriminators are even relevant (they only matter in the tachy / VT zone;
    a VF-rate is treated as dangerous regardless of stability).

    Returns one of RATE_ZONES. NaN / non-finite HR returns "brady" (fail-safe:
    a rate we cannot read is treated as the least-permissive slow state, and
    the caller's reliability / hard-rule logic handles the missing data).
    """
    cfg = get_species_config(species)
    if not np.isfinite(hr_bpm) or hr_bpm <= 0:
        return "brady"
    if hr_bpm >= cfg["vf_bpm"]:
        return "vf_zone"
    if hr_bpm >= cfg["vt_bpm"]:
        return "vt_zone"
    if hr_bpm >= cfg["tachy_bpm"]:
        return "tachy"
    # Below tachy: split normal vs brady at the long-RR (slow) boundary.
    brady_bpm = 60_000.0 / cfg["long_rr_thresh_ms"]
    if hr_bpm < brady_bpm:
        return "brady"
    return "normal"


def get_species_config(species: str) -> Dict:
    """Return the config dict for a species, raising ValueError for unknown species."""
    key = species.lower().strip()
    if key not in _SPECIES_CONFIGS:
        raise ValueError(
            f"Unknown species '{species}'. "
            f"Supported: {sorted(_SPECIES_CONFIGS.keys())}"
        )
    return _SPECIES_CONFIGS[key]


# ---------------------------------------------------------------------------
# Spectral HRV (pure NumPy)
# ---------------------------------------------------------------------------

_SPECTRAL_FEATURE_NAMES = (
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
    "hrv_lf_norm",
    "hrv_hf_norm",
    "hrv_total_power",
)


def _nan_spectral() -> Dict[str, float]:
    """Return a dict of NaN for all spectral features (used when computation is skipped)."""
    return {k: float("nan") for k in _SPECTRAL_FEATURE_NAMES}


def _spectral_hrv(
    t_beats_s: np.ndarray,
    rr_ms: np.ndarray,
    lf_band: Tuple[float, float],
    hf_band: Tuple[float, float],
    resample_hz: float,
) -> Dict[str, float]:
    """
    Compute LF and HF HRV power from an RR interval series.

    Strategy
    --------
    1. Interpolate rr_ms onto a uniformly spaced time grid (linear).
    2. Apply a Hann window to reduce spectral leakage.
    3. Compute one-sided power spectral density via FFT.
    4. Integrate PSD over LF and HF bands.

    Parameters
    ----------
    t_beats_s : timestamps of the RR intervals in seconds, shape (N,).
        t_beats_s[i] is the time of the R-peak that ENDS interval rr_ms[i].
    rr_ms : RR intervals in milliseconds, shape (N,).
    lf_band, hf_band : (low_hz, high_hz) tuples.
    resample_hz : rate of the interpolation grid in Hz.

    Returns
    -------
    Dict with hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio,
              hrv_lf_norm, hrv_hf_norm, hrv_total_power.
    All values NaN if computation is not feasible.
    """
    # Minimum recording duration: 2 full periods at the lowest LF frequency.
    min_duration_s = 2.0 / lf_band[0]
    duration_s = float(t_beats_s[-1] - t_beats_s[0])
    if duration_s < min_duration_s:
        return _nan_spectral()

    # Build interpolation grid
    t_grid = np.arange(t_beats_s[0], t_beats_s[-1], 1.0 / resample_hz)
    if len(t_grid) < 8:
        return _nan_spectral()

    rr_interp = np.interp(t_grid, t_beats_s, rr_ms.astype(float))
    rr_interp -= rr_interp.mean()  # demean

    # Windowed FFT power spectral density
    N = len(rr_interp)
    window = np.hanning(N)
    # Normalise so PSD is in ms² / Hz
    window_power = float(np.sum(window ** 2)) / N
    fft_vals = np.fft.rfft(rr_interp * window)
    psd = (np.abs(fft_vals) ** 2) / (resample_hz * N * window_power)
    # Double non-DC, non-Nyquist bins to recover one-sided PSD
    psd[1:-1] *= 2.0

    freqs = np.fft.rfftfreq(N, d=1.0 / resample_hz)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

    lf_power = float(np.sum(psd[lf_mask]) * df)
    hf_power = float(np.sum(psd[hf_mask]) * df)
    total_power = lf_power + hf_power

    return {
        "hrv_lf_power": lf_power,
        "hrv_hf_power": hf_power,
        "hrv_lf_hf_ratio": lf_power / hf_power if hf_power > 1e-12 else float("nan"),
        "hrv_lf_norm": lf_power / total_power if total_power > 1e-12 else float("nan"),
        "hrv_hf_norm": hf_power / total_power if total_power > 1e-12 else float("nan"),
        "hrv_total_power": total_power,
    }


# ---------------------------------------------------------------------------
# Onset & stability discriminators (ICD/AED-style)
# ---------------------------------------------------------------------------

_ONSET_STABILITY_FEATURE_NAMES = (
    "onset_accel_frac",
    "stability_ms",
    "tachy_fraction",
)


def _nan_onset_stability() -> Dict[str, float]:
    return {k: float("nan") for k in _ONSET_STABILITY_FEATURE_NAMES}


def onset_stability_features(rr: np.ndarray, tachy_bpm: float) -> Dict[str, float]:
    """
    Onset and stability discriminators computed from an RR series (ms).

    These mirror the two classic implantable-defibrillator SVT/VT
    discriminators (Swerdlow et al. 1994; Boston Scientific / Medtronic
    "Onset" and "Stability" algorithms):

    onset_accel_frac
        Fractional shortening of RR from the first half of the window to the
        second half: (median_first - median_second) / median_first, clipped to
        [-1, 1]. A large positive value means the rate accelerated abruptly
        within the window (sudden onset -> VT-like); values near 0 mean a
        gradual or steady rate (sinus tachycardia-like).

    stability_ms
        Mean absolute successive RR difference (ms). Low = regular / stable
        fast rhythm (monomorphic VT); high = irregular (AF conducted fast).
        Stability is what separates a dangerous organised VT from irregular AF
        at the same mean rate.

    tachy_fraction
        Fraction of RR intervals at or above the species tachy rate
        (rr <= 60000 / tachy_bpm). How much of the window is actually fast.

    Returns NaN for features that need >=2 intervals when fewer are available.
    """
    rr = np.asarray(rr, dtype=float).ravel()
    n = len(rr)
    if n == 0:
        return _nan_onset_stability()

    # tachy_fraction is defined for any n >= 1.
    tachy_rr_ms = 60_000.0 / tachy_bpm if tachy_bpm > 0 else float("nan")
    tachy_fraction = (
        float(np.mean(rr <= tachy_rr_ms)) if np.isfinite(tachy_rr_ms) else float("nan")
    )

    if n < 2:
        return {
            "onset_accel_frac": float("nan"),
            "stability_ms": float("nan"),
            "tachy_fraction": tachy_fraction,
        }

    stability_ms = float(np.mean(np.abs(np.diff(rr))))

    half = n // 2
    if half >= 1 and (n - half) >= 1:
        med_first = float(np.median(rr[:half]))
        med_second = float(np.median(rr[half:]))
        onset = (med_first - med_second) / med_first if med_first > 1e-9 else float("nan")
        onset_accel_frac = float(np.clip(onset, -1.0, 1.0)) if np.isfinite(onset) else float("nan")
    else:
        onset_accel_frac = float("nan")

    return {
        "onset_accel_frac": onset_accel_frac,
        "stability_ms": stability_ms,
        "tachy_fraction": tachy_fraction,
    }


# ---------------------------------------------------------------------------
# Main feature function
# ---------------------------------------------------------------------------

def rhythm_features(
    r_peaks_s: Optional[np.ndarray] = None,
    rr_ms: Optional[np.ndarray] = None,
    species: str = "human",
    short_rr_thresh_ms: Optional[float] = None,
    long_rr_thresh_ms: Optional[float] = None,
    compute_spectral: bool = True,
    min_rr_for_spectral: int = 10,
    compute_onset_stability: bool = False,
) -> Dict[str, float]:
    """
    Compute RR-interval and rhythm features from R-peak timestamps or RR intervals.

    At least one of r_peaks_s or rr_ms must be provided.
    If rr_ms is given it is used directly; if only r_peaks_s is given, rr_ms is derived
    from successive differences.

    Parameters
    ----------
    r_peaks_s : R-peak timestamps in seconds, 1D array. Optional.
    rr_ms : RR intervals in milliseconds, 1D array. Optional.
        If both are given, rr_ms is used for interval features; r_peaks_s is used
        to construct absolute timestamps for the spectral interpolation.
    species : one of "human", "rat", "pig". Sets default thresholds and HRV bands.
    short_rr_thresh_ms : override short-RR threshold (ms). None → species default.
    long_rr_thresh_ms : override long-RR threshold (ms). None → species default.
    compute_spectral : if False, spectral HRV features are skipped (faster).
    min_rr_for_spectral : minimum number of RR intervals required to attempt spectral HRV.

    Returns
    -------
    Dict[str, float] with keys:
        rr_count, rr_mean_ms, hr_bpm,
        rr_sdnn_ms, rr_rmssd_ms, rr_cv,
        rr_min_ms, rr_max_ms, rr_range_ms,
        rr_diff_abs_mean_ms, rr_diff_abs_max_ms,
        short_rr_fraction, long_rr_fraction,
        hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio,  ← NaN if not computed
        hrv_lf_norm, hrv_hf_norm, hrv_total_power.

    Raises
    ------
    ValueError : if neither r_peaks_s nor rr_ms is provided, or inputs are malformed.
    """
    # ---- Input validation and RR derivation ---------------------------------
    if r_peaks_s is None and rr_ms is None:
        raise ValueError(
            "At least one of r_peaks_s or rr_ms must be provided."
        )

    if rr_ms is not None:
        rr = np.asarray(rr_ms, dtype=float).ravel()
    else:
        peaks = np.asarray(r_peaks_s, dtype=float).ravel()
        if len(peaks) < 2:
            raise ValueError(
                f"r_peaks_s must contain at least 2 R-peaks to derive RR intervals; "
                f"got {len(peaks)}."
            )
        rr = np.diff(peaks) * 1000.0  # seconds → milliseconds

    if len(rr) == 0:
        raise ValueError("RR interval array is empty.")

    # ---- Species config and threshold overrides -----------------------------
    cfg = get_species_config(species)
    thr_short = short_rr_thresh_ms if short_rr_thresh_ms is not None else cfg["short_rr_thresh_ms"]
    thr_long  = long_rr_thresh_ms  if long_rr_thresh_ms  is not None else cfg["long_rr_thresh_ms"]

    # ---- Scalar RR features -------------------------------------------------
    n = len(rr)
    mean_rr = float(np.mean(rr))
    hr_bpm = 60_000.0 / mean_rr if mean_rr > 0 else float("nan")

    # SDNN: standard deviation of NN intervals
    sdnn = float(np.std(rr, ddof=1)) if n >= 2 else float("nan")

    # RMSSD: root-mean-square of successive differences
    if n >= 2:
        diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        diff_abs_mean = float(np.mean(np.abs(diffs)))
        diff_abs_max  = float(np.max(np.abs(diffs)))
    else:
        rmssd = float("nan")
        diff_abs_mean = float("nan")
        diff_abs_max  = float("nan")

    # Coefficient of variation (dimensionless)
    rr_cv = (sdnn / mean_rr) if (mean_rr > 0 and np.isfinite(sdnn)) else float("nan")

    # Fraction of beats with short / long RR intervals
    short_frac = float(np.mean(rr < thr_short))
    long_frac  = float(np.mean(rr > thr_long))

    features: Dict[str, float] = {
        "rr_count":            float(n),
        "rr_mean_ms":          mean_rr,
        "hr_bpm":              hr_bpm,
        "rr_sdnn_ms":          sdnn,
        "rr_rmssd_ms":         rmssd,
        "rr_cv":               rr_cv,
        "rr_min_ms":           float(np.min(rr)),
        "rr_max_ms":           float(np.max(rr)),
        "rr_range_ms":         float(np.max(rr) - np.min(rr)),
        "rr_diff_abs_mean_ms": diff_abs_mean,
        "rr_diff_abs_max_ms":  diff_abs_max,
        "short_rr_fraction":   short_frac,
        "long_rr_fraction":    long_frac,
    }

    # ---- Onset / stability discriminators (opt-in) --------------------------
    if compute_onset_stability:
        features.update(onset_stability_features(rr, tachy_bpm=cfg["tachy_bpm"]))

    # ---- Spectral HRV -------------------------------------------------------
    if not compute_spectral or n < min_rr_for_spectral:
        features.update(_nan_spectral())
        return features

    # Build beat timestamps for the spectral interpolation.
    # Each rr[i] ends at the beat whose timestamp is t_beats_s[i].
    if r_peaks_s is not None and rr_ms is None:
        # Derived rr from r_peaks_s → timestamps are r_peaks_s[1:]
        peaks_arr = np.asarray(r_peaks_s, dtype=float).ravel()
        t_beats_s = peaks_arr[1:]
    elif r_peaks_s is not None and rr_ms is not None:
        # Both provided: use r_peaks_s for absolute timestamps
        peaks_arr = np.asarray(r_peaks_s, dtype=float).ravel()
        # rr_ms may have a different length; truncate to shorter
        n_common = min(len(rr), len(peaks_arr) - 1)
        rr_for_spectral = rr[:n_common]
        t_beats_s = peaks_arr[1 : 1 + n_common]
        features.update(
            _spectral_hrv(
                t_beats_s, rr_for_spectral,
                lf_band=cfg["lf_band"],
                hf_band=cfg["hf_band"],
                resample_hz=cfg["resample_hz"],
            )
        )
        return features
    else:
        # Only rr_ms: reconstruct timestamps as cumulative sum
        # rr[0] ends at t=rr[0]/1000; rr[i] ends at sum(rr[:i+1])/1000
        t_beats_s = np.cumsum(rr) / 1000.0

    features.update(
        _spectral_hrv(
            t_beats_s, rr,
            lf_band=cfg["lf_band"],
            hf_band=cfg["hf_band"],
            resample_hz=cfg["resample_hz"],
        )
    )
    return features
