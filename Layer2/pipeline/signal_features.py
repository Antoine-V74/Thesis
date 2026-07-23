"""
Layer 2 — feature extraction additions.

Adds wavelet decomposition and entropy features on top of the existing
rhythm feature vector from rhythm_features.py.

Design philosophy:
    Pure NumPy / scipy / pywt — no deep learning. Pynapse-compatible
    (no PyTorch in the runtime path). Every feature is interpretable
    in physiological terms.

Wavelet features:
    ECG is non-stationary. DWT resolves QRS transients (sharp, high-freq),
    T-waves (smoother, mid-freq), and baseline wander (low-freq) at
    different scales. We extract log-energy and Shannon entropy per
    decomposition level — 10 features for db4 at 4 levels (approximation
    plus 4 detail coefficient bands).

Entropy features:
    Sample entropy measures predictability without assuming any specific
    waveform shape. Highly species-portable: low for organized rhythms
    (sinus), high for chaotic ones (VF), in any species.

Usage:
    from signal_features import signal_features
    extra = signal_features(window_signal)   # window is 1D numpy array
    full_features = {**existing_features, **extra}
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pywt


# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------

def wavelet_features(
    window: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> Dict[str, float]:
    """
    Compute log-energy and Shannon entropy per DWT decomposition level.

    Parameters
    ----------
    window : 1D signal array
    wavelet : pywt wavelet name (default 'db4' — well-studied for ECG)
    level : number of decomposition levels (default 4)

    Returns
    -------
    Dict with keys per level:
        wave_A_log_energy,  wave_A_shannon_ent     (approximation)
        wave_D{i}_log_energy, wave_D{i}_shannon_ent (detail bands)
    """
    window = np.asarray(window, dtype=float)
    if len(window) < 2 ** level:
        window = np.pad(window, (0, 2 ** level - len(window)))

    coeffs = pywt.wavedec(window, wavelet, level=level)
    # coeffs ordering: [approximation, detail_L, detail_L-1, ..., detail_1]
    features: Dict[str, float] = {}
    for i, c in enumerate(coeffs):
        label = "A" if i == 0 else f"D{level - i + 1}"
        c = np.asarray(c, dtype=float)
        energy = float(np.sum(c ** 2)) + 1e-12
        features[f"wave_{label}_log_energy"] = float(np.log(energy))

        # Normalize squared coefficients to a probability distribution
        p = (c ** 2) / energy
        p = p[p > 1e-12]
        if p.size > 0:
            features[f"wave_{label}_shannon_ent"] = float(-np.sum(p * np.log(p)))
        else:
            features[f"wave_{label}_shannon_ent"] = 0.0
    return features


# ---------------------------------------------------------------------------
# Entropy features
# ---------------------------------------------------------------------------

def _chebyshev_distance_matrix(templates: np.ndarray) -> np.ndarray:
    """Pairwise Chebyshev (L-infinity) distance matrix for an N x m template array."""
    return np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)


def sample_entropy(window: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample entropy (Richman & Moorman 2000).

    Lower for predictable signals, higher for chaotic ones.

    Parameters
    ----------
    window : 1D signal
    m : embedding dimension (typical: 2)
    r : tolerance as fraction of signal std (typical: 0.2)
    """
    x = np.asarray(window, dtype=float)
    N = len(x)
    if N <= m + 1:
        return 0.0
    sd = np.std(x)
    if sd == 0:
        return 0.0
    tol = r * sd

    def _count_matches(m_: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(x, m_)
        dists = _chebyshev_distance_matrix(templates)
        mask = (dists <= tol) & ~np.eye(len(templates), dtype=bool)
        return float(np.sum(mask)) / 2.0  # unique pairs only

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if A == 0 or B == 0:
        return 0.0
    return float(-np.log(A / B))


def approximate_entropy(window: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate entropy (Pincus 1991). Includes self-matches; slightly biased."""
    x = np.asarray(window, dtype=float)
    N = len(x)
    if N <= m + 1:
        return 0.0
    sd = np.std(x)
    if sd == 0:
        return 0.0
    tol = r * sd

    def _phi(m_: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(x, m_)
        n = len(templates)
        dists = _chebyshev_distance_matrix(templates)
        counts = np.sum(dists <= tol, axis=1)  # includes self
        return float(np.mean(np.log(counts / n)))

    return float(_phi(m) - _phi(m + 1))


def entropy_features(window: np.ndarray) -> Dict[str, float]:
    """Compute sample and approximate entropy of the signal window."""
    return {
        "sample_entropy": sample_entropy(window),
        "approx_entropy": approximate_entropy(window),
    }


# ---------------------------------------------------------------------------
# Signal-quality index (SQI) ensemble
# ---------------------------------------------------------------------------
#
# Rationale (Clifford et al. 2012; Behar et al. 2013; Li et al. 2008):
#     No single SQI is robust to every artifact type. ICD/AED and clinical
#     wearable practice combines several complementary indices and inhibits
#     when any one flags poor quality. Here we add three interpretable,
#     signal-only (no-label, causal) indices to complement the existing
#     hf_noise_ratio / lf_wander_ratio FFT gates:
#
#         kSQI  kurtosis of the window. Clean ECG is highly peaked (large
#               positive kurtosis) because of sharp QRS complexes. EMG / motion
#               / saturation noise flattens the distribution -> kurtosis drops.
#         pSQI  relative spectral power in the QRS band (5-15 Hz) vs the wider
#               5-40 Hz band. High for QRS-dominated clean ECG; low when
#               broadband noise or wander dominates.
#         bSQI  agreement between two independent R-peak detectors on the same
#               window (Li et al. 2008). High agreement -> trustworthy beats;
#               low agreement -> the peak train (and every rr__ / morph__
#               feature derived from it) is unreliable. Requires two detectors,
#               so it is computed by the caller and passed in, not derived from
#               the raw window alone.
#
# All three are OFF by default in full_features(); enabling them adds
# signal__ksqi / signal__psqi / signal__bsqi to the feature vector. The hard-rule
# limits live in decision/config.py and only fire when these keys are present,
# so existing frozen benchmarks are unaffected until the ensemble is enabled.


def kurtosis_sqi(window: np.ndarray) -> float:
    """
    kSQI: Fisher (excess) kurtosis of the window.

    Clean single-lead ECG is leptokurtic (sharp QRS spikes on a quiet
    baseline) with kurtosis typically well above 0. Broadband EMG, motion, or
    saturation artifact flatten the amplitude distribution and drive kurtosis
    toward or below 0. Returns NaN for degenerate windows.
    """
    x = np.asarray(window, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan")
    sd = np.std(x)
    if sd < 1e-12:
        return float("nan")
    z = (x - x.mean()) / sd
    # Excess kurtosis (0 for a Gaussian).
    return float(np.mean(z ** 4) - 3.0)


def power_spectrum_sqi(
    window: np.ndarray,
    fs: float,
    qrs_band: tuple = (5.0, 15.0),
    ref_band: tuple = (5.0, 40.0),
) -> float:
    """
    pSQI: fraction of spectral power in the QRS band relative to a wider band.

    pSQI = P[qrs_band] / P[ref_band]. QRS-dominated clean ECG concentrates
    power in ~5-15 Hz, so pSQI is high (typically > 0.5). Broadband noise
    spreads power across the reference band and lowers pSQI. Returns NaN when
    the window is too short or fs is invalid.
    """
    x = np.asarray(window, dtype=float)
    n = len(x)
    if n < 32 or not (np.isfinite(fs) and fs > 0):
        return float("nan")
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    ref_mask = (freqs >= ref_band[0]) & (freqs < ref_band[1])
    ref_power = float(np.sum(psd[ref_mask]))
    if ref_power < 1e-12:
        return float("nan")
    qrs_mask = (freqs >= qrs_band[0]) & (freqs < qrs_band[1])
    qrs_power = float(np.sum(psd[qrs_mask]))
    return qrs_power / ref_power


def beat_detector_agreement_sqi(
    peaks_ref_s: np.ndarray,
    peaks_alt_s: np.ndarray,
    tol_ms: float = 150.0,
) -> float:
    """
    bSQI: agreement between two R-peak detectors (Li et al. 2008).

    bSQI = N_matched / (N_ref + N_alt - N_matched), i.e. matched peaks over the
    union, in [0, 1]. A peak in one train matches a peak in the other if they
    fall within ``tol_ms`` of each other (greedy one-to-one matching in time
    order). 1.0 = perfect agreement; low values mean the beat train is
    unreliable and rr__/morph__ features should not be trusted.

    Both inputs are peak timestamps in seconds. Returns:
        1.0  if both detectors agree there are no beats (both empty),
        0.0  if exactly one detector found beats (maximal disagreement),
        NaN  is never returned (fail-closed handled by the caller's hard rule).
    """
    a = np.sort(np.asarray(peaks_ref_s, dtype=float).ravel())
    b = np.sort(np.asarray(peaks_alt_s, dtype=float).ravel())
    n_a, n_b = len(a), len(b)
    if n_a == 0 and n_b == 0:
        return 1.0
    if n_a == 0 or n_b == 0:
        return 0.0
    tol_s = float(tol_ms) / 1000.0
    i = j = matched = 0
    while i < n_a and j < n_b:
        dt = a[i] - b[j]
        if abs(dt) <= tol_s:
            matched += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    union = n_a + n_b - matched
    return float(matched) / float(union) if union > 0 else 1.0


def signal_quality_index_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute the signal-only SQI ensemble (kSQI, pSQI).

    bSQI needs a second R-peak detector and is added separately by the caller
    (see full_features(..., bsqi=...)).
    """
    return {
        "ksqi": kurtosis_sqi(window),
        "psqi": power_spectrum_sqi(window, fs),
    }


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def signal_features(
    window: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> Dict[str, float]:
    """
    Compute all signal-only Layer 2 features (wavelet + entropy).

    To be merged with the existing rhythm_features.compute(...) output
    for the full Layer 2 feature vector.
    """
    features: Dict[str, float] = {}
    features.update(wavelet_features(window, wavelet=wavelet, level=level))
    features.update(entropy_features(window))
    return features


# Backward-compatible alias for old notebooks/scripts.
layer2_extra_features = signal_features
