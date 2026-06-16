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
