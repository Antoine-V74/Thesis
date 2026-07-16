"""
Layer 3 — physiology-aware ECG augmentations for SSL pretraining.

DESIGN PRINCIPLE: every augmentation must preserve the *semantic content*
of the ECG (whether it's sinus, AF, VT, VFib, or noise) while introducing
nuisance variability that the encoder should learn to be invariant to.

Augmentations included (3KG-inspired, adapted to single-lead):
    - Time warping (small, ±10% rate)
    - Gaussian noise (SNR-controlled)
    - Baseline wander injection (low-frequency sinusoid)
    - Bandpass cutoff perturbation
    - Random crop with re-padding

NOT included (would change ECG semantics):
    - Polarity inversion (changes Q/S morphology meaning)
    - Aggressive frequency masking (destroys QRS)
    - Strong amplitude rescaling (could collapse sinus/VFib distinguishability)
    - Random permutation (destroys temporal structure)

Reference:
    Gopal et al. 2021, "3KG: Contrastive Learning of 12-Lead ECGs
    using Physiologically-Inspired Augmentations" — for the augmentation
    selection rationale. We use a subset valid for single-lead signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt


@dataclass
class AugmentConfig:
    fs: int = 250
    time_warp_range: float = 0.03
    noise_snr_db_range: tuple = (20.0, 35.0)
    wander_amp_range: tuple = (0.0, 0.05)
    wander_freq_range: tuple = (0.1, 0.5)
    bandpass_lo_range: tuple = (0.3, 0.8)
    bandpass_hi_range: tuple = (35.0, 50.0)
    crop_frac_range: tuple = (0.95, 1.0)
    p_time_warp: float = 0.0
    p_noise: float = 0.7
    p_wander: float = 0.5
    p_bandpass: float = 0.0
    p_crop: float = 0.1

    # Safety-veto default: keep morphology-changing transforms as ablations.


# ---------------------------------------------------------------------------
# Individual augmentations (operate on 1D torch tensors or numpy arrays)
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy(), True
    return np.asarray(x, dtype=np.float32), False


def _back_to_torch(x, was_tensor, like=None):
    if was_tensor:
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
        if like is not None:
            t = t.to(like.device, dtype=like.dtype)
        return t
    return x


def time_warp(window, fs: int, warp: float, rng: np.random.Generator):
    """Stretch or compress the time axis by a random factor in [1-warp, 1+warp]."""
    x, was_t = _to_numpy(window)
    factor = float(rng.uniform(1.0 - warp, 1.0 + warp))
    n_new = max(8, int(round(len(x) * factor)))
    # Linear interpolation onto a different number of samples, then resample back to original length
    t_new = np.linspace(0, len(x) - 1, n_new)
    warped = np.interp(t_new, np.arange(len(x)), x)
    # Restore original length
    t_back = np.linspace(0, n_new - 1, len(x))
    out = np.interp(t_back, np.arange(n_new), warped).astype(np.float32)
    return _back_to_torch(out, was_t, like=window if was_t else None)


def add_gaussian_noise(window, snr_db_range, rng: np.random.Generator):
    """Add Gaussian noise at a randomly chosen SNR within range."""
    x, was_t = _to_numpy(window)
    snr_db = float(rng.uniform(*snr_db_range))
    sig_power = float(np.mean(x ** 2) + 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return _back_to_torch(x + noise, was_t, like=window if was_t else None)


def baseline_wander(window, fs: int, amp_range, freq_range, rng: np.random.Generator):
    """Add a low-frequency sinusoid simulating respiratory baseline wander."""
    x, was_t = _to_numpy(window)
    amp = float(rng.uniform(*amp_range))
    freq = float(rng.uniform(*freq_range))
    phase = float(rng.uniform(0.0, 2 * np.pi))
    t = np.arange(len(x), dtype=np.float32) / fs
    wander = amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    return _back_to_torch(x + wander, was_t, like=window if was_t else None)


def bandpass_perturbation(window, fs: int, lo_range, hi_range, rng: np.random.Generator):
    """Apply a bandpass filter with randomly chosen cutoffs (mild)."""
    x, was_t = _to_numpy(window)
    lo = float(rng.uniform(*lo_range))
    hi = float(rng.uniform(*hi_range))
    if hi >= fs / 2:
        hi = fs / 2 - 1
    if lo >= hi:
        return _back_to_torch(x, was_t, like=window if was_t else None)
    sos = butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x).astype(np.float32)
    return _back_to_torch(y, was_t, like=window if was_t else None)


def random_crop_repad(window, fs: int, crop_frac_range, rng: np.random.Generator):
    """Crop to a fraction of original length, then pad back with zero-tail to original length."""
    x, was_t = _to_numpy(window)
    n = len(x)
    frac = float(rng.uniform(*crop_frac_range))
    n_keep = max(8, int(round(frac * n)))
    start_max = n - n_keep
    start = int(rng.integers(0, max(1, start_max + 1)))
    cropped = x[start:start + n_keep]
    out = np.zeros(n, dtype=np.float32)
    pad_left = (n - n_keep) // 2
    out[pad_left:pad_left + n_keep] = cropped
    return _back_to_torch(out, was_t, like=window if was_t else None)


# ---------------------------------------------------------------------------
# Composite augmentor
# ---------------------------------------------------------------------------

class ECGAugmentor:
    """
    Stochastic composition of physiology-aware augmentations.

    Each augmentation is applied independently with probability p_*.
    Call .augment(window) twice on the same window to get two contrastive views.
    """

    def __init__(self, cfg: Optional[AugmentConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def augment(self, window):
        """Apply a stochastic chain of augmentations to a single window."""
        cfg = self.cfg
        x = window
        if self.rng.random() < cfg.p_time_warp:
            x = time_warp(x, cfg.fs, cfg.time_warp_range, self.rng)
        if self.rng.random() < cfg.p_noise:
            x = add_gaussian_noise(x, cfg.noise_snr_db_range, self.rng)
        if self.rng.random() < cfg.p_wander:
            x = baseline_wander(x, cfg.fs, cfg.wander_amp_range,
                                cfg.wander_freq_range, self.rng)
        if self.rng.random() < cfg.p_bandpass:
            x = bandpass_perturbation(x, cfg.fs, cfg.bandpass_lo_range,
                                      cfg.bandpass_hi_range, self.rng)
        if self.rng.random() < cfg.p_crop:
            x = random_crop_repad(x, cfg.fs, cfg.crop_frac_range, self.rng)
        return x

    def augment_pair(self, window):
        """Produce two independently augmented views of the same window."""
        return self.augment(window), self.augment(window)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fs = 250
    t = np.arange(5 * fs) / fs
    # Synthetic QRS-train
    x = np.zeros_like(t, dtype=np.float32)
    for k in range(int(t[-1] * 1.2)):
        i = int(k / 1.2 * fs)
        if i + 5 < len(x):
            x[i:i+5] += np.array([0.0, 0.3, 1.0, -0.5, 0.1], dtype=np.float32)

    aug = ECGAugmentor(seed=0)
    v1, v2 = aug.augment_pair(x)

    print(f"Original window: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"View 1:          shape={v1.shape}, mean={v1.mean():.4f}, std={v1.std():.4f}")
    print(f"View 2:          shape={v2.shape}, mean={v2.mean():.4f}, std={v2.std():.4f}")
    print(f"v1 == v2? {np.allclose(v1, v2)}   (should be False — augmentations are stochastic)")
    print(f"v1 == x?  {np.allclose(v1, x)}    (should be False — augmentation occurred)")
