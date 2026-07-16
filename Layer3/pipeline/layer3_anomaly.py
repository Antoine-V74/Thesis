"""
Layer 3 — anomaly head on top of the pretrained encoder.

Implements Deep SVDD (Ruff et al. 2018) as an optional ablation:

    The main Layer 3 path is a learned ECG embedding anomaly veto with a
    healthy-baseline distance model such as Mahalanobis or kNN. Deep SVDD is
    useful for comparison, but should not be presented as the primary runtime
    method unless it wins that ablation.

    Given a pretrained encoder f(x) → R^d,
    learn a small head g(z) → R^k that maps healthy-rat embeddings
    as close as possible to a fixed center c in R^k.

    Anomaly score at test time:  s(x) = || g(f(x)) - c ||_2
    Decision:                    inhibit if s(x) > threshold

The center c is computed once from a forward pass over the healthy
calibration data (standard Deep SVDD initialization). The head is then
trained to minimize squared distance to c on healthy data.

Why Deep SVDD vs autoencoder:
    - Faster, smaller, more interpretable score
    - No decoder to maintain
    - Suited to embedded deployment (the head can be <10K parameters)

Per-rat calibration matches the Layer 2 philosophy: each rat gets its
own anomaly head trained on its own healthy baseline. The pretrained
encoder is shared across all rats; only the head is rat-specific.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from layer3_encoder import ECGEncoder1D


# ---------------------------------------------------------------------------
# Head architecture
# ---------------------------------------------------------------------------

class DeepSVDDHead(nn.Module):
    """
    Small MLP head mapping encoder embeddings to the SVDD output space.

    NO BIAS TERMS: Ruff et al. show bias terms can cause "hypersphere
    collapse" where the head learns to output the constant c regardless
    of input. Removing biases avoids this trivial solution.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Dataset wrapper for healthy ECG windows
# ---------------------------------------------------------------------------

class HealthyWindowDataset(Dataset):
    """Wraps a numpy array of healthy windows for the SVDD training loop."""

    def __init__(self, windows: np.ndarray):
        # windows: (N, T) float array of 1-channel signals
        if windows.ndim != 2:
            raise ValueError(f"Expected (N, T) array, got shape {windows.shape}")
        # Z-score per window
        m = windows.mean(axis=1, keepdims=True)
        s = windows.std(axis=1, keepdims=True) + 1e-8
        self.x = ((windows - m) / s).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.x[i]).unsqueeze(0)  # (1, T)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class SVDDConfig:
    embedding_dim: int = 128
    hidden_dim: int = 64
    out_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    batch_size: int = 64
    threshold_quantile: float = 0.999


def _compute_center(
    encoder: ECGEncoder1D,
    head: DeepSVDDHead,
    loader: DataLoader,
    device: str,
    eps: float = 0.1,
) -> torch.Tensor:
    """
    Initialize SVDD center c as the mean of head(encoder(x)) over healthy data.

    Small-eps push-away rule from Ruff et al.: components of c near zero are
    nudged away from zero to prevent collapse.
    """
    encoder.eval()
    head.eval()
    out_sum = None
    n = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            z = encoder(x)
            o = head(z)
            out_sum = o.sum(dim=0) if out_sum is None else out_sum + o.sum(dim=0)
            n += o.size(0)
    c = (out_sum / max(1, n)).detach()
    # Avoid trivial zero center
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c


def fit_svdd(
    encoder: ECGEncoder1D,
    healthy_windows: np.ndarray,
    cfg: Optional[SVDDConfig] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DeepSVDDHead, torch.Tensor, float]:
    """
    Train a Deep SVDD head on healthy rat windows.

    Returns
    -------
    head : trained DeepSVDDHead
    center : fixed center vector c
    threshold : score threshold from in-baseline quantile
    """
    cfg = cfg or SVDDConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    head = DeepSVDDHead(
        in_dim=cfg.embedding_dim,
        hidden=cfg.hidden_dim,
        out_dim=cfg.out_dim,
    ).to(device)

    ds = HealthyWindowDataset(healthy_windows)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Initialize center c
    center = _compute_center(encoder, head, loader, device)
    if verbose:
        print(f"SVDD center initialized, shape={tuple(center.shape)}, "
              f"||c||={center.norm().item():.3f}")

    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        head.train()
        ep_loss = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                z = encoder(x)
            o = head(z)
            dist2 = torch.sum((o - center) ** 2, dim=1)
            loss = dist2.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  mean_dist2={ep_loss / max(1, n):.4f}")

    # Set threshold from in-baseline score quantile
    head.eval()
    scores = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            z = encoder(x)
            o = head(z)
            d = torch.sqrt(torch.sum((o - center) ** 2, dim=1))
            scores.append(d.cpu().numpy())
    scores = np.concatenate(scores)
    threshold = float(np.quantile(scores, cfg.threshold_quantile))
    if verbose:
        print(f"Baseline score stats: mean={scores.mean():.3f}, "
              f"std={scores.std():.3f}, threshold@{cfg.threshold_quantile:.3f}={threshold:.3f}")

    return head, center, threshold


# ---------------------------------------------------------------------------
# Scoring + gate
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_windows(
    encoder: ECGEncoder1D,
    head: DeepSVDDHead,
    center: torch.Tensor,
    windows: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute SVDD anomaly scores for a batch of windows."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    center = center.to(device)
    ds = HealthyWindowDataset(windows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for x in loader:
        x = x.to(device)
        z = encoder(x)
        o = head(z)
        d = torch.sqrt(torch.sum((o - center) ** 2, dim=1))
        out.append(d.cpu().numpy())
    return np.concatenate(out)


def decide_window(
    score: float,
    threshold: float,
) -> dict:
    """Per-window permit/inhibit decision with reasoning."""
    inhibit = bool(score > threshold)
    return {
        "permit": not inhibit,
        "inhibit": inhibit,
        "anomaly_score": float(score),
        "threshold": float(threshold),
        "score_over_threshold_ratio": float(score / threshold) if threshold > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_svdd(
    path: str | Path,
    head: DeepSVDDHead,
    center: torch.Tensor,
    threshold: float,
    config: SVDDConfig,
) -> None:
    """Save head weights + center + threshold to a single .pt file."""
    torch.save({
        "head_state_dict": head.state_dict(),
        "center": center.cpu(),
        "threshold": float(threshold),
        "config": asdict(config),
    }, path)


def load_svdd(
    path: str | Path,
    device: Optional[str] = None,
) -> Tuple[DeepSVDDHead, torch.Tensor, float, SVDDConfig]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device)
    cfg = SVDDConfig(**blob["config"])
    head = DeepSVDDHead(
        in_dim=cfg.embedding_dim,
        hidden=cfg.hidden_dim,
        out_dim=cfg.out_dim,
    ).to(device)
    head.load_state_dict(blob["head_state_dict"])
    center = blob["center"].to(device)
    threshold = float(blob["threshold"])
    return head, center, threshold, cfg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fake healthy rat data: 200 windows of 1250 samples
    rng = np.random.default_rng(0)
    fs = 250
    T = 5 * fs

    # Healthy = periodic pulses + small noise
    healthy = np.zeros((200, T), dtype=np.float32)
    for i in range(200):
        rate_hz = rng.uniform(5.0, 7.0)  # ~300-420 bpm, rat range
        x = 0.05 * rng.standard_normal(T)
        for k in range(int(T / fs * rate_hz)):
            j = int(k / rate_hz * fs)
            if j + 5 < T:
                x[j:j+5] += np.array([0, 0.4, 1.0, -0.4, 0.1], dtype=np.float32)
        healthy[i] = x

    # Anomalous = chaotic high-variance noise
    anomalous = 1.5 * rng.standard_normal((40, T)).astype(np.float32)

    # Train encoder briefly (random init for this smoke test)
    enc = ECGEncoder1D()
    print(f"Encoder: {sum(p.numel() for p in enc.parameters()):,} params")

    # Fit SVDD on healthy
    cfg = SVDDConfig(epochs=10, batch_size=32)
    head, center, threshold = fit_svdd(enc, healthy, cfg=cfg, verbose=True)

    # Score test windows
    healthy_scores = score_windows(enc, head, center, healthy[:40])
    anomalous_scores = score_windows(enc, head, center, anomalous)
    print(f"\nHealthy scores:   mean={healthy_scores.mean():.3f}, "
          f"max={healthy_scores.max():.3f}")
    print(f"Anomalous scores: mean={anomalous_scores.mean():.3f}, "
          f"max={anomalous_scores.max():.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Anomalous detected as inhibit: "
          f"{(anomalous_scores > threshold).sum()}/{len(anomalous_scores)}")
    print(f"Healthy false-inhibit:          "
          f"{(healthy_scores > threshold).sum()}/{len(healthy_scores)}")

    print("\nNOTE: With a randomly-initialized encoder, scores are not meaningful — "
          "this only verifies the training loop runs end-to-end. "
          "After SSL pretraining, healthy < anomalous should hold.")
