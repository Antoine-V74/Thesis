"""
Layer 3 — ECG encoder architecture.

A small 1D ResNet that takes raw ECG windows and produces a fixed-dimensional
embedding. Designed to be the SHARED encoder across:

    - SSL contrastive pretraining (layer3_pretrain.py)
    - Anomaly head on healthy rat (layer3_anomaly.py)
    - Optional downstream supervised head on labeled human data

Architecture decisions:
    - Input:  1 channel × N samples (5s @ 250Hz = 1250 samples by default)
    - Output: 128-dim L2-normalizable embedding
    - ~500K parameters — fits comfortably on Pynapse server, fast inference.
    - GroupNorm preferred over BatchNorm because (a) we sometimes run with
      very small batches at inference time, (b) BN running stats are awkward
      with the patient-balanced batch sampling used during contrastive
      pretraining.

The projection head is only used during contrastive pretraining and is
discarded for downstream tasks (standard SimCLR / SupCon practice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Pre-activation 1D residual block with GroupNorm."""

    def __init__(self, channels: int, kernel: int = 7, groups: int = 8):
        super().__init__()
        pad = kernel // 2
        self.gn1 = nn.GroupNorm(min(groups, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.gn2 = nn.GroupNorm(min(groups, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(F.relu(self.gn1(x)))
        out = self.conv2(F.relu(self.gn2(out)))
        return out + identity


class DownsampleBlock(nn.Module):
    """Strided 1D conv that doubles channels and halves length."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, groups: int = 8):
        super().__init__()
        pad = kernel // 2
        self.gn = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=2, padding=pad)
        # 1x1 projection on the identity path so the skip matches dimensions
        self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv(F.relu(self.gn(x)))
        return out + identity


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    in_channels: int = 1
    base_channels: int = 32
    n_stages: int = 4            # each stage doubles channels and halves length
    blocks_per_stage: int = 2
    embedding_dim: int = 128
    kernel: int = 7
    groups: int = 8              # GroupNorm group count
    pooling_mode: str = "global_avg"  # locked default; exploratory: avg_max, causal_local_global
    local_fraction: float = 0.125     # final-map tail used by causal_local_global


class ECGEncoder1D(nn.Module):
    """
    Small 1D ResNet for ECG.

    For 5s @ 250Hz input (1250 samples) with default config:
        stem: 1 -> 32 channels, length 1250
        stage 1: 32 -> 64,  length 625
        stage 2: 64 -> 128, length ~312
        stage 3: 128 -> 256, length ~156
        stage 4: 256 -> 512, length ~78
        global pool -> Linear -> 128-dim embedding
        ~700K params
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        super().__init__()
        cfg = cfg or EncoderConfig()
        self.cfg = cfg

        ch = cfg.base_channels
        self.stem = nn.Conv1d(cfg.in_channels, ch, kernel_size=15, padding=7)

        stages = []
        for stage in range(cfg.n_stages):
            for _ in range(cfg.blocks_per_stage):
                stages.append(ResBlock1D(ch, kernel=cfg.kernel, groups=cfg.groups))
            next_ch = ch * 2
            stages.append(DownsampleBlock(ch, next_ch, kernel=cfg.kernel, groups=cfg.groups))
            ch = next_ch
        self.stages = nn.Sequential(*stages)

        self.final_gn = nn.GroupNorm(min(cfg.groups, ch), ch)
        pooling_mode = str(cfg.pooling_mode).lower()
        if pooling_mode not in {"global_avg", "avg_max", "causal_local_global"}:
            raise ValueError(
                f"Unsupported pooling_mode={cfg.pooling_mode!r}; "
                "use global_avg, avg_max, or causal_local_global"
            )
        self.pooling_mode = pooling_mode
        pooled_channels = ch if pooling_mode == "global_avg" else 2 * ch
        self.head_linear = nn.Linear(pooled_channels, cfg.embedding_dim)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, T) where T is window length in samples
        returns : (B, embedding_dim)
        """
        h = self.stem(x)
        h = self.stages(h)
        h = F.relu(self.final_gn(h))
        global_avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        if self.pooling_mode == "global_avg":
            pooled = global_avg
        elif self.pooling_mode == "avg_max":
            # Max pooling preserves short high-response morphology that global
            # averaging can dilute across an 8 s rhythm window.
            global_max = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            pooled = torch.cat([global_avg, global_max], dim=1)
        else:
            # Beat-synchronous causal windows place the current trigger near the
            # right edge. Pair full-window rhythm context with a local tail pool.
            fraction = float(min(1.0, max(1e-3, self.cfg.local_fraction)))
            local_n = max(1, int(round(h.shape[-1] * fraction)))
            local_avg = h[..., -local_n:].mean(dim=-1)
            pooled = torch.cat([global_avg, local_avg], dim=1)
        z = self.head_linear(pooled)
        return z


# ---------------------------------------------------------------------------
# Projection head (used only during contrastive pretraining)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP projection head, output L2-normalized for NT-Xent."""

    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=1)


class EncoderWithProjection(nn.Module):
    """Wraps the encoder + projection head together for pretraining."""

    def __init__(self, encoder: ECGEncoder1D, projection: ProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        p = self.projection(z)
        return z, p


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = EncoderConfig()
    enc = ECGEncoder1D(cfg)
    proj = ProjectionHead(in_dim=cfg.embedding_dim)
    model = EncoderWithProjection(enc, proj)

    # Test forward pass on a fake batch
    fs = 250
    x = torch.randn(8, 1, 5 * fs)  # 8 windows, 1 channel, 5s @ 250Hz
    z, p = model(x)
    assert z.shape == (8, cfg.embedding_dim), z.shape
    assert p.shape == (8, 64), p.shape
    assert torch.allclose(p.norm(dim=1), torch.ones(8), atol=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder + projection: {n_params:,} parameters")
    print(f"Embedding shape: {z.shape}, projection shape: {p.shape}")
    print(f"Projection L2-normalized: yes (norms = {p.norm(dim=1)[:3].tolist()})")
