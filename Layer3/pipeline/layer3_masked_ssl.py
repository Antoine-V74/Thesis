"""
ZEROSHOT-inspired masked reconstruction + subject-contrastive SSL utilities.

This module is training-only. The reconstruction decoder is discarded after
pretraining; downstream Layer 3 anomaly scores must come from encoder embedding
distance (Mahalanobis/kNN), never reconstruction error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
for path in (PROJECT_ROOT, LAYER3_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from layer3_encoder import ECGEncoder1D, EncoderConfig, ProjectionHead  # noqa: E402


@dataclass
class MaskedSSLConfig:
    embedding_dim: int = 128
    projection_dim: int = 64
    mask_ratio: float = 0.75
    patch_size: int = 25
    subject_contrastive_lambda: float = 0.30
    temperature: float = 0.1
    decoder_base_channels: int = 128


def make_patch_mask(
    x: torch.Tensor,
    *,
    mask_ratio: float = 0.75,
    patch_size: int = 25,
) -> torch.Tensor:
    """Return a boolean mask shaped like x, masking whole temporal patches."""
    if x.ndim != 3:
        raise ValueError(f"Expected x as (B, C, T), got {tuple(x.shape)}")
    b, c, t = x.shape
    patch_size = max(1, int(patch_size))
    n_patches = int((t + patch_size - 1) // patch_size)
    mask_ratio = float(min(max(mask_ratio, 0.0), 1.0))
    n_mask = int(round(mask_ratio * n_patches))
    patch_mask = torch.zeros((b, n_patches), dtype=torch.bool, device=x.device)
    if n_mask > 0:
        noise = torch.rand((b, n_patches), device=x.device)
        idx = torch.argsort(noise, dim=1)[:, :n_mask]
        patch_mask.scatter_(1, idx, True)
    sample_mask = patch_mask.repeat_interleave(patch_size, dim=1)[:, :t]
    return sample_mask[:, None, :].expand(b, c, t)


class MaskedConvDecoder1D(nn.Module):
    """Small ConvTranspose decoder from global embedding to a 1D signal."""

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        output_len: int = 1000,
        base_channels: int = 128,
    ):
        super().__init__()
        self.output_len = int(output_len)
        self.base_len = max(1, int((self.output_len + 15) // 16))
        self.base_channels = int(base_channels)
        self.fc = nn.Linear(int(embedding_dim), self.base_channels * self.base_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(self.base_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, output_len: int | None = None) -> torch.Tensor:
        target_len = self.output_len if output_len is None else int(output_len)
        h = self.fc(z).view(z.shape[0], self.base_channels, self.base_len)
        y = self.net(h)
        if y.shape[-1] > target_len:
            y = y[..., :target_len]
        elif y.shape[-1] < target_len:
            y = F.pad(y, (0, target_len - y.shape[-1]))
        return y


class SubjectContrastiveLoss(nn.Module):
    """Supervised contrastive loss with positives sharing the same subject id."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, projections: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        z = F.normalize(projections, dim=1)
        labels = subject_ids.view(-1)
        n = int(z.shape[0])
        if n < 2:
            return z.new_tensor(0.0)

        sim = z @ z.t() / self.temperature
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        same = labels[:, None].eq(labels[None, :]) & ~self_mask
        if not same.any():
            return z.new_tensor(0.0)

        logits = sim.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        per_anchor = -(log_prob.masked_fill(~same, 0.0).sum(dim=1) / same.sum(dim=1).clamp_min(1))
        valid = same.any(dim=1)
        return per_anchor[valid].mean()


class MaskedSubjectContrastiveModel(nn.Module):
    """Encoder + training-only reconstruction decoder + projection head."""

    def __init__(
        self,
        *,
        encoder: ECGEncoder1D,
        output_len: int,
        config: MaskedSSLConfig | None = None,
    ):
        super().__init__()
        self.config = config or MaskedSSLConfig()
        self.encoder = encoder
        self.decoder = MaskedConvDecoder1D(
            embedding_dim=self.config.embedding_dim,
            output_len=int(output_len),
            base_channels=self.config.decoder_base_channels,
        )
        self.projection = ProjectionHead(
            in_dim=self.config.embedding_dim,
            out_dim=self.config.projection_dim,
        )
        self.subject_loss = SubjectContrastiveLoss(temperature=self.config.temperature)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = make_patch_mask(
            x,
            mask_ratio=self.config.mask_ratio,
            patch_size=self.config.patch_size,
        )
        x_masked = x.masked_fill(mask, 0.0)
        z = self.encoder(x_masked)
        recon = self.decoder(z, output_len=x.shape[-1])
        if mask.any():
            recon_loss = F.mse_loss(recon[mask], x[mask])
        else:
            recon_loss = F.mse_loss(recon, x)
        proj = self.projection(z)
        contrastive_loss = self.subject_loss(proj, subject_ids)
        total = recon_loss + float(self.config.subject_contrastive_lambda) * contrastive_loss
        return total, {
            "loss": total.detach(),
            "reconstruction_loss": recon_loss.detach(),
            "subject_contrastive_loss": contrastive_loss.detach(),
            "mask_fraction": mask.float().mean().detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MaskedSSLConfig(mask_ratio=0.75, patch_size=25)
    encoder = ECGEncoder1D(EncoderConfig(embedding_dim=cfg.embedding_dim))
    model = MaskedSubjectContrastiveModel(encoder=encoder, output_len=1000, config=cfg)
    x = torch.randn(8, 1, 1000)
    subject_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    loss, logs = model(x, subject_ids)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert 0.70 <= float(logs["mask_fraction"]) <= 0.80, logs["mask_fraction"]
    print("layer3_masked_ssl smoke test: OK")
