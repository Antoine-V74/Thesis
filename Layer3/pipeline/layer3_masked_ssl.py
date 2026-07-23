"""
Masked-reconstruction SSL utilities for the Layer 3 "B" arm family.

Two objectives live here, both training-only (the reconstruction decoder is
discarded after pretraining; downstream Layer 3 anomaly scores must come from
encoder embedding distance (Mahalanobis/kNN), NEVER reconstruction error):

    B  (primary)  — MaskedConsistencyModel:
        masked reconstruction + non-contrastive *same-window* consistency.
        Two independent masks of the SAME window form two views; a VICReg-style
        (invariance + variance + covariance) term aligns their embeddings.
        The invariance unit is the WINDOW, never the subject/record, so healthy
        and abnormal beats from one recording are not pulled together.
        Spirit of NERULA (masked + non-contrastive); distinct from A1 because B
        keeps a reconstruction objective and views come from masking, not augs.

    B1 (ablation) — MaskedSubjectContrastiveModel:
        masked reconstruction + subject/record contrastive (Yu / ZEROSHOT-style).
        Kept only as an ablation because subject-level positives can incorrectly
        align normal and abnormal windows from the same record. Prefer running it
        with healthy-only pretraining.

See Layer3/reports/LAYER3_ARM_B_B1_SPEC.md for the arm rationale and protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Tuple

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
from layer3_vicreg import VICRegConfig, VICRegExpander, vicreg_loss  # noqa: E402


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


@dataclass
class MaskedConsistencyConfig:
    """Config for the primary B arm: masked recon + same-window consistency."""

    embedding_dim: int = 128
    mask_ratio: float = 0.75
    patch_size: int = 25
    decoder_base_channels: int = 128
    # Weight on the non-contrastive same-window consistency term, relative to the
    # reconstruction MSE. The consistency term itself is internally weighted by
    # the VICReg sim/var/cov coefficients below.
    consistency_lambda: float = 1.0
    expander_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    vicreg_sim_coeff: float = 25.0
    vicreg_var_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0


class MaskedConsistencyModel(nn.Module):
    """Encoder + training-only decoder + expander for the primary B arm.

    Forward pass builds TWO independent masks of the SAME window, reconstructs
    each masked view, and aligns the two embeddings with a VICReg-style
    (non-contrastive) consistency term. There is no subject/record label and no
    contrastive negative: the only invariance asserted is "two masked views of
    the same window should embed similarly", which is safe for anomaly detection.
    """

    def __init__(
        self,
        *,
        encoder: ECGEncoder1D,
        output_len: int,
        config: MaskedConsistencyConfig | None = None,
    ):
        super().__init__()
        self.config = config or MaskedConsistencyConfig()
        self.encoder = encoder
        self.decoder = MaskedConvDecoder1D(
            embedding_dim=self.config.embedding_dim,
            output_len=int(output_len),
            base_channels=self.config.decoder_base_channels,
        )
        self.expander = VICRegExpander(self.config.embedding_dim, self.config.expander_dims)
        self._vicreg_cfg = VICRegConfig(
            embedding_dim=self.config.embedding_dim,
            expander_dims=list(self.config.expander_dims),
            sim_coeff=self.config.vicreg_sim_coeff,
            var_coeff=self.config.vicreg_var_coeff,
            cov_coeff=self.config.vicreg_cov_coeff,
        )

    def _masked_recon(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return z, recon_loss, mask.float().mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_a, recon_a, frac_a = self._masked_recon(x)
        z_b, recon_b, frac_b = self._masked_recon(x)
        recon_loss = 0.5 * (recon_a + recon_b)

        p_a = self.expander(z_a)
        p_b = self.expander(z_b)
        consistency_loss, cons_logs = vicreg_loss(p_a, p_b, self._vicreg_cfg)

        total = recon_loss + float(self.config.consistency_lambda) * consistency_loss
        return total, {
            "loss": total.detach(),
            "reconstruction_loss": recon_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "invariance_loss": cons_logs["invariance_loss"],
            "variance_loss": cons_logs["variance_loss"],
            "covariance_loss": cons_logs["covariance_loss"],
            "embedding_std": cons_logs["embedding_std"],
            "mask_fraction": (0.5 * (frac_a + frac_b)).detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    output_len = 1000

    # B1 (ablation): masked reconstruction + subject contrastive.
    cfg_b1 = MaskedSSLConfig(mask_ratio=0.75, patch_size=25)
    enc_b1 = ECGEncoder1D(EncoderConfig(embedding_dim=cfg_b1.embedding_dim))
    model_b1 = MaskedSubjectContrastiveModel(encoder=enc_b1, output_len=output_len, config=cfg_b1)
    x = torch.randn(8, 1, output_len)
    subject_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    loss_b1, logs_b1 = model_b1(x, subject_ids)
    assert torch.isfinite(loss_b1), loss_b1
    loss_b1.backward()
    assert 0.70 <= float(logs_b1["mask_fraction"]) <= 0.80, logs_b1["mask_fraction"]

    # B (primary): masked reconstruction + non-contrastive same-window consistency.
    cfg_b = MaskedConsistencyConfig(mask_ratio=0.75, patch_size=25)
    enc_b = ECGEncoder1D(EncoderConfig(embedding_dim=cfg_b.embedding_dim))
    model_b = MaskedConsistencyModel(encoder=enc_b, output_len=output_len, config=cfg_b)
    loss_b, logs_b = model_b(x)
    assert torch.isfinite(loss_b), loss_b
    loss_b.backward()
    assert 0.70 <= float(logs_b["mask_fraction"]) <= 0.80, logs_b["mask_fraction"]
    assert float(logs_b["embedding_std"]) > 0.0, logs_b["embedding_std"]

    print(
        "layer3_masked_ssl smoke test: OK "
        f"(B1 loss={float(logs_b1['loss']):.4f}; "
        f"B loss={float(logs_b['loss']):.4f}, recon={float(logs_b['reconstruction_loss']):.4f}, "
        f"cons={float(logs_b['consistency_loss']):.4f})"
    )
