"""
Layer 3 — VICReg (A1) non-contrastive SSL utilities.

VICReg (Bardes, Ponce, LeCun 2022) learns embeddings from two augmented views
of the same input using three terms and NO negative pairs and NO EMA teacher:

    invariance : MSE between the two views' expander outputs
    variance   : hinge keeping each dimension's std above a floor (anti-collapse)
    covariance : off-diagonal covariance penalty (decorrelate dimensions)

This module is training-only. The expander is discarded after pretraining;
downstream Layer 3 anomaly scores must come from encoder embedding distance
(Mahalanobis/kNN), never from any VICReg loss term or expander output.

See Layer3/reports/VICREG_A1_IMPLEMENTATION_PLAN.md for the arm rationale.
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

from layer3_encoder import ECGEncoder1D, EncoderConfig  # noqa: E402


@dataclass
class VICRegConfig:
    embedding_dim: int = 128
    # Expander is wider than the embedding, per the VICReg convention.
    expander_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    sim_coeff: float = 25.0   # invariance weight (lambda)
    var_coeff: float = 25.0   # variance weight (mu)
    cov_coeff: float = 1.0    # covariance weight (nu)
    var_gamma: float = 1.0    # target std floor per dimension
    eps: float = 1e-4         # variance sqrt stabilizer


class VICRegExpander(nn.Module):
    """MLP expander: Linear -> BN -> ReLU blocks, final Linear (no activation)."""

    def __init__(self, in_dim: int, hidden_dims: List[int]):
        super().__init__()
        dims = [int(in_dim)] + [int(d) for d in hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _variance_loss(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    """Hinge keeping each dimension's standard deviation above gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariances, normalized by dimension."""
    n, d = z.shape
    if n < 2:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    cfg: VICRegConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the VICReg loss and its components for two expander outputs."""
    inv = F.mse_loss(z_a, z_b)
    var = _variance_loss(z_a, cfg.var_gamma, cfg.eps) + _variance_loss(z_b, cfg.var_gamma, cfg.eps)
    cov = _covariance_loss(z_a) + _covariance_loss(z_b)
    total = cfg.sim_coeff * inv + cfg.var_coeff * var + cfg.cov_coeff * cov
    logs = {
        "loss": total.detach(),
        "invariance_loss": inv.detach(),
        "variance_loss": var.detach(),
        "covariance_loss": cov.detach(),
        # Mean per-dimension std of view A: a live collapse monitor (should stay
        # near var_gamma, not drift toward 0).
        "embedding_std": torch.sqrt(z_a.var(dim=0) + cfg.eps).mean().detach(),
    }
    return total, logs


class VICRegModel(nn.Module):
    """Shared encoder + training-only expander for VICReg pretraining."""

    def __init__(
        self,
        *,
        encoder: ECGEncoder1D,
        config: VICRegConfig | None = None,
    ):
        super().__init__()
        self.config = config or VICRegConfig()
        self.encoder = encoder
        self.expander = VICRegExpander(self.config.embedding_dim, self.config.expander_dims)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_a = self.expander(self.encoder(x_a))
        z_b = self.expander(self.encoder(x_b))
        return vicreg_loss(z_a, z_b, self.config)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = VICRegConfig()
    encoder = ECGEncoder1D(EncoderConfig(embedding_dim=cfg.embedding_dim))
    model = VICRegModel(encoder=encoder, config=cfg)
    x_a = torch.randn(8, 1, 1000)
    x_b = x_a + 0.01 * torch.randn_like(x_a)
    loss, logs = model(x_a, x_b)
    assert torch.isfinite(loss), loss
    loss.backward()
    # Embeddings should not have collapsed on a single forward pass.
    assert float(logs["embedding_std"]) > 0.0, logs["embedding_std"]
    print(
        "layer3_vicreg smoke test: OK "
        f"(loss={float(logs['loss']):.4f}, inv={float(logs['invariance_loss']):.4f}, "
        f"var={float(logs['variance_loss']):.4f}, cov={float(logs['covariance_loss']):.4f}, "
        f"emb_std={float(logs['embedding_std']):.4f})"
    )
