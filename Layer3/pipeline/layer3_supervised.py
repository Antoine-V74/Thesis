"""
Supervised representation utilities for Layer 3 Arm C.

Primary objective: Supervised Contrastive (SupCon / Khosla et al. 2020).
Public safety labels shape the embedding offline; the projection head is
discarded after pretraining. Deployment remains label-free (per-record healthy
Mahalanobis/kNN + conformal) — identical contract to A0 / A / A1 / B / B1.

See Layer3/reports/LAYER3_ARM_C_SUPERVISED_SPEC.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LABEL_MAP = (
    "NORMAL=normal,DANGEROUS=unsafe,NOISE=unsafe,"
    "BENIGN_ABNORMAL=benign,AF_CONTEXT=drop"
)


def parse_label_map(spec: str) -> Dict[str, str]:
    """Parse 'SRC=dst,SRC2=dst2' into a mapping. Destination 'drop' excludes rows."""
    out: Dict[str, str] = {}
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid --label-map entry {chunk!r}; expected SRC=dst "
                f"(e.g. NORMAL=normal or AF_CONTEXT=drop)."
            )
        src, dst = chunk.split("=", 1)
        src = src.strip()
        dst = dst.strip().lower()
        if not src or not dst:
            raise ValueError(f"Invalid --label-map entry {chunk!r}.")
        out[src] = dst
    if not out:
        raise ValueError("--label-map is empty.")
    return out


def apply_label_map(
    df: pd.DataFrame,
    *,
    label_col: str,
    label_map: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Map raw label_col values → supervised class ids.

    Returns
    -------
    filtered_df : rows with destination != 'drop', plus column `_supcon_label_id`
    class_to_id : sorted kept class name → int id
    raw_counts  : destination name → count after mapping (including drop)
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found in window index. "
            "Rebuild with build_window_index.py (writes safety_group)."
        )
    raw = df[label_col].astype(str)
    mapped = raw.map(label_map)
    unknown = sorted(set(raw[mapped.isna()].unique().tolist()))
    if unknown:
        raise ValueError(
            f"Label values not covered by --label-map: {unknown}. "
            f"Map: {label_map}"
        )
    counts = mapped.value_counts().to_dict()
    keep = mapped != "drop"
    kept = df.loc[keep].copy()
    kept_labels = mapped.loc[keep]
    if kept.empty:
        raise ValueError(
            "After applying --label-map every window was dropped "
            "(all destinations were 'drop' or filtered out)."
        )
    class_names = sorted(kept_labels.unique().tolist())
    # Fail closed if a non-drop destination from the map never appears.
    expected_kept = sorted({v for v in label_map.values() if v != "drop"})
    missing = [c for c in expected_kept if c not in class_names]
    if missing:
        raise ValueError(
            f"After filtering, supervised class(es) are empty: {missing}. "
            f"Observed kept classes: {class_names}. Counts (incl. drop): {counts}. "
            "Refusing to train Arm C without all mapped classes present."
        )
    class_to_id = {name: i for i, name in enumerate(class_names)}
    kept = kept.copy()
    kept["_supcon_label_id"] = kept_labels.map(class_to_id).astype(int).values
    return kept, class_to_id, {str(k): int(v) for k, v in counts.items()}


class SupConLoss(nn.Module):
    """Khosla-style supervised contrastive loss (multi-positive InfoNCE).

    Expects two L2-normalized projection views of the same batch plus integer
    class labels. Concatenates the views so same-label pairs across views and
    within a view count as positives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1/z2 shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
        if labels.ndim != 1 or labels.shape[0] != z1.shape[0]:
            raise ValueError(
                f"labels must be (B,), got {tuple(labels.shape)} for batch {z1.shape[0]}"
            )
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        features = torch.cat([z1, z2], dim=0)  # (2B, D)
        labels = torch.cat([labels, labels], dim=0)  # (2B,)
        device = features.device
        n = features.shape[0]
        if n < 2:
            raise ValueError("SupConLoss needs at least 2 samples (after view concat).")

        sim = torch.matmul(features, features.T) / self.temperature
        # Mask self-comparisons.
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        logits_mask = ~self_mask
        # Numerical stability
        logits_max, _ = torch.max(sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        label_col = labels.view(-1, 1)
        pos_mask = (label_col == label_col.T) & logits_mask
        # Anchors with no other positive would yield nan; fail closed.
        n_pos = pos_mask.sum(dim=1)
        if (n_pos == 0).any():
            raise RuntimeError(
                "SupConLoss: at least one anchor has no positive pair in the batch. "
                "Increase batch size or ensure each class appears ≥2 times per batch."
            )

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / n_pos.float()
        return -mean_log_prob_pos.mean()


# ---------------------------------------------------------------------------
# Arm C ladder (C1/C2/C3): SVDD compactness + Outlier-Exposure terms.
#
# These operate on the ENCODER EMBEDDING h (the vector the deployment scorer
# actually reads), NOT the projection, so the geometry they shape is the geometry
# the per-record Mahalanobis/kNN veto sees. The center `c` is a fixed constant
# (mean of normal embeddings, computed once before training and frozen). See
# Layer3/reports/LAYER3_ARM_C_LADDER_SPEC.md.
#   C1 supcon_oe    = SupCon(z) + oe_weight * OE(h)
#   C2 deepsad      = svdd_weight * compactness(h) + oe_weight * OE(h)
#   C3 supcon_hybrid= SupCon(z) + svdd_weight * compactness(h) + oe_weight * OE(h)
# ---------------------------------------------------------------------------


def svdd_compactness_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    center: torch.Tensor,
    normal_id: int,
) -> torch.Tensor:
    """Deep-SVDD compactness (Ruff et al. 2018): pull NORMAL embeddings toward the
    fixed center c. Grad-safe zero if the batch contains no normal windows."""
    mask = labels == int(normal_id)
    if not bool(mask.any()):
        return h.sum() * 0.0
    d2 = ((h[mask] - center.unsqueeze(0)) ** 2).sum(dim=1)
    return d2.mean()


def outlier_exposure_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    center: torch.Tensor,
    unsafe_id: int,
    eps: float = 1.0,
) -> torch.Tensor:
    """Deep-SAD / Outlier-Exposure push (Ruff et al. 2020; Hendrycks et al. 2019):
    UNSAFE embeddings should be FAR from c. Penalize small distance via bounded
    inverse-distance 1/(||h-c||^2 + eps) in (0, 1/eps]; minimized as unsafe distance
    grows. Grad-safe zero if the batch contains no unsafe windows."""
    mask = labels == int(unsafe_id)
    if not bool(mask.any()):
        return h.sum() * 0.0
    d2 = ((h[mask] - center.unsqueeze(0)) ** 2).sum(dim=1)
    return (1.0 / (d2 + float(eps))).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = SupConLoss(temperature=0.1)
    z1 = F.normalize(torch.randn(8, 64, requires_grad=True), dim=1)
    z2 = F.normalize(torch.randn(8, 64, requires_grad=True), dim=1)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1], dtype=torch.long)
    loss = loss_fn(z1, z2, y)
    assert torch.isfinite(loss), loss
    loss.backward()
    print("supcon_ok", float(loss.detach()))

    # Arm C ladder losses: normal_id=1, unsafe_id=2 (benign=0).
    h = torch.randn(8, 64, requires_grad=True)
    center = h.detach()[y == 1].mean(dim=0)
    svdd = svdd_compactness_loss(h, y, center, normal_id=1)
    oe = outlier_exposure_loss(h, y, center, unsafe_id=2)
    hybrid = svdd + 1.0 * oe
    assert torch.isfinite(hybrid), hybrid
    hybrid.backward()
    # Grad-safe empty-class path (no unsafe in batch).
    y_no_unsafe = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.long)
    oe_empty = outlier_exposure_loss(h.detach().requires_grad_(True), y_no_unsafe, center, unsafe_id=2)
    assert float(oe_empty.detach()) == 0.0, oe_empty
    print("ladder_ok", float(svdd.detach()), float(oe.detach()))
