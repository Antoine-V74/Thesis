"""
Layer 3 — SSL contrastive pretraining loop.

Pretrains the ECGEncoder1D on a pool of unlabeled ECG windows using
NT-Xent contrastive loss. The default positive pair is two safe augmented
views of the same window; same-record positives are an explicit ablation:

    1. CLOCS-CMSC ablation (Kiyasseh et al. 2021):
       Positives are two non-overlapping segments from the SAME RECORD.
       Captures patient-invariance for free (same recording → same patient
       → same cardiac physiology, regardless of which 5s slice we picked).

    2. SimCLR-style default:
       Positives are two augmented views of the SAME WINDOW.
       Captures invariance to augmentation nuisances (noise, baseline
       wander, mild time warp).

Negatives are all other anchors in the batch.

This file provides:
    - NTXentLoss class
    - ContrastiveECGDataset for windowed records
    - pretrain() training loop with periodic linear-probe validation

Cluster usage:
    python Layer3/tools/pretrain_encoder.py \\
        --features-csv path/to/window_index.csv \\
        --signal-dir   path/to/signals/ \\
        --epochs 100 \\
        --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
for path in (PROJECT_ROOT, LAYER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Layer3._bootstrap import setup_layer3_paths  # noqa: E402

setup_layer3_paths(include_tools=True)

from layer3_encoder import ECGEncoder1D, EncoderConfig, EncoderWithProjection, ProjectionHead
from layer3_augmentations import AugmentConfig, ECGAugmentor
from layer3_masked_ssl import (
    MaskedConsistencyConfig,
    MaskedConsistencyModel,
    MaskedSSLConfig,
    MaskedSubjectContrastiveModel,
)
from layer3_vicreg import VICRegConfig, VICRegModel
from layer3_supervised import (
    DEFAULT_LABEL_MAP,
    SupConLoss,
    apply_label_map,
    outlier_exposure_loss,
    parse_label_map,
    svdd_compactness_loss,
)

# Arm C family: supervised-label objectives that share the label-map pipeline,
# class-balanced sampler, projection/head-discard, and encoder-only checkpoint.
#   supcon         = C  (SupCon baseline)
#   supcon_oe      = C1 (SupCon + outlier exposure)
#   deepsad        = C2 (SVDD compact normal + outlier exposure; no SupCon)
#   supcon_hybrid  = C3 (SupCon + SVDD compactness + outlier exposure)
C_FAMILY = ("supcon", "supcon_oe", "deepsad", "supcon_hybrid")
# Objectives that use the SupCon contrastive term (on projections).
_SUPCON_USERS = ("supcon", "supcon_oe", "supcon_hybrid")
# Objectives that need a frozen SVDD/SAD center c (computed from normal embeddings).
_CENTER_USERS = ("supcon_oe", "deepsad", "supcon_hybrid")
# Objectives that use the SVDD compactness term.
_SVDD_USERS = ("deepsad", "supcon_hybrid")
# Objectives that use the outlier-exposure term.
_OE_USERS = ("supcon_oe", "deepsad", "supcon_hybrid")


def write_json_provenance(path: Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_bool_column(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a CSV boolean column without pandas string truthiness traps.

    `Series.astype(bool)` treats any non-empty string as True, so the string
    "False" would become True. That is unacceptable for `--healthy-only`.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) != 0.0

    values = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none", "null"}
    valid = values.isin(true_values | false_values)
    if not bool(valid.all()):
        bad = sorted(values[~valid].dropna().unique().tolist())
        raise ValueError(f"Column {column_name!r} has non-boolean values: {bad[:10]}")
    return values.isin(true_values)


def robust_normalize_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window median/MAD normalization.

    MUST match Layer3/validation/layer3_validation_utils.robust_normalize_window
    so the encoder sees the same input distribution at pretraining and at Phase 1
    evaluation. Do not switch this back to mean/std z-score without also changing
    the evaluation path.
    """
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x)
    x = x - med
    mad = np.nanmedian(np.abs(x))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    x = x / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss (SimCLR).

    Given a batch of N anchor projections z1 and N positive projections z2,
    treats z2[i] as the positive for z1[i] and all other 2N-1 projections as
    negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2 already L2-normalized by the projection head
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)                  # (2N, D)
        sim = z @ z.t() / self.temperature              # (2N, 2N)
        # Mask self-similarities
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))
        # Positives: index i ↔ i+N
        targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
        return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveECGDataset(Dataset):
    """
    Yields (anchor_view, positive_view) pairs for contrastive pretraining.

    Default positive strategy: same window with two independent safe
    augmentations. Same-record positives are available as an explicit ablation,
    but can pull healthy and abnormal beats from one record too close together.

    Expects a window index DataFrame with at least these columns:
        record_id   : identifier shared by windows from the same recording
        signal_path : path to a .npy file containing the full record signal
        start_idx   : start sample index of the window in the record
        n_samples   : number of samples in the window

    The .npy signal files are memory-mapped to keep RAM low when the corpus
    of full-record signals is large.
    """

    def __init__(
        self,
        window_index: pd.DataFrame,
        augmentor: Optional[ECGAugmentor] = None,
        same_record_positive: bool = False,
        return_subject_id: bool = False,
        return_label: bool = False,
        label_id_col: str = "_supcon_label_id",
        subject_col: str = "record_id",
        apply_augmentations: bool = True,
        augment_fs: float = 125.0,
        rng_seed: int = 0,
        use_mmap: bool = True,
    ):
        if not {"record_id", "signal_path", "start_idx", "n_samples"}.issubset(window_index.columns):
            raise ValueError("window_index must contain: record_id, signal_path, start_idx, n_samples")
        self.idx = window_index.reset_index(drop=True)
        # Validate signal_path values to fail fast and informatively.
        if self.idx["signal_path"].astype(str).eq("").any():
            raise ValueError(
                "ContrastiveECGDataset requires non-empty signal_path values for every row. "
                "Re-run build_window_index.py without --no-signal-cache."
            )
        # Augmentations (baseline wander, bandpass) are frequency-dependent, so the
        # augmentor sampling rate MUST match the cached window fs (target_fs used by
        # build_window_index.py, default 125 Hz). A mismatch injects wander at the
        # wrong physical frequency. Seed the augmentor RNG so --seed / --deterministic
        # actually reproduces augmented views.
        self.augmentor = augmentor or ECGAugmentor(
            AugmentConfig(fs=int(round(float(augment_fs)))), seed=int(rng_seed)
        )
        self.same_record_positive = same_record_positive
        self.return_subject_id = bool(return_subject_id)
        self.return_label = bool(return_label)
        self.label_id_col = str(label_id_col)
        self.subject_col = str(subject_col)
        self.apply_augmentations = bool(apply_augmentations)
        self.rng = np.random.default_rng(rng_seed)
        self.use_mmap = bool(use_mmap)
        self.window_len = int(self.idx["n_samples"].iloc[0])
        if self.return_label and self.return_subject_id:
            raise ValueError("return_label and return_subject_id cannot both be True.")
        if self.return_label and self.label_id_col not in self.idx.columns:
            raise ValueError(
                f"return_label=True requires column {self.label_id_col!r} in window_index "
                "(apply_label_map before building the dataset)."
            )
        self.subject_to_id = {
            value: i for i, value in enumerate(sorted(self.idx.get(self.subject_col, self.idx["record_id"]).astype(str).unique()))
        }

        # Build index of window-row positions per record_id for fast positive sampling
        self.record_to_rows: dict = {}
        for i, rec in enumerate(self.idx["record_id"].tolist()):
            self.record_to_rows.setdefault(rec, []).append(i)
        # Records with at least 2 windows can produce same-record positives
        self.records_with_pairs = [r for r, rows in self.record_to_rows.items() if len(rows) >= 2]

        # Lazy cache of loaded signals to avoid re-reading the same .npy
        self._signal_cache: dict = {}

    def _load_window(self, row_idx: int) -> np.ndarray:
        row = self.idx.iloc[row_idx]
        sig_path = row["signal_path"]
        sig = self._signal_cache.get(sig_path)
        if sig is None:
            if self.use_mmap:
                sig = np.load(sig_path, mmap_mode="r")
            else:
                sig = np.load(sig_path).astype(np.float32)
            self._signal_cache[sig_path] = sig
        start = int(row["start_idx"])
        n = int(row["n_samples"])
        # Force a contiguous float32 copy of the slice so downstream augmentations
        # can write in place without touching the mmap.
        return np.array(sig[start:start + n], dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = self._load_window(idx)
        rec = self.idx.iloc[idx]["record_id"]

        if self.same_record_positive and rec in self.records_with_pairs:
            # Sample a different window from the same record
            choices = [r for r in self.record_to_rows[rec] if r != idx]
            pos_idx = int(self.rng.choice(choices))
            positive = self._load_window(pos_idx)
        else:
            # Fall back to same window, just augmented differently
            positive = anchor.copy()

        # Robust median/MAD normalize each window independently. This matches the
        # Phase 1 evaluation normalization (layer3_validation_utils) so the encoder
        # sees the same input distribution at train and test time.
        anchor = robust_normalize_window(anchor)
        positive = robust_normalize_window(positive)

        if self.apply_augmentations:
            a_aug = self.augmentor.augment(anchor)
            p_aug = self.augmentor.augment(positive)
        else:
            a_aug = anchor
            p_aug = positive

        # Return as (1, T) tensors — single channel
        a = torch.from_numpy(np.ascontiguousarray(a_aug, dtype=np.float32)).unsqueeze(0)
        p = torch.from_numpy(np.ascontiguousarray(p_aug, dtype=np.float32)).unsqueeze(0)
        if self.return_label:
            lid = torch.tensor(int(self.idx.iloc[idx][self.label_id_col]), dtype=torch.long)
            return a, p, lid
        if self.return_subject_id:
            subject_value = str(self.idx.iloc[idx].get(self.subject_col, rec))
            sid = torch.tensor(self.subject_to_id.get(subject_value, 0), dtype=torch.long)
            return a, p, sid
        return a, p


# ---------------------------------------------------------------------------
# Linear probe for periodic monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(encoder: ECGEncoder1D, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    embs = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x)
        embs.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def linear_probe_accuracy(
    encoder: ECGEncoder1D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Train an LR on frozen embeddings; return validation accuracy."""
    from sklearn.linear_model import LogisticRegression
    Z_train, y_train = extract_embeddings(encoder, train_loader, device)
    Z_val, y_val = extract_embeddings(encoder, val_loader, device)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Z_train, y_train)
    return float(clf.score(Z_val, y_val))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    temperature: float = 0.1
    embedding_dim: int = 128
    projection_dim: int = 64
    log_every_n_steps: int = 50
    probe_every_n_epochs: int = 10
    checkpoint_dir: str = "./layer3_checkpoints"
    seed: int = 0
    num_workers: int = 0
    deterministic: bool = False
    positive_mode: str = "same_window"
    ssl_objective: str = "ntxent"
    supcon_temperature: float = 0.1
    oe_weight: float = 1.0
    svdd_weight: float = 1.0
    center_init_windows: int = 2048
    normal_class_id: int = -1
    unsafe_class_id: int = -1
    mask_ratio: float = 0.75
    mask_patch_size: int = 25
    subject_contrastive_lambda: float = 0.30
    consistency_lambda: float = 1.0
    subject_col: str = "record_id"
    vicreg_sim_coeff: float = 25.0
    vicreg_var_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    vicreg_expander_dims: str = "512,512,512"
    encoder_pooling_mode: str = "global_avg"
    encoder_local_fraction: float = 0.125
    window_n_samples: int = 0
    window_target_fs: float = 125.0
    window_s: float = 0.0


def pretrain(
    train_dataset: ContrastiveECGDataset,
    cfg: PretrainConfig,
    probe_loaders: Optional[Tuple[DataLoader, DataLoader]] = None,
    device: Optional[str] = None,
) -> ECGEncoder1D:
    """
    Pretrain the encoder via NT-Xent on contrastive pairs.

    probe_loaders : optional (labeled_train, labeled_val) DataLoader pair that
        yields (x, y) for linear-probe accuracy monitoring during pretraining.
    """
    # Lazy import to keep top-of-file importable even if utils itself is broken.
    from layer3_validation_utils import (
        get_logger,
        make_torch_generator,
        resolve_torch_device,
        set_seed,
        worker_init_fn,
    )
    logger = get_logger("layer3.pretrain")

    device = resolve_torch_device(device or "auto")
    logger.info("Using device: %s (cuda_available=%s)", device, torch.cuda.is_available())
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    enc = ECGEncoder1D(
        EncoderConfig(
            embedding_dim=cfg.embedding_dim,
            pooling_mode=cfg.encoder_pooling_mode,
            local_fraction=cfg.encoder_local_fraction,
        )
    )
    ssl_objective = str(cfg.ssl_objective).lower()
    if ssl_objective == "ntxent" or ssl_objective in C_FAMILY:
        proj = ProjectionHead(in_dim=cfg.embedding_dim, out_dim=cfg.projection_dim)
        model = EncoderWithProjection(enc, proj).to(device)
    elif ssl_objective == "mae_subject_contrastive":
        masked_cfg = MaskedSSLConfig(
            embedding_dim=cfg.embedding_dim,
            projection_dim=cfg.projection_dim,
            mask_ratio=cfg.mask_ratio,
            patch_size=cfg.mask_patch_size,
            subject_contrastive_lambda=cfg.subject_contrastive_lambda,
            temperature=cfg.temperature,
        )
        model = MaskedSubjectContrastiveModel(
            encoder=enc,
            output_len=int(getattr(train_dataset, "window_len", 1000)),
            config=masked_cfg,
        ).to(device)
    elif ssl_objective == "mae_consistency":
        expander_dims = [int(d) for d in str(cfg.vicreg_expander_dims).split(",") if str(d).strip()]
        consistency_cfg = MaskedConsistencyConfig(
            embedding_dim=cfg.embedding_dim,
            mask_ratio=cfg.mask_ratio,
            patch_size=cfg.mask_patch_size,
            consistency_lambda=cfg.consistency_lambda,
            expander_dims=expander_dims,
            vicreg_sim_coeff=cfg.vicreg_sim_coeff,
            vicreg_var_coeff=cfg.vicreg_var_coeff,
            vicreg_cov_coeff=cfg.vicreg_cov_coeff,
        )
        model = MaskedConsistencyModel(
            encoder=enc,
            output_len=int(getattr(train_dataset, "window_len", 1000)),
            config=consistency_cfg,
        ).to(device)
    elif ssl_objective == "vicreg":
        expander_dims = [int(d) for d in str(cfg.vicreg_expander_dims).split(",") if str(d).strip()]
        vicreg_cfg = VICRegConfig(
            embedding_dim=cfg.embedding_dim,
            expander_dims=expander_dims,
            sim_coeff=cfg.vicreg_sim_coeff,
            var_coeff=cfg.vicreg_var_coeff,
            cov_coeff=cfg.vicreg_cov_coeff,
        )
        model = VICRegModel(encoder=enc, config=vicreg_cfg).to(device)
    else:
        raise ValueError(f"Unsupported --ssl-objective: {cfg.ssl_objective}")

    generator = make_torch_generator(cfg.seed)
    # Arm C (supcon): class-balanced sampling so the scarce danger labels are seen
    # in every batch (spec LAYER3_ARM_C_SUPERVISED_SPEC.md §9.2). At natural
    # prevalence (~5% unsafe) SupCon would shape mostly normal/benign geometry and
    # underuse the danger labels. Sqrt-tempered inverse-frequency weights rebalance
    # toward rare classes without pathologically replicating the few unsafe windows
    # across 100 epochs (full inverse-freq would over-replicate them).
    train_sampler = None
    if ssl_objective in C_FAMILY:
        label_ids = train_dataset.idx[train_dataset.label_id_col].astype(int).to_numpy()
        class_counts = np.bincount(label_ids)
        class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
        sample_weights = class_weights[label_ids]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(label_ids),
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    if ssl_objective in _SUPCON_USERS:
        loss_fn = SupConLoss(temperature=cfg.supcon_temperature)
    else:
        loss_fn = NTXentLoss(temperature=cfg.temperature)

    # Arm C ladder (C1/C2/C3): freeze an SVDD/SAD center c = mean of NORMAL encoder
    # embeddings, computed once before training. c shapes the geometry the deployment
    # Mahalanobis/kNN veto reads; it is a constant (never receives gradient).
    center = None
    if ssl_objective in _CENTER_USERS:
        normal_id = int(cfg.normal_class_id)
        if normal_id < 0:
            raise RuntimeError(
                f"{ssl_objective} needs a 'normal' class id for the SVDD/SAD center; "
                "got normal_class_id=-1 (check --label-map maps a class to 'normal')."
            )
        model.eval()
        collected = []
        n_have = 0
        with torch.no_grad():
            for c_batch in loader:
                c_a, _c_p, c_labels = c_batch
                c_a = c_a.to(device, non_blocking=True)
                h_c, _z_c = model(c_a)
                m = c_labels.to(device) == normal_id
                if bool(m.any()):
                    collected.append(h_c[m].detach())
                    n_have += int(m.sum().item())
                if n_have >= int(cfg.center_init_windows):
                    break
        if not collected:
            raise RuntimeError(
                "Cannot initialize SVDD/SAD center: no NORMAL windows appeared in the "
                "loader. Check the label map and class balance."
            )
        center = torch.cat(collected, dim=0)[: int(cfg.center_init_windows)].mean(dim=0).detach()
        model.train()
        logger.info(
            "Arm C center c initialized from %d normal embeddings (dim=%d).",
            min(n_have, int(cfg.center_init_windows)), int(center.shape[0]),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    history = []

    def _save_checkpoint(path: Path, epoch: int) -> None:
        torch.save(
            {
                "encoder_state_dict": enc.state_dict(),
                "epoch": int(epoch),
                "config": cfg.__dict__,
                "ssl_objective": ssl_objective,
                "decoder_discarded": ssl_objective in ("mae_subject_contrastive", "mae_consistency"),
                "expander_discarded": ssl_objective in ("vicreg", "mae_consistency"),
                "projection_discarded": ssl_objective == "ntxent" or ssl_objective in C_FAMILY,
                "head_discarded": ssl_objective in C_FAMILY,
                "labels_used_in_pretraining_only": ssl_objective in C_FAMILY,
                "anomaly_score_note": "Downstream anomaly scores use encoder embedding distance only; reconstruction error / SSL projections / class logits are never used for permit/inhibit.",
            },
            path,
        )

    for epoch in range(cfg.epochs):
        # Linear warmup of the learning rate
        if epoch < cfg.warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (epoch + 1) / cfg.warmup_epochs

        model.train()
        ep_loss = 0.0
        ep_n = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            if ssl_objective == "mae_subject_contrastive":
                a, p, subject_ids = batch
                labels = None
            elif ssl_objective in C_FAMILY:
                a, p, labels = batch
                subject_ids = None
            else:
                a, p = batch
                subject_ids = None
                labels = None
            a = a.to(device, non_blocking=True)
            if ssl_objective == "ntxent":
                p = p.to(device, non_blocking=True)
                _, z_a = model(a)
                _, z_p = model(p)
                loss = loss_fn(z_a, z_p)
                loss_logs = {}
            elif ssl_objective in C_FAMILY:
                assert labels is not None
                labels = labels.to(device, non_blocking=True)
                h_a, z_a = model(a)
                loss = a.sum() * 0.0  # grad-safe zero base
                loss_logs = {}
                if ssl_objective in _SUPCON_USERS:
                    p = p.to(device, non_blocking=True)
                    _, z_p = model(p)
                    supcon_term = loss_fn(z_a, z_p, labels)
                    loss = loss + supcon_term
                    loss_logs["supcon"] = float(supcon_term.detach())
                if ssl_objective in _SVDD_USERS:
                    svdd_term = svdd_compactness_loss(h_a, labels, center, cfg.normal_class_id)
                    loss = loss + cfg.svdd_weight * svdd_term
                    loss_logs["svdd"] = float(svdd_term.detach())
                if ssl_objective in _OE_USERS:
                    oe_term = outlier_exposure_loss(h_a, labels, center, cfg.unsafe_class_id)
                    loss = loss + cfg.oe_weight * oe_term
                    loss_logs["oe"] = float(oe_term.detach())
            elif ssl_objective == "vicreg":
                p = p.to(device, non_blocking=True)
                loss, loss_logs = model(a, p)
            elif ssl_objective == "mae_consistency":
                # Two masked views of the SAME window are generated inside the
                # model; only the anchor window is needed here.
                loss, loss_logs = model(a)
            else:
                assert subject_ids is not None
                subject_ids = subject_ids.to(device, non_blocking=True)
                loss, loss_logs = model(a, subject_ids)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += float(loss.item()) * a.size(0)
            ep_n += a.size(0)
            if step % cfg.log_every_n_steps == 0:
                extra = ""
                if ssl_objective == "mae_subject_contrastive" and loss_logs:
                    extra = " recon=%.4f subj=%.4f mask=%.2f" % (
                        float(loss_logs["reconstruction_loss"]),
                        float(loss_logs["subject_contrastive_loss"]),
                        float(loss_logs["mask_fraction"]),
                    )
                elif ssl_objective == "vicreg" and loss_logs:
                    extra = " inv=%.4f var=%.4f cov=%.4f emb_std=%.3f" % (
                        float(loss_logs["invariance_loss"]),
                        float(loss_logs["variance_loss"]),
                        float(loss_logs["covariance_loss"]),
                        float(loss_logs["embedding_std"]),
                    )
                elif ssl_objective == "mae_consistency" and loss_logs:
                    extra = " recon=%.4f cons=%.4f emb_std=%.3f mask=%.2f" % (
                        float(loss_logs["reconstruction_loss"]),
                        float(loss_logs["consistency_loss"]),
                        float(loss_logs["embedding_std"]),
                        float(loss_logs["mask_fraction"]),
                    )
                elif ssl_objective in C_FAMILY and loss_logs:
                    extra = " " + " ".join(f"{k}={v:.4f}" for k, v in loss_logs.items())
                logger.info("epoch %d step %d loss=%.4f%s lr=%.2e",
                            epoch, step, loss.item(), extra, opt.param_groups[0]["lr"])

        sched.step()
        avg_loss = ep_loss / max(1, ep_n)
        dt = time.time() - t0
        log_row = {"epoch": epoch, "loss": avg_loss, "seconds": dt,
                   "lr": opt.param_groups[0]["lr"]}

        if probe_loaders is not None and (epoch % cfg.probe_every_n_epochs == 0 or epoch == cfg.epochs - 1):
            acc = linear_probe_accuracy(enc, probe_loaders[0], probe_loaders[1], device)
            log_row["linear_probe_acc"] = acc
            logger.info("epoch %d loss=%.4f linear_probe_acc=%.4f", epoch, avg_loss, acc)
        else:
            logger.info("epoch %d loss=%.4f (%.1fs)", epoch, avg_loss, dt)

        history.append(log_row)

        # Save checkpoint at every probe interval and on the last epoch.
        if epoch % cfg.probe_every_n_epochs == 0 or epoch == cfg.epochs - 1:
            ckpt_path = Path(cfg.checkpoint_dir) / f"encoder_epoch{epoch:03d}.pt"
            _save_checkpoint(ckpt_path, epoch)

    # Always also write a stable "last" checkpoint and a stable named copy of the
    # final epoch, matching the example CLI which references encoder_epoch099.pt
    # when --epochs 100 is used.
    last_ckpt = Path(cfg.checkpoint_dir) / "encoder_last.pt"
    _save_checkpoint(last_ckpt, cfg.epochs - 1)

    # Save training history
    pd.DataFrame(history).to_csv(Path(cfg.checkpoint_dir) / "pretrain_history.csv", index=False)
    return enc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-index", type=str, required=True,
                        help="CSV with columns record_id, signal_path, start_idx, n_samples")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="./layer3_checkpoints")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0. auto uses CUDA if available.")
    parser.add_argument("--ssl-objective",
                        choices=["ntxent", "vicreg", "mae_consistency", "mae_subject_contrastive",
                                 "supcon", "supcon_oe", "deepsad", "supcon_hybrid"],
                        default="ntxent",
                        help="SSL objective / arm. ntxent = A (SimCLR contrastive); vicreg = A1 (non-contrastive); "
                             "mae_consistency = B PRIMARY (masked recon + non-contrastive same-window consistency); "
                             "mae_subject_contrastive = B1 ABLATION (masked recon + subject/record contrastive); "
                             "supcon = C (supervised contrastive; labels at pretrain only); "
                             "supcon_oe = C1 (SupCon + outlier exposure); "
                             "deepsad = C2 (SVDD compact normal + outlier exposure); "
                             "supcon_hybrid = C3 (SupCon + SVDD + outlier exposure).")
    parser.add_argument("--label-col", default="safety_group",
                        help="Window-index column used as supervised class for --ssl-objective supcon.")
    parser.add_argument("--label-map", default=DEFAULT_LABEL_MAP,
                        help="Comma-separated SRC=dst map for --ssl-objective supcon. "
                             "Use dst=drop to exclude a class. Default maps NORMAL/unsafe/benign.")
    parser.add_argument("--supcon-temperature", type=float, default=0.1,
                        help="Temperature for SupCon loss (Arm C).")
    parser.add_argument("--oe-weight", type=float, default=1.0,
                        help="Weight on the outlier-exposure term (C1 supcon_oe / C2 deepsad / C3 supcon_hybrid).")
    parser.add_argument("--svdd-weight", type=float, default=1.0,
                        help="Weight on the SVDD compactness term (C2 deepsad / C3 supcon_hybrid).")
    parser.add_argument("--center-init-windows", type=int, default=2048,
                        help="Number of NORMAL embeddings used to initialize the frozen SVDD/SAD center c "
                             "(C1/C2/C3). Larger = more stable center estimate.")
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Fraction of temporal patches to mask (mae_consistency / mae_subject_contrastive).")
    parser.add_argument("--mask-patch-size", type=int, default=25,
                        help="Masked SSL temporal patch size in samples (25 samples = 200 ms at 125 Hz).")
    parser.add_argument("--consistency-lambda", type=float, default=1.0,
                        help="Weight on the non-contrastive same-window consistency term for mae_consistency (B).")
    parser.add_argument("--subject-contrastive-lambda", type=float, default=0.30,
                        help="Weight on subject/record contrastive loss for mae_subject_contrastive (B1).")
    parser.add_argument("--subject-col", default="record_id",
                        help="Window-index column used as subject/record identity for subject contrastive loss.")
    parser.add_argument("--vicreg-sim-coeff", type=float, default=25.0,
                        help="VICReg invariance (MSE) weight (vicreg only).")
    parser.add_argument("--vicreg-var-coeff", type=float, default=25.0,
                        help="VICReg variance (anti-collapse) weight (vicreg only).")
    parser.add_argument("--vicreg-cov-coeff", type=float, default=1.0,
                        help="VICReg covariance (decorrelation) weight (vicreg only).")
    parser.add_argument("--vicreg-expander-dims", default="512,512,512",
                        help="Comma-separated expander hidden dims for VICReg (vicreg only). Expander is discarded after pretraining.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. Use 0 for CPU-laptop / Windows compatibility.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-mmap", action="store_true",
                        help="Disable mmap signal loading (loads each .npy fully into RAM).")
    parser.add_argument("--positive-mode", choices=["same_window", "same_record"], default="same_window",
                        help="same_window is the safer default for anomaly-veto embeddings; same_record is a CLOCS-style ablation.")
    parser.add_argument("--healthy-only", action="store_true",
                        help="Restrict contrastive pretraining to is_healthy_window=True rows. "
                             "Use this if you want pretraining to see only baseline-safe morphology. "
                             "Fails if the window index has no is_healthy_window column.")
    parser.add_argument("--augment-fs", type=float, default=125.0,
                        help="Sampling rate (Hz) of the cached windows, used to calibrate frequency-dependent "
                             "augmentations (baseline wander / bandpass). MUST match build_window_index --target-fs.")
    parser.add_argument("--augment-p-noise", type=float, default=0.70,
                        help="Probability of Gaussian-noise augmentation. Locked-arm default: 0.70.")
    parser.add_argument("--augment-p-wander", type=float, default=0.50,
                        help="Probability of baseline-wander augmentation. Locked-arm default: 0.50.")
    parser.add_argument("--augment-p-crop", type=float, default=0.10,
                        help="Probability of random crop/re-pad augmentation. Locked-arm default: 0.10.")
    parser.add_argument("--augment-noise-snr-min-db", type=float, default=20.0)
    parser.add_argument("--augment-noise-snr-max-db", type=float, default=35.0)
    parser.add_argument("--encoder-pooling-mode",
                        choices=["global_avg", "avg_max", "causal_local_global"],
                        default="global_avg",
                        help="Encoder temporal pooling. global_avg preserves locked checkpoints; other modes are exploratory.")
    parser.add_argument("--encoder-local-fraction", type=float, default=0.125,
                        help="Tail fraction pooled by causal_local_global (0.125 = 1 s of an 8 s window).")
    parser.add_argument("--exclude-records-csv", default=None,
                        help="Optional CSV with columns dataset,record. Windows from these records are DROPPED "
                             "from pretraining. Use it to hold the gold evaluation records out of SSL so Phase 1 "
                             "can claim an unseen-subject split (prevents pretrain/eval leakage).")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Restrict to first N rows (smoke testing).")
    args = parser.parse_args()

    if not Path(args.window_index).exists():
        raise SystemExit(f"--window-index not found: {args.window_index}. Run build_window_index.py first.")
    idx_df = pd.read_csv(args.window_index)
    required = {"record_id", "signal_path", "start_idx", "n_samples"}
    missing = required - set(idx_df.columns)
    if missing:
        raise SystemExit(
            f"Window index is missing required pretrain columns: {sorted(missing)}. "
            f"Re-run build_window_index.py (do NOT pass --no-signal-cache)."
        )
    if args.exclude_records_csv:
        # Reuse the same allowlist loader used by evaluation to define the eval set.
        sys.path.insert(0, str(LAYER3_ROOT / "validation"))
        from layer3_validation_utils import load_record_allowlist  # noqa: E402
        exclude_pairs = load_record_allowlist(args.exclude_records_csv)
        exclude_ids = {f"{ds}/{rec}" for ds, rec in exclude_pairs}
        before = len(idx_df)
        # Match only on the fully-qualified record_id (dataset/record). Do NOT match
        # bare record names: ids like "100" collide across datasets and could hold
        # out the wrong records.
        rid = idx_df["record_id"].astype(str)
        mask_keep = ~rid.isin(exclude_ids)
        if "dataset" in idx_df.columns and "record" in idx_df.columns:
            pair_id = idx_df["dataset"].astype(str) + "/" + idx_df["record"].astype(str)
            mask_keep &= ~pair_id.isin(exclude_ids)
        idx_df = idx_df[mask_keep].copy()
        print(
            f"[INFO] --exclude-records-csv dropped {before - len(idx_df)} windows "
            f"({len(exclude_pairs)} eval records held out of pretraining); {len(idx_df)} windows remain.",
            flush=True,
        )
        if idx_df.empty:
            raise SystemExit("--exclude-records-csv removed every window; check the CSV and window index.")
    if args.healthy_only:
        if "is_healthy_window" not in idx_df.columns:
            raise SystemExit(
                "--healthy-only requested but the window index has no 'is_healthy_window' column. "
                "Rebuild the index with build_window_index.py (which writes safety labels), or drop --healthy-only. "
                "Refusing to silently pretrain on mixed data."
            )
        before = len(idx_df)
        try:
            healthy_mask = parse_bool_column(idx_df["is_healthy_window"], "is_healthy_window")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        idx_df = idx_df[healthy_mask].copy()
        print(f"[INFO] --healthy-only kept {len(idx_df)}/{before} windows", flush=True)
        if idx_df.empty:
            raise SystemExit("--healthy-only removed every window; the index has no is_healthy_window=True rows.")

    label_map_parsed = None
    class_to_id = None
    label_counts = None
    if args.ssl_objective in C_FAMILY:
        try:
            label_map_parsed = parse_label_map(args.label_map)
            before = len(idx_df)
            idx_df, class_to_id, label_counts = apply_label_map(
                idx_df,
                label_col=args.label_col,
                label_map=label_map_parsed,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"[INFO] Arm C-family ({args.ssl_objective}): label_col={args.label_col!r} "
            f"kept {len(idx_df)}/{before} windows; classes={class_to_id}; counts={label_counts}",
            flush=True,
        )

    if args.max_windows is not None:
        n_keep = int(args.max_windows)
        if args.ssl_objective in C_FAMILY and class_to_id is not None:
            # Stratified sample so smoke truncations still keep every mapped class.
            full = idx_df
            parts = []
            n_cls = max(1, len(class_to_id))
            per = max(2, n_keep // n_cls)
            for cid in sorted(class_to_id.values()):
                sub = full[full["_supcon_label_id"].astype(int) == int(cid)]
                if len(sub) < 2:
                    raise SystemExit(
                        f"Supervised class id {cid} has <2 windows after filtering; "
                        "cannot form SupCon positives."
                    )
                take = min(len(sub), per)
                parts.append(sub.sample(n=take, random_state=int(args.seed)))
            sampled = pd.concat(parts, axis=0)
            if len(sampled) < n_keep:
                remaining = full.drop(index=sampled.index, errors="ignore")
                need = min(len(remaining), n_keep - len(sampled))
                if need > 0:
                    sampled = pd.concat(
                        [sampled, remaining.sample(n=need, random_state=int(args.seed))],
                        axis=0,
                    )
            idx_df = sampled.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
            print(
                f"[INFO] --max-windows stratified sample for supcon: {len(idx_df)} windows "
                f"(class counts={idx_df['_supcon_label_id'].value_counts().to_dict()})",
                flush=True,
            )
        else:
            idx_df = idx_df.head(n_keep).copy()
    sample_counts = pd.to_numeric(idx_df["n_samples"], errors="coerce").dropna().astype(int).unique()
    if len(sample_counts) != 1:
        raise SystemExit(
            f"Pretraining requires one uniform n_samples value, got {sorted(sample_counts.tolist())[:10]}"
        )
    window_n_samples = int(sample_counts[0])
    window_s = float(window_n_samples / float(args.augment_fs))
    print(f"Loaded {len(idx_df)} windows across {idx_df['record_id'].nunique()} records.", flush=True)
    if args.subject_col not in idx_df.columns:
        raise SystemExit(f"--subject-col {args.subject_col!r} not found in window index columns")

    same_record_positive = args.positive_mode == "same_record"
    if same_record_positive:
        print(
            "[WARN] --positive-mode same_record can pull healthy and abnormal beats from the same record together; treat as an ablation.",
            flush=True,
        )
    if args.ssl_objective == "mae_subject_contrastive" and not args.healthy_only:
        print(
            "[WARN] mae_subject_contrastive (B1) uses record/subject labels. Without --healthy-only, same-subject positives may include abnormal windows; treat as a deliberate ablation.",
            flush=True,
        )
    if args.ssl_objective == "mae_consistency" and not args.healthy_only:
        print(
            "[WARN] mae_consistency (B) reconstructs masked ECG. On mixed (healthy+abnormal) data the decoder can learn to reconstruct pathology, making danger look 'normal'. Healthy-only pretraining is the preferred primary for B; mixed is a robustness ablation.",
            flush=True,
        )
    if not 0.0 <= float(args.augment_p_noise) <= 1.0:
        raise SystemExit("--augment-p-noise must be in [0, 1]")
    if not 0.0 <= float(args.augment_p_wander) <= 1.0:
        raise SystemExit("--augment-p-wander must be in [0, 1]")
    if not 0.0 <= float(args.augment_p_crop) <= 1.0:
        raise SystemExit("--augment-p-crop must be in [0, 1]")
    if float(args.augment_noise_snr_min_db) > float(args.augment_noise_snr_max_db):
        raise SystemExit("--augment-noise-snr-min-db must be <= --augment-noise-snr-max-db")
    augment_cfg = AugmentConfig(
        fs=int(round(float(args.augment_fs))),
        p_noise=float(args.augment_p_noise),
        p_wander=float(args.augment_p_wander),
        p_crop=float(args.augment_p_crop),
        noise_snr_db_range=(
            float(args.augment_noise_snr_min_db),
            float(args.augment_noise_snr_max_db),
        ),
    )
    ds = ContrastiveECGDataset(
        idx_df,
        augmentor=ECGAugmentor(augment_cfg, seed=int(args.seed)),
        rng_seed=args.seed,
        use_mmap=not args.no_mmap,
        same_record_positive=same_record_positive,
        return_subject_id=args.ssl_objective == "mae_subject_contrastive",
        return_label=args.ssl_objective in C_FAMILY,
        subject_col=args.subject_col,
        apply_augmentations=args.ssl_objective in ("ntxent", "vicreg") or args.ssl_objective in C_FAMILY,
        augment_fs=args.augment_fs,
    )
    # Provenance: record which records the encoder actually saw, so the eval side
    # can verify the pretrain/eval split (leakage check).
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "ssl_objective": args.ssl_objective,
        "healthy_only": bool(args.healthy_only),
        "augment_fs": float(args.augment_fs),
        "augmentation": {
            "p_noise": float(args.augment_p_noise),
            "p_wander": float(args.augment_p_wander),
            "p_crop": float(args.augment_p_crop),
            "noise_snr_db_range": [
                float(args.augment_noise_snr_min_db),
                float(args.augment_noise_snr_max_db),
            ],
        },
        "encoder_pooling_mode": str(args.encoder_pooling_mode),
        "encoder_local_fraction": float(args.encoder_local_fraction),
        "window_n_samples": window_n_samples,
        "window_target_fs": float(args.augment_fs),
        "window_s": window_s,
        "excluded_records_csv": args.exclude_records_csv,
        "n_windows": int(len(idx_df)),
        "n_records": int(idx_df["record_id"].nunique()),
        "record_ids": sorted(idx_df["record_id"].astype(str).unique().tolist()),
    }
    if args.ssl_objective in C_FAMILY:
        provenance.update(
            {
                "label_col": str(args.label_col),
                "label_map": label_map_parsed,
                "class_to_id": class_to_id,
                "label_counts": label_counts,
                "labels_used_in_pretraining_only": True,
                "head_discarded": True,
                "supcon_temperature": float(args.supcon_temperature),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "class_balanced_sampler": "sqrt_inverse_frequency",
            }
        )
        if args.ssl_objective in _OE_USERS or args.ssl_objective in _SVDD_USERS:
            provenance.update(
                {
                    "oe_weight": float(args.oe_weight),
                    "svdd_weight": float(args.svdd_weight),
                    "center_init_windows": int(args.center_init_windows),
                    "center": "frozen_mean_of_normal_embeddings",
                }
            )
    write_json_provenance(ckpt_dir / "pretrain_records.json", provenance)

    # Arm C ladder: resolve the class ids the SVDD/SAD terms need. Fail closed if a
    # center-using objective lacks a 'normal' or 'unsafe' class in the label map.
    normal_class_id = int(class_to_id["normal"]) if (class_to_id and "normal" in class_to_id) else -1
    unsafe_class_id = int(class_to_id["unsafe"]) if (class_to_id and "unsafe" in class_to_id) else -1
    if args.ssl_objective in _CENTER_USERS and (normal_class_id < 0 or unsafe_class_id < 0):
        raise SystemExit(
            f"{args.ssl_objective} requires both 'normal' and 'unsafe' classes in --label-map; "
            f"got class_to_id={class_to_id}. Map DANGEROUS/NOISE→unsafe and NORMAL→normal."
        )

    cfg = PretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        positive_mode=args.positive_mode,
        ssl_objective=args.ssl_objective,
        supcon_temperature=args.supcon_temperature,
        oe_weight=args.oe_weight,
        svdd_weight=args.svdd_weight,
        center_init_windows=args.center_init_windows,
        normal_class_id=normal_class_id,
        unsafe_class_id=unsafe_class_id,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        subject_contrastive_lambda=args.subject_contrastive_lambda,
        consistency_lambda=args.consistency_lambda,
        subject_col=args.subject_col,
        vicreg_sim_coeff=args.vicreg_sim_coeff,
        vicreg_var_coeff=args.vicreg_var_coeff,
        vicreg_cov_coeff=args.vicreg_cov_coeff,
        vicreg_expander_dims=args.vicreg_expander_dims,
        encoder_pooling_mode=args.encoder_pooling_mode,
        encoder_local_fraction=args.encoder_local_fraction,
        window_n_samples=window_n_samples,
        window_target_fs=args.augment_fs,
        window_s=window_s,
    )
    pretrain(ds, cfg, device=args.device)


if __name__ == "__main__":
    main()
