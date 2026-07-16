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
from layer3_masked_ssl import MaskedSSLConfig, MaskedSubjectContrastiveModel
from layer3_vicreg import VICRegConfig, VICRegModel


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
        subject_col: str = "record_id",
        apply_augmentations: bool = True,
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
        self.augmentor = augmentor or ECGAugmentor()
        self.same_record_positive = same_record_positive
        self.return_subject_id = bool(return_subject_id)
        self.subject_col = str(subject_col)
        self.apply_augmentations = bool(apply_augmentations)
        self.rng = np.random.default_rng(rng_seed)
        self.use_mmap = bool(use_mmap)
        self.window_len = int(self.idx["n_samples"].iloc[0])
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

        # Z-score normalize each window independently
        anchor = (anchor - anchor.mean()) / (anchor.std() + 1e-8)
        positive = (positive - positive.mean()) / (positive.std() + 1e-8)

        if self.apply_augmentations:
            a_aug = self.augmentor.augment(anchor)
            p_aug = self.augmentor.augment(positive)
        else:
            a_aug = anchor
            p_aug = positive

        # Return as (1, T) tensors — single channel
        a = torch.from_numpy(np.ascontiguousarray(a_aug, dtype=np.float32)).unsqueeze(0)
        p = torch.from_numpy(np.ascontiguousarray(p_aug, dtype=np.float32)).unsqueeze(0)
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
    mask_ratio: float = 0.75
    mask_patch_size: int = 25
    subject_contrastive_lambda: float = 0.30
    subject_col: str = "record_id"
    vicreg_sim_coeff: float = 25.0
    vicreg_var_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    vicreg_expander_dims: str = "512,512,512"


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

    enc = ECGEncoder1D(EncoderConfig(embedding_dim=cfg.embedding_dim))
    ssl_objective = str(cfg.ssl_objective).lower()
    if ssl_objective == "ntxent":
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
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    loss_fn = NTXentLoss(temperature=cfg.temperature)
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
                "decoder_discarded": ssl_objective == "mae_subject_contrastive",
                "expander_discarded": ssl_objective == "vicreg",
                "anomaly_score_note": "Downstream anomaly scores use encoder embedding distance only; reconstruction error / SSL projections are never used for permit/inhibit.",
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
            else:
                a, p = batch
                subject_ids = None
            a = a.to(device, non_blocking=True)
            if ssl_objective == "ntxent":
                p = p.to(device, non_blocking=True)
                _, z_a = model(a)
                _, z_p = model(p)
                loss = loss_fn(z_a, z_p)
                loss_logs = {}
            elif ssl_objective == "vicreg":
                p = p.to(device, non_blocking=True)
                loss, loss_logs = model(a, p)
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
    parser.add_argument("--ssl-objective", choices=["ntxent", "mae_subject_contrastive", "vicreg"], default="ntxent",
                        help="SSL objective. ntxent is contrastive (A); mae_subject_contrastive is ZEROSHOT-inspired "
                             "masked reconstruction + subject contrastive (B); vicreg is non-contrastive VICReg (A1).")
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Masked SSL fraction of temporal patches to mask (mae_subject_contrastive only).")
    parser.add_argument("--mask-patch-size", type=int, default=25,
                        help="Masked SSL temporal patch size in samples (25 samples = 200 ms at 125 Hz).")
    parser.add_argument("--subject-contrastive-lambda", type=float, default=0.30,
                        help="Weight on subject/record contrastive loss for mae_subject_contrastive.")
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
                             "Use this if you want pretraining to see only baseline-safe morphology.")
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
    if args.healthy_only and "is_healthy_window" in idx_df.columns:
        before = len(idx_df)
        idx_df = idx_df[idx_df["is_healthy_window"].astype(bool)].copy()
        print(f"[INFO] --healthy-only kept {len(idx_df)}/{before} windows", flush=True)
    if args.max_windows is not None:
        idx_df = idx_df.head(int(args.max_windows)).copy()
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
            "[WARN] mae_subject_contrastive uses record/subject labels. Without --healthy-only, same-subject positives may include abnormal windows; treat as a deliberate ablation.",
            flush=True,
        )
    ds = ContrastiveECGDataset(
        idx_df,
        rng_seed=args.seed,
        use_mmap=not args.no_mmap,
        same_record_positive=same_record_positive,
        return_subject_id=args.ssl_objective == "mae_subject_contrastive",
        subject_col=args.subject_col,
        apply_augmentations=args.ssl_objective in ("ntxent", "vicreg"),
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
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        subject_contrastive_lambda=args.subject_contrastive_lambda,
        subject_col=args.subject_col,
        vicreg_sim_coeff=args.vicreg_sim_coeff,
        vicreg_var_coeff=args.vicreg_var_coeff,
        vicreg_cov_coeff=args.vicreg_cov_coeff,
        vicreg_expander_dims=args.vicreg_expander_dims,
    )
    pretrain(ds, cfg, device=args.device)


if __name__ == "__main__":
    main()
