#!/usr/bin/env python3
"""
Window-level Layer 3 embedding anomaly-veto validation.

Default behavior:
- Uses a pretrained ECGEncoder1D if available from Layer3/pipeline/layer3_encoder.py.
- Falls back to a randomly initialized encoder only for end-to-end smoke testing.
- Fits healthy-baseline anomaly models on calibration embeddings.
- Main scoring options are Mahalanobis and kNN distance.
- Sets thresholds from held-out healthy validation embeddings.
- Scores later windows and outputs permit/inhibit decisions.

Safety framing:
This is an offline validation script for a stimulation safety veto. It is not a clinical
arrhythmia classifier and it never commands stimulation by itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
for path in (PROJECT_ROOT, LAYER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Layer3._bootstrap import setup_layer3_paths  # noqa: E402

setup_layer3_paths(include_validation=True)

from layer3_embedding_mahalanobis import (  # noqa: E402
    EmbeddingKNNBaseline,
    EmbeddingKNNConfig,
    EmbeddingMahalanobisBaseline,
    EmbeddingMahalanobisConfig,
    EmbeddingPreprocessConfig,
    EmbeddingPreprocessor,
)
from layer3_group_metrics import add_normal_vs_danger_auroc, add_policy_columns, policy_decision_metrics  # noqa: E402
from label_grouping import INHIBIT_EXPECTED  # noqa: E402
from layer3_validation_utils import (  # noqa: E402
    SAFETY_DISCLAIMER,
    add_auroc_auprc,
    build_encoder,
    compute_guard_samples,
    decision_metrics,
    encode_windows,
    get_logger,
    parse_csv_list,
    read_wfdb_window,
    resolve_record_path,
    select_decision_threshold,
    set_seed,
    write_json,
    write_threshold_coverage,
)

LOG = get_logger("layer3.validate")


def load_windows(df: pd.DataFrame, data_dir: str | Path, lead_index: int, target_fs: float | None) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    for row in df.itertuples(index=False):
        record_path = resolve_record_path(data_dir, row.dataset, row.record, getattr(row, "record_path", None))
        x = read_wfdb_window(
            record_path=record_path,
            start_sample=int(row.start_sample),
            end_sample=int(row.end_sample),
            lead_index=lead_index,
            target_fs=target_fs,
        )
        windows.append(x)
    return windows


def assign_global_splits(df: pd.DataFrame, fit_record_frac: float, val_record_frac: float) -> pd.Series:
    """Record-level split to avoid leakage across overlapping windows."""
    records = sorted(df["record_key"].unique())
    n = len(records)
    if n == 0:
        return pd.Series([], dtype=str)
    n_fit = max(1, int(round(n * fit_record_frac)))
    n_val = max(1, int(round(n * val_record_frac))) if n > 1 else 0
    fit_records = set(records[:n_fit])
    val_records = set(records[n_fit : n_fit + n_val])
    split = []
    for rk in df["record_key"]:
        if rk in fit_records:
            split.append("fit")
        elif rk in val_records:
            split.append("val")
        else:
            split.append("test")
    return pd.Series(split, index=df.index, dtype=str)


def build_anomaly_baseline(
    anomaly_model: str,
    threshold_quantile: float,
    shrinkage: float,
    eps: float,
    knn_k: int,
) -> object:
    model = str(anomaly_model).strip().lower()
    if model == "mahalanobis":
        return EmbeddingMahalanobisBaseline(
            EmbeddingMahalanobisConfig(
                threshold_quantile=threshold_quantile,
                shrinkage=shrinkage,
                eps=eps,
            )
        )
    if model == "knn":
        return EmbeddingKNNBaseline(
            EmbeddingKNNConfig(
                k=int(knn_k),
                threshold_quantile=threshold_quantile,
            )
        )
    raise ValueError(f"Unsupported Layer 3 anomaly model: {anomaly_model}")


def fit_baseline_with_pruning(
    fit_embeddings: np.ndarray,
    anomaly_model: str,
    threshold_quantile: float,
    shrinkage: float,
    eps: float,
    knn_k: int,
    calibration_outlier_frac: float,
    min_keep: int,
) -> object:
    baseline = build_anomaly_baseline(
        anomaly_model=anomaly_model,
        threshold_quantile=threshold_quantile,
        shrinkage=shrinkage,
        eps=eps,
        knn_k=knn_k,
    )
    if float(calibration_outlier_frac) > 0.0:
        return baseline.fit_robust(
            fit_embeddings,
            outlier_frac=float(calibration_outlier_frac),
            min_keep=int(min_keep),
        )
    return baseline.fit(fit_embeddings)


def fit_transform_embeddings(
    fit_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    l2_normalize: bool,
    pca_dim: int,
    pca_whiten: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fit preprocessing on healthy calibration embeddings only, then transform all rows."""
    pca = None if int(pca_dim) <= 0 else int(pca_dim)
    pre = EmbeddingPreprocessor(
        EmbeddingPreprocessConfig(
            l2_normalize=bool(l2_normalize),
            pca_dim=pca,
            pca_whiten=bool(pca_whiten),
        )
    )
    fit_z = pre.fit_transform(fit_embeddings)
    all_z = pre.transform(all_embeddings)
    return fit_z, all_z, pre.to_dict()


def add_score_columns(g: pd.DataFrame, scores: np.ndarray, threshold: float, anomaly_model: str) -> pd.DataFrame:
    """Write generic score columns plus model-specific compatibility aliases."""
    scores = np.asarray(scores, dtype=float)
    g["layer3_anomaly_model"] = str(anomaly_model)
    g["layer3_score"] = scores
    g["anomaly_score"] = scores
    g["layer3_threshold"] = float(threshold)
    g["threshold"] = float(threshold)
    g["score_over_threshold_ratio"] = (scores / float(threshold)) if threshold > 0 else np.nan
    if str(anomaly_model).lower() == "mahalanobis":
        g["layer3_mahal_score"] = scores
        g["layer3_knn_score"] = np.nan
    else:
        g["layer3_mahal_score"] = np.nan
        g["layer3_knn_score"] = scores
    return g


def fit_score_one_group(
    group: pd.DataFrame,
    embeddings: np.ndarray,
    threshold_quantile: float,
    shrinkage: float,
    eps: float,
    anomaly_model: str,
    knn_k: int,
    calibration_outlier_frac: float,
    min_fit: int,
    min_val: int,
    calibration_fit_frac: float,
    calibration_val_frac: float,
    guard_s: float | None = None,
    l2_normalize: bool = False,
    pca_dim: int = 32,
    pca_whiten: bool = False,
    threshold_method: str = "conformal",
    conformal_alpha: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fit one per-record healthy embedding baseline and score all windows in that record.

    Temporal ordering inside a record is enforced by `center_sample`:
        early healthy windows  -> fit (baseline)
        next healthy windows   -> val (threshold calibration)
        any window whose center is within `guard_s` after the last val center
        is marked `guard_excluded` to prevent window overlap with calibration.
        windows after that are scored against the threshold.

    `embeddings[i]` must correspond to the row at position i in the original
    (un-grouped, un-sorted) DataFrame; we use the DataFrame's integer index
    (after the caller's reset_index) to gather embeddings for this group.
    """
    g = group.sort_values("center_sample").copy()
    idx = g.index.to_numpy()
    emb = embeddings[idx]

    # Per-record sampling rate. build_window_index.py always writes the native fs
    # to the `fs` column; we fall back to the cached_fs only if `fs` is missing.
    if "fs" in g.columns and pd.notna(g["fs"].iloc[0]):
        fs = float(g["fs"].iloc[0])
    elif "cached_fs" in g.columns and pd.notna(g["cached_fs"].iloc[0]):
        fs = float(g["cached_fs"].iloc[0])
    else:
        fs = 250.0  # last-resort default; should not happen in practice

    healthy = g["is_healthy_window"].astype(bool).to_numpy()
    healthy_positions = np.flatnonzero(healthy)
    n_healthy = int(len(healthy_positions))
    if n_healthy < max(min_fit + min_val, 2):
        g["split"] = "unscored_insufficient_healthy_calibration"
        g["layer3_anomaly_model"] = str(anomaly_model)
        g["layer3_score"] = np.nan
        g["anomaly_score"] = np.nan
        g["layer3_mahal_score"] = np.nan
        g["layer3_knn_score"] = np.nan
        g["layer3_threshold"] = np.nan
        g["threshold"] = np.nan
        g["score_over_threshold_ratio"] = np.nan
        g["decision"] = "inhibit"  # Conservative: fail-safe inhibit.
        meta = {
            "record_key": str(g["record_key"].iloc[0]),
            "status": "insufficient_healthy_calibration",
            "anomaly_model": str(anomaly_model),
            "n_healthy": n_healthy,
            "n_fit": 0,
            "n_val": 0,
            "n_outlier_removed": 0,
            "guard_samples": 0,
            "threshold": np.nan,
        }
        return g, meta

    n_fit = max(min_fit, int(round(n_healthy * calibration_fit_frac)))
    n_val = max(min_val, int(round(n_healthy * calibration_val_frac)))
    if n_fit + n_val > n_healthy:
        n_fit = max(min_fit, n_healthy - min_val)
        n_val = max(min_val, n_healthy - n_fit)
    n_fit = int(max(1, min(n_fit, n_healthy - 1)))
    n_val = int(max(1, min(n_val, n_healthy - n_fit)))

    fit_pos = healthy_positions[:n_fit]
    val_pos = healthy_positions[n_fit : n_fit + n_val]
    calibration_end_center = int(g.iloc[val_pos[-1]]["center_sample"])
    guard_samples = compute_guard_samples(window_s=float(g["window_s"].iloc[0]) if "window_s" in g.columns else float("nan"),
                                          guard_s=guard_s, fs=fs)
    guard_end_center = calibration_end_center + guard_samples

    split = np.array(["test"] * len(g), dtype=object)
    split[fit_pos] = "fit"
    split[val_pos] = "val"
    centers = g["center_sample"].to_numpy()
    # Exclude any window before or at the end of the calibration segment.
    early_mask = centers <= calibration_end_center
    split[(early_mask) & (split == "test")] = "calibration_excluded"
    # Guard zone: prevent calibration/test window overlap by requiring a temporal
    # buffer at least as long as the window itself between the last val center
    # and any test center.
    guard_mask = (centers > calibration_end_center) & (centers <= guard_end_center)
    split[(guard_mask) & (split == "test")] = "guard_excluded"
    g["split"] = split

    fit_emb, emb_transformed, preprocess_meta = fit_transform_embeddings(
        fit_embeddings=emb[fit_pos],
        all_embeddings=emb,
        l2_normalize=l2_normalize,
        pca_dim=pca_dim,
        pca_whiten=pca_whiten,
    )

    baseline = fit_baseline_with_pruning(
        fit_embeddings=fit_emb,
        anomaly_model=anomaly_model,
        threshold_quantile=threshold_quantile,
        shrinkage=shrinkage,
        eps=eps,
        knn_k=knn_k,
        calibration_outlier_frac=calibration_outlier_frac,
        min_keep=min_fit,
    )
    fit_scores = baseline.score(emb_transformed[fit_pos])
    val_scores = baseline.score(emb_transformed[val_pos])
    thr_info = select_decision_threshold(
        val_scores=val_scores,
        method=threshold_method,
        threshold_quantile=threshold_quantile,
        conformal_alpha=conformal_alpha,
    )
    threshold = float(thr_info["threshold"])
    threshold_ok = str(thr_info["status"]) == "ok" and np.isfinite(threshold)
    if threshold_ok:
        baseline.threshold_ = threshold

    scores = baseline.score(emb_transformed)
    g = add_score_columns(g, scores, threshold, anomaly_model)
    if threshold_ok:
        g["decision"] = np.where(scores <= threshold, "permit", "inhibit")
    else:
        # Fail-safe: an unusable threshold (e.g. conformal alpha infeasible for
        # the healthy calibration size) must inhibit, per uncertainty -> inhibit.
        g["decision"] = "inhibit"
    # Conservative runtime rule: if a row is not scorable/evaluable, mark calibration_no_stim.
    g.loc[g["split"].isin(["fit", "val", "calibration_excluded", "guard_excluded"]), "decision"] = "calibration_no_stim"

    meta = {
        "record_key": str(g["record_key"].iloc[0]),
        "status": "ok" if threshold_ok else "threshold_unavailable",
        "anomaly_model": str(anomaly_model),
        "n_windows": int(len(g)),
        "n_healthy": n_healthy,
        "n_fit": int(len(fit_pos)),
        "n_fit_after_pruning": int(getattr(baseline, "n_fit_", len(fit_pos))),
        "n_outlier_removed": int(getattr(baseline, "n_outlier_removed_", 0)),
        "n_val": int(len(val_pos)),
        "fs": fs,
        "calibration_end_center_sample": calibration_end_center,
        "guard_samples": int(guard_samples),
        "guard_end_center_sample": int(guard_end_center),
        "threshold": float(threshold),
        "threshold_method": str(thr_info["method"]),
        "threshold_status": str(thr_info["status"]),
        "target_false_inhibit": float(thr_info["target_false_inhibit"]),
        "conformal_alpha": float(thr_info["conformal_alpha"]),
        "conformal_alpha_min": float(thr_info["conformal_alpha_min"]),
        "fit_score_median": float(np.median(fit_scores)),
        "val_score_quantile": float(np.quantile(val_scores, threshold_quantile)),
        "calibration_outlier_frac": float(calibration_outlier_frac),
        "knn_k": int(knn_k) if str(anomaly_model).lower() == "knn" else np.nan,
        **{f"preprocess_{k}": v for k, v in preprocess_meta.items()},
    }
    return g, meta


def run_per_record_validation(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored_groups: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, object]] = []
    for _, group in df.groupby("record_key", sort=True):
        scored, meta = fit_score_one_group(
            group=group,
            embeddings=embeddings,
            threshold_quantile=args.threshold_quantile,
            shrinkage=args.shrinkage,
            eps=args.eps,
            anomaly_model=args.anomaly_model,
            knn_k=args.knn_k,
            calibration_outlier_frac=args.calibration_outlier_frac,
            min_fit=args.min_fit_windows,
            min_val=args.min_val_windows,
            calibration_fit_frac=args.calibration_fit_frac,
            calibration_val_frac=args.calibration_val_frac,
            guard_s=args.guard_s,
            l2_normalize=args.l2_normalize_embeddings,
            pca_dim=args.pca_dim,
            pca_whiten=args.pca_whiten,
            threshold_method=args.threshold_method,
            conformal_alpha=args.conformal_alpha,
        )
        scored_groups.append(scored)
        threshold_rows.append(meta)
    return pd.concat(scored_groups, axis=0).sort_index(), pd.DataFrame(threshold_rows)


def run_global_validation(df: pd.DataFrame, embeddings: np.ndarray, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df.copy()
    g["split"] = assign_global_splits(g, args.global_fit_record_frac, args.global_val_record_frac)
    fit_mask = (g["split"] == "fit") & g["is_healthy_window"].astype(bool)
    val_mask = (g["split"] == "val") & g["is_healthy_window"].astype(bool)
    if fit_mask.sum() < args.min_fit_windows:
        raise RuntimeError(
            f"Not enough healthy fit windows for global validation: have {int(fit_mask.sum())}, need {args.min_fit_windows}. "
            f"Try --per-record-calibration or lower --min-fit-windows."
        )
    if val_mask.sum() < args.min_val_windows:
        # Carve val out of the fit tail; mark those rows' split as 'val' so they
        # are correctly excluded from the test eval. Records used for fit/val
        # remain disjoint from test records by construction of assign_global_splits.
        healthy_fit_idx = np.flatnonzero(fit_mask.to_numpy())
        n_val = max(args.min_val_windows, len(healthy_fit_idx) // 5)
        n_val = min(n_val, len(healthy_fit_idx) - args.min_fit_windows)
        n_val = max(1, n_val)
        val_idx = healthy_fit_idx[-n_val:]
        fit_idx = healthy_fit_idx[:-n_val]
        g.loc[g.index[val_idx], "split"] = "val"
        LOG.warning("Global validation: too few val-record healthy windows; carving %d from fit tail.", n_val)
    else:
        fit_idx = np.flatnonzero(fit_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())

    fit_emb, embeddings_transformed, preprocess_meta = fit_transform_embeddings(
        fit_embeddings=embeddings[fit_idx],
        all_embeddings=embeddings,
        l2_normalize=args.l2_normalize_embeddings,
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
    )
    baseline = fit_baseline_with_pruning(
        fit_embeddings=fit_emb,
        anomaly_model=args.anomaly_model,
        threshold_quantile=args.threshold_quantile,
        shrinkage=args.shrinkage,
        eps=args.eps,
        knn_k=args.knn_k,
        calibration_outlier_frac=args.calibration_outlier_frac,
        min_keep=args.min_fit_windows,
    )
    val_scores = baseline.score(embeddings_transformed[val_idx])
    thr_info = select_decision_threshold(
        val_scores=val_scores,
        method=args.threshold_method,
        threshold_quantile=args.threshold_quantile,
        conformal_alpha=args.conformal_alpha,
    )
    threshold = float(thr_info["threshold"])
    threshold_ok = str(thr_info["status"]) == "ok" and np.isfinite(threshold)
    if threshold_ok:
        baseline.threshold_ = threshold
    scores = baseline.score(embeddings_transformed)
    g = add_score_columns(g, scores, threshold, args.anomaly_model)
    if threshold_ok:
        g["decision"] = np.where(scores <= threshold, "permit", "inhibit")
    else:
        g["decision"] = "inhibit"
    g.loc[g["split"].isin(["fit", "val"]), "decision"] = "calibration_no_stim"
    thresholds = pd.DataFrame([
        {
            "record_key": "GLOBAL",
            "status": "ok" if threshold_ok else "threshold_unavailable",
            "anomaly_model": str(args.anomaly_model),
            "n_windows": int(len(g)),
            "n_fit": int(len(fit_idx)),
            "n_fit_after_pruning": int(getattr(baseline, "n_fit_", len(fit_idx))),
            "n_outlier_removed": int(getattr(baseline, "n_outlier_removed_", 0)),
            "n_val": int(len(val_idx)),
            "threshold": float(threshold),
            "threshold_method": str(thr_info["method"]),
            "threshold_status": str(thr_info["status"]),
            "target_false_inhibit": float(thr_info["target_false_inhibit"]),
            "conformal_alpha": float(thr_info["conformal_alpha"]),
            "conformal_alpha_min": float(thr_info["conformal_alpha_min"]),
            "calibration_outlier_frac": float(args.calibration_outlier_frac),
            "knn_k": int(args.knn_k) if str(args.anomaly_model).lower() == "knn" else np.nan,
            **{f"preprocess_{k}": v for k, v in preprocess_meta.items()},
        }
    ])
    return g, thresholds


def write_metrics(scored: pd.DataFrame, out_dir: Path) -> None:
    eval_df = scored[(scored["split"] == "test") & scored["decision"].isin(["permit", "inhibit"])].copy()
    eval_df["is_healthy"] = eval_df["is_healthy_window"].astype(bool)

    legacy_overall = decision_metrics(eval_df, decision_col="decision", healthy_col="is_healthy")
    overall = policy_decision_metrics(eval_df, decision_col="decision")
    if not eval_df.empty:
        legacy_overall = add_auroc_auprc(
            legacy_overall,
            y_abnormal=(~eval_df["is_healthy"].to_numpy()).astype(int),
            scores=eval_df["anomaly_score"].to_numpy(),
        )
        overall = add_normal_vs_danger_auroc(overall, eval_df)
    pd.DataFrame([overall]).to_csv(out_dir / "metrics_overall.csv", index=False)
    pd.DataFrame([legacy_overall]).to_csv(out_dir / "metrics_legacy_healthy_vs_abnormal.csv", index=False)

    by_record = []
    for rk, gg in eval_df.groupby("record_key"):
        row = {"record_key": rk}
        row.update(policy_decision_metrics(gg, decision_col="decision"))
        by_record.append(row)
    pd.DataFrame(by_record).to_csv(out_dir / "metrics_by_record.csv", index=False)

    by_label = []
    if "dominant_label" in eval_df.columns:
        for label, gg in eval_df.groupby("dominant_label"):
            row = {"dominant_label": str(label)}
            row.update(policy_decision_metrics(gg, decision_col="decision"))
            by_label.append(row)
    pd.DataFrame(by_label).to_csv(out_dir / "metrics_by_label.csv", index=False)

    by_group = []
    if "safety_group" in eval_df.columns:
        for group_name, gg in eval_df.groupby("safety_group"):
            row = {"safety_group": str(group_name)}
            row.update(policy_decision_metrics(gg, decision_col="decision"))
            by_group.append(row)
    pd.DataFrame(by_group).to_csv(out_dir / "metrics_by_safety_group.csv", index=False)

    fp_cols = [
        "window_id", "dataset", "record", "record_key",
        "start_sample", "end_sample", "center_sample", "center_s",
        "dominant_label", "safety_group", "n_abnormal_beats", "n_normal_beats",
        "layer3_anomaly_model", "layer3_score", "anomaly_score",
        "layer3_mahal_score", "layer3_knn_score",
        "layer3_threshold", "score_over_threshold_ratio",
        "split", "decision",
    ]
    policy_eval = add_policy_columns(eval_df)
    false_permits = policy_eval[
        (policy_eval["safety_expectation_policy"] == INHIBIT_EXPECTED)
        & (policy_eval["decision"] == "permit")
    ].copy()
    false_permits["reason"] = "score_below_threshold_on_inhibit_expected_window"
    keep = [c for c in fp_cols if c in false_permits.columns] + ["reason"]
    false_permits[keep].to_csv(out_dir / "false_permits_detail.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate Layer 3 learned ECG embedding anomaly veto on ECG windows.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--window-index", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-record-calibration", action="store_true")
    p.add_argument("--lead-index", type=int, default=0)
    p.add_argument("--target-fs", type=float, default=125.0,
                   help="Resample windows to this Hz (default: 125 Hz for 8 s rhythm windows); set <=0 to keep native fs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0. auto uses CUDA if available.")
    p.add_argument("--threshold-method", default="conformal", choices=["conformal", "healthy_quantile"],
                   help="Decision threshold on healthy calibration scores. conformal (default) gives a "
                        "stated healthy false-inhibit budget (--conformal-alpha); healthy_quantile is the "
                        "legacy quantile method (--threshold-quantile).")
    p.add_argument("--conformal-alpha", type=float, default=0.10,
                   help="Target healthy false-inhibit budget for --threshold-method conformal. If alpha is "
                        "infeasible for the healthy calibration size, the record fails safe to inhibit.")
    p.add_argument("--threshold-quantile", type=float, default=0.99,
                   help="Healthy calibration quantile for --threshold-method healthy_quantile.")
    p.add_argument("--anomaly-model", default="mahalanobis", choices=["mahalanobis", "knn"],
                   help="Healthy-baseline embedding anomaly scorer.")
    p.add_argument("--shrinkage", type=float, default=0.10)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--knn-k", type=int, default=5,
                   help="Number of healthy calibration neighbors for --anomaly-model knn.")
    p.add_argument("--calibration-outlier-frac", type=float, default=0.0,
                   help="Optional fraction of most anomalous healthy-fit embeddings to remove before refitting the baseline, e.g. 0.05.")
    p.add_argument("--l2-normalize-embeddings", action="store_true",
                   help="L2-normalize encoder embeddings before PCA/baseline scoring. Required ablation for SSL objectives.")
    p.add_argument("--pca-dim", type=int, default=32,
                   help="PCA dimension fitted on healthy calibration embeddings before the baseline. Set <=0 to disable PCA.")
    p.add_argument("--pca-whiten", action="store_true",
                   help="Whiten PCA components before Mahalanobis/kNN. Off by default because Ledoit-Wolf models covariance downstream.")
    p.add_argument("--min-fit-windows", type=int, default=20)
    p.add_argument("--min-val-windows", type=int, default=10)
    p.add_argument("--calibration-fit-frac", type=float, default=0.60)
    p.add_argument("--calibration-val-frac", type=float, default=0.20)
    p.add_argument("--guard-s", type=float, default=None,
                   help="Temporal buffer (in seconds) between the last calibration window center and the first test window center. Default: --window-s, so calibration and test windows cannot overlap.")
    p.add_argument("--global-fit-record-frac", type=float, default=0.50)
    p.add_argument("--global-val-record-frac", type=float, default=0.20)
    p.add_argument("--max-windows", type=int, default=None, help="Truncate to first N windows after record-level filtering (smoke testing). Use --max-records for safer record-aware truncation.")
    p.add_argument("--max-records", type=int, default=None, help="Keep only the first N records per dataset (smoke testing). Preferred over --max-windows.")
    p.add_argument("--no-random-fallback", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Request torch deterministic algorithms (best effort)")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_fs = None if args.target_fs is not None and args.target_fs <= 0 else args.target_fs
    datasets = set(parse_csv_list(args.datasets))
    if not Path(args.window_index).exists():
        raise RuntimeError(f"--window-index file not found: {args.window_index}. Run build_window_index.py first.")
    df = pd.read_csv(args.window_index)
    if "dataset" not in df.columns:
        raise RuntimeError(f"--window-index CSV does not contain a 'dataset' column. Got: {list(df.columns)[:20]}")
    df = df[df["dataset"].astype(str).isin(datasets)].copy()
    if df.empty:
        raise RuntimeError(f"No windows selected for datasets={sorted(datasets)} from {args.window_index}.")
    if args.guard_s is None:
        if "window_s" in df.columns:
            window_s_values = pd.to_numeric(df["window_s"], errors="coerce")
            window_s_values = window_s_values[np.isfinite(window_s_values) & (window_s_values > 0)]
            args.guard_s = float(window_s_values.median()) if not window_s_values.empty else 5.0
        else:
            args.guard_s = 5.0

    # Record-aware filtering (preferred) before any positional truncation.
    if args.max_records is not None:
        keep = (
            df.groupby("dataset")["record"]
            .apply(lambda s: sorted(s.unique())[: int(args.max_records)])
            .to_dict()
        )
        mask = df.apply(lambda r: r["record"] in keep.get(r["dataset"], []), axis=1)
        df = df[mask].copy()
        if df.empty:
            raise RuntimeError("--max-records produced an empty selection.")
    if args.max_windows is not None:
        df = df.head(int(args.max_windows)).copy()

    # Embeddings are stored in the same row order as this DataFrame. Resetting the
    # index prevents sparse CSV/window_id values from being accidentally used as
    # numpy positions during per-record scoring.
    df = df.reset_index(drop=True)
    df["record_key"] = df["dataset"].astype(str) + "/" + df["record"].astype(str)
    df["is_healthy_window"] = df["is_healthy_window"].astype(bool)
    LOG.info("Selected %d windows across %d records.", len(df), df["record_key"].nunique())

    model, encoder_info = build_encoder(
        checkpoint=args.checkpoint,
        device=args.device,
        allow_random_fallback=not args.no_random_fallback,
    )
    resolved_device = str(encoder_info.get("device", args.device))
    encoder_info["seed"] = int(args.seed)
    write_json(out_dir / "encoder_info.json", encoder_info)
    if encoder_info.get("source", "").startswith("RandomConvEncoder"):
        LOG.warning("Validation will use a randomly-initialized fallback encoder; results are pipeline smoke-test only.")

    t0 = time.perf_counter()
    windows = load_windows(df, data_dir=args.data_dir, lead_index=args.lead_index, target_fs=target_fs)
    load_s = time.perf_counter() - t0
    LOG.info("Loaded %d windows in %.1fs", len(windows), load_s)

    t1 = time.perf_counter()
    embeddings = encode_windows(model, windows, batch_size=args.batch_size, device=resolved_device)
    encode_s = time.perf_counter() - t1
    LOG.info("Encoded %d windows in %.1fs (embedding dim=%d)", len(windows), encode_s, embeddings.shape[1] if embeddings.size else 0)
    np.save(out_dir / "embeddings.npy", embeddings)

    if args.per_record_calibration:
        scored, thresholds = run_per_record_validation(df, embeddings, args)
        calibration_mode = "per_record_calibration"
    else:
        scored, thresholds = run_global_validation(df, embeddings, args)
        calibration_mode = "global_record_split"

    scored.to_csv(out_dir / "per_window.csv", index=False)
    thresholds.to_csv(out_dir / "thresholds.csv", index=False)

    target_false_inhibit = (
        float(args.conformal_alpha)
        if args.threshold_method == "conformal"
        else float(1.0 - float(args.threshold_quantile))
    )
    write_threshold_coverage(
        scored,
        out_dir,
        threshold_method=args.threshold_method,
        target_false_inhibit=target_false_inhibit,
        healthy_col="is_healthy_window",
    )

    score_cols = [
        "window_id", "dataset", "record", "record_key",
        "start_sample", "end_sample", "center_sample",
        "split", "decision",
        "layer3_anomaly_model", "layer3_score", "anomaly_score",
        "layer3_mahal_score", "layer3_knn_score",
        "layer3_threshold", "threshold",
        "score_over_threshold_ratio",
        "is_healthy_window", "dominant_label", "safety_group",
    ]
    existing_score_cols = [c for c in score_cols if c in scored.columns]
    scored[existing_score_cols].to_csv(out_dir / "embedding_scores.csv", index=False)

    write_metrics(scored, out_dir)

    runtime = {
        "n_windows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "load_seconds": float(load_s),
        "encode_seconds": float(encode_s),
        "seconds_per_window_encode_only": float(encode_s / max(1, len(df))),
        "calibration_mode": calibration_mode,
        "anomaly_model": str(args.anomaly_model),
        "knn_k": int(args.knn_k),
        "calibration_outlier_frac": float(args.calibration_outlier_frac),
        "l2_normalize_embeddings": bool(args.l2_normalize_embeddings),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "guard_s": float(args.guard_s),
        "target_fs": target_fs,
        "safety_grouping": "If present in the window index, safety_group comes from Layer3/pipeline/label_grouping.py rhythm-span labels.",
        "lead_index": args.lead_index,
        "seed": int(args.seed),
        "safety_framing": "Layer 3 is an offline/research anomaly veto; uncertainty/failure should inhibit and it must not command stimulation alone.",
    }
    write_json(out_dir / "runtime_summary.json", runtime)

    with (out_dir / "FINAL_LAYER3_SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("# Layer 3 window-level validation summary\n\n")
        f.write("This evaluates an ECG embedding anomaly veto, not a clinical arrhythmia classifier.\n\n")
        f.write(SAFETY_DISCLAIMER)
        f.write("\n## Configuration\n\n")
        f.write(f"- Calibration mode: `{calibration_mode}`\n")
        f.write(f"- Anomaly model: `{args.anomaly_model}`\n")
        if args.anomaly_model == "knn":
            f.write(f"- kNN neighbors: `{args.knn_k}`\n")
        f.write(f"- L2-normalize embeddings: `{args.l2_normalize_embeddings}`\n")
        f.write(f"- PCA dimension before baseline: `{args.pca_dim}` (`<=0` disables PCA)\n")
        f.write(f"- PCA whitening: `{args.pca_whiten}`\n")
        f.write(f"- Calibration outlier pruning: `{args.calibration_outlier_frac}`\n")
        f.write(f"- Guard between calibration and test windows: `{args.guard_s}` s\n")
        f.write(f"- Windows scored: {len(df)}\n")
        f.write(f"- Embedding shape: {list(embeddings.shape)}\n")
        f.write(f"- Encoder source: {encoder_info.get('source')}\n")
        f.write(f"- Checkpoint loaded: {encoder_info.get('checkpoint_loaded')}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write("\nNote: `anomaly_score` is the model-agnostic embedding distance to the healthy baseline. "
                "`layer3_mahal_score` or `layer3_knn_score` is populated according to the selected anomaly model.\n")

    LOG.info("Wrote Layer 3 window validation outputs to %s", out_dir)


if __name__ == "__main__":
    main()
