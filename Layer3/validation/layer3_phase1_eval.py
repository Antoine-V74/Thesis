#!/usr/bin/env python3
"""Phase 1 A0/Layer3 evaluation helpers for beat-synchronous validation.

This module keeps the research comparison code out of run_beat_validation.py:
A0 handcrafted Layer 2 features and Layer 3 embeddings share the same
Mahalanobis/kNN scorers and threshold reporting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
LAYER1_DIR = PROJECT_ROOT / "Layer1"
for path in (PROJECT_ROOT, LAYER3_ROOT, LAYER1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Layer2._bootstrap import setup_layer2_paths  # noqa: E402

from run_window_validation import fit_baseline_with_pruning, fit_transform_embeddings  # noqa: E402
from layer3_validation_utils import (  # noqa: E402
    DEFAULT_DANGEROUS_SYMBOLS,
    DEFAULT_NORMAL_SYMBOLS,
    compute_guard_samples,
    conformal_threshold_from_scores,
    import_wfdb,
    parse_csv_list,
    phase1_label_group,
    resolve_record_path,
    safe_auroc,
    wilson_ci,
)
from label_grouping import (  # noqa: E402
    AF_CONTEXT,
    BENIGN_ABNORMAL,
    DANGEROUS,
    NORMAL,
    NOISE,
    UNLABELED,
)

def _phase1_prepare_group(
    group: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, object]]:
    g = group.sort_values("center_sample").copy()
    if "fs" in g.columns and pd.notna(g["fs"].iloc[0]):
        fs = float(g["fs"].iloc[0])
    else:
        fs = 250.0
    healthy = g["is_healthy_window"].astype(bool).to_numpy()
    healthy_positions = np.flatnonzero(healthy)
    n_healthy = int(len(healthy_positions))
    meta: Dict[str, object] = {
        "record_key": str(g["record_key"].iloc[0]),
        "n_beats": int(len(g)),
        "n_healthy": n_healthy,
        "fs": fs,
    }
    if n_healthy < max(args.min_fit_beats + args.min_val_beats, 2):
        g["split"] = "unscored_insufficient_healthy_calibration"
        meta.update({
            "status": "insufficient_healthy_calibration",
            "n_fit": 0,
            "n_val": 0,
            "guard_samples": 0,
        })
        return g, np.array([], dtype=int), np.array([], dtype=int), meta

    n_fit = max(args.min_fit_beats, int(round(n_healthy * args.calibration_fit_frac)))
    n_val = max(args.min_val_beats, int(round(n_healthy * args.calibration_val_frac)))
    if n_fit + n_val > n_healthy:
        n_fit = max(args.min_fit_beats, n_healthy - args.min_val_beats)
        n_val = max(args.min_val_beats, n_healthy - n_fit)
    n_fit = int(max(1, min(n_fit, n_healthy - 1)))
    n_val = int(max(1, min(n_val, n_healthy - n_fit)))

    fit_pos = healthy_positions[:n_fit]
    val_pos = healthy_positions[n_fit:n_fit + n_val]
    calibration_end_center = int(g.iloc[val_pos[-1]]["center_sample"])
    guard_samples = compute_guard_samples(
        window_s=float(g["window_s"].iloc[0]) if "window_s" in g.columns else float("nan"),
        guard_s=args.guard_s,
        fs=fs,
    )
    guard_end_center = calibration_end_center + guard_samples

    split = np.array(["test"] * len(g), dtype=object)
    split[fit_pos] = "fit"
    split[val_pos] = "val"
    centers = g["center_sample"].to_numpy()
    split[(centers <= calibration_end_center) & (split == "test")] = "calibration_excluded"
    split[(centers > calibration_end_center) & (centers <= guard_end_center) & (split == "test")] = "guard_excluded"
    g["split"] = split
    meta.update({
        "status": "ok",
        "n_fit": int(len(fit_pos)),
        "n_val": int(len(val_pos)),
        "calibration_end_center_sample": calibration_end_center,
        "guard_samples": int(guard_samples),
        "guard_end_center_sample": int(guard_end_center),
    })
    return g, fit_pos, val_pos, meta


def _score_phase1_matrix(
    values: np.ndarray,
    fit_pos: np.ndarray,
    scorer: str,
    args: argparse.Namespace,
    *,
    apply_embedding_preprocess: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scorer = str(scorer).lower()
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Phase 1 scorer expects a 2D feature matrix")

    fit_values = x[fit_pos]
    preprocess_meta: Dict[str, object] = {}
    if apply_embedding_preprocess:
        fit_values, x, preprocess_meta = fit_transform_embeddings(
            fit_embeddings=fit_values,
            all_embeddings=x,
            l2_normalize=args.l2_normalize_embeddings,
            pca_dim=args.pca_dim,
            pca_whiten=args.pca_whiten,
        )

    baseline = fit_baseline_with_pruning(
        fit_embeddings=fit_values,
        anomaly_model=scorer,
        threshold_quantile=args.threshold_quantile,
        shrinkage=args.shrinkage,
        eps=args.eps,
        knn_k=args.knn_k,
        calibration_outlier_frac=args.calibration_outlier_frac,
        min_keep=args.min_fit_beats,
    )
    scores = baseline.score(x)
    meta = {
        "n_fit_after_pruning": int(getattr(baseline, "n_fit_", len(fit_pos))),
        "n_outlier_removed": int(getattr(baseline, "n_outlier_removed_", 0)),
        "knn_k": int(args.knn_k) if scorer == "knn" else np.nan,
    }
    meta.update({f"preprocess_{k}": v for k, v in preprocess_meta.items()})
    return scores.astype(float), meta


def _score_phase1_prepared_group(
    g: pd.DataFrame,
    values: np.ndarray,
    fit_pos: np.ndarray,
    val_pos: np.ndarray,
    base_meta: Dict[str, object],
    args: argparse.Namespace,
    arm: str,
    scorer: str,
    feature_meta: Dict[str, object] | None = None,
    *,
    apply_embedding_preprocess: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = g.copy()
    meta = dict(base_meta)
    meta.update({"arm": arm, "scorer": scorer})
    if feature_meta:
        meta.update(feature_meta)

    if meta.get("status") != "ok":
        out["anomaly_score"] = np.nan
        out["threshold_healthy_quantile"] = np.nan
        out["threshold_conformal"] = np.nan
        out["decision_healthy_quantile"] = "inhibit"
        out["decision_conformal"] = "inhibit"
        out["score_over_threshold_healthy_quantile"] = np.nan
        out["score_over_threshold_conformal"] = np.nan
        out["conformal_alpha"] = float(args.conformal_alpha)
        out["conformal_status"] = str(meta.get("status"))
        meta.update({
            "healthy_quantile_threshold": np.nan,
            "conformal_threshold": np.nan,
            "conformal_status": str(meta.get("status")),
        })
        return out, meta

    try:
        scores, score_meta = _score_phase1_matrix(
            values,
            fit_pos,
            scorer,
            args,
            apply_embedding_preprocess=apply_embedding_preprocess,
        )
        val_scores = scores[val_pos]
        finite_val_scores = val_scores[np.isfinite(val_scores)]
        if finite_val_scores.size == 0:
            raise RuntimeError("No finite healthy validation scores for Phase 1 thresholding")
        q_threshold = float(np.quantile(finite_val_scores, float(args.threshold_quantile)))
        conformal = conformal_threshold_from_scores(finite_val_scores, float(args.conformal_alpha))
        c_threshold = float(conformal["threshold"]) if conformal["status"] == "ok" else np.nan

        out["anomaly_score"] = scores.astype(float)
        out["threshold_healthy_quantile"] = q_threshold
        out["threshold_conformal"] = c_threshold
        out["score_over_threshold_healthy_quantile"] = scores / q_threshold if q_threshold > 0 else np.nan
        out["score_over_threshold_conformal"] = (
            scores / c_threshold if np.isfinite(c_threshold) and c_threshold > 0 else np.nan
        )

        q_decision = np.where(scores <= q_threshold, "permit", "inhibit")
        if conformal["status"] == "ok":
            c_decision = np.where(scores <= c_threshold, "permit", "inhibit")
        else:
            c_decision = np.array(["inhibit"] * len(out), dtype=object)
        no_stim = out["split"].isin(["fit", "val", "calibration_excluded", "guard_excluded"])
        q_decision[no_stim.to_numpy()] = "calibration_no_stim"
        c_decision[no_stim.to_numpy()] = "calibration_no_stim"
        out["decision_healthy_quantile"] = q_decision
        out["decision_conformal"] = c_decision
        out["conformal_alpha"] = float(args.conformal_alpha)
        out["conformal_status"] = str(conformal["status"])

        meta.update(score_meta)
        meta.update({
            "status": "ok",
            "healthy_quantile_threshold": q_threshold,
            "conformal_threshold": c_threshold,
            "conformal_alpha": float(args.conformal_alpha),
            "conformal_status": str(conformal["status"]),
            "conformal_n": int(conformal["n"]),
            "conformal_rank": int(conformal["rank"]),
            "conformal_alpha_min": float(conformal["alpha_min"]),
            "fit_score_median": float(np.median(scores[fit_pos])),
            "val_score_quantile": float(np.quantile(finite_val_scores, float(args.threshold_quantile))),
        })
    except Exception as exc:
        out["anomaly_score"] = np.nan
        out["threshold_healthy_quantile"] = np.nan
        out["threshold_conformal"] = np.nan
        out["decision_healthy_quantile"] = "inhibit"
        out["decision_conformal"] = "inhibit"
        out["score_over_threshold_healthy_quantile"] = np.nan
        out["score_over_threshold_conformal"] = np.nan
        out["conformal_alpha"] = float(args.conformal_alpha)
        out["conformal_status"] = "scorer_error"
        meta.update({
            "status": "scorer_error",
            "error": str(exc),
            "healthy_quantile_threshold": np.nan,
            "conformal_threshold": np.nan,
            "conformal_status": "scorer_error",
        })
    return out, meta


def _phase1_wide_to_long(scored_wide: pd.DataFrame, arm: str, scorer: str) -> pd.DataFrame:
    base_cols = [
        "beat_id", "dataset", "record", "record_key", "fs", "signal_len",
        "beat_index", "trigger_index", "beat_sample", "trigger_sample",
        "center_sample", "start_sample", "end_sample", "beat_time_s", "trigger_time_s",
        "window_s", "beat_symbol", "dominant_label", "safety_group", "safety_expectation",
        "is_healthy_beat", "is_healthy_window", "matched_annotation",
        "matched_annotation_sample", "matched_annotation_dt_s", "has_labels",
        "window_mode", "mode", "trigger_source", "layer1_filter",
        "split", "phase1_label_group", "a0_n_beats_morph_window", "anomaly_score",
    ]
    rows = []
    for method, threshold_col, decision_col, ratio_col in [
        ("healthy_quantile", "threshold_healthy_quantile", "decision_healthy_quantile", "score_over_threshold_healthy_quantile"),
        ("conformal", "threshold_conformal", "decision_conformal", "score_over_threshold_conformal"),
    ]:
        tmp = scored_wide[[c for c in base_cols if c in scored_wide.columns]].copy()
        tmp["arm"] = arm
        tmp["scorer"] = scorer
        tmp["threshold_method"] = method
        tmp["threshold"] = scored_wide[threshold_col].to_numpy()
        tmp["decision"] = scored_wide[decision_col].to_numpy()
        tmp["score_over_threshold_ratio"] = scored_wide[ratio_col].to_numpy()
        if method == "conformal":
            tmp["conformal_alpha"] = (
                scored_wide["conformal_alpha"].to_numpy()
                if "conformal_alpha" in scored_wide.columns
                else np.nan
            )
            status = (
                scored_wide["conformal_status"].astype(str)
                if "conformal_status" in scored_wide.columns
                else pd.Series([""] * len(scored_wide), index=scored_wide.index)
            )
            tmp["conformal_status"] = status.to_numpy()
            tmp["conformal_alpha_infeasible"] = status.eq("alpha_infeasible").to_numpy()
        else:
            tmp["conformal_alpha"] = np.nan
            tmp["conformal_status"] = ""
            tmp["conformal_alpha_infeasible"] = False
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def _load_layer2_beat_module():
    setup_layer2_paths(include_validation=True, include_archive=False)
    layer2_file = PROJECT_ROOT / "Layer2" / "validation" / "run_beat_validation.py"
    spec = importlib.util.spec_from_file_location("layer2_phase1_run_beat_validation", layer2_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Layer 2 beat validation module at {layer2_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _phase1_a0_feature_mode(args: argparse.Namespace) -> str:
    mode = str(args.a0_feature_window_mode).lower()
    if mode == "auto":
        return "causal" if args.causal_window else "centered"
    return mode


def compute_a0_feature_dicts(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[List[Dict[str, float]], List[int]]:
    layer2_mod = _load_layer2_beat_module()
    wfdb = import_wfdb()
    feature_mode = _phase1_a0_feature_mode(args)
    post_r_s = float(args.a0_post_r_lookahead_ms) / 1000.0
    feature_dicts: List[Dict[str, float]] = [dict() for _ in range(len(df))]
    n_beats_win: List[int] = [0 for _ in range(len(df))]
    for _, group in df.groupby("record_key", sort=True):
        first = group.iloc[0]
        record_path = resolve_record_path(args.data_dir, first["dataset"], first["record"], first.get("record_path", None))
        rec = wfdb.rdrecord(str(record_path), channels=[int(args.lead_index)])
        if rec.p_signal is not None:
            raw = rec.p_signal[:, 0].astype(float)
        elif rec.d_signal is not None:
            raw = rec.d_signal[:, 0].astype(float)
        else:
            raise RuntimeError(f"No signal found in {record_path}")
        fs = float(rec.fs)
        filt = (
            layer2_mod._causal_layer2_filter(raw, fs)
            if feature_mode == "causal"
            else layer2_mod.apply_filters(raw, fs)
        )
        time_col = "trigger_time_s" if "trigger_time_s" in group.columns else "beat_time_s"
        peaks = np.sort(pd.to_numeric(group[time_col], errors="coerce").dropna().to_numpy(dtype=float))
        for idx, row in group.iterrows():
            beat_time = float(row.get(time_col, row.get("beat_time_s")))
            feats, n_win = layer2_mod.extract_beat_features(
                filt,
                fs,
                beat_time,
                peaks,
                morphology_window_s=float(args.a0_feature_window_s),
                rr_lookback_s=float(args.a0_rr_lookback_s),
                species="human",
                feature_window_mode=feature_mode,
                post_r_lookahead_s=post_r_s,
            )
            feature_dicts[int(idx)] = feats
            n_beats_win[int(idx)] = int(n_win)
    return feature_dicts, n_beats_win


def _a0_matrix_for_group(
    g: pd.DataFrame,
    fit_pos: np.ndarray,
    feature_dicts: Sequence[Dict[str, float]],
    min_finite_frac: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, object]]:
    fit_indices = g.index.to_numpy()[fit_pos]
    fit_dicts = [feature_dicts[int(i)] for i in fit_indices]
    keys = sorted({k for f in fit_dicts for k in f.keys()})
    selected: List[str] = []
    medians: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([f.get(key, np.nan) for f in fit_dicts], dtype=float)
        finite = np.isfinite(vals)
        if float(finite.mean()) >= float(min_finite_frac) and finite.any():
            selected.append(key)
            medians[key] = float(np.median(vals[finite]))
    if not selected:
        raise RuntimeError("A0 feature selection produced zero finite calibration features")

    matrix = np.empty((len(g), len(selected)), dtype=np.float64)
    for row_i, df_idx in enumerate(g.index.to_numpy()):
        feats = feature_dicts[int(df_idx)]
        for col_i, key in enumerate(selected):
            val = float(feats.get(key, np.nan))
            matrix[row_i, col_i] = val if np.isfinite(val) else medians[key]

    fit_values = matrix[fit_pos]
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    matrix = (matrix - mean) / std
    return matrix, {
        "feature_dim": int(len(selected)),
        "feature_names_json": json.dumps(selected),
    }


def _phase1_group_for_row(row: pd.Series, normal_symbols: set[str], dangerous_symbols: set[str]) -> str:
    safety_group = str(row.get("safety_group", ""))
    if safety_group in {NORMAL, DANGEROUS, BENIGN_ABNORMAL, NOISE, AF_CONTEXT, UNLABELED}:
        return safety_group
    return phase1_label_group(row.get("beat_symbol", ""), normal_symbols, dangerous_symbols)


def run_phase1_eval(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal_symbols = set(parse_csv_list(args.normal_symbols)) or set(DEFAULT_NORMAL_SYMBOLS)
    dangerous_symbols = set(parse_csv_list(args.dangerous_symbols)) or set(DEFAULT_DANGEROUS_SYMBOLS)
    arms = [a.lower() for a in parse_csv_list(args.phase1_arms)]
    scorers = [s.lower() for s in parse_csv_list(args.phase1_scorers)]
    bad_arms = sorted(set(arms) - {"layer3", "a0", "a0_layer2_features"})
    bad_scorers = sorted(set(scorers) - {"mahalanobis", "knn"})
    if bad_arms:
        raise ValueError(f"Unsupported Phase 1 arms: {bad_arms}")
    if bad_scorers:
        raise ValueError(f"Unsupported Phase 1 scorers: {bad_scorers}")

    df = df.copy()
    df["phase1_label_group"] = [
        _phase1_group_for_row(row, normal_symbols, dangerous_symbols)
        for _, row in df.iterrows()
    ]

    a0_features: List[Dict[str, float]] | None = None
    a0_n_beats: List[int] | None = None
    if "a0" in arms or "a0_layer2_features" in arms:
        a0_features, a0_n_beats = compute_a0_feature_dicts(df, args)

    scored_parts: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, object]] = []
    for _, group in df.groupby("record_key", sort=True):
        g, fit_pos, val_pos, base_meta = _phase1_prepare_group(group, args)
        for arm in arms:
            arm_name = "a0_layer2_features" if arm == "a0" else arm
            for scorer in scorers:
                feature_meta: Dict[str, object] = {}
                apply_embedding_preprocess = False
                if arm_name == "layer3":
                    values = embeddings[g.index.to_numpy()]
                    feature_meta = {"feature_dim": int(values.shape[1])}
                    apply_embedding_preprocess = True
                elif arm_name == "a0_layer2_features":
                    if a0_features is None:
                        raise RuntimeError("A0 features were not computed")
                    if base_meta.get("status") == "ok":
                        values, feature_meta = _a0_matrix_for_group(
                            g,
                            fit_pos,
                            a0_features,
                            min_finite_frac=float(args.a0_min_finite_frac),
                        )
                        if a0_n_beats is not None:
                            g = g.copy()
                            g["a0_n_beats_morph_window"] = [a0_n_beats[int(i)] for i in g.index.to_numpy()]
                    else:
                        values = np.zeros((len(g), 1), dtype=float)
                        feature_meta = {"feature_dim": 0, "feature_names_json": "[]"}
                else:
                    raise ValueError(f"Unsupported Phase 1 arm: {arm}")

                wide, meta = _score_phase1_prepared_group(
                    g=g,
                    values=values,
                    fit_pos=fit_pos,
                    val_pos=val_pos,
                    base_meta=base_meta,
                    args=args,
                    arm=arm_name,
                    scorer=scorer,
                    feature_meta=feature_meta,
                    apply_embedding_preprocess=apply_embedding_preprocess,
                )
                scored_parts.append(_phase1_wide_to_long(wide, arm_name, scorer))
                threshold_rows.append(meta)
    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(threshold_rows)


def _phase1_metrics_for_group(g: pd.DataFrame, dataset: str) -> Dict[str, object]:
    label = g["phase1_label_group"].astype(str)
    normal = label.eq(NORMAL)
    dangerous = label.eq(DANGEROUS)
    all_abnormal = ~normal
    permit = g["decision"].astype(str).str.lower().eq("permit")

    fp_danger = int((dangerous & permit).sum())
    n_danger = int(dangerous.sum())
    fi_normal = int((normal & ~permit).sum())
    n_normal = int(normal.sum())
    fp_lo, fp_hi = wilson_ci(fp_danger, n_danger)
    fi_lo, fi_hi = wilson_ci(fi_normal, n_normal)
    nd = normal | dangerous
    row: Dict[str, object] = {
        "dataset": dataset,
        "arm": g["arm"].iloc[0],
        "scorer": g["scorer"].iloc[0],
        "threshold_method": g["threshold_method"].iloc[0],
        "n": int(len(g)),
        "n_NORMAL": n_normal,
        "n_DANGEROUS": n_danger,
        "n_all_abnormal": int(all_abnormal.sum()),
        "false_permit_DANGEROUS_n": fp_danger,
        "false_permit_DANGEROUS": float(fp_danger / n_danger) if n_danger else float("nan"),
        "false_permit_DANGEROUS_ci_low": fp_lo,
        "false_permit_DANGEROUS_ci_high": fp_hi,
        "false_inhibit_NORMAL_n": fi_normal,
        "false_inhibit_NORMAL": float(fi_normal / n_normal) if n_normal else float("nan"),
        "false_inhibit_NORMAL_ci_low": fi_lo,
        "false_inhibit_NORMAL_ci_high": fi_hi,
        "inhibit_rate_BENIGN_ABNORMAL": float((~permit[label.eq(BENIGN_ABNORMAL)]).mean()) if label.eq(BENIGN_ABNORMAL).any() else float("nan"),
        "inhibit_rate_AF_CONTEXT": float((~permit[label.eq(AF_CONTEXT)]).mean()) if label.eq(AF_CONTEXT).any() else float("nan"),
        "inhibit_rate_NOISE": float((~permit[label.eq(NOISE)]).mean()) if label.eq(NOISE).any() else float("nan"),
        "conformal_alpha_infeasible_n": int(g.get("conformal_alpha_infeasible", pd.Series(False, index=g.index)).astype(bool).sum()),
        "auroc_NORMAL_vs_DANGEROUS": safe_auroc(dangerous[nd].to_numpy(), g.loc[nd, "anomaly_score"].to_numpy()),
        "auroc_NORMAL_vs_all_abnormal": safe_auroc(all_abnormal.to_numpy(), g["anomaly_score"].to_numpy()),
    }
    return row


def _danger_fpr_threshold_ratio(ratios: pd.Series, target_fpr: float) -> Tuple[float, int, int]:
    """Pick an offline ratio threshold targeting false permits on DANGEROUS rows.

    This uses labeled DANGEROUS test rows and is therefore not deployable. It is
    only an offline operating-point analysis for the thesis safety table.
    """
    x = pd.to_numeric(ratios, errors="coerce").to_numpy(dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = int(x.size)
    if n <= 0:
        return float("nan"), 0, 0
    target = float(np.clip(target_fpr, 0.0, 1.0))
    max_permits = int(np.floor(target * n))
    if max_permits <= 0:
        return float(np.nextafter(x[0], -np.inf)), 0, n
    if max_permits >= n:
        return float(x[-1]), n, n

    # Choose the largest finite threshold that keeps the realized permit count
    # <= max_permits, accounting for ties in the dangerous ratios.
    unique = np.unique(x)
    best = float(np.nextafter(x[0], -np.inf))
    best_count = 0
    for cand in unique:
        count = int((x <= cand).sum())
        if count <= max_permits:
            best = float(cand)
            best_count = count
        else:
            break
    return best, best_count, n


def _add_phase1_offline_danger_operating_point(
    scored: pd.DataFrame,
    *,
    target_fpr: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append a non-deployable danger-targeted threshold method to Phase 1 rows."""
    base_all = scored[scored["threshold_method"] == "healthy_quantile"].copy()
    base_eval = base_all[
        (base_all["split"] == "test")
        & base_all["decision"].isin(["permit", "inhibit"])
    ].copy()
    if base_eval.empty:
        return scored, pd.DataFrame()

    op_rows: List[Dict[str, object]] = []
    extra_rows: List[pd.DataFrame] = []
    for (arm, scorer), gg_eval in base_eval.groupby(["arm", "scorer"], dropna=False):
        label = gg_eval["phase1_label_group"].astype(str)
        danger = label.eq(DANGEROUS)
        threshold_ratio, allowed_permits, n_danger = _danger_fpr_threshold_ratio(
            gg_eval.loc[danger, "score_over_threshold_ratio"],
            target_fpr,
        )
        tmp = base_all[(base_all["arm"] == arm) & (base_all["scorer"] == scorer)].copy()
        tmp["threshold_method"] = "danger_2pct_offline"
        tmp["threshold"] = threshold_ratio
        tmp["offline_non_deployable_threshold"] = True
        tmp["offline_target_false_permit_DANGEROUS"] = float(target_fpr)
        tmp["offline_threshold_basis"] = "DANGEROUS test labels; do not use for deployment calibration"
        if np.isfinite(threshold_ratio):
            ratio = pd.to_numeric(tmp["score_over_threshold_ratio"], errors="coerce").to_numpy(dtype=float)
            tmp["decision"] = np.where(ratio <= threshold_ratio, "permit", "inhibit")
            tmp.loc[~tmp["split"].eq("test"), "decision"] = "calibration_no_stim"
        else:
            tmp["decision"] = "inhibit"
            tmp.loc[~tmp["split"].eq("test"), "decision"] = "calibration_no_stim"
        extra_rows.append(tmp)
        tmp_eval = tmp[(tmp["split"] == "test") & tmp["decision"].isin(["permit", "inhibit"])].copy()
        tmp_danger = tmp_eval["phase1_label_group"].astype(str).eq(DANGEROUS)
        actual_fp = int((tmp_danger & tmp_eval["decision"].astype(str).str.lower().eq("permit")).sum())
        ci_lo, ci_hi = wilson_ci(actual_fp, n_danger)
        op_rows.append({
            "arm": arm,
            "scorer": scorer,
            "threshold_method": "danger_2pct_offline",
            "offline_non_deployable_threshold": True,
            "target_false_permit_DANGEROUS": float(target_fpr),
            "threshold_ratio": threshold_ratio,
            "n_DANGEROUS": int(n_danger),
            "allowed_false_permits_DANGEROUS_n": int(allowed_permits),
            "realized_false_permits_DANGEROUS_n": int(actual_fp),
            "realized_false_permit_DANGEROUS": float(actual_fp / n_danger) if n_danger else float("nan"),
            "realized_false_permit_DANGEROUS_ci_low": ci_lo,
            "realized_false_permit_DANGEROUS_ci_high": ci_hi,
            "note": "Offline labeled operating point only; deployment thresholds must use healthy calibration only.",
        })

    if not extra_rows:
        return scored, pd.DataFrame(op_rows)
    out = pd.concat([scored, *extra_rows], ignore_index=True)
    return out, pd.DataFrame(op_rows)


def write_phase1_outputs(
    scored: pd.DataFrame,
    thresholds: pd.DataFrame,
    out_dir: Path,
    *,
    offline_danger_fpr_target: float = 0.02,
) -> None:
    scored, offline_ops = _add_phase1_offline_danger_operating_point(
        scored,
        target_fpr=float(offline_danger_fpr_target),
    )
    scored.to_csv(out_dir / "phase1_per_beat.csv", index=False)
    thresholds.to_csv(out_dir / "phase1_thresholds.csv", index=False)
    offline_ops.to_csv(out_dir / "phase1_offline_operating_points.csv", index=False)
    eval_df = scored[(scored["split"] == "test") & scored["decision"].isin(["permit", "inhibit"])].copy()
    eval_df = eval_df[~eval_df["phase1_label_group"].eq(UNLABELED)].copy()

    metric_rows = []
    group_cols = ["dataset", "arm", "scorer", "threshold_method"]
    for keys, gg in eval_df.groupby(group_cols, dropna=False):
        metric_rows.append(_phase1_metrics_for_group(gg, dataset=str(keys[0])))
    metrics_by_dataset = pd.DataFrame(metric_rows)
    metrics_by_dataset.to_csv(out_dir / "phase1_metrics_by_dataset.csv", index=False)

    overall_rows = []
    for _, gg in eval_df.groupby(["arm", "scorer", "threshold_method"], dropna=False):
        overall_rows.append(_phase1_metrics_for_group(gg, dataset="ALL"))
    metrics_overall = pd.DataFrame(overall_rows)
    metrics_overall.to_csv(out_dir / "phase1_metrics_overall.csv", index=False)

    auroc_cols = [
        "dataset", "arm", "scorer", "threshold_method", "n_NORMAL", "n_DANGEROUS",
        "n_all_abnormal", "auroc_NORMAL_vs_DANGEROUS", "auroc_NORMAL_vs_all_abnormal",
    ]
    all_metrics = pd.concat([metrics_by_dataset, metrics_overall], ignore_index=True)
    for col in auroc_cols:
        if col not in all_metrics.columns:
            all_metrics[col] = np.nan
    all_metrics[auroc_cols].to_csv(
        out_dir / "phase1_aurocs.csv",
        index=False,
    )

