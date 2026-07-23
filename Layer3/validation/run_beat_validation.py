#!/usr/bin/env python3
"""
Beat-synchronous Layer 3 embedding anomaly validation.

For each annotated beat (oracle mode) this script extracts an ECG morphology window,
embeds it, fits a healthy-baseline anomaly model, and outputs one permit/inhibit
veto decision per beat/trigger.

Important: centered windows use future samples and are offline only. Use --causal-window
for a stricter real-time simulation.
"""
from __future__ import annotations

import argparse
import sys
import time
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

from Layer3._bootstrap import setup_layer3_paths  # noqa: E402
from Layer1._bootstrap import setup_layer1_paths  # noqa: E402

setup_layer3_paths(include_validation=True)
setup_layer1_paths(include_archive=True)

from run_window_validation import fit_score_one_group  # noqa: E402
from layer3_group_metrics import add_normal_vs_danger_auroc, add_policy_columns, policy_decision_metrics  # noqa: E402
from layer3_phase1_eval import run_phase1_eval, write_phase1_outputs  # noqa: E402
from layer3_validation_utils import (  # noqa: E402
    DEFAULT_NORMAL_SYMBOLS,
    KNOWN_BEAT_SYMBOLS,
    SAFETY_DISCLAIMER,
    add_auroc_auprc,
    build_encoder,
    decision_metrics,
    encode_windows,
    get_logger,
    import_wfdb,
    load_record_allowlist,
    parse_csv_list,
    read_wfdb_window,
    resolve_record_path,
    set_seed,
    write_json,
    write_threshold_coverage,
)
from label_grouping import (  # noqa: E402
    INHIBIT_EXPECTED,
    NORMAL,
    NOISE,
    UNLABELED,
    build_rhythm_spans,
    group_for_beat,
    safety_expectation,
)

LOG = get_logger("layer3.beat_sync")

try:
    from main_pipeline import layer1_r_peaks  # noqa: E402
    _ADAPTIVE_LAYER1_OK = True
except Exception:
    layer1_r_peaks = None
    _ADAPTIVE_LAYER1_OK = False


def zero_phase_layer1_filter(raw: np.ndarray, fs: float) -> np.ndarray:
    """Offline filter matching the Layer 2 beat-sync validation trigger path.

    This uses zero-phase filtering and is therefore not deployable real-time code.
    It is used only to reproduce the existing offline Layer 1 validation trigger
    stream for fair Layer 2 / Layer 3 comparison.
    """
    try:
        from scipy.signal import butter, filtfilt, iirnotch
    except Exception as exc:
        raise RuntimeError("scipy is required for --mode layer1_adaptive_gated trigger filtering") from exc

    x = np.where(np.isfinite(np.asarray(raw, dtype=float)), raw, 0.0)
    nyq = 0.5 * float(fs)
    lo = max(1e-4, 5.0 / nyq)
    hi = min(0.499, min(20.0, 0.45 * float(fs)) / nyq)
    b, a = butter(4, [lo, hi], btype="band")
    filt = filtfilt(b, a, x)
    if fs > 110:
        b, a = iirnotch(50.0 / (0.5 * fs), 30.0)
        filt = filtfilt(b, a, filt)
    if fs > 125:
        b, a = iirnotch(60.0 / (0.5 * fs), 30.0)
        filt = filtfilt(b, a, filt)
    return filt


def find_wfdb_records(dataset_dir: Path) -> List[str]:
    records: List[str] = []
    for hea in sorted(dataset_dir.rglob("*.hea")):
        records.append(str(hea.relative_to(dataset_dir).with_suffix("")).replace("\\", "/"))
    return records


def read_record_beats(
    data_dir: Path,
    dataset: str,
    record: str,
    ann_ext: str,
    window_s: float,
    normal_symbols: set[str],
    causal_window: bool,
    lookahead_ms: float,
) -> List[Dict[str, object]]:
    wfdb = import_wfdb()
    record_path = data_dir / dataset / record
    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    sig_len = int(header.sig_len)
    window_n = int(round(window_s * fs))

    try:
        ann = wfdb.rdann(str(record_path), ann_ext)
    except Exception as exc:
        print(f"[WARN] no annotation {ann_ext} for {dataset}/{record}: {exc}", file=sys.stderr)
        return []
    ann_samples = np.asarray(ann.sample, dtype=np.int64)
    ann_symbols = [str(s) for s in ann.symbol]
    ann_aux = [str(a) for a in (ann.aux_note if getattr(ann, "aux_note", None) is not None else [""] * len(ann_symbols))]
    if ann_samples.size:
        order = np.argsort(ann_samples, kind="mergesort")
        ann_samples = ann_samples[order]
        ann_symbols = [ann_symbols[i] for i in order]
        ann_aux = [ann_aux[i] for i in order]
    rhythm_spans = build_rhythm_spans(ann_samples, ann_symbols, ann_aux, sig_len)

    rows: List[Dict[str, object]] = []
    for beat_i, (sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
        symbol = str(symbol)
        if symbol not in KNOWN_BEAT_SYMBOLS:
            continue
        sample = int(sample)
        if causal_window:
            lookahead_n = int(round(max(0.0, float(lookahead_ms)) * fs / 1000.0))
            end = sample + lookahead_n + 1
            start = end - window_n
        else:
            start = sample - window_n // 2
            end = start + window_n
        if start < 0 or end > sig_len:
            continue
        safety_group = group_for_beat(sample, symbol, rhythm_spans, dataset=dataset)
        is_healthy = safety_group == NORMAL
        rows.append(
            {
                "dataset": dataset,
                "record": record,
                "record_path": str((Path(dataset) / record)).replace("\\", "/"),
                "record_key": f"{dataset}/{record}",
                "fs": fs,
                "signal_len": sig_len,
                "beat_index": int(beat_i),
                "beat_sample": sample,
                "center_sample": sample,
                "start_sample": int(start),
                "end_sample": int(end),
                "beat_time_s": float(sample / fs),
                "window_s": float(window_s),
                "beat_symbol": symbol,
                "dominant_label": symbol,
                "safety_group": safety_group,
                "safety_expectation": safety_expectation(safety_group),
                "is_healthy_beat": bool(is_healthy),
                # Alias used by the shared per-record scorer.
                "is_healthy_window": bool(is_healthy),
                "has_labels": True,
                "lookahead_ms": float(lookahead_ms) if causal_window else 0.0,
                "window_mode": (
                    "causal_with_lookahead" if causal_window and lookahead_ms > 0
                    else "causal" if causal_window
                    else "centered_offline"
                ),
            }
        )
    return rows


def _nearest_annotation_label(
    sample: int,
    ann_samples: np.ndarray,
    ann_symbols: Sequence[str],
    rhythm_spans,
    dataset: str,
    fs: float,
    normal_symbols: set[str],
    tolerance_s: float,
) -> Dict[str, object]:
    if len(ann_samples) == 0:
        return {
            "beat_symbol": "?",
            "dominant_label": "unmatched_trigger",
            "is_healthy_beat": False,
            "safety_group": "UNLABELED",
            "safety_expectation": "ignore",
            "matched_annotation_sample": np.nan,
            "matched_annotation_dt_s": np.nan,
            "matched_annotation": False,
        }
    sample = int(sample)
    pos = int(np.searchsorted(ann_samples, sample))
    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < len(ann_samples):
        candidates.append(pos)
    best_i = min(candidates, key=lambda i: abs(int(ann_samples[i]) - sample))
    best_sample = int(ann_samples[best_i])
    dt_s = abs(best_sample - sample) / float(fs)
    symbol = str(ann_symbols[best_i])
    matched = bool(dt_s <= float(tolerance_s) and symbol in KNOWN_BEAT_SYMBOLS)
    safety_group = group_for_beat(sample, symbol if matched else "?", rhythm_spans, dataset=dataset)
    # A Layer 1 trigger that does not match a beat should not become a healthy
    # calibration/example merely because it lies in a normal rhythm span. Keep
    # unmatched triggers labelable only for fail-safe groups needed by trigger
    # mode (VF/VT/noise/AF context); otherwise ignore them for metrics.
    span_or_symbol_labeled = safety_group in {DANGEROUS, NOISE, AF_CONTEXT}
    if not matched and not span_or_symbol_labeled:
        safety_group = UNLABELED
    if not matched:
        return {
            "beat_symbol": symbol if symbol else "?",
            "dominant_label": safety_group if span_or_symbol_labeled else "unmatched_trigger",
            "is_healthy_beat": False,
            "safety_group": safety_group,
            "safety_expectation": safety_expectation(safety_group),
            "matched_annotation_sample": best_sample,
            "matched_annotation_dt_s": float(dt_s),
            "matched_annotation": bool(span_or_symbol_labeled),
        }
    return {
        "beat_symbol": symbol,
        "dominant_label": symbol,
        "is_healthy_beat": bool(safety_group == NORMAL),
        "safety_group": safety_group,
        "safety_expectation": safety_expectation(safety_group),
        "matched_annotation_sample": best_sample,
        "matched_annotation_dt_s": float(dt_s),
        "matched_annotation": True,
    }


def read_record_layer1_adaptive_triggers(
    data_dir: Path,
    dataset: str,
    record: str,
    ann_ext: str,
    window_s: float,
    normal_symbols: set[str],
    causal_window: bool,
    lookahead_ms: float,
    lead_index: int,
    annotation_match_tolerance_s: float,
) -> List[Dict[str, object]]:
    if not _ADAPTIVE_LAYER1_OK or layer1_r_peaks is None:
        raise RuntimeError(
            "Layer 1 imports failed. Ensure Layer1/pipeline/main_pipeline.py is available."
        )

    wfdb = import_wfdb()
    record_path = data_dir / dataset / record
    rec = wfdb.rdrecord(str(record_path), channels=[int(lead_index)])
    if rec.p_signal is None or rec.p_signal.shape[0] == 0:
        return []
    fs = float(rec.fs)
    sig_len = int(rec.sig_len)
    raw = rec.p_signal[:, 0].astype(float)
    filt = zero_phase_layer1_filter(raw, fs)
    trigger_times_s = layer1_r_peaks(filt, fs)
    trigger_samples = np.round(np.asarray(trigger_times_s, dtype=float) * fs).astype(int)
    trigger_samples = trigger_samples[(trigger_samples >= 0) & (trigger_samples < sig_len)]
    window_n = int(round(window_s * fs))

    try:
        ann = wfdb.rdann(str(record_path), ann_ext)
        ann_samples = np.asarray(ann.sample, dtype=np.int64)
        ann_symbols = [str(s) for s in ann.symbol]
        ann_aux = [str(a) for a in (ann.aux_note if getattr(ann, "aux_note", None) is not None else [""] * len(ann_symbols))]
        if ann_samples.size:
            order = np.argsort(ann_samples, kind="mergesort")
            ann_samples = ann_samples[order]
            ann_symbols = [ann_symbols[i] for i in order]
            ann_aux = [ann_aux[i] for i in order]
    except Exception as exc:
        print(f"[WARN] no annotation {ann_ext} for trigger labeling {dataset}/{record}: {exc}", file=sys.stderr)
        ann_samples = np.asarray([], dtype=np.int64)
        ann_symbols = []
        ann_aux = []
    rhythm_spans = build_rhythm_spans(ann_samples, ann_symbols, ann_aux, sig_len)

    rows: List[Dict[str, object]] = []
    for trigger_i, sample in enumerate(trigger_samples):
        sample = int(sample)
        if causal_window:
            lookahead_n = int(round(max(0.0, float(lookahead_ms)) * fs / 1000.0))
            end = sample + lookahead_n + 1
            start = end - window_n
        else:
            start = sample - window_n // 2
            end = start + window_n
        if start < 0 or end > sig_len:
            continue
        label = _nearest_annotation_label(
            sample=sample,
            ann_samples=ann_samples,
            ann_symbols=ann_symbols,
            rhythm_spans=rhythm_spans,
            dataset=dataset,
            fs=fs,
            normal_symbols=normal_symbols,
            tolerance_s=annotation_match_tolerance_s,
        )
        rows.append(
            {
                "dataset": dataset,
                "record": record,
                "record_path": str((Path(dataset) / record)).replace("\\", "/"),
                "record_key": f"{dataset}/{record}",
                "fs": fs,
                "signal_len": sig_len,
                "beat_index": int(trigger_i),
                "trigger_index": int(trigger_i),
                "beat_sample": sample,
                "trigger_sample": sample,
                "center_sample": sample,
                "start_sample": int(start),
                "end_sample": int(end),
                "beat_time_s": float(sample / fs),
                "trigger_time_s": float(sample / fs),
                "window_s": float(window_s),
                "beat_symbol": str(label["beat_symbol"]),
                "dominant_label": str(label["dominant_label"]),
                "safety_group": str(label["safety_group"]),
                "safety_expectation": str(label["safety_expectation"]),
                "is_healthy_beat": bool(label["is_healthy_beat"]),
                "is_healthy_window": bool(label["is_healthy_beat"]),
                "matched_annotation": bool(label["matched_annotation"]),
                "matched_annotation_sample": label["matched_annotation_sample"],
                "matched_annotation_dt_s": label["matched_annotation_dt_s"],
                "has_labels": bool(label["matched_annotation"]),
                "lookahead_ms": float(lookahead_ms) if causal_window else 0.0,
                "window_mode": (
                    "causal_with_lookahead" if causal_window and lookahead_ms > 0
                    else "causal" if causal_window
                    else "centered_offline"
                ),
                "mode": "layer1_adaptive_gated",
                "trigger_source": "layer1_adaptive_gated",
                "layer1_filter": "zero_phase_offline_validation",
            }
        )
    return rows


def build_beat_table(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    datasets = parse_csv_list(args.datasets)
    normal_symbols = set(parse_csv_list(args.normal_symbols)) or set(DEFAULT_NORMAL_SYMBOLS)
    allowlist: set[Tuple[str, str]] | None = None
    if getattr(args, "records_csv", None):
        allowlist = load_record_allowlist(args.records_csv)
        print(
            f"[INFO] record allowlist: {len(allowlist)} (dataset, record) pairs from {args.records_csv}",
            file=sys.stderr,
        )
    all_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds_dir = data_dir / dataset
        if not ds_dir.exists():
            print(f"[WARN] missing dataset directory: {ds_dir}", file=sys.stderr)
            continue
        records = find_wfdb_records(ds_dir)
        if allowlist is not None:
            allowed_recs = {rec for ds, rec in allowlist if ds == dataset}
            # Also allow basename match if WFDB path is nested.
            before = len(records)
            records = [
                r for r in records
                if r in allowed_recs or Path(r).name in allowed_recs
            ]
            print(
                f"[INFO] {dataset}: allowlist kept {len(records)}/{before} records",
                file=sys.stderr,
            )
        if args.max_records is not None:
            records = records[: int(args.max_records)]
        print(f"[INFO] {dataset}: found {len(records)} WFDB records", file=sys.stderr)
        for record in records:
            if args.mode == "oracle":
                rows = read_record_beats(
                    data_dir=data_dir,
                    dataset=dataset,
                    record=record,
                    ann_ext=args.ann_ext,
                    window_s=args.window_s,
                    normal_symbols=normal_symbols,
                    causal_window=args.causal_window,
                    lookahead_ms=args.lookahead_ms,
                )
            elif args.mode == "layer1_adaptive_gated":
                rows = read_record_layer1_adaptive_triggers(
                    data_dir=data_dir,
                    dataset=dataset,
                    record=record,
                    ann_ext=args.ann_ext,
                    window_s=args.window_s,
                    normal_symbols=normal_symbols,
                    causal_window=args.causal_window,
                    lookahead_ms=args.lookahead_ms,
                    lead_index=args.lead_index,
                    annotation_match_tolerance_s=args.annotation_match_tolerance_s,
                )
            else:
                raise ValueError(f"Unsupported beat-sync mode: {args.mode}")
            all_rows.extend(rows)
            unit = "beat" if args.mode == "oracle" else "Layer 1 trigger"
            print(f"[INFO] {dataset}/{record}: {len(rows)} {unit} windows", file=sys.stderr)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.insert(0, "beat_id", np.arange(len(df), dtype=int))
    if args.max_beats is not None and not df.empty:
        df = df.head(int(args.max_beats)).copy()
    return df


def load_beat_windows(df: pd.DataFrame, data_dir: str | Path, lead_index: int, target_fs: float | None) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    for row in df.itertuples(index=False):
        record_path = resolve_record_path(data_dir, row.dataset, row.record, getattr(row, "record_path", None))
        windows.append(
            read_wfdb_window(
                record_path=record_path,
                start_sample=int(row.start_sample),
                end_sample=int(row.end_sample),
                lead_index=lead_index,
                target_fs=target_fs,
            )
        )
    return windows


def run_per_record(df: pd.DataFrame, embeddings: np.ndarray, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored_groups = []
    thresholds = []
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
            min_fit=args.min_fit_beats,
            min_val=args.min_val_beats,
            calibration_fit_frac=args.calibration_fit_frac,
            calibration_val_frac=args.calibration_val_frac,
            guard_s=args.guard_s,
            l2_normalize=args.l2_normalize_embeddings,
            pca_dim=args.pca_dim,
            pca_whiten=args.pca_whiten,
            threshold_method=args.threshold_method,
            conformal_alpha=args.conformal_alpha,
            covariance_estimator=args.covariance_estimator,
        )
        scored_groups.append(scored)
        thresholds.append(meta)
    out = pd.concat(scored_groups, axis=0).sort_index()
    out["is_healthy_beat"] = out["is_healthy_window"].astype(bool)
    return out, pd.DataFrame(thresholds)

def write_beat_metrics(scored: pd.DataFrame, out_dir: Path) -> None:
    eval_df = scored[(scored["split"] == "test") & scored["decision"].isin(["permit", "inhibit"])].copy()
    eval_df["is_healthy"] = eval_df["is_healthy_beat"].astype(bool)

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
    for label, gg in eval_df.groupby("beat_symbol"):
        row = {"beat_symbol": str(label)}
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
        "beat_id", "dataset", "record", "record_key",
        "beat_index", "trigger_index", "beat_sample", "trigger_sample",
        "beat_time_s", "trigger_time_s", "beat_symbol",
        "safety_group", "safety_expectation",
        "matched_annotation", "matched_annotation_sample", "matched_annotation_dt_s",
        "layer3_anomaly_model", "layer3_score", "anomaly_score",
        "layer3_mahal_score", "layer3_knn_score",
        "layer3_threshold", "score_over_threshold_ratio",
        "window_mode", "mode", "trigger_source", "split", "decision",
    ]
    policy_eval = add_policy_columns(eval_df)
    false_permits = policy_eval[
        (policy_eval["safety_expectation_policy"] == INHIBIT_EXPECTED)
        & (policy_eval["decision"] == "permit")
    ].copy()
    false_permits["reason"] = "score_below_threshold_on_inhibit_expected_beat"
    keep = [c for c in fp_cols if c in false_permits.columns] + ["reason"]
    false_permits[keep].to_csv(out_dir / "false_permits_detail.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Beat-synchronous Layer 3 learned ECG embedding anomaly-veto validation.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument(
        "--records-csv",
        default=None,
        help="Optional CSV with columns dataset,record — only these records are scored "
             "(e.g. Results/layer3/transition_analysis/pilot_primary_mitbih_gold.csv).",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", default="oracle", choices=["oracle", "layer1_adaptive_gated"],
                   help="oracle = annotated-beat offline upper bound. layer1_adaptive_gated = one decision per accepted adaptive Layer 1 trigger, with oracle labels used only for offline metrics.")
    p.add_argument("--per-record-calibration", action="store_true", help="Accepted for CLI symmetry; beat-sync validation is per-record by default.")
    p.add_argument("--ann-ext", default="atr")
    p.add_argument("--normal-symbols", default="N",
                   help="Legacy beat-symbol rule retained for compatibility. Safety calibration uses safety_group==NORMAL from label_grouping.py.")
    p.add_argument("--dangerous-symbols", default="V,F,E,/,f,!",
                   help="Fallback Phase 1 DANGEROUS beat-symbol list when safety_group is unavailable; does not change clinical policy.")
    p.add_argument("--window-s", type=float, default=8.0,
                   help="Primary rhythm-context window length in seconds (default: 8.0).")
    p.add_argument("--causal-window", action="store_true", help="Use only samples up to the beat; otherwise centered offline window")
    p.add_argument("--lookahead-ms", type=float, default=0.0,
                   help="When --causal-window is used, include this many milliseconds after the beat/trigger. Offline therapy-latency simulation, not zero-latency causal.")
    p.add_argument("--annotation-match-tolerance-s", type=float, default=0.080,
                   help="Offline-only tolerance for labeling Layer 1 triggers against annotations. Unmatched triggers are treated as non-healthy for metrics.")
    p.add_argument("--lead-index", type=int, default=0)
    p.add_argument("--target-fs", type=float, default=125.0,
                   help="Resample windows to this Hz (default: 125 Hz for 8 s rhythm windows). Set <=0 to keep native fs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0. auto uses CUDA if available.")
    p.add_argument("--encoder-pooling-mode",
                   choices=["auto", "global_avg", "avg_max", "causal_local_global"],
                   default="auto",
                   help="Infer from checkpoint by default; explicit values must match. Legacy checkpoints resolve to global_avg.")
    p.add_argument("--encoder-local-fraction", type=float, default=0.125,
                   help="Tail fraction used by causal_local_global; 0.125 corresponds to 1 s of an 8 s window.")
    p.add_argument("--threshold-method", default="conformal", choices=["conformal", "healthy_quantile"],
                   help="Main-decision threshold on healthy calibration scores. conformal (default) gives a "
                        "stated healthy false-inhibit budget (--conformal-alpha); healthy_quantile is the "
                        "legacy quantile method (--threshold-quantile). Phase 1 always reports both methods.")
    p.add_argument("--threshold-quantile", type=float, default=0.99,
                   help="Healthy calibration quantile for --threshold-method healthy_quantile.")
    p.add_argument("--anomaly-model", default="mahalanobis", choices=["mahalanobis", "knn"],
                   help="Healthy-baseline embedding anomaly scorer.")
    p.add_argument("--shrinkage", type=float, default=0.10)
    p.add_argument("--covariance-estimator",
                   choices=["ledoit_wolf", "oas", "diagonal", "diagonal_shrinkage"],
                   default="ledoit_wolf",
                   help="Mahalanobis covariance estimator. ledoit_wolf is the locked default.")
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
    p.add_argument("--min-fit-beats", type=int, default=20)
    p.add_argument("--min-val-beats", type=int, default=10)
    p.add_argument("--calibration-fit-frac", type=float, default=0.60)
    p.add_argument("--calibration-val-frac", type=float, default=0.20)
    p.add_argument("--guard-s", type=float, default=None,
                   help="Temporal buffer between last calibration beat and first test beat (in seconds). Default: --window-s, so beat windows cannot overlap calibration beats.")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--max-beats", type=int, default=None)
    p.add_argument("--no-random-fallback", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--phase1-eval", action="store_true",
                   help="Write additive Phase 1 evaluation outputs: A0/Layer3 arms, Mahalanobis/kNN, quantile/conformal metrics.")
    p.add_argument("--phase1-arms", default="layer3,a0",
                   help="Comma-separated Phase 1 arms. Supported: layer3, a0.")
    p.add_argument("--phase1-scorers", default="mahalanobis,knn",
                   help="Comma-separated Phase 1 scorers. Supported: mahalanobis, knn.")
    p.add_argument("--conformal-alpha", type=float, default=0.10,
                   help="Per-record conformal false-inhibit alpha on healthy calibration scores.")
    p.add_argument("--offline-danger-fpr-target", type=float, default=0.02,
                   help="Offline-only labeled operating point target for DANGEROUS false permits. Not deployable calibration.")
    p.add_argument("--phase1-bootstrap-n", type=int, default=2000,
                   help="Record-cluster bootstrap resamples for the headline false-permit CI (Phase 1).")
    p.add_argument("--phase1-bootstrap-seed", type=int, default=12345,
                   help="Seed for the Phase 1 record-cluster bootstrap.")
    p.add_argument("--a0-feature-window-s", type=float, default=5.0,
                   help="Handcrafted A0 Layer-2 morphology feature window in seconds.")
    p.add_argument("--a0-rr-lookback-s", type=float, default=30.0,
                   help="Handcrafted A0 RR lookback in seconds.")
    p.add_argument("--a0-feature-window-mode", default="auto", choices=["auto", "causal", "centered"],
                   help="A0 feature window mode. auto follows --causal-window.")
    p.add_argument("--a0-post-r-lookahead-ms", type=float, default=80.0,
                   help="Post-R lookahead for causal A0 feature extraction in milliseconds.")
    p.add_argument("--a0-min-finite-frac", type=float, default=0.95,
                   help="Minimum finite fraction in healthy fit beats for keeping an A0 feature.")
    args = p.parse_args()
    if args.guard_s is None:
        args.guard_s = float(args.window_s)

    set_seed(args.seed, deterministic=args.deterministic)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_fs = None if args.target_fs is not None and args.target_fs <= 0 else args.target_fs

    df = build_beat_table(args)
    if df.empty:
        raise RuntimeError("No annotated beats found. Check data path, dataset name, and annotation extension.")
    df = df.reset_index(drop=True)
    LOG.info("Built beat table: %d beats across %d records.", len(df), df["record_key"].nunique())

    model, encoder_info = build_encoder(
        checkpoint=args.checkpoint,
        device=args.device,
        allow_random_fallback=not args.no_random_fallback,
        pooling_mode=args.encoder_pooling_mode,
        local_fraction=args.encoder_local_fraction,
    )
    resolved_device = str(encoder_info.get("device", args.device))
    encoder_info["seed"] = int(args.seed)
    checkpoint_window_s = encoder_info.get("checkpoint_window_s")
    if checkpoint_window_s is not None and not np.isclose(
        float(checkpoint_window_s), float(args.window_s), rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            f"Checkpoint was pretrained on {checkpoint_window_s}s windows but evaluation requested "
            f"{args.window_s}s. Use a matching checkpoint/window or explicitly create a documented "
            "cross-window ablation instead of silently shifting distributions."
        )
    write_json(out_dir / "encoder_info.json", encoder_info)
    if encoder_info.get("source", "").startswith("RandomConvEncoder"):
        LOG.warning("Beat-sync validation is using a random fallback encoder; results are smoke-test only.")

    t0 = time.perf_counter()
    windows = load_beat_windows(df, data_dir=args.data_dir, lead_index=args.lead_index, target_fs=target_fs)
    load_s = time.perf_counter() - t0
    LOG.info("Loaded %d beat windows in %.1fs", len(windows), load_s)

    t1 = time.perf_counter()
    embeddings = encode_windows(model, windows, batch_size=args.batch_size, device=resolved_device)
    encode_s = time.perf_counter() - t1
    LOG.info("Encoded %d beat windows in %.1fs", len(windows), encode_s)
    np.save(out_dir / "embeddings.npy", embeddings)

    scored, thresholds = run_per_record(df, embeddings, args)
    scored.to_csv(out_dir / "per_beat.csv", index=False)
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
        healthy_col="is_healthy_beat",
    )

    score_cols = [
        "beat_id", "dataset", "record", "record_key",
        "beat_index", "trigger_index", "beat_sample", "trigger_sample",
        "beat_time_s", "trigger_time_s", "beat_symbol",
        "safety_group", "safety_expectation",
        "matched_annotation", "matched_annotation_sample", "matched_annotation_dt_s",
        "split", "decision",
        "layer3_anomaly_model", "layer3_score", "anomaly_score",
        "layer3_mahal_score", "layer3_knn_score",
        "layer3_threshold", "threshold",
        "score_over_threshold_ratio",
        "is_healthy_beat", "window_s", "window_mode", "mode", "trigger_source",
    ]
    scored[[c for c in score_cols if c in scored.columns]].to_csv(out_dir / "embedding_scores.csv", index=False)
    write_beat_metrics(scored, out_dir)

    phase1_files: List[str] = []
    if args.phase1_eval:
        phase1_scored, phase1_thresholds = run_phase1_eval(df, embeddings, args)
        write_phase1_outputs(
            phase1_scored,
            phase1_thresholds,
            out_dir,
            offline_danger_fpr_target=args.offline_danger_fpr_target,
            bootstrap_n=args.phase1_bootstrap_n,
            bootstrap_seed=args.phase1_bootstrap_seed,
        )
        phase1_files = [
            "phase1_per_beat.csv",
            "phase1_thresholds.csv",
            "phase1_offline_operating_points.csv",
            "phase1_metrics_by_dataset.csv",
            "phase1_metrics_overall.csv",
            "phase1_aurocs.csv",
            "phase1_metrics_bootstrap.csv",
            "phase1_metrics_by_record.csv",
            "phase1_metrics_by_danger_subtype.csv",
            "phase1_cav_l2_l3.csv",
        ]

    runtime = {
        "n_beats": int(len(scored)),
        "embedding_shape": list(embeddings.shape),
        "load_seconds": float(load_s),
        "encode_seconds": float(encode_s),
        "seconds_per_beat_encode_only": float(encode_s / max(1, len(scored))),
        "window_mode": (
            "causal_with_lookahead" if args.causal_window and args.lookahead_ms > 0
            else "causal" if args.causal_window
            else "centered_offline_noncausal"
        ),
        "mode": args.mode,
        "lookahead_ms": float(args.lookahead_ms),
        "annotation_match_tolerance_s": float(args.annotation_match_tolerance_s),
        "anomaly_model": str(args.anomaly_model),
        "knn_k": int(args.knn_k),
        "calibration_outlier_frac": float(args.calibration_outlier_frac),
        "l2_normalize_embeddings": bool(args.l2_normalize_embeddings),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "guard_s": float(args.guard_s),
        "target_fs": target_fs,
        "safety_grouping": "Layer 3 danger-grouped labels from rhythm spans + beat symbols; AF_CONTEXT default is inhibit.",
        "lead_index": args.lead_index,
        "seed": int(args.seed),
        "phase1_eval": bool(args.phase1_eval),
        "phase1_arms": str(args.phase1_arms),
        "phase1_scorers": str(args.phase1_scorers),
        "conformal_alpha": float(args.conformal_alpha),
        "offline_danger_fpr_target": float(args.offline_danger_fpr_target),
        "phase1_files": phase1_files,
        "safety_framing": "One Layer 3 veto decision per beat; calibration windows never stimulate; Layer 3 must not command stimulation alone.",
    }
    write_json(out_dir / "runtime_summary.json", runtime)

    with (out_dir / "FINAL_LAYER3_SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("# Layer 3 beat-synchronous validation summary\n\n")
        f.write("This evaluates one permit/inhibit anomaly-veto decision per annotated beat/trigger. It is not a clinical arrhythmia classifier.\n\n")
        f.write(SAFETY_DISCLAIMER)
        if args.mode == "layer1_adaptive_gated":
            f.write("\nLayer 1 adaptive gated mode scores accepted Layer 1 triggers. Oracle annotations are used only after detection for offline metric labels; unmatched triggers are treated as non-healthy for safety-gate reporting. The Layer 1 filter used here is the existing zero-phase offline validation filter, so this is still not an embedded deployment trace.\n")
        if args.causal_window and args.lookahead_ms > 0:
            f.write(f"\n> NOTE: `--lookahead-ms {args.lookahead_ms}` includes post-trigger samples. "
                    "This is appropriate only as an offline stimulation-latency simulation when therapy would occur after the R peak.\n")
        if not args.causal_window:
            f.write("\n> WARNING: Centered windows are offline/non-causal because they include future samples around the beat. "
                    "Use `--causal-window` for stricter real-time simulation.\n")
        f.write("\n## Configuration\n\n")
        f.write(f"- Beats scored: {len(scored)}\n")
        f.write(f"- Embedding shape: {list(embeddings.shape)}\n")
        f.write(f"- Anomaly model: `{args.anomaly_model}`\n")
        if args.anomaly_model == "knn":
            f.write(f"- kNN neighbors: `{args.knn_k}`\n")
        f.write(f"- L2-normalize embeddings: `{args.l2_normalize_embeddings}`\n")
        f.write(f"- PCA dimension before baseline: `{args.pca_dim}` (`<=0` disables PCA)\n")
        f.write(f"- PCA whitening: `{args.pca_whiten}`\n")
        f.write(f"- Calibration outlier pruning: `{args.calibration_outlier_frac}`\n")
        f.write(f"- Encoder source: {encoder_info.get('source')}\n")
        f.write(f"- Checkpoint loaded: {encoder_info.get('checkpoint_loaded')}\n")
        f.write(f"- Window mode: `{'causal_with_lookahead' if args.causal_window and args.lookahead_ms > 0 else 'causal' if args.causal_window else 'centered_offline_noncausal'}`\n")
        f.write(f"- Lookahead: `{args.lookahead_ms}` ms\n")
        f.write(f"- Trigger/beat mode: `{args.mode}`\n")
        f.write(f"- Guard between calibration and test beats: `{args.guard_s}` s\n")
        f.write(f"- Annotation match tolerance: `{args.annotation_match_tolerance_s}` s\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write("\nNote: `anomaly_score` is the model-agnostic embedding distance to the healthy baseline. "
                "`layer3_mahal_score` or `layer3_knn_score` is populated according to the selected anomaly model.\n")
        if args.phase1_eval:
            f.write("\n## Phase 1 Evaluation Outputs\n\n")
            f.write("- Arms: `" + str(args.phase1_arms) + "`\n")
            f.write("- Scorers: `" + str(args.phase1_scorers) + "`\n")
            f.write(f"- Conformal alpha: `{args.conformal_alpha}`\n")
            f.write("- Conformal guarantee is only for false-inhibit / false alarm on healthy data under exchangeability; it does not guarantee false-permit on dangerous beats.\n")
            f.write(f"- Offline labeled DANGEROUS operating point target: `{args.offline_danger_fpr_target}` false permits. This is non-deployable and uses DANGEROUS test labels.\n")
            f.write("- Headline false-permit CI is `phase1_metrics_bootstrap.csv` (record-cluster bootstrap); Wilson beat CIs are transparency only.\n")
            f.write("- `phase1_cav_l2_l3.csv` reports A0 vs Layer 3 conditional added value + healthy score correlation (L2 vs L3 redundancy).\n")
            f.write("- `phase1_metrics_by_danger_subtype.csv` splits false-permit into rhythm / morphology / noise origin.\n")
            f.write("- Files: `phase1_per_beat.csv`, `phase1_thresholds.csv`, `phase1_offline_operating_points.csv`, `phase1_metrics_by_dataset.csv`, `phase1_metrics_overall.csv`, `phase1_aurocs.csv`, `phase1_metrics_bootstrap.csv`, `phase1_metrics_by_record.csv`, `phase1_metrics_by_danger_subtype.csv`, `phase1_cav_l2_l3.csv`\n")

    LOG.info("Wrote Layer 3 beat-sync validation outputs to %s", out_dir)


if __name__ == "__main__":
    main()
