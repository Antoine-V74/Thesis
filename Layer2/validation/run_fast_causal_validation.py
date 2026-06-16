"""
Layer 2-only validation using a standalone fast causal R-peak detector.

This script intentionally does not import Layer 1. It reports two separate
questions:

1. How well did the fast causal detector find annotated R-peaks?
2. On correctly matched detected beats, how often did Layer 2 inhibit
   arrhythmia/noise beats and falsely inhibit healthy beats?

Smoke example:
    .venv\\Scripts\\python.exe Layer2\\validation\\run_fast_causal_validation.py `
        --data-dir data --datasets mitdb --record-limit 2 --max-beats-per-record 500 `
        --out-dir Results\\layer2\\fast_causal_layer2_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    sys.exit("Missing dependency: pip install wfdb")

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_L2))
sys.path.insert(0, str(_DATA))

from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()

from decision import BaselineCalibrator, FROZEN_COUPLING_THRESHOLD  # noqa: E402
from full_features import full_features  # noqa: E402
from r_peak_detector import causal_filter_ecg, detect_r_peaks  # noqa: E402
from dataset_registry import dataset_dir, resolve_dataset  # noqa: E402


FeatureDict = Dict[str, float]


def beat_label(symbol: str, dataset: str) -> str:
    """Map annotation symbols to Layer 2 safety labels."""
    info = resolve_dataset(dataset)
    if info.group == "vt_vfib":
        return "abnormal_v"
    if info.folder == "noise_stress_test":
        return "abnormal_noise"
    if symbol in info.normal_beats:
        return "healthy"
    if symbol in info.abnormal_beats:
        return "abnormal_v"
    return "mixed"


def extract_beat_features(
    filt: np.ndarray,
    raw: np.ndarray,
    fs: float,
    beat_time_s: float,
    peaks_all_s: np.ndarray,
    morphology_window_s: float,
    rr_lookback_s: float,
    post_r_lookahead_s: float,
    species: str,
) -> Tuple[FeatureDict, int]:
    """Extract causal Layer 2 features around one detected beat."""
    start_s = max(0.0, beat_time_s - morphology_window_s)
    end_s = min(len(filt) / fs, beat_time_s + post_r_lookahead_s)
    start = int(start_s * fs)
    end = max(start + 1, int(end_s * fs))
    window = filt[start:end]

    rr_lb = max(0.0, beat_time_s - rr_lookback_s)
    rr_peaks = peaks_all_s[(peaks_all_s >= rr_lb) & (peaks_all_s <= beat_time_s)]
    rr_rel = rr_peaks - start_s

    visible_peaks = peaks_all_s[(peaks_all_s >= start_s) & (peaks_all_s <= beat_time_s)]
    n_beats = int(len(visible_peaks))
    focus_peak_s = beat_time_s - start_s

    feats, _groups = full_features(
        window=window,
        r_peaks_s=rr_rel,
        fs=fs,
        species=species,
        compute_spectral_hrv=False,
        compute_entropy=False,
        focus_peak_s=focus_peak_s,
    )

    raw_window = raw[start:end]
    if len(raw_window) == len(window) and len(raw_window) > 3:
        # Preserve raw high-frequency noise hard rule used by the frozen gate.
        raw_feats, _ = full_features(
            window=raw_window,
            r_peaks_s=rr_rel,
            fs=fs,
            species=species,
            compute_spectral_hrv=False,
            compute_entropy=False,
            focus_peak_s=focus_peak_s,
        )
        if "signal__raw_hf_noise_ratio" in raw_feats:
            feats["signal__raw_hf_noise_ratio"] = raw_feats["signal__raw_hf_noise_ratio"]

    prev_peaks = peaks_all_s[peaks_all_s < (beat_time_s - 0.003)]
    if len(prev_peaks) >= 4:
        current_rr_s = float(beat_time_s - prev_peaks[-1])
        recent_rrs = np.diff(prev_peaks[-min(20, len(prev_peaks)):])
        valid_rrs = recent_rrs[(recent_rrs > 0.20) & (recent_rrs < 3.0)]
        if len(valid_rrs) >= 3 and current_rr_s > 0.20:
            median_rr = float(np.median(valid_rrs))
            if median_rr > 0.20:
                feats["rr__beat_coupling_ratio"] = current_rr_s / median_rr

    return feats, n_beats


def select_finite_features(
    features_list: List[FeatureDict],
    min_finite_frac: float = 0.95,
) -> Tuple[List[str], List[FeatureDict]]:
    """Choose stable feature columns and median-impute occasional NaNs."""
    all_keys = list(features_list[0].keys())
    finite_frac = {
        k: float(np.mean([np.isfinite(f.get(k, float("nan"))) for f in features_list]))
        for k in all_keys
    }
    names = [k for k, frac in finite_frac.items() if frac >= min_finite_frac]
    for key in names:
        vals = [f[key] for f in features_list if np.isfinite(f.get(key, float("nan")))]
        med = float(np.median(vals)) if vals else 0.0
        for f in features_list:
            if not np.isfinite(f.get(key, float("nan"))):
                f[key] = med
    return names, features_list


def fit_calibrator(
    features_list: List[FeatureDict],
    feature_names: List[str],
    threshold_quantile: float,
    feature_set: str,
) -> BaselineCalibrator:
    cal = BaselineCalibrator()
    return cal.fit(
        baseline_features=features_list,
        feature_names=feature_names,
        threshold_quantile=threshold_quantile,
        feature_set=feature_set,
    )


def align_features(feats: FeatureDict, cal: BaselineCalibrator) -> FeatureDict:
    aligned = {k: feats.get(k, float("nan")) for k in cal.feature_names}
    for k in cal.hard_rules:
        aligned[k] = feats.get(k, float("nan"))
    return aligned


def score_layer2(features: FeatureDict, cal: BaselineCalibrator) -> Dict[str, object]:
    if not all(np.isfinite(features.get(k, float("nan"))) for k in cal.mahal_feature_names):
        return {
            "permit": False,
            "reason": "missing_features",
            "mahalanobis": float("nan"),
            "mahalanobis_threshold": cal.threshold_mahalanobis,
            "max_zscore": float("nan"),
            "zscore_threshold": cal.threshold_max_zscore,
            "hard_rule_violated": "",
        }
    d = cal.decide(features)
    return {
        "permit": bool(d["permit"]),
        "reason": str(d["reason"]),
        "mahalanobis": float(d["mahalanobis"]),
        "mahalanobis_threshold": float(d["mahalanobis_threshold"]),
        "max_zscore": float(d["max_abs_zscore"]),
        "zscore_threshold": float(d["zscore_threshold"]),
        "hard_rule_violated": str(d.get("hard_rule_violated") or ""),
    }


def greedy_match_peaks(
    ref_s: np.ndarray,
    det_s: np.ndarray,
    tol_s: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """One-to-one greedy match between annotations and detected peaks."""
    pairs: List[Tuple[float, int, int]] = []
    for i, r in enumerate(ref_s):
        lo = r - tol_s
        hi = r + tol_s
        cand = np.where((det_s >= lo) & (det_s <= hi))[0]
        for j in cand:
            pairs.append((abs(float(det_s[j] - r)), i, int(j)))
    pairs.sort(key=lambda x: x[0])

    used_ref = set()
    used_det = set()
    matches: List[Tuple[int, int, float]] = []
    for dt, i, j in pairs:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        matches.append((i, j, float(det_s[j] - ref_s[i])))

    missed = [i for i in range(len(ref_s)) if i not in used_ref]
    extras = [j for j in range(len(det_s)) if j not in used_det]
    return matches, missed, extras


def summarize_detector(
    rows: List[Dict[str, object]],
    extra_rows: List[Dict[str, object]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = []
    for (dataset, record, label), grp in df.groupby(["dataset", "record", "label"]):
        n_true = len(grp)
        n_matched = int(grp["detected"].sum())
        timing = grp.loc[grp["detected"], "timing_error_ms"].dropna().abs()
        extras = 0
        if extra_rows:
            extras = sum(
                1 for r in extra_rows
                if r["dataset"] == dataset and r["record"] == record
            )
        out.append({
            "dataset": dataset,
            "record": record,
            "label": label,
            "n_true_beats": n_true,
            "n_matched_beats": n_matched,
            "n_missed_beats": n_true - n_matched,
            "sensitivity": round(n_matched / n_true, 4) if n_true else float("nan"),
            "n_extra_peaks_record": extras,
            "median_abs_timing_error_ms": round(float(timing.median()), 2) if len(timing) else float("nan"),
        })
    return pd.DataFrame(out)


def summarize_layer2(per_beat: pd.DataFrame) -> pd.DataFrame:
    if per_beat.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, feature_set, label), grp in per_beat.groupby(["dataset", "feature_set", "label"]):
        n = len(grp)
        n_permit = int(grp["permit"].sum())
        n_inhibit = n - n_permit
        rows.append({
            "dataset": dataset,
            "feature_set": feature_set,
            "label": label,
            "n_detected_matched_beats": n,
            "n_permit": n_permit,
            "n_inhibit": n_inhibit,
            "permit_rate": round(n_permit / n, 4) if n else float("nan"),
            "inhibit_rate": round(n_inhibit / n, 4) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def summarize_end_to_end(
    detector_rows: List[Dict[str, object]],
    per_beat: pd.DataFrame,
) -> pd.DataFrame:
    det = pd.DataFrame(detector_rows)
    if det.empty or per_beat.empty:
        return pd.DataFrame()
    rows = []
    for dataset, grp in det.groupby("dataset"):
        for label in ("healthy", "abnormal_v", "abnormal_noise"):
            g = grp[grp["label"] == label]
            if g.empty:
                continue
            total = len(g)
            matched = int(g["detected"].sum())
            for fset, fgrp in per_beat[(per_beat["dataset"] == dataset) & (per_beat["label"] == label)].groupby("feature_set"):
                permit = int(fgrp["permit"].sum())
                inhibit = int((~fgrp["permit"]).sum())
                rows.append({
                    "dataset": dataset,
                    "feature_set": fset,
                    "label": label,
                    "n_true_beats": total,
                    "n_detected_by_fast_causal": matched,
                    "n_missed_by_fast_causal": total - matched,
                    "n_inhibited_by_layer2": inhibit,
                    "n_permitted_by_layer2": permit,
                    "detector_sensitivity": round(matched / total, 4) if total else float("nan"),
                    "layer2_inhibit_on_detected_rate": round(inhibit / matched, 4) if matched else float("nan"),
                    "end_to_end_permit_fraction_of_true": round(permit / total, 4) if total else float("nan"),
                })
    return pd.DataFrame(rows)


def validate_record(
    stem: Path,
    dataset: str,
    out_rows: List[Dict[str, object]],
    detector_rows: List[Dict[str, object]],
    extra_rows: List[Dict[str, object]],
    *,
    feature_sets: Iterable[str],
    cal_frac: float,
    threshold_quantile: float,
    morphology_window_s: float,
    rr_lookback_s: float,
    post_r_lookahead_s: float,
    match_tol_s: float,
    max_beats_per_record: Optional[int],
    species: str,
) -> None:
    info = resolve_dataset(dataset)
    rec = wfdb.rdrecord(str(stem))
    ann = wfdb.rdann(str(stem), info.ann_ext)
    ch = min(info.channel, rec.p_signal.shape[1] - 1)
    raw = rec.p_signal[:, ch].astype(float)
    fs = float(rec.fs)
    filt = causal_filter_ecg(raw, fs)

    det = detect_r_peaks(raw, fs)
    det_s = det.peak_samples.astype(float) / fs

    ref = []
    for sample, sym in zip(ann.sample, ann.symbol):
        lbl = beat_label(sym, dataset)
        if lbl == "mixed":
            continue
        ref.append((sample / fs, sym, lbl))
        if max_beats_per_record and len(ref) >= max_beats_per_record:
            break
    if not ref:
        return

    ref_s = np.array([r[0] for r in ref], dtype=float)
    matches, missed, extras = greedy_match_peaks(ref_s, det_s, match_tol_s)
    rec_name = stem.name

    match_by_ref = {i: (j, err_s) for i, j, err_s in matches}
    for i, (t_ref, sym, lbl) in enumerate(ref):
        matched = i in match_by_ref
        j, err_s = match_by_ref[i] if matched else (-1, float("nan"))
        detector_rows.append({
            "dataset": dataset,
            "record": rec_name,
            "beat_time_s": round(float(t_ref), 3),
            "beat_symbol": sym,
            "label": lbl,
            "detected": bool(matched),
            "detected_time_s": round(float(det_s[j]), 3) if matched else float("nan"),
            "timing_error_ms": round(float(err_s * 1000.0), 3) if matched else float("nan"),
        })

    matched_det = {j for _, j, _ in matches}
    for j in extras:
        if max_beats_per_record and det_s[j] > ref_s[-1] + match_tol_s:
            continue
        extra_rows.append({
            "dataset": dataset,
            "record": rec_name,
            "detected_time_s": round(float(det_s[j]), 3),
        })

    healthy_times = [t for t, _sym, lbl in ref if lbl == "healthy"]
    n_cal = max(5, int(len(healthy_times) * cal_frac))
    if len(healthy_times) < 5:
        logging.warning("  %s: <5 healthy beats, skip Layer 2", rec_name)
        return
    cal_end_t = healthy_times[min(n_cal, len(healthy_times)) - 1]

    cal_feats = []
    for t_beat in healthy_times[:n_cal]:
        feats, _ = extract_beat_features(
            filt, raw, fs, t_beat, det_s,
            morphology_window_s, rr_lookback_s, post_r_lookahead_s, species,
        )
        cal_feats.append(feats)
    if len(cal_feats) < 5:
        return

    feature_names, cal_feats = select_finite_features(cal_feats)
    calibrators = {
        fset: fit_calibrator(cal_feats, feature_names, threshold_quantile, fset)
        for fset in feature_sets
    }

    for i_ref, j_det, err_s in matches:
        t_ref, sym, lbl = ref[i_ref]
        t_det = float(det_s[j_det])
        if lbl == "healthy" and t_ref <= cal_end_t:
            continue
        feats, n_beats = extract_beat_features(
            filt, raw, fs, t_det, det_s,
            morphology_window_s, rr_lookback_s, post_r_lookahead_s, species,
        )
        base = {
            "dataset": dataset,
            "record": rec_name,
            "beat_time_s": round(float(t_ref), 3),
            "detected_time_s": round(t_det, 3),
            "timing_error_ms": round(float(err_s * 1000.0), 3),
            "beat_symbol": sym,
            "label": lbl,
            "n_beats_morph_window": n_beats,
            "eval_mode": "fast_causal_detected",
        }
        for fset, cal in calibrators.items():
            aligned = align_features(feats, cal)
            score = score_layer2(aligned, cal)
            out_rows.append({**base, "feature_set": fset, **score})


def run_validation(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "fast_causal_validation.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    per_beat_rows: List[Dict[str, object]] = []
    detector_rows: List[Dict[str, object]] = []
    extra_rows: List[Dict[str, object]] = []

    datasets = [resolve_dataset(d).folder for d in args.datasets]
    for dataset in datasets:
        ds_dir = dataset_dir(args.data_dir, dataset)
        records = sorted(ds_dir.glob("*.hea"))
        if args.record_limit:
            records = records[:args.record_limit]
        logging.info("[%s] %d records", dataset, len(records))
        for hea in records:
            t0 = time.time()
            stem = hea.with_suffix("")
            try:
                validate_record(
                    stem,
                    dataset,
                    per_beat_rows,
                    detector_rows,
                    extra_rows,
                    feature_sets=args.feature_sets,
                    cal_frac=args.cal_frac,
                    threshold_quantile=args.threshold_quantile,
                    morphology_window_s=args.morphology_window_s,
                    rr_lookback_s=args.rr_lookback_s,
                    post_r_lookahead_s=args.post_r_lookahead_s,
                    match_tol_s=args.match_tol_ms / 1000.0,
                    max_beats_per_record=args.max_beats_per_record,
                    species=args.species,
                )
                logging.info("  %s %.1fs", hea.stem, time.time() - t0)
            except Exception as exc:
                logging.exception("  %s failed: %s", hea.stem, exc)

    per_beat = pd.DataFrame(per_beat_rows)
    detector_detail = pd.DataFrame(detector_rows)
    extras = pd.DataFrame(extra_rows)
    detector_summary = summarize_detector(detector_rows, extra_rows)
    layer2_summary = summarize_layer2(per_beat)
    end_to_end = summarize_end_to_end(detector_rows, per_beat)

    per_beat.to_csv(args.out_dir / "per_beat.csv", index=False)
    detector_detail.to_csv(args.out_dir / "detector_detail.csv", index=False)
    extras.to_csv(args.out_dir / "detector_extra_peaks.csv", index=False)
    detector_summary.to_csv(args.out_dir / "detector_summary_by_record_label.csv", index=False)
    layer2_summary.to_csv(args.out_dir / "layer2_summary_by_dataset_label.csv", index=False)
    end_to_end.to_csv(args.out_dir / "end_to_end_summary.csv", index=False)

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    config["data_dir"] = str(args.data_dir)
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logging.info("Wrote %d Layer 2 beat rows", len(per_beat))
    logging.info("Done -> %s", args.out_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/fast_causal_layer2"))
    p.add_argument("--datasets", nargs="+", default=["mitdb"])
    p.add_argument("--feature-sets", nargs="+", default=["all"])
    p.add_argument("--record-limit", type=int, default=None)
    p.add_argument("--max-beats-per-record", type=int, default=None)
    p.add_argument("--cal-frac", type=float, default=0.6)
    p.add_argument("--threshold-quantile", type=float, default=0.999)
    p.add_argument("--morphology-window-s", type=float, default=5.0)
    p.add_argument("--rr-lookback-s", type=float, default=30.0)
    p.add_argument("--post-r-lookahead-s", type=float, default=0.08)
    p.add_argument("--match-tol-ms", type=float, default=80.0)
    p.add_argument("--species", default="human")
    args = p.parse_args(argv)
    run_validation(args)


if __name__ == "__main__":
    main()
