"""
Layer 2 beat-synchronous validation.

Scores one permit/inhibit decision per R-peak trigger (not per 5 s clock window).

For each normal annotated beat (or each Layer1 accepted peak):
  - morphology features from a deployment-causal ECG slice (default: causal mode)
  - RR features from peaks in [t_beat - rr_lookback_s, t_beat]
  - Layer 2 gate decision for THAT trigger

Two window modes
----------------
causal (default, deployment-realistic)
    Window = [beat - morphology_window_s,  beat + post_r_lookahead_s]
    Filter  = causal IIR bandpass (lfilter, no future samples)
    Template uses only R-peaks BEFORE beat_time_s + 3 ms

centered (offline oracle upper-bound)
    Window = [beat - half,  beat + half]
    Filter  = zero-phase filtfilt (non-causal, not for deployment)


Usage
-----
    cd "ECG Processing"
    .venv\\Scripts\\python Layer2\\validation\\run_beat_validation.py `
        --data-dir data --datasets mitdb `
        --out-dir Results/layer2_beat_sync `
        --per-record-calibration
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_L2))
sys.path.insert(0, str(_DATA))
from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()
_L1 = _ROOT / "Layer1"
_l1_bootstrap = importlib.util.spec_from_file_location(
    "layer1_bootstrap", _L1 / "_bootstrap.py",
)
if _l1_bootstrap is None or _l1_bootstrap.loader is None:
    raise ImportError("Cannot load Layer1 bootstrap")
layer1_bootstrap = importlib.util.module_from_spec(_l1_bootstrap)
_l1_bootstrap.loader.exec_module(layer1_bootstrap)
sys.path.insert(0, str(_L1))
setup_layer1_paths = layer1_bootstrap.setup_layer1_paths
setup_layer1_paths(include_archive=True)

from full_features import full_features  # noqa: E402

# Reuse shared validation utilities
from dataset_registry import DATASET_GROUPS, dataset_dir, resolve_dataset  # noqa: E402
from common import (  # noqa: E402
    ABNORMAL_BEATS,
    CALIBRATION_RECORDS,
    NORMAL_BEATS,
    apply_filters,
    layer1_r_peaks,
    score_one,
    score_one_hybrid,
    _fit_calibrator,
    _select_finite_features,
    align_features_for_scoring,
)

_ADAPTIVE_OK = False

try:
    from scipy.signal import butter, iirnotch, lfilter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Causal ECG filter (deployment-compatible — no future samples)
# ---------------------------------------------------------------------------

def _causal_bandpass(x: np.ndarray, fs: float,
                     low: float = 5.0, high: float = 20.0) -> np.ndarray:
    x = np.where(np.isfinite(np.asarray(x, dtype=float)), x, 0.0)
    nyq = 0.5 * fs
    lo = max(1e-4, low / nyq)
    hi = min(0.499, min(high, 0.45 * fs) / nyq)
    b, a = butter(2, [lo, hi], btype="band")
    return lfilter(b, a, x)


def _causal_notch(x: np.ndarray, fs: float,
                  f0: float, q: float = 30.0) -> np.ndarray:
    if f0 >= 0.5 * fs:
        return x
    b, a = iirnotch(f0 / (0.5 * fs), q)
    return lfilter(b, a, x)


def _causal_layer2_filter(raw: np.ndarray, fs: float) -> np.ndarray:
    """Causal ECG filter (lfilter only) — safe for real-time deployment."""
    if not _SCIPY_OK:
        return np.asarray(raw, dtype=float)
    filt = _causal_bandpass(raw, fs)
    if fs > 110:
        filt = _causal_notch(filt, fs, 50.0)
    if fs > 125:
        filt = _causal_notch(filt, fs, 60.0)
    return filt


def beat_label(symbol: str, dataset: str) -> str:
    info = resolve_dataset(dataset)
    if info.group == "vt_vfib":
        return "abnormal_v"
    if info.folder == "noise_stress_test":
        return "abnormal_noise"
    if symbol in NORMAL_BEATS:
        return "healthy"
    if symbol in ABNORMAL_BEATS:
        return "abnormal_v"
    return "mixed"


def extract_beat_features(
    filt: np.ndarray,
    fs: float,
    beat_time_s: float,
    peaks_all_s: np.ndarray,
    morphology_window_s: float = 5.0,
    rr_lookback_s: float = 30.0,
    species: str = "human",
    feature_window_mode: str = "causal",
    post_r_lookahead_s: float = 0.08,
) -> Tuple[Dict[str, float], int]:
    """
    Extract beat features using either causal or centered window.

    causal (deployment-realistic, default)
        Window  = [beat - morphology_window_s, beat + post_r_lookahead_s]
        Template peaks limited to beat_time_s (no future R-peaks used)

    centered (offline oracle upper-bound)
        Window  = [beat - half, beat + half]
        Template peaks include future R-peaks inside the window

    Returns (features, n_beats_in_morphology_window).
    """
    if feature_window_mode == "causal":
        start_s = max(0.0, beat_time_s - morphology_window_s)
        end_s = min(len(filt) / fs, beat_time_s + post_r_lookahead_s)
        visible_peak_cutoff_s = beat_time_s
    elif feature_window_mode == "centered":
        half = morphology_window_s / 2.0
        start_s = max(0.0, beat_time_s - half)
        end_s = min(len(filt) / fs, beat_time_s + half)
        visible_peak_cutoff_s = end_s
    else:
        raise ValueError(f"feature_window_mode must be 'causal' or 'centered', got {feature_window_mode!r}")

    start = int(start_s * fs)
    end = max(start + 1, int(end_s * fs))
    w = filt[start:end]

    rr_lb = max(0.0, beat_time_s - rr_lookback_s)
    rr_peaks = peaks_all_s[(peaks_all_s >= rr_lb) & (peaks_all_s <= beat_time_s)]
    rr_rel = rr_peaks - start_s

    win_peaks = peaks_all_s[(peaks_all_s >= start_s) & (peaks_all_s <= visible_peak_cutoff_s)]
    n_beats_win = len(win_peaks)

    focus_peak_s = beat_time_s - start_s
    feats, _ = full_features(
        w,
        r_peaks_s=rr_rel,
        fs=fs,
        species=species,
        compute_spectral_hrv=False,
        compute_entropy=False,
        focus_peak_s=focus_peak_s,
    )

    # Per-beat RR coupling ratio: current_rr / median(recent_rrs).
    # Catches isolated PVCs (short coupling) which aggregate RR features miss
    # because 1 PVC in a 30 s window only shifts the mean by ~3-5%.
    prev_peaks = peaks_all_s[peaks_all_s < (beat_time_s - 0.003)]
    if len(prev_peaks) >= 4:
        current_rr_s = float(beat_time_s - prev_peaks[-1])
        recent_rrs = np.diff(prev_peaks[-min(20, len(prev_peaks)):])
        # filter physiologically implausible intervals (noise)
        valid_rrs = recent_rrs[(recent_rrs > 0.20) & (recent_rrs < 3.0)]
        if len(valid_rrs) >= 3 and current_rr_s > 0.20:
            median_rr = float(np.median(valid_rrs))
            if median_rr > 0.20:
                feats["rr__beat_coupling_ratio"] = current_rr_s / median_rr

    return feats, n_beats_win


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fset, mode, lbl), grp in df.groupby(["feature_set", "mode", "label"]):
        n = len(grp)
        if lbl == "healthy":
            tp = int(grp["permit"].sum())
            fn = n - tp
            rows.append({
                "feature_set": fset,
                "mode": mode,
                "label": lbl,
                "n_beats": n,
                "n_permit": tp,
                "n_inhibit": fn,
                "healthy_permit_rate": round(tp / n, 4) if n else float("nan"),
                "false_inhibit_rate": round(fn / n, 4) if n else float("nan"),
            })
        elif lbl in ("abnormal_v", "abnormal_noise"):
            tn = int((~grp["permit"]).sum())
            fp = int(grp["permit"].sum())
            rows.append({
                "feature_set": fset,
                "mode": mode,
                "label": lbl,
                "n_beats": n,
                "n_permit": fp,
                "n_inhibit": tn,
                "abnormal_inhibit_rate": round(tn / n, 4) if n else float("nan"),
                "false_permit_rate": round(fp / n, 4) if n else float("nan"),
            })
    return pd.DataFrame(rows)


def _consec_inhibit_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution of consecutive inhibited beat run lengths."""
    rows = []
    for (fset, mode), grp in df.groupby(["feature_set", "mode"]):
        g = grp.sort_values("beat_time_s")["permit"].values
        run = 0
        runs = []
        for p in g:
            if not p:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
        runs = np.array(runs, dtype=int)
        rows.append({
            "feature_set": fset,
            "mode": mode,
            "n_runs": len(runs),
            "max_consec_inhibit": int(runs.max()) if len(runs) else 0,
            "mean_consec_inhibit": round(float(runs.mean()), 2) if len(runs) else 0.0,
            "frac_runs_gt3": round(float((runs > 3).mean()), 4) if len(runs) else 0.0,
            "frac_runs_gt10": round(float((runs > 10).mean()), 4) if len(runs) else 0.0,
        })
    return pd.DataFrame(rows)


def _inhibit_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Count why beats were inhibited, by reason and class."""
    rows = []
    inhibited = df[~df["permit"]]
    for (fset, mode, reason, iclass), grp in inhibited.groupby(
        ["feature_set", "mode", "reason", "inhibit_class"], dropna=False
    ):
        rows.append({
            "feature_set": fset,
            "mode": mode,
            "reason": reason,
            "inhibit_class": iclass,
            "n_inhibited": len(grp),
        })
    return pd.DataFrame(rows)


def _top_inhibit_features(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify features most correlated with inhibition (healthy beats only)."""
    rows = []
    healthy_inh = df[(df["label"] == "healthy") & (~df["permit"])]
    healthy_all = df[df["label"] == "healthy"]
    feat_cols = [c for c in df.columns if c.startswith("f_")]  # if enriched later
    # Fall back: use columns that look like numeric feature values
    if not feat_cols:
        skip = {"dataset", "group", "record", "beat_time_s", "beat_symbol",
                "label", "n_beats_morph_window", "feature_set", "mode",
                "permit", "distance", "is_outlier"}
        feat_cols = [c for c in df.columns if c not in skip and df[c].dtype.kind == "f"]
    for (fset, mode), grp_inh in healthy_inh.groupby(["feature_set", "mode"]):
        grp_all = healthy_all[(healthy_all["feature_set"] == fset) &
                               (healthy_all["mode"] == mode)]
        correlations = {}
        for col in feat_cols:
            if col in grp_all.columns:
                vals = grp_all[col].dropna()
                inhibit_flag = (~grp_all.loc[vals.index, "permit"]).astype(float)
                if len(vals) > 5 and inhibit_flag.std() > 0:
                    correlations[col] = float(np.corrcoef(vals, inhibit_flag)[0, 1])
        top = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        for feat, corr in top:
            rows.append({"feature_set": fset, "mode": mode,
                         "feature": feat, "corr_with_inhibit": round(corr, 4)})
    return pd.DataFrame(rows)


def _classify_inhibit(reason: str) -> str:
    """
    Coarse class for false-inhibit analysis:
      - technical: missing data / unreliable trigger path
      - rr: rhythm-history or RR-gate driven inhibit
      - morphology: signal/morphology anomaly on current beat
      - hard_rule: explicit hard veto
      - threshold_other: threshold exceedance not clearly in rr/morphology
      - permit: non-inhibited beats
    """
    if not reason:
        return "threshold_other"
    if reason == "within_baseline" or reason == "all_safe":
        return "permit"
    if reason == "hard_rule":
        return "hard_rule"
    if "missing" in reason or "unreliable" in reason:
        return "technical"
    if reason.startswith("rr_") or "mahalanobis" in reason and "rr" in reason:
        return "rr"
    if reason in ("signal_not_safe", "max_zscore_exceeded"):
        return "morphology"
    if reason in ("mahalanobis_exceeded",):
        return "threshold_other"
    return "threshold_other"


def _per_record_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-record permit/inhibit rates and Layer 1 proxy sensitivity."""
    rows = []
    for (fset, mode, rec), grp in df.groupby(["feature_set", "mode", "record"]):
        h = grp[grp["label"] == "healthy"]
        ab = grp[grp["label"].isin(["abnormal_v", "abnormal_noise"])]
        row = {
            "feature_set": fset,
            "mode": mode,
            "record": rec,
            "dataset": grp["dataset"].iloc[0],
            "group": grp["group"].iloc[0],
            "n_healthy": len(h),
            "n_abnormal": len(ab),
            "healthy_permit_rate": round(h["permit"].mean(), 4) if len(h) else float("nan"),
            "abnormal_inhibit_rate": round((~ab["permit"]).mean(), 4) if len(ab) else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _per_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fset, mode, grp_name), grp in df.groupby(["feature_set", "mode", "group"]):
        h = grp[grp["label"] == "healthy"]
        ab = grp[grp["label"].isin(["abnormal_v", "abnormal_noise"])]
        rows.append({
            "feature_set": fset,
            "mode": mode,
            "group": grp_name,
            "n_healthy": len(h),
            "n_abnormal": len(ab),
            "healthy_permit_rate": round(h["permit"].mean(), 4) if len(h) else float("nan"),
            "abnormal_inhibit_rate": round((~ab["permit"]).mean(), 4) if len(ab) else float("nan"),
        })
    return pd.DataFrame(rows)


def run_beat_sync_validation(
    data_dir: Path,
    out_dir: Path,
    datasets: Optional[List[str]] = None,
    feature_sets: Optional[List[str]] = None,
    morphology_window_s: float = 5.0,
    rr_lookback_s: float = 30.0,
    per_record_calibration: bool = True,
    cal_frac: float = 0.6,
    threshold_quantile: float = 0.999,
    species: str = "human",
    include_adaptive: bool = True,
    window_limit: Optional[int] = None,
    abnormal_target_inhibit: Optional[float] = None,
    max_healthy_false_inhibit: float = 0.80,
    all_use_hybrid_gate: bool = False,
    feature_window_mode: str = "causal",
    post_r_lookahead_s: float = 0.08,
) -> None:
    import wfdb

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "beat_sync.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("beat_sync")

    if feature_sets is None:
        feature_sets = ["all", "signal_only", "hybrid_rewarming"]

    target = (
        {resolve_dataset(d).folder for d in datasets}
        if datasets else {"mit_bih_arrhythmia"}
    )
    rows: List[Dict] = []

    for dataset in sorted(target):
        if dataset not in DATASET_GROUPS:
            continue
        ds_dir = dataset_dir(data_dir, dataset)
        if not ds_dir.is_dir():
            continue
        group = DATASET_GROUPS[dataset]
        hea_files = sorted(ds_dir.glob("*.hea"))
        log.info(f"[{dataset}] {len(hea_files)} records")

        for hea in hea_files:
            t0 = time.time()
            stem = str(hea.with_suffix(""))
            try:
                rec = wfdb.rdrecord(stem)
                ann = wfdb.rdann(stem, "atr")
            except Exception as exc:
                log.warning(f"  skip {hea.stem}: {exc}")
                continue

            raw = rec.p_signal[:, 0].astype(float)
            fs = float(rec.fs)
            filt = (
                _causal_layer2_filter(raw, fs)
                if feature_window_mode == "causal"
                else apply_filters(raw, fs)
            )

            l1_peaks = layer1_r_peaks(filt, fs)
            ad_peaks = (
                adaptive_layer1_r_peaks(filt, fs, deployment_adaptive_config())
                if include_adaptive and _ADAPTIVE_OK else np.array([])
            )

            # Per-record calibrators from first healthy annotated beats (oracle peaks)
            oracle_peaks = np.array([
                s / fs for s, sym in zip(ann.sample, ann.symbol)
                if sym in (NORMAL_BEATS | ABNORMAL_BEATS)
            ], dtype=float)

            record_cals: Dict[str, Optional[object]] = {f: None for f in feature_sets}
            cal_end_t = 0.0
            if per_record_calibration:
                healthy_times = sorted(
                    s / fs for s, sym in zip(ann.sample, ann.symbol)
                    if beat_label(sym, dataset) == "healthy"
                )
                n_cal = max(5, int(len(healthy_times) * cal_frac))
                cal_feats: List[Dict] = []
                for t_beat in healthy_times[:n_cal]:
                    feats_o, _ = extract_beat_features(
                        filt, fs, t_beat, oracle_peaks,
                        morphology_window_s, rr_lookback_s, species,
                        feature_window_mode, post_r_lookahead_s,
                    )
                    cal_feats.append(feats_o)
                cal_end_t = 0.0
                if len(cal_feats) >= 5:
                    fn, cal_feats = _select_finite_features(cal_feats)
                    cal_end_t = healthy_times[min(n_cal, len(healthy_times)) - 1]
                    abn_cal_feats: List[Dict] = []
                    h_val_feats: List[Dict] = []
                    if abnormal_target_inhibit is not None:
                        for s, sym in zip(ann.sample, ann.symbol):
                            t_b = s / fs
                            if t_b <= cal_end_t:
                                continue
                            if beat_label(sym, dataset) != "abnormal_v":
                                continue
                            feats_a, _ = extract_beat_features(
                                filt, fs, t_b, oracle_peaks,
                                morphology_window_s, rr_lookback_s, species,
                                feature_window_mode, post_r_lookahead_s,
                            )
                            abn_cal_feats.append(feats_a)
                        for t_b in healthy_times[n_cal:]:
                            feats_h, _ = extract_beat_features(
                                filt, fs, t_b, oracle_peaks,
                                morphology_window_s, rr_lookback_s, species,
                                feature_window_mode, post_r_lookahead_s,
                            )
                            h_val_feats.append(feats_h)

                    for fset in feature_sets:
                        cal = _fit_calibrator(
                            cal_feats, fn, threshold_quantile, fset)
                        if (
                            abnormal_target_inhibit is not None
                            and fset in ("all", "hybrid_rewarming")
                            and len(abn_cal_feats) >= 5
                        ):
                            abn_aligned = []
                            for f in abn_cal_feats:
                                abn_aligned.append(
                                    {k: f.get(k, float("nan")) for k in cal.feature_names}
                                )
                            h_aligned = [
                                {k: f.get(k, float("nan")) for k in cal.feature_names}
                                for f in h_val_feats
                            ] if h_val_feats else None
                            cal.calibrate_thresholds_for_abnormal_inhibit(
                                abn_aligned,
                                target_inhibit_rate=abnormal_target_inhibit,
                                healthy_validation_features=h_aligned,
                                max_healthy_false_inhibit_rate=max_healthy_false_inhibit,
                            )
                        record_cals[fset] = cal

            def _score_trigger(
                beat_time_s: float,
                sym: str,
                lbl: str,
                peaks: np.ndarray,
                mode: str,
            ) -> None:
                if per_record_calibration and lbl == "healthy" and beat_time_s <= cal_end_t:
                    return
                feats, n_beats_win = extract_beat_features(
                    filt, fs, beat_time_s, peaks,
                    morphology_window_s, rr_lookback_s, species,
                    feature_window_mode, post_r_lookahead_s,
                )
                base = {
                    "dataset": dataset,
                    "group": group,
                    "record": hea.stem,
                    "beat_time_s": round(beat_time_s, 3),
                    "beat_symbol": sym,
                    "label": lbl,
                    "n_beats_morph_window": n_beats_win,
                }
                for fset in feature_sets:
                    cal = record_cals.get(fset)
                    if cal is None:
                        continue
                    aligned = align_features_for_scoring(feats, cal)
                    if fset == "hybrid_rewarming":
                        sc = score_one_hybrid(
                            aligned, cal,
                            n_beats_in_window=max(1, n_beats_win),
                            n_recent_clean_beats=10,
                        )
                    elif fset == "all" and all_use_hybrid_gate:
                        sc = score_one_hybrid(
                            aligned, cal,
                            n_beats_in_window=max(1, n_beats_win),
                            n_recent_clean_beats=10,
                        )
                    else:
                        sc = score_one(aligned, cal)
                    r = {**base, "feature_set": fset, "mode": mode, **sc}
                    r["inhibit_class"] = _classify_inhibit(str(r.get("reason", "")))
                    rows.append(r)

            tol_s = 0.080
            n_scored = 0

            # Oracle mode: one decision per annotated beat
            for s, sym in zip(ann.sample, ann.symbol):
                lbl = beat_label(sym, dataset)
                if lbl == "mixed":
                    continue
                t_beat = s / fs
                if window_limit and n_scored >= window_limit:
                    break
                _score_trigger(t_beat, sym, lbl, oracle_peaks, "oracle")
                # Layer2 at each true heartbeat using Layer1 RR stream (fair RR test)
                if len(l1_peaks):
                    _score_trigger(t_beat, sym, lbl, l1_peaks, "layer1_rr_at_beat")
                if len(ad_peaks):
                    _score_trigger(t_beat, sym, lbl, ad_peaks, "layer1_adaptive_rr_at_beat")
                n_scored += 1

            # Layer1 modes: one decision per accepted trigger
            def _label_at(t_s: float) -> Tuple[str, str]:
                best_sym, best_d = "?", "mixed"
                best_dt = tol_s + 1
                for s, sym in zip(ann.sample, ann.symbol):
                    dt = abs(s / fs - t_s)
                    if dt < best_dt:
                        best_dt = dt
                        best_sym = sym
                        best_d = beat_label(sym, dataset)
                if best_dt > tol_s:
                    return best_sym, "mixed"
                return best_sym, best_d

            for peaks, mode in ((l1_peaks, "layer1"),):
                if len(peaks) == 0:
                    continue
                for t_beat in peaks:
                    sym, lbl = _label_at(float(t_beat))
                    _score_trigger(float(t_beat), sym, lbl, peaks, mode)

            if len(ad_peaks):
                for t_beat in ad_peaks:
                    sym, lbl = _label_at(float(t_beat))
                    _score_trigger(float(t_beat), sym, lbl, ad_peaks, "layer1_adaptive_gated")

            log.info(f"  {hea.stem}: {n_scored} oracle beats  {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No rows collected — check data_dir and datasets.")
        return

    df.to_csv(out_dir / "per_beat.csv", index=False)

    metrics = build_metrics(df)
    metrics.to_csv(out_dir / "metrics_by_label.csv", index=False)

    per_rec = _per_record_metrics(df)
    per_rec.to_csv(out_dir / "per_record.csv", index=False)

    per_grp = _per_group_metrics(df)
    per_grp.to_csv(out_dir / "per_group.csv", index=False)

    # Overall metrics (aggregated across all records)
    overall_rows = []
    for (fset, mode), grp in df.groupby(["feature_set", "mode"]):
        h = grp[grp["label"] == "healthy"]
        ab = grp[grp["label"].isin(["abnormal_v", "abnormal_noise"])]
        overall_rows.append({
            "feature_set": fset,
            "mode": mode,
            "n_healthy": len(h),
            "n_abnormal": len(ab),
            "healthy_permit_rate": round(h["permit"].mean(), 4) if len(h) else float("nan"),
            "false_inhibit_rate": round((~h["permit"]).mean(), 4) if len(h) else float("nan"),
            "abnormal_inhibit_rate": round((~ab["permit"]).mean(), 4) if len(ab) else float("nan"),
            "false_permit_rate": round(ab["permit"].mean(), 4) if len(ab) else float("nan"),
        })
    metrics_overall = pd.DataFrame(overall_rows)
    metrics_overall.to_csv(out_dir / "metrics_overall.csv", index=False)

    consec = _consec_inhibit_dist(df[df["label"] == "healthy"] if len(df) else df)
    consec.to_csv(out_dir / "consec_inhibit_dist.csv", index=False)

    inh_reasons = _inhibit_reasons(df)
    inh_reasons.to_csv(out_dir / "inhibit_reasons.csv", index=False)
    if len(inh_reasons):
        inh_reasons.groupby(["feature_set", "mode", "inhibit_class"], as_index=False)[
            "n_inhibited"
        ].sum().to_csv(out_dir / "inhibit_class_breakdown.csv", index=False)

    top_feats = _top_inhibit_features(df)
    top_feats.to_csv(out_dir / "top_inhibit_features.csv", index=False)

    # Side-by-side pivot for healthy beats
    if len(metrics):
        h = metrics[metrics["label"] == "healthy"]
        if len(h):
            piv = h.pivot_table(
                index=["feature_set", "mode"],
                values="healthy_permit_rate",
                aggfunc="first",
            )
            piv.to_csv(out_dir / "healthy_permit_by_mode.csv")
            log.info("Healthy beat permit rates:\n" + piv.to_string())

    if len(metrics_overall):
        log.info("Overall metrics:\n" + metrics_overall.to_string(index=False))

    log.info(f"Done -> {out_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2_beat_sync"))
    p.add_argument("--datasets", nargs="+", default=["mit_bih_arrhythmia"],
                   help="Dataset folder or PhysioNet alias (see data/README.md)")
    p.add_argument("--feature-sets", nargs="+",
                   default=["all", "signal_only", "hybrid_rewarming"])
    p.add_argument("--morphology-window-s", type=float, default=5.0)
    p.add_argument("--rr-lookback-s", type=float, default=30.0)
    p.add_argument("--per-record-calibration", action="store_true", default=True)
    p.add_argument("--no-per-record-calibration", action="store_true")
    p.add_argument("--cal-frac", type=float, default=0.6)
    p.add_argument("--threshold-quantile", type=float, default=0.999)
    p.add_argument("--include-adaptive", action="store_true", default=True)
    p.add_argument("--window-limit", type=int, default=None)
    p.add_argument(
        "--abnormal-target-inhibit", type=float, default=None,
        help="If set (e.g. 0.95), tune Mahalanobis thresholds per record using "
             "labeled abnormal beats after calibration (healthy mean/cov unchanged).",
    )
    p.add_argument(
        "--max-healthy-false-inhibit", type=float, default=0.80,
        help="Max healthy false-inhibit when using --abnormal-target-inhibit "
             "(0.80 = at least 20%% healthy permit allowed by threshold tuning).",
    )
    p.add_argument(
        "--all-use-hybrid-gate", action="store_true", default=False,
        help="Use two-stage hybrid gate for feature_set='all' "
             "(morphology-first, RR only when reliable).",
    )
    p.add_argument(
        "--feature-window-mode",
        choices=["causal", "centered"],
        default="causal",
        help="causal (default): deployment-realistic window [beat-window, beat+lookahead] "
             "with causal IIR filter. "
             "centered: offline oracle window beat±half with zero-phase filtfilt.",
    )
    p.add_argument(
        "--post-r-lookahead-s", type=float, default=0.08,
        help="Post-R lookahead in seconds used in causal mode (default 0.08 = 80 ms). "
             "Must be <= stimulation delay to remain causal for the trigger beat.",
    )
    args = p.parse_args(argv)

    run_beat_sync_validation(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        datasets=args.datasets,
        feature_sets=args.feature_sets,
        morphology_window_s=args.morphology_window_s,
        rr_lookback_s=args.rr_lookback_s,
        per_record_calibration=not args.no_per_record_calibration,
        cal_frac=args.cal_frac,
        threshold_quantile=args.threshold_quantile,
        include_adaptive=args.include_adaptive,
        window_limit=args.window_limit,
        abnormal_target_inhibit=args.abnormal_target_inhibit,
        max_healthy_false_inhibit=args.max_healthy_false_inhibit,
        all_use_hybrid_gate=args.all_use_hybrid_gate,
        feature_window_mode=args.feature_window_mode,
        post_r_lookahead_s=args.post_r_lookahead_s,
    )


if __name__ == "__main__":
    main()
