"""
Fast Pareto test: beat-sync on a 10-record MIT-BIH subset (~2 min).

Used by run_pareto_sweep.py quick mode before committing to a full 48-record run.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_L1 = _ROOT / "Layer1"
sys.path.insert(0, str(_L2))
from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()

_l1_bootstrap = importlib.util.spec_from_file_location(
    "layer1_bootstrap", _L1 / "_bootstrap.py",
)
if _l1_bootstrap is None or _l1_bootstrap.loader is None:
    raise ImportError("Cannot load Layer1 bootstrap")
layer1_bootstrap = importlib.util.module_from_spec(_l1_bootstrap)
_l1_bootstrap.loader.exec_module(layer1_bootstrap)
sys.path.insert(0, str(_L1))
layer1_bootstrap.setup_layer1_paths(include_archive=True)

from common import (  # noqa: E402
    ABNORMAL_BEATS,
    NORMAL_BEATS,
    apply_filters,
    score_one,
    score_one_hybrid,
    _fit_calibrator,
    _select_finite_features,
)
from run_beat_validation import (  # noqa: E402
    beat_label,
    extract_beat_features,
    _classify_inhibit,
)

FAST_RECORDS = ["100", "115", "117", "122", "108", "116", "222", "205", "200", "213"]


def run_quick(
    data_dir: Path,
    out_dir: Path,
    abnormal_target: Optional[float] = 0.95,
    healthy_fi_cap: float = 0.15,
    cal_frac: float = 0.6,
    threshold_quantile: float = 0.999,
    records: Optional[List[str]] = None,
    use_hybrid_gate: bool = True,
) -> None:
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = data_dir / "mit_bih_arrhythmia"
    recs = records or FAST_RECORDS
    rows: List[Dict] = []

    for rec in recs:
        hea = ds_dir / f"{rec}.hea"
        if not hea.exists():
            print(f"  [skip] {rec}")
            continue
        stem = str(hea.with_suffix(""))
        wfdb_rec = wfdb.rdrecord(stem)
        ann = wfdb.rdann(stem, "atr")
        raw = wfdb_rec.p_signal[:, 0].astype(float)
        fs = float(wfdb_rec.fs)
        filt = apply_filters(raw, fs)

        oracle_peaks = np.array(
            [s / fs for s, sym in zip(ann.sample, ann.symbol)
             if sym in (NORMAL_BEATS | ABNORMAL_BEATS)], dtype=float)
        ad_peaks = np.array([])

        healthy_times = sorted(
            s / fs for s, sym in zip(ann.sample, ann.symbol)
            if beat_label(sym, "mit_bih_arrhythmia") == "healthy")

        n_cal = max(5, int(len(healthy_times) * cal_frac))
        cal_feats: List[Dict] = []
        for t_b in healthy_times[:n_cal]:
            f, _ = extract_beat_features(filt, fs, t_b, oracle_peaks)
            cal_feats.append(f)

        if len(cal_feats) < 5:
            print(f"  {rec}: too few cal feats, skip")
            continue

        fn, cal_feats = _select_finite_features(cal_feats)
        cal_end_t = healthy_times[n_cal - 1]

        abn_cal: List[Dict] = []
        h_val: List[Dict] = []
        if abnormal_target is not None:
            for s, sym in zip(ann.sample, ann.symbol):
                t_b = s / fs
                if t_b <= cal_end_t:
                    continue
                if beat_label(sym, "mit_bih_arrhythmia") == "abnormal_v":
                    f, _ = extract_beat_features(filt, fs, t_b, oracle_peaks)
                    abn_cal.append(f)
            for t_b in healthy_times[n_cal:]:
                f, _ = extract_beat_features(filt, fs, t_b, oracle_peaks)
                h_val.append(f)

        cal = _fit_calibrator(cal_feats, fn, threshold_quantile, "all")
        if abnormal_target is not None and len(abn_cal) >= 5:
            h_aligned = [{k: f.get(k, float("nan")) for k in cal.feature_names} for f in h_val]
            a_aligned = [{k: f.get(k, float("nan")) for k in cal.feature_names} for f in abn_cal]
            cal.calibrate_thresholds_for_abnormal_inhibit(
                a_aligned,
                target_inhibit_rate=abnormal_target,
                healthy_validation_features=h_aligned if h_aligned else None,
                max_healthy_false_inhibit_rate=healthy_fi_cap,
            )

        tol_s = 0.080

        def _label_at(t_s: float) -> Tuple[str, str]:
            best_sym, best_dt = "?", tol_s + 1
            for s, sym in zip(ann.sample, ann.symbol):
                dt = abs(s / fs - t_s)
                if dt < best_dt:
                    best_dt = dt
                    best_sym = sym
            if best_dt > tol_s:
                return best_sym, "mixed"
            return best_sym, beat_label(best_sym, "mit_bih_arrhythmia")

        for peaks, mode in [(oracle_peaks, "oracle"), (ad_peaks, "layer1_adaptive_gated")]:
            if len(peaks) == 0:
                continue
            for t_b in peaks:
                sym, lbl = _label_at(float(t_b))
                if lbl == "mixed":
                    continue
                if lbl == "healthy" and float(t_b) <= cal_end_t:
                    continue
                feats, n_win = extract_beat_features(filt, fs, float(t_b), peaks)
                aligned = {k: feats.get(k, float("nan")) for k in cal.feature_names}
                if use_hybrid_gate:
                    sc = score_one_hybrid(aligned, cal, n_beats_in_window=max(1, n_win),
                                         n_recent_clean_beats=10)
                else:
                    sc = score_one(aligned, cal)
                r = {"record": rec, "mode": mode, "label": lbl,
                     "beat_time_s": round(float(t_b), 3),
                     "beat_symbol": sym, **sc}
                r["inhibit_class"] = _classify_inhibit(str(r.get("reason", "")))
                rows.append(r)

        print(f"  {rec}: done  {time.time()-t_start:.0f}s total")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows — check data dir.")
        return

    df.to_csv(out_dir / "per_beat.csv", index=False)

    summary_rows = []
    for (mode,), grp in df.groupby(["mode"]):
        h = grp[grp.label == "healthy"]
        ab = grp[grp.label == "abnormal_v"]
        summary_rows.append({
            "mode": mode,
            "n_healthy": len(h),
            "n_abnormal": len(ab),
            "healthy_permit": round(h.permit.mean(), 4) if len(h) else float("nan"),
            "abnormal_inhibit": round((~ab.permit).mean(), 4) if len(ab) else float("nan"),
            "false_permit": round(ab.permit.mean(), 4) if len(ab) else float("nan"),
            "false_inhibit": round((~h.permit).mean(), 4) if len(h) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    inh = df[~df.permit]
    inh.groupby(["mode", "inhibit_class"]).size().reset_index(name="n").to_csv(
        out_dir / "inhibit_class.csv", index=False)

    print(f"\n=== Results (target={abnormal_target}, fi_cap={healthy_fi_cap}) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {out_dir}")
