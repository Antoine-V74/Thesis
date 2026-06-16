"""
Per-record failure breakdown — why does Layer 1 miss or reject beats?

For each recording, decomposes oracle vs fixed-Layer1 mismatches into:

  Detection path (no candidate near oracle beat):
    - threshold_too_high   amplitude under chosen-polarity threshold
    - wrong_polarity       opposite polarity would cross threshold
    - slope_gate_or_other  amplitude OK but no candidate (slope gate / logic)

  Supervisor path (candidate exists but not accepted):
    - supervisor_out_of_band
    - supervisor_rr_short / supervisor_rr_long
    - supervisor_refractory_blanking
    - supervisor_recovery
    - supervisor_calibrating
    - supervisor_other

  Extra detections (no oracle match):
    - extra_threshold_oversense
    - extra_wrong_polarity
    - extra_supervisor_should_have_rejected (rare)

Also reports fixed vs best-polarity sensitivity to flag polarity lock errors.

Usage
-----
    cd "ECG Processing"
    .venv\\Scripts\\python Layer1\\tools\\analyze_records.py `
        --data-dir data --datasets mit_bih_arrhythmia `
        --out-dir Results/layer1_record_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_L1 = _HERE.parent
sys.path.insert(0, str(_L1))
from _bootstrap import setup_layer1_paths  # noqa: E402

setup_layer1_paths(include_archive=True)

from analysis_helpers import (  # noqa: E402
    apply_filters,
    load_record,
    match_peaks,
    run_layer1_detector,
)

from main_pipeline import run_layer1
from rhythm_supervisor import RRSupervisor

NORMAL = set("NLRAaJSVFeEjf/!Qn?")


def _decision_map(supervisor: RRSupervisor) -> Dict[int, str]:
    return {int(d.sample): d.decision for d in supervisor.state.decisions}


def _classify_supervisor_reason(decision: str) -> str:
    d = decision.lower()
    if "out_of_band" in d or "reject_out_of_band" in d:
        return "supervisor_out_of_band"
    if "reject_short" in d:
        return "supervisor_rr_short"
    if "reject_long" in d:
        return "supervisor_rr_long"
    if "refractory" in d or "blanking" in d or "post_stim" in d:
        return "supervisor_refractory_blanking"
    if "recovery" in d:
        return "supervisor_recovery"
    if "calibration" in d or "wait_arm" in d:
        return "supervisor_calibrating"
    return "supervisor_other"


def _amp_at(filt: np.ndarray, fs: float, t_s: float, polarity: str) -> float:
    idx = int(round(t_s * fs))
    idx = max(0, min(len(filt) - 1, idx))
    v = float(filt[idx])
    return v if polarity == "positive" else -v


def _nearest_candidate_sample(
    t_s: float,
    fs: float,
    candidates_s: np.ndarray,
    tol_ms: float,
) -> Optional[int]:
    if len(candidates_s) == 0:
        return None
    oi = int(round(t_s * fs))
    tol = int(round(tol_ms * 1e-3 * fs))
    best = None
    best_d = tol + 1
    for c in candidates_s:
        ci = int(round(c * fs))
        d = abs(ci - oi)
        if d <= tol and d < best_d:
            best_d = d
            best = ci
    return best


def analyze_record(
    dataset: str,
    record: str,
    data_dir: Path,
    tol_ms: float = 80.0,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Return (summary row, per-miss rows, per-extra rows)."""
    stem = str(data_dir / dataset / record)
    raw, fs, ann = load_record(stem)
    filt = apply_filters(raw, fs)

    if ann is None:
        raise RuntimeError("no annotations")

    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=int)
    mask = np.isin(symbols, list(NORMAL))
    oracle_s = samples[mask] / fs

    l1 = run_layer1(filt, fs, already_filtered=True)
    cand_s = l1.candidate_samples.astype(float) / fs
    acc_s = l1.accepted_samples.astype(float) / fs
    sup = l1.supervisor
    dec_map = _decision_map(sup)
    acc_set = set(int(round(p * fs)) for p in acc_s)
    best_pol = l1.detector.polarity
    fixed_pol = best_pol
    thr_arr = l1.detector.thresholds
    fixed_thr = float(np.median(thr_arr[thr_arr > 0])) if np.any(thr_arr > 0) else 0.03

    m = match_peaks(oracle_s, acc_s, fs, tol_ms)
    best_sens = round(float(m["sensitivity"]), 4)
    ad_sens, ad_ppv, ad_polarity = float("nan"), float("nan"), "n/a"

    miss_rows: List[Dict] = []
    extra_rows: List[Dict] = []

    miss_counts: Dict[str, int] = {}
    extra_counts: Dict[str, int] = {}

    def _bump(bucket: Dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    # --- missed oracle beats ---
    matched_oracle = set()
    for o_samp, t_samp, _ in m["matches"]:
        matched_oracle.add(int(o_samp))

    oracle_set = set(int(s) for s in samples[mask])
    for o_samp in sorted(oracle_set):
        if o_samp in matched_oracle:
            continue
        t_s = o_samp / fs
        cand_samp = _nearest_candidate_sample(t_s, fs, cand_s, tol_ms)

        if cand_samp is None:
            amp_pos = _amp_at(filt, fs, t_s, "positive")
            amp_neg = _amp_at(filt, fs, t_s, "negative")
            amp_chosen = amp_pos if fixed_pol == "positive" else amp_neg
            amp_alt = amp_neg if fixed_pol == "positive" else amp_pos
            if amp_chosen < fixed_thr and amp_alt >= fixed_thr:
                reason = "wrong_polarity"
            elif amp_chosen < fixed_thr:
                reason = "threshold_too_high"
            else:
                reason = "slope_gate_or_other"
            _bump(miss_counts, reason)
            miss_rows.append({
                "dataset": dataset,
                "record": record,
                "beat_time_s": round(t_s, 3),
                "failure_class": "detection",
                "failure_reason": reason,
                "fixed_polarity": fixed_pol,
                "fixed_threshold": round(fixed_thr, 4),
                "amp_chosen": round(amp_chosen, 4),
                "amp_opposite": round(amp_alt, 4),
                "amp_ratio_chosen": round(amp_chosen / fixed_thr, 3) if fixed_thr else float("nan"),
                "supervisor_decision": "",
            })
            continue

        if cand_samp not in acc_set:
            decision = dec_map.get(cand_samp, "unknown")
            reason = _classify_supervisor_reason(decision)
            _bump(miss_counts, reason)
            miss_rows.append({
                "dataset": dataset,
                "record": record,
                "beat_time_s": round(t_s, 3),
                "failure_class": "supervisor",
                "failure_reason": reason,
                "fixed_polarity": fixed_pol,
                "fixed_threshold": round(fixed_thr, 4),
                "amp_chosen": round(_amp_at(filt, fs, t_s, fixed_pol), 4),
                "amp_opposite": round(_amp_at(filt, fs, t_s,
                                               "negative" if fixed_pol == "positive" else "positive"), 4),
                "amp_ratio_chosen": float("nan"),
                "supervisor_decision": decision,
            })

    # --- extra accepted peaks ---
    oracle_idx = set(int(s) for s in samples[mask])
    tol_samp = int(round(tol_ms * 1e-3 * fs))
    for a_samp in sorted(int(round(p * fs)) for p in acc_s):
        if any(abs(a_samp - oi) <= tol_samp for oi in oracle_idx):
            continue
        t_s = a_samp / fs
        amp_pos = _amp_at(filt, fs, t_s, "positive")
        amp_neg = _amp_at(filt, fs, t_s, "negative")
        amp_chosen = amp_pos if fixed_pol == "positive" else amp_neg
        if amp_chosen < 0.6 * fixed_thr:
            reason = "extra_threshold_oversense"
        elif (fixed_pol == "positive" and amp_neg > amp_pos) or (fixed_pol == "negative" and amp_pos > amp_neg):
            reason = "extra_wrong_polarity"
        else:
            reason = "extra_other"
        _bump(extra_counts, reason)
        extra_rows.append({
            "dataset": dataset,
            "record": record,
            "beat_time_s": round(t_s, 3),
            "failure_class": "extra",
            "failure_reason": reason,
            "fixed_polarity": fixed_pol,
            "fixed_threshold": round(fixed_thr, 4),
            "amp_chosen": round(amp_chosen, 4),
        })

    n_miss = int(m["fn"])
    n_extra = int(m["fp"])
    n_oracle = len(oracle_s)

    def _pct(k: str, n: int) -> float:
        return round(100.0 * miss_counts.get(k, 0) / n, 1) if n else 0.0

    def _pct_e(k: str, n: int) -> float:
        return round(100.0 * extra_counts.get(k, 0) / n, 1) if n else 0.0

    det_miss = sum(v for k, v in miss_counts.items()
                   if k in ("threshold_too_high", "wrong_polarity", "slope_gate_or_other"))
    sup_miss = n_miss - det_miss

    summary = {
        "dataset": dataset,
        "record": record,
        "n_oracle_beats": n_oracle,
        "n_fixed_accepted": len(acc_s),
        "sensitivity": round(float(m["sensitivity"]), 4),
        "ppv": round(float(m["ppv"]), 4),
        "adaptive_sensitivity": ad_sens,
        "adaptive_ppv": ad_ppv,
        "fixed_polarity": fixed_pol,
        "adaptive_polarity": ad_polarity,
        "best_polarity_by_sensitivity": best_pol,
        "best_polarity_sensitivity": round(float(best_sens), 4),
        "polarity_mismatch": fixed_pol != best_pol,
        "adaptive_polarity_mismatch": ad_polarity != best_pol,
        "fixed_threshold": round(fixed_thr, 4),
        "n_missed": n_miss,
        "n_extra": n_extra,
        "pct_miss_detection": round(100 * det_miss / n_miss, 1) if n_miss else 0.0,
        "pct_miss_supervisor": round(100 * sup_miss / n_miss, 1) if n_miss else 0.0,
        "pct_miss_threshold_too_high": _pct("threshold_too_high", n_miss),
        "pct_miss_wrong_polarity": _pct("wrong_polarity", n_miss),
        "pct_miss_slope_gate_or_other": _pct("slope_gate_or_other", n_miss),
        "pct_miss_supervisor_out_of_band": _pct("supervisor_out_of_band", n_miss),
        "pct_miss_supervisor_rr_short": _pct("supervisor_rr_short", n_miss),
        "pct_miss_supervisor_rr_long": _pct("supervisor_rr_long", n_miss),
        "pct_miss_supervisor_refractory": _pct("supervisor_refractory_blanking", n_miss),
        "pct_miss_supervisor_recovery": _pct("supervisor_recovery", n_miss),
        "pct_miss_supervisor_calibrating": _pct("supervisor_calibrating", n_miss),
        "pct_miss_supervisor_other": _pct("supervisor_other", n_miss),
        "pct_extra_total": round(100 * n_extra / max(len(acc_s), 1), 1),
        "pct_extra_threshold_oversense": _pct_e("extra_threshold_oversense", n_extra),
        "pct_extra_wrong_polarity": _pct_e("extra_wrong_polarity", n_extra),
        "primary_work_area": _primary_work_area(miss_counts, extra_counts, fixed_pol, best_pol, n_miss, n_extra),
    }
    return summary, miss_rows, extra_rows


def _primary_work_area(
    miss_counts: Dict[str, int],
    extra_counts: Dict[str, int],
    fixed_pol: str,
    best_pol: str,
    n_miss: int,
    n_extra: int,
) -> str:
    if fixed_pol != best_pol and miss_counts.get("wrong_polarity", 0) > 0:
        return "polarity_selection"
    if n_miss == 0 and n_extra == 0:
        return "ok"
    det = sum(miss_counts.get(k, 0) for k in (
        "threshold_too_high", "wrong_polarity", "slope_gate_or_other"))
    sup = n_miss - det
    if det >= sup and det > 0:
        if miss_counts.get("threshold_too_high", 0) >= miss_counts.get("wrong_polarity", 0):
            return "threshold_calibration"
        return "polarity_or_detection"
    if sup > det and sup > 0:
        return "supervisor_logic"
    if n_extra > n_miss:
        return "threshold_too_low_or_oversensing"
    return "mixed"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer1_record_diagnostics"))
    p.add_argument("--datasets", nargs="+", default=["mitdb"])
    p.add_argument("--match-tolerance-ms", type=float, default=80.0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []
    all_miss: List[Dict] = []
    all_extra: List[Dict] = []

    for dataset in args.datasets:
        ds_dir = args.data_dir / dataset
        if not ds_dir.is_dir():
            print(f"[WARN] missing {ds_dir}")
            continue
        hea_files = sorted(ds_dir.glob("*.hea"))
        print(f"{dataset}: {len(hea_files)} records")
        for hea in hea_files:
            rec = hea.stem
            try:
                summary, miss_rows, extra_rows = analyze_record(
                    dataset, rec, args.data_dir, args.match_tolerance_ms)
                summaries.append(summary)
                all_miss.extend(miss_rows)
                all_extra.extend(extra_rows)
                print(f"  {rec}: sens={summary['sensitivity']:.2f}  "
                      f"work={summary['primary_work_area']}  "
                      f"miss det/sup={summary['pct_miss_detection']}/"
                      f"{summary['pct_miss_supervisor']}%")
            except Exception as exc:
                print(f"  [WARN] {rec}: {exc}")

    df_sum = pd.DataFrame(summaries).sort_values("n_missed", ascending=False)
    df_sum.to_csv(args.out_dir / "record_failure_taxonomy.csv", index=False)

    if all_miss:
        pd.DataFrame(all_miss).to_csv(args.out_dir / "missed_beats_detail.csv", index=False)
    if all_extra:
        pd.DataFrame(all_extra).to_csv(args.out_dir / "extra_beats_detail.csv", index=False)

    # Aggregate recommendation table
    if len(df_sum):
        agg = {
            "n_records": len(df_sum),
            "records_polarity_mismatch": int(df_sum["polarity_mismatch"].sum()),
            "records_primary_supervisor": int((df_sum["primary_work_area"] == "supervisor_logic").sum()),
            "records_primary_threshold": int((df_sum["primary_work_area"] == "threshold_calibration").sum()),
            "records_primary_polarity": int((df_sum["primary_work_area"] == "polarity_selection").sum()),
            "mean_sensitivity": round(float(df_sum["sensitivity"].mean()), 4),
        }
        pd.DataFrame([agg]).to_csv(args.out_dir / "summary_overall.csv", index=False)

    print(f"\nWrote {args.out_dir / 'record_failure_taxonomy.csv'}")


if __name__ == "__main__":
    main()
