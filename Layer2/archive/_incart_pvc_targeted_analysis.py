"""
Targeted INCART analysis: PVC-specific rules vs global z-score tightening.

Extracts per-beat features for INCART (oracle), then post-hoc sweeps:
  1. Global max_zscore threshold scale
  2. PVC morphology hard rules (template_corr, neighbor_corr, coupling)
  3. Combined PVC bundle
  4. Per-record adaptive template threshold

Outputs:
  Results/incart_pvc_analysis/pareto_strategies.csv
  Results/incart_pvc_analysis/worst_records.csv
  Results/incart_pvc_analysis/summary.txt
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Layer2"))

import wfdb  # noqa: E402

from decision import BaselineCalibrator, FROZEN_ZSCORE_QUANTILE  # noqa: E402
from common import _fit_calibrator, _select_finite_features, apply_filters, score_one  # noqa: E402
from run_cross_dataset_benchmark import (  # noqa: E402
    DATASET_ABNORMAL,
    DATASET_CHANNEL,
    DATASET_NORMAL,
    beat_label,
    extract_beat_features,
)

DATA_DIR = ROOT / "data" / "incartdb"
OUT_DIR = ROOT / "Results" / "incart_pvc_analysis"
WORST_RECORDS = ["I51", "I42", "I30", "I31", "I29", "I08", "I36", "I12", "I24", "I26"]

CAL_FRAC = 0.6  # must match run_cross_dataset_benchmark default
THRESHOLD_QUANTILE = 0.999
MORPHOLOGY_WINDOW_S = 5.0
RR_LOOKBACK_S = 30.0
SPECIES = "human"


def _load_record(stem: str) -> Optional[Tuple]:
    try:
        rec = wfdb.rdrecord(stem)
        ann = wfdb.rdann(stem, "atr")
    except Exception as exc:
        logging.warning("skip %s: %s", stem, exc)
        return None
    ch = min(DATASET_CHANNEL["incartdb"], rec.p_signal.shape[1] - 1)
    raw = rec.p_signal[:, ch].astype(float)
    fs = float(rec.fs)
    filt = apply_filters(raw, fs)
    dataset = "incartdb"
    all_peaks = np.array([
        s / fs for s, sym in zip(ann.sample, ann.symbol)
        if sym in (DATASET_NORMAL[dataset] | DATASET_ABNORMAL[dataset])
    ], dtype=float)
    healthy_times = sorted(
        s / fs for s, sym in zip(ann.sample, ann.symbol)
        if beat_label(sym, dataset) == "healthy"
    )
    if len(healthy_times) < 10:
        return None
    n_cal = max(5, int(len(healthy_times) * CAL_FRAC))
    cal_end_t = healthy_times[min(n_cal, len(healthy_times)) - 1]
    return rec, ann, raw, filt, fs, all_peaks, healthy_times, n_cal, cal_end_t, dataset


def extract_record_beats(stem: str) -> List[Dict]:
    loaded = _load_record(stem)
    if loaded is None:
        return []
    _, ann, raw, filt, fs, all_peaks, healthy_times, n_cal, cal_end_t, dataset = loaded

    cal_feats: List[Dict] = []
    for t_beat in healthy_times[:n_cal]:
        feats, _ = extract_beat_features(
            filt, fs, t_beat, all_peaks,
            MORPHOLOGY_WINDOW_S, RR_LOOKBACK_S, SPECIES, raw=raw,
        )
        cal_feats.append(feats)
    fn, cal_feats = _select_finite_features(cal_feats)
    if len(cal_feats) < 5:
        return []

    cal = _fit_calibrator(cal_feats, fn, THRESHOLD_QUANTILE, "all")
    cal_z_thr = cal.threshold_max_zscore
    cal_m_thr = cal.threshold_mahalanobis

    rows: List[Dict] = []
    for s, sym in zip(ann.sample, ann.symbol):
        t_b = s / fs
        lbl = beat_label(sym, dataset)
        if lbl == "mixed":
            continue
        if lbl not in ("healthy", "abnormal_v"):
            continue
        if lbl == "healthy" and t_b <= cal_end_t:
            continue
        feats, _ = extract_beat_features(
            filt, fs, t_b, all_peaks,
            MORPHOLOGY_WINDOW_S, RR_LOOKBACK_S, SPECIES, raw=raw,
        )
        aligned = {k: feats.get(k, float("nan")) for k in cal.feature_names}
        sc = score_one(aligned, cal)
        row = {
            "record": Path(stem).name,
            "beat_time_s": round(t_b, 3),
            "beat_symbol": sym,
            "label": lbl,
            "baseline_permit": sc["permit"],
            "mahalanobis": sc.get("mahalanobis"),
            "max_zscore": sc.get("max_zscore"),
            "zscore_threshold": cal_z_thr,
            "mahalanobis_threshold": cal_m_thr,
            "hard_rule": sc.get("hard_rule_violated") or None,
        }
        for k in cal.feature_names:
            row[k] = feats.get(k, float("nan"))
        if "rr__beat_coupling_ratio" in feats:
            row["rr__beat_coupling_ratio"] = feats["rr__beat_coupling_ratio"]
        rows.append(row)
    return rows


def _hard_rule_ok(df: pd.DataFrame) -> pd.Series:
    return df["hard_rule"].fillna("").astype(str).eq("")


def _baseline_permit(df: pd.DataFrame) -> pd.Series:
    if "baseline_permit" in df.columns:
        return df["baseline_permit"].astype(bool)
    permit = _hard_rule_ok(df)
    permit &= ~(df["max_zscore"].fillna(0) > df["zscore_threshold"])
    permit &= ~(df["mahalanobis"].fillna(0) > df["mahalanobis_threshold"])
    return permit


def _metrics(permit: pd.Series, df: pd.DataFrame) -> Tuple[float, float, float]:
    h = df["label"] == "healthy"
    ab = df["label"] == "abnormal_v"
    hp = float(permit[h].mean()) if h.any() else float("nan")
    ai = float((~permit[ab]).mean()) if ab.any() else float("nan")
    fp = float(permit[ab].mean()) if ab.any() else float("nan")
    return hp, ai, fp


def _core_permit(df: pd.DataFrame, zscore_scale: float = 1.0) -> pd.Series:
    """Hard rules + mahal + scaled zscore (rebuilds zscore gate)."""
    permit = _hard_rule_ok(df)
    permit &= ~(df["mahalanobis"].fillna(0) > df["mahalanobis_threshold"])
    permit &= ~(df["max_zscore"].fillna(0) > df["zscore_threshold"] * zscore_scale)
    return permit


def apply_pvc_rules(
    df: pd.DataFrame,
    base_permit: pd.Series,
    template_min: Optional[float] = None,
    neighbor_min: Optional[float] = None,
    coupling_min: Optional[float] = None,
    amp_vs_median_max_z: Optional[float] = None,
    qrs_width_min_z: Optional[float] = None,
    zscore_scale: Optional[float] = None,
    per_record_template_p: Optional[float] = None,
) -> pd.Series:
    if zscore_scale is not None:
        permit = _core_permit(df, zscore_scale=zscore_scale)
    else:
        permit = base_permit.copy()

    if template_min is not None and "morph__template_corr" in df.columns:
        permit &= df["morph__template_corr"].fillna(1.0) >= template_min

    if neighbor_min is not None and "morph__neighbor_corr" in df.columns:
        permit &= df["morph__neighbor_corr"].fillna(1.0) >= neighbor_min

    if coupling_min is not None and "rr__beat_coupling_ratio" in df.columns:
        # Missing coupling -> do not extra-inhibit (conservative)
        c = df["rr__beat_coupling_ratio"]
        permit &= c.isna() | (c >= coupling_min)

    if amp_vs_median_max_z is not None and "morph__amp_vs_median" in df.columns:
        # Simple z vs record healthy median/IQR
        for rec, g in df.groupby("record"):
            h = g[g["label"] == "healthy"]["morph__amp_vs_median"].dropna()
            if len(h) < 10:
                continue
            med = h.median()
            iqr = max(h.quantile(0.75) - h.quantile(0.25), 1e-6)
            z = (g["morph__amp_vs_median"] - med) / iqr
            permit.loc[g.index] &= z.abs() <= amp_vs_median_max_z

    if qrs_width_min_z is not None and "morph__qrs_width_ms" in df.columns:
        for rec, g in df.groupby("record"):
            h = g[g["label"] == "healthy"]["morph__qrs_width_ms"].dropna()
            if len(h) < 10:
                continue
            med = h.median()
            iqr = max(h.quantile(0.75) - h.quantile(0.25), 1e-6)
            z = (g["morph__qrs_width_ms"] - med) / iqr
            permit.loc[g.index] &= z.abs() <= qrs_width_min_z

    if per_record_template_p is not None and "morph__template_corr" in df.columns:
        for rec, g in df.groupby("record"):
            h = g[g["label"] == "healthy"]["morph__template_corr"].dropna()
            if len(h) < 10:
                continue
            thr = float(h.quantile(per_record_template_p))
            permit.loc[g.index] &= g["morph__template_corr"].fillna(1.0) >= thr

    return permit


def sweep_strategies(df: pd.DataFrame) -> pd.DataFrame:
    base = _baseline_permit(df)
    hp0, ai0, fp0 = _metrics(base, df)
    rows = [{
        "strategy": "baseline_zero_shot",
        "params": "",
        "healthy_permit": hp0,
        "abnormal_inhibit": ai0,
        "false_permit": fp0,
    }]

    def add(name: str, params: str, permit: pd.Series):
        hp, ai, fp = _metrics(permit, df)
        rows.append({
            "strategy": name,
            "params": params,
            "healthy_permit": round(hp, 4),
            "abnormal_inhibit": round(ai, 4),
            "false_permit": round(fp, 4),
        })

    # Global z-score tightening only
    for scale in [0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.10]:
        p = apply_pvc_rules(df, base, zscore_scale=scale)
        add("global_zscore", f"scale={scale}", p)

    # PVC template_corr alone (raise MIT-BIH 0.55 -> higher)
    for tmin in [0.60, 0.65, 0.70, 0.75, 0.80]:
        p = apply_pvc_rules(df, base, template_min=tmin)
        add("pvc_template_corr", f"min={tmin}", p)

    # Coupling alone
    for cmin in [0.80, 0.85, 0.90, 0.95]:
        p = apply_pvc_rules(df, base, coupling_min=cmin)
        add("pvc_coupling", f"min={cmin}", p)

    # Neighbor corr
    for nmin in [0.55, 0.60, 0.65, 0.70]:
        p = apply_pvc_rules(df, base, neighbor_min=nmin)
        add("pvc_neighbor_corr", f"min={nmin}", p)

    # Per-record adaptive template (healthy percentile)
    for pctl in [0.05, 0.10, 0.15, 0.20]:
        p = apply_pvc_rules(df, base, per_record_template_p=pctl)
        add("adaptive_template", f"healthy_p={pctl}", p)

    # Morph z-score rules (record-relative)
    for z in [2.0, 2.5, 3.0, 3.5]:
        p = apply_pvc_rules(df, base, qrs_width_min_z=z)
        add("pvc_qrs_width_z", f"max_abs_z={z}", p)
        p2 = apply_pvc_rules(df, base, amp_vs_median_max_z=z)
        add("pvc_amp_vs_median_z", f"max_abs_z={z}", p2)

    # Combined bundles (best expected trade-offs)
    combos = [
        ("bundle_a", dict(template_min=0.65, coupling_min=0.85)),
        ("bundle_b", dict(template_min=0.70, coupling_min=0.85)),
        ("bundle_c", dict(template_min=0.65, coupling_min=0.85, neighbor_min=0.60)),
        ("bundle_d", dict(template_min=0.70, per_record_template_p=0.10)),
        ("bundle_e", dict(template_min=0.65, coupling_min=0.85, qrs_width_min_z=2.5)),
        ("bundle_f", dict(template_min=0.65, zscore_scale=0.5)),
        ("bundle_g", dict(template_min=0.70, coupling_min=0.85, zscore_scale=0.3)),
    ]
    for name, kw in combos:
        p = apply_pvc_rules(df, base, **kw)
        add(name, str(kw), p)

    out = pd.DataFrame(rows)
    return out.sort_values(["abnormal_inhibit", "healthy_permit"], ascending=[False, False])


def analyze_worst_records(df: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    base = _baseline_permit(df)
    rows = []
    for rec in WORST_RECORDS:
        sub = df[df["record"] == rec]
        if sub.empty:
            continue
        hp0, ai0, _ = _metrics(base.loc[sub.index], sub)
        ab = sub[sub["label"] == "abnormal_v"]
        ab_permit = base.loc[ab.index]
        fp = ab[~ab_permit]
        tn = ab[ab_permit]
        rows.append({
            "record": rec,
            "n_abnormal": len(ab),
            "baseline_ai": ai0,
            "baseline_hp": hp0,
            "fp_n": len(fp),
            "fp_template_corr_p50": fp["morph__template_corr"].median() if len(fp) else float("nan"),
            "fp_coupling_p50": fp["rr__beat_coupling_ratio"].median()
            if len(fp) and "rr__beat_coupling_ratio" in fp.columns else float("nan"),
            "fp_max_zscore_p50": fp["max_zscore"].median() if len(fp) else float("nan"),
            "tn_max_zscore_p50": tn["max_zscore"].median() if len(tn) else float("nan"),
        })

    worst = pd.DataFrame(rows)

    # Best strategy per worst record at AI>=0.90 and AI>=0.95
    best_rows = []
    strategies = [
        ("baseline", {}),
        ("template_0.65", dict(template_min=0.65)),
        ("template_0.70", dict(template_min=0.70)),
        ("coupling_0.85", dict(coupling_min=0.85)),
        ("bundle_a", dict(template_min=0.65, coupling_min=0.85)),
        ("bundle_g", dict(template_min=0.70, coupling_min=0.85, zscore_scale=0.3)),
        ("zscore_0.2", dict(zscore_scale=0.2)),
    ]
    for rec in WORST_RECORDS:
        sub = df[df["record"] == rec]
        if sub.empty:
            continue
        bsub = base.loc[sub.index]
        for sname, kw in strategies:
            p = apply_pvc_rules(sub, bsub, **kw)
            hp, ai, _ = _metrics(p, sub)
            best_rows.append({
                "record": rec, "strategy": sname,
                "healthy_permit": hp, "abnormal_inhibit": ai,
            })
    pd.DataFrame(best_rows).to_csv(OUT_DIR / "worst_record_strategies.csv", index=False)
    return worst


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "beat_features.csv"

    if cache.exists():
        logging.info("Loading cached features from %s", cache)
        df = pd.read_csv(cache, low_memory=False)
    else:
        stems = sorted(p.with_suffix("").as_posix() for p in DATA_DIR.glob("*.dat"))
        all_rows: List[Dict] = []
        t0 = time.time()
        for i, stem in enumerate(stems, 1):
            rec_name = Path(stem).name
            t1 = time.time()
            rows = extract_record_beats(stem)
            all_rows.extend(rows)
            logging.info("[%d/%d] %s  beats=%d  %.1fs", i, len(stems), rec_name, len(rows), time.time() - t1)
        df = pd.DataFrame(all_rows)
        df.to_csv(cache, index=False)
        logging.info("Extracted %d beats in %.0fs", len(df), time.time() - t0)

    pareto = sweep_strategies(df)
    pareto.to_csv(OUT_DIR / "pareto_strategies.csv", index=False)

    worst = analyze_worst_records(df, pareto)
    worst.to_csv(OUT_DIR / "worst_records.csv", index=False)

    # Summary text
    base = pareto[pareto.strategy == "baseline_zero_shot"].iloc[0]
    lines = [
        "INCART targeted PVC analysis",
        f"beats: {len(df):,}  records: {df.record.nunique()}",
        "",
        f"Baseline: HP={base.healthy_permit:.1%}  AI={base.abnormal_inhibit:.1%}  FP={base.false_permit:.1%}",
        "",
        "Best strategies with AI >= 95%:",
    ]
    hi = pareto[pareto.abnormal_inhibit >= 0.95].sort_values("healthy_permit", ascending=False)
    if hi.empty:
        lines.append("  (none reach 95% AI without large HP cost)")
    else:
        for _, r in hi.head(8).iterrows():
            lines.append(f"  {r.strategy} {r.params}: HP={r.healthy_permit:.1%} AI={r.abnormal_inhibit:.1%}")

    lines += ["", "Best PVC-only (no zscore scale) with AI >= 90%:"]
    pvc = pareto[
        (pareto.abnormal_inhibit >= 0.90)
        & (~pareto.strategy.str.startswith("global_zscore"))
        & (pareto.strategy != "baseline_zero_shot")
    ].sort_values("healthy_permit", ascending=False)
    for _, r in pvc.head(8).iterrows():
        lines.append(f"  {r.strategy} {r.params}: HP={r.healthy_permit:.1%} AI={r.abnormal_inhibit:.1%}")

    lines += ["", "Global zscore at AI ~90-95%:"]
    gz = pareto[pareto.strategy == "global_zscore"].sort_values("abnormal_inhibit")
    for _, r in gz.iterrows():
        if 0.88 <= r.abnormal_inhibit <= 0.96:
            lines.append(f"  {r.params}: HP={r.healthy_permit:.1%} AI={r.abnormal_inhibit:.1%}")

    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
