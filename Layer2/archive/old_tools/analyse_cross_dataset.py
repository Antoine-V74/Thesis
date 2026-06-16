"""
Deep-dive analysis: why is abnormal inhibit low on external datasets?

Compares per-beat feature distributions of:
  healthy beats   (should be permitted, check for overlap)
  abnormal beats  (should be inhibited)

across datasets (mitdb vs nstdb vs svdb), using the scores already stored
in per_beat.csv from run_cross_dataset_benchmark.py.

Outputs
-------
  <out_dir>/score_distributions.csv   -- quantiles of key scores per (dataset, label)
  <out_dir>/hard_rule_rates.csv       -- hard-rule hit rate per (dataset, label, rule)
  <out_dir>/feature_separation.csv    -- AUROC of each score for healthy vs abnormal
  <out_dir>/analysis_report.txt       -- human-readable narrative
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------

def _auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC of (pos_scores > threshold) classifier. Higher = more separable."""
    from itertools import product
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # Mann–Whitney U statistic
    n_p, n_n = len(pos_scores), len(neg_scores)
    pos_scores = np.sort(pos_scores)
    neg_scores = np.sort(neg_scores)
    # Count: for each positive, how many negatives are below it
    u = np.sum(np.searchsorted(neg_scores, pos_scores, side="right"))
    return float(u) / (n_p * n_n)


def run_analysis(per_beat_csv: Path, out_dir: Path, eval_mode: str = "oracle") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {per_beat_csv} ...")
    df = pd.read_csv(per_beat_csv, low_memory=False)

    # Focus on oracle mode, all features, zero_shot
    sub = df[
        (df["eval_mode"] == eval_mode)
        & (df["feature_set"] == "all")
        & (df["benchmark_mode"] == "zero_shot")
    ].copy()

    print(f"  {len(sub)} rows after filter (eval_mode={eval_mode})")
    print(f"  Datasets: {sorted(sub['dataset'].unique())}")
    print(f"  Labels: {sub['label'].value_counts().to_dict()}\n")

    report_lines = []
    report_lines.append("=== Cross-dataset gate analysis ===\n")

    # ── 1. Overall permit rates ──────────────────────────────────────────────
    report_lines.append("--- 1. Overall permit rates (zero_shot, oracle, all features) ---")
    rows = []
    for (ds, lbl), grp in sub.groupby(["dataset", "label"]):
        rows.append({
            "dataset": ds,
            "label": lbl,
            "n_beats": len(grp),
            "permit_rate": round(grp["permit"].mean(), 3),
        })
    perm_df = pd.DataFrame(rows)
    report_lines.append(perm_df.to_string(index=False))
    report_lines.append("")

    # ── 2. Score distributions ───────────────────────────────────────────────
    score_cols = [c for c in ["mahalanobis", "signal_mahal_proxy",
                               "rr_mahal_proxy", "max_zscore"] if c in sub.columns]
    dist_rows = []
    for (ds, lbl), grp in sub.groupby(["dataset", "label"]):
        row = {"dataset": ds, "label": lbl, "n": len(grp)}
        for col in score_cols:
            vals = grp[col].dropna()
            if len(vals):
                row[f"{col}_p25"] = round(float(np.percentile(vals, 25)), 2)
                row[f"{col}_p50"] = round(float(np.percentile(vals, 50)), 2)
                row[f"{col}_p75"] = round(float(np.percentile(vals, 75)), 2)
                row[f"{col}_p90"] = round(float(np.percentile(vals, 90)), 2)
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "score_distributions.csv", index=False)

    report_lines.append("--- 2. Score distributions (median / p90) ---")
    for ds in sorted(sub["dataset"].unique()):
        report_lines.append(f"  [{ds}]")
        for lbl in ["healthy", "abnormal_v"]:
            d = dist_df[(dist_df["dataset"] == ds) & (dist_df["label"] == lbl)]
            if len(d) == 0:
                continue
            d = d.iloc[0]
            parts = []
            for col in score_cols:
                p50 = d.get(f"{col}_p50", float("nan"))
                p90 = d.get(f"{col}_p90", float("nan"))
                parts.append(f"{col}: p50={p50:.1f} p90={p90:.1f}")
            report_lines.append(f"    {lbl:12s}: " + "  |  ".join(parts))
    report_lines.append("")

    # ── 3. AUROC of each score for healthy vs abnormal ───────────────────────
    report_lines.append("--- 3. Score separability (AUROC, higher=better, 0.5=random) ---")
    auroc_rows = []
    for ds in sorted(sub["dataset"].unique()):
        ds_df = sub[sub["dataset"] == ds]
        h_vals = ds_df[ds_df["label"] == "healthy"]
        ab_vals = ds_df[ds_df["label"] == "abnormal_v"]
        if len(ab_vals) == 0:
            continue
        row = {"dataset": ds, "n_healthy": len(h_vals), "n_abnormal": len(ab_vals)}
        for col in score_cols:
            h_s = h_vals[col].dropna().values
            ab_s = ab_vals[col].dropna().values
            row[f"auroc_{col}"] = round(_auroc(ab_s, h_s), 3)
        auroc_rows.append(row)
    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_csv(out_dir / "feature_separation.csv", index=False)

    report_lines.append(auroc_df.to_string(index=False))
    report_lines.append("")

    # ── 4. Hard-rule hit rates ───────────────────────────────────────────────
    if "hard_rule_violated" in sub.columns:
        report_lines.append("--- 4. Hard-rule hit rates per dataset / label ---")
        hr_rows = []
        for (ds, lbl), grp in sub.groupby(["dataset", "label"]):
            n = len(grp)
            hr = grp[grp["inhibit_class"] == "hard_rule"]
            # Count which rules fire
            rule_counts = hr["hard_rule_violated"].value_counts()
            for rule, cnt in rule_counts.items():
                hr_rows.append({
                    "dataset": ds,
                    "label": lbl,
                    "rule": rule,
                    "n_fired": cnt,
                    "rate": round(cnt / n, 4),
                })
        if hr_rows:
            hr_df = pd.DataFrame(hr_rows).sort_values(
                ["dataset", "label", "rate"], ascending=[True, True, False]
            )
            hr_df.to_csv(out_dir / "hard_rule_rates.csv", index=False)
            report_lines.append(hr_df.to_string(index=False))
        report_lines.append("")

    # ── 5. Inhibit class breakdown ───────────────────────────────────────────
    report_lines.append("--- 5. Inhibit class breakdown (abnormal beats only) ---")
    ab_only = sub[sub["label"] == "abnormal_v"]
    inh_only = ab_only[ab_only["permit"] == 0]
    for ds in sorted(ab_only["dataset"].unique()):
        n_total = len(ab_only[ab_only["dataset"] == ds])
        n_inh = len(inh_only[inh_only["dataset"] == ds])
        ds_inh = inh_only[inh_only["dataset"] == ds]
        report_lines.append(f"  [{ds}] {n_inh}/{n_total} inhibited ({n_inh/max(n_total,1):.1%})")
        if "inhibit_class" in ds_inh.columns:
            for cls, cnt in ds_inh["inhibit_class"].value_counts().items():
                report_lines.append(f"    {cls:20s}: {cnt:5d}  ({cnt/max(n_total,1):.1%})")
    report_lines.append("")

    # ── 6. Score threshold comparison ───────────────────────────────────────
    report_lines.append("--- 6. Why abnormal beats pass (zero-shot threshold mismatch) ---")
    if "mahalanobis" in sub.columns and "mahalanobis_threshold" in sub.columns:
        for ds in sorted(sub["dataset"].unique()):
            ab_ds = sub[(sub["dataset"] == ds) & (sub["label"] == "abnormal_v")]
            if len(ab_ds) == 0:
                continue
            passed = ab_ds[ab_ds["permit"] == 1]
            # median threshold vs median score for abnormal beats
            thr_med = ab_ds["mahalanobis_threshold"].dropna().median()
            score_med = ab_ds["mahalanobis"].dropna().median()
            passed_score_med = passed["mahalanobis"].dropna().median() if len(passed) else float("nan")
            report_lines.append(
                f"  [{ds}] median Mahal threshold={thr_med:.1f}  "
                f"median abnormal score={score_med:.1f}  "
                f"median score of PASSED abnormals={passed_score_med:.1f}"
            )
    report_lines.append("")

    # ── 7. NSTDB by SNR ─────────────────────────────────────────────────────
    nstdb_rows = sub[sub["dataset"] == "nstdb"].copy()
    if len(nstdb_rows):
        NSTDB_SNR = {
            "118e_6": -6, "119e_6": -6, "118e00": 0, "119e00": 0,
            "118e06": 6, "119e06": 6, "118e12": 12, "119e12": 12,
            "118e18": 18, "119e18": 18, "118e24": 24, "119e24": 24,
        }
        nstdb_rows["snr_db"] = nstdb_rows["record"].map(NSTDB_SNR)
        report_lines.append("--- 7. NSTDB abnormal inhibit by SNR (lower SNR = more noise) ---")
        snr_rows = []
        for (snr, lbl), grp in nstdb_rows.groupby(["snr_db", "label"]):
            if lbl not in ("healthy", "abnormal_v"):
                continue
            snr_rows.append({
                "snr_db": snr,
                "label": lbl,
                "n": len(grp),
                "permit_rate": round(grp["permit"].mean(), 3),
                "inhibit_rate": round(1 - grp["permit"].mean(), 3),
                "mahal_p50": round(grp["mahalanobis"].median(), 1),
            })
        snr_df = pd.DataFrame(snr_rows).sort_values(["snr_db", "label"])
        report_lines.append(snr_df.to_string(index=False))
        report_lines.append("")

    # ── 8. Key diagnostic: what fraction of abnormals have LOW mahal? ────────
    report_lines.append("--- 8. Abnormal beats that look 'normal' (mahal < median healthy mahal) ---")
    for ds in sorted(sub["dataset"].unique()):
        h_ds = sub[(sub["dataset"] == ds) & (sub["label"] == "healthy")]
        ab_ds = sub[(sub["dataset"] == ds) & (sub["label"] == "abnormal_v")]
        if len(h_ds) == 0 or len(ab_ds) == 0:
            continue
        h_med = h_ds["mahalanobis"].dropna().median()
        n_ab_below = (ab_ds["mahalanobis"].dropna() < h_med).sum()
        n_ab = len(ab_ds["mahalanobis"].dropna())
        report_lines.append(
            f"  [{ds}] {n_ab_below}/{n_ab} = {n_ab_below/max(n_ab,1):.1%} of abnormals "
            f"score BELOW median healthy mahal ({h_med:.1f})"
        )
    report_lines.append("")

    # ── Write report ─────────────────────────────────────────────────────────
    report_txt = "\n".join(report_lines)
    (out_dir / "analysis_report.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)
    print(f"\nWrote analysis to {out_dir}")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-beat", type=Path,
                   default=Path("Results/cross_dataset_full/per_beat.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("Results/cross_dataset_full/analysis"))
    p.add_argument("--eval-mode", default="oracle",
                   choices=["oracle", "layer1_adaptive_gated"])
    args = p.parse_args()
    run_analysis(args.per_beat, args.out_dir, args.eval_mode)


if __name__ == "__main__":
    main()
