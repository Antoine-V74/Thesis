"""
Post-hoc Pareto sweep on already-extracted per_beat.csv scores.

Runs in seconds (no feature re-extraction).

For each combination of:
  - mahalanobis threshold scale (multiplier on the calibrated per-record threshold)
  - signal_proxy threshold scale
  - whether to skip morph__* hard-rule violations

Reports healthy_permit vs abnormal_inhibit Pareto curve and writes
  <out_dir>/pareto_posthoc.csv

Usage
-----
    .venv\\Scripts\\python Layer2\\pareto_sweep_posthoc.py `
        --per-beat Results/final_mitbih_validation/beat_sync/per_beat.csv `
        --out-dir  Results/final_mitbih_validation/pareto_posthoc `
        --mode layer1_adaptive_gated
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _apply_thresholds(
    df: pd.DataFrame,
    mahal_scale: float,
    sig_scale: float,
    ignore_morph_hard_rules: bool,
    max_zscore_scale: float,
) -> pd.Series:
    """
    Re-decide permit/inhibit for each beat using scaled thresholds.

    Returns a boolean Series (True = permit).
    """
    # Start from "permit" for every beat
    permit = pd.Series(True, index=df.index)

    # Hard rule inhibits that are NOT morph-related → always keep
    if "hard_rule_violated" in df.columns:
        hrv = df["hard_rule_violated"].fillna("").astype(str)
        is_morph_hard = hrv.str.startswith("morph__")
        is_other_hard = hrv.str.startswith(("rr__", "signal__")) & ~hrv.eq("")
        if ignore_morph_hard_rules:
            permit &= ~is_other_hard
        else:
            permit &= (hrv == "")

    # Mahalanobis threshold: mahalanobis > threshold_calibrated * mahal_scale → inhibit
    if "mahalanobis" in df.columns and "mahalanobis_threshold" in df.columns:
        effective_thr = df["mahalanobis_threshold"] * mahal_scale
        permit &= ~(df["mahalanobis"].fillna(0) > effective_thr)

    # Signal proxy threshold
    if "signal_mahal_proxy" in df.columns:
        if (df["signal_mahal_proxy"] > 0).any():
            # Estimate per-record proxy threshold from calibrated mahal ratio
            # Use signal/mahal ratio from the calibration
            # Fallback: use 95th pct of healthy signal proxy as threshold, scaled
            h_proxy = df.loc[df["label"] == "healthy", "signal_mahal_proxy"].dropna()
            if len(h_proxy):
                proxy_thr = float(np.quantile(h_proxy, 0.999)) * sig_scale
                permit &= ~(df["signal_mahal_proxy"].fillna(0) > proxy_thr)

    # Max zscore
    if "max_zscore" in df.columns and "zscore_threshold" in df.columns:
        effective_z = df["zscore_threshold"] * max_zscore_scale
        permit &= ~(df["max_zscore"].fillna(0) > effective_z)

    return permit


def sweep(
    df: pd.DataFrame,
    mahal_scales: list,
    sig_scales: list,
    zscore_scales: list,
    ignore_morph: bool,
) -> pd.DataFrame:
    rows = []
    for ms in mahal_scales:
        for ss in sig_scales:
            for zs in zscore_scales:
                permit = _apply_thresholds(df, ms, ss, ignore_morph, zs)
                h = df[df["label"] == "healthy"]
                ab = df[df["label"].isin(["abnormal_v", "abnormal_noise"])]
                hp = permit[h.index].mean() if len(h) else float("nan")
                ai = (~permit[ab.index]).mean() if len(ab) else float("nan")
                fp = permit[ab.index].mean() if len(ab) else float("nan")
                fi = (~permit[h.index]).mean() if len(h) else float("nan")
                rows.append({
                    "mahal_scale": ms,
                    "sig_scale": ss,
                    "zscore_scale": zs,
                    "ignore_morph_hard_rules": ignore_morph,
                    "healthy_permit": round(float(hp), 4),
                    "abnormal_inhibit": round(float(ai), 4),
                    "false_permit": round(float(fp), 4),
                    "false_inhibit": round(float(fi), 4),
                    "meets_95": float(ai) >= 0.95 if not np.isnan(ai) else False,
                    "meets_82hp": float(hp) >= 0.82 if not np.isnan(hp) else False,
                })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--per-beat",
        type=Path,
        default=Path("Results/final_mitbih_validation/beat_sync/per_beat.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Results/final_mitbih_validation/pareto_posthoc"),
    )
    p.add_argument("--mode", default="layer1_adaptive_gated")
    p.add_argument("--feature-set", default="all")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.per_beat} ...")
    df = pd.read_csv(args.per_beat, low_memory=False)
    df = df[(df["mode"] == args.mode) & (df["feature_set"] == args.feature_set)].copy()
    print(f"  {len(df)} rows, labels: {df['label'].value_counts().to_dict()}")

    # Check columns available
    has_mahal = "mahalanobis" in df.columns and "mahalanobis_threshold" in df.columns
    has_sig = "signal_mahal_proxy" in df.columns
    has_z = "max_zscore" in df.columns and "zscore_threshold" in df.columns
    has_hr = "hard_rule_violated" in df.columns
    print(f"  has_mahal={has_mahal}, has_sig={has_sig}, has_z={has_z}, has_hr={has_hr}")

    # Sweep
    mahal_scales = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    sig_scales   = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    zscore_scales = [1.0]

    print(f"Running sweep: {len(mahal_scales)} x {len(sig_scales)} x 2 (morph ignore) = "
          f"{len(mahal_scales)*len(sig_scales)*2} combinations ...")

    results = []
    for ignore_morph in (False, True):
        batch = sweep(df, mahal_scales, sig_scales, zscore_scales, ignore_morph)
        results.append(batch)

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["meets_95", "meets_82hp", "healthy_permit"], ascending=[False, False, False])
    out.to_csv(args.out_dir / "pareto_posthoc.csv", index=False)
    print(f"\nWrote: {args.out_dir / 'pareto_posthoc.csv'}")

    # Show Pareto frontier (abnormal ≥95%, sort by healthy permit)
    target = out[(out["abnormal_inhibit"] >= 0.95) & (out["false_permit"] <= 0.05)]
    if len(target):
        best = target.sort_values("healthy_permit", ascending=False)
        print("\n=== Operating points meeting >=95% abnormal inhibit AND <=5% false permit ===")
        print(best.head(10).to_string(index=False))
    else:
        # Relax to ≥93%
        target93 = out[out["abnormal_inhibit"] >= 0.93].sort_values("healthy_permit", ascending=False)
        print("\n=== No point meets both targets. Best >=93% abnormal inhibit ===")
        print(target93.head(10).to_string(index=False))

    # Show full curve (Pareto frontier): for each abnormal_inhibit bucket, best healthy permit
    print("\n=== Pareto curve (best healthy permit per abnormal-inhibit bin) ===")
    out["abn_bin"] = (out["abnormal_inhibit"] * 20).round() / 20
    curve = out.groupby("abn_bin", as_index=False)["healthy_permit"].max()
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
