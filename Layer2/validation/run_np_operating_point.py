"""
Neyman-Pearson operating-point selection and worst-record report (OFFLINE).

Purpose
-------
Given an already-scored ``per_beat.csv`` (from run_beat_validation.py), pick the
Layer 2 primary-distance operating point using a Neyman-Pearson rule and report
the worst per-record dangerous false-permit rate at that point.

Neyman-Pearson framing (why)
----------------------------
The deployable Layer 2 threshold is set label-free by conformal calibration on
HEALTHY baseline windows only (see calibration.py). It answers "how far is this
window from the healthy baseline?" but it does not, by itself, know how much
danger leaks through. This script is the OFFLINE, LABEL-AWARE audit that closes
that loop: using the danger labels that exist in public datasets (MIT-BIH etc.)
it measures the achievable trade-off and selects the operating point that:

    minimise   normal false-inhibit rate      (Type II, comfort / therapy uptime)
    subject to danger false-permit rate <= budget   (Type I, the safety metric)

This is the classical Neyman-Pearson test: bound the dangerous error, then be as
permissive as possible under that bound. It is a DESIGN / REPORTING tool, not a
runtime calibrator: deployment on animals has no danger labels, so the chosen
conformal-alpha (or equivalent distance threshold) must be transferred, and this
script quantifies what that choice costs in danger leakage on labeled data.

Worst-record reporting
-----------------------
A good pooled false-permit rate can still hide one catastrophic record. We
therefore also report the per-record dangerous permit rate at the selected
operating point and flag the worst record. Judge Layer 2 by worst-record danger
leakage, not the mean.

Inputs
------
per_beat.csv columns used (produced by run_beat_validation.py):
    primary_distance (or mahalanobis), record, split, safety_group / is_healthy,
    hard_rule_violated, feature_set, mode.

Usage
-----
    python Layer2/validation/run_np_operating_point.py \
        --per-beat Results/layer2_beat_validation/per_beat.csv \
        --out-dir  Results/layer2_beat_validation/np_operating_point \
        --danger-budget 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _score_column(df: pd.DataFrame) -> str:
    """Prefer primary_distance; fall back to mahalanobis."""
    for col in ("primary_distance", "mahalanobis"):
        if col in df.columns and df[col].notna().any():
            return col
    raise SystemExit(
        "per_beat.csv has neither 'primary_distance' nor 'mahalanobis' with values."
    )


def _danger_mask(df: pd.DataFrame, danger_groups: List[str]) -> pd.Series:
    if "safety_group" in df.columns:
        return df["safety_group"].astype(str).str.upper().isin(
            [g.upper() for g in danger_groups]
        )
    # Legacy fallback: treat labeled ventricular/noise abnormals as danger.
    if "label" in df.columns:
        return df["label"].astype(str).isin(["abnormal_v", "abnormal_noise"])
    return ~df.get("is_healthy", pd.Series(False, index=df.index)).astype(bool)


def _normal_mask(df: pd.DataFrame) -> pd.Series:
    if "safety_group" in df.columns:
        return df["safety_group"].astype(str).str.upper().eq("NORMAL")
    if "is_healthy" in df.columns:
        return df["is_healthy"].astype(bool)
    if "label" in df.columns:
        return df["label"].astype(str).eq("healthy")
    raise SystemExit("Cannot identify NORMAL beats (no safety_group / is_healthy / label).")


def _hard_inhibit(df: pd.DataFrame) -> pd.Series:
    """Beats vetoed by a hard rule fire regardless of the distance threshold."""
    if "hard_rule_violated" not in df.columns:
        return pd.Series(False, index=df.index)
    hrv = df["hard_rule_violated"].fillna("").astype(str)
    return ~hrv.eq("") & ~hrv.str.lower().isin(["nan", "none"])


def build_frontier(
    df: pd.DataFrame, score_col: str, danger_groups: List[str], n_grid: int
) -> pd.DataFrame:
    """Sweep the primary-distance threshold and tabulate the danger/normal trade-off."""
    danger = _danger_mask(df, danger_groups)
    normal = _normal_mask(df)
    hard = _hard_inhibit(df)
    score = df[score_col].astype(float)

    n_danger = int(danger.sum())
    n_normal = int(normal.sum())
    if n_danger == 0 or n_normal == 0:
        raise SystemExit(
            f"Need both danger and normal beats; got danger={n_danger}, normal={n_normal}."
        )

    finite = score[np.isfinite(score)]
    lo, hi = float(finite.min()), float(finite.max())
    grid = np.unique(
        np.concatenate([
            np.linspace(lo, hi, n_grid),
            np.quantile(finite, np.linspace(0.0, 1.0, min(n_grid, len(finite)))),
        ])
    )

    rows = []
    for thr in grid:
        # permit iff not hard-vetoed AND distance within threshold.
        permit = (~hard) & (score <= thr)
        danger_fp = float(permit[danger].mean())
        normal_fi = float((~permit[normal]).mean())
        rows.append({
            "threshold": float(thr),
            "danger_false_permit": round(danger_fp, 5),
            "normal_false_inhibit": round(normal_fi, 5),
            "normal_permit": round(float(permit[normal].mean()), 5),
            "n_danger": n_danger,
            "n_normal": n_normal,
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_np_operating_point(frontier: pd.DataFrame, danger_budget: float) -> dict:
    """
    Neyman-Pearson: among thresholds meeting the danger false-permit budget,
    choose the one with the lowest normal false-inhibit (most permissive-safe).
    """
    feasible = frontier[frontier["danger_false_permit"] <= float(danger_budget)]
    if len(feasible) == 0:
        # Budget infeasible: fail safe to the most conservative point.
        best = frontier.iloc[frontier["danger_false_permit"].idxmin()]
        status = "budget_infeasible_fail_safe"
    else:
        best = feasible.iloc[feasible["normal_false_inhibit"].idxmin()]
        status = "ok"
    return {
        "status": status,
        "danger_budget": float(danger_budget),
        "selected_threshold": float(best["threshold"]),
        "danger_false_permit": float(best["danger_false_permit"]),
        "normal_false_inhibit": float(best["normal_false_inhibit"]),
        "normal_permit": float(best["normal_permit"]),
    }


def worst_record_report(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    danger_groups: List[str],
    record_col: str = "record",
) -> pd.DataFrame:
    """Per-record dangerous permit (leak) rate at the selected threshold."""
    danger = _danger_mask(df, danger_groups)
    hard = _hard_inhibit(df)
    score = df[score_col].astype(float)
    permit = (~hard) & (score <= threshold)

    dd = df[danger].copy()
    dd["_permit"] = permit[danger].values
    rows = []
    for rec, grp in dd.groupby(record_col):
        n = int(len(grp))
        leaked = int(grp["_permit"].sum())
        rows.append({
            record_col: str(rec),
            "n_danger_beats": n,
            "danger_permitted": leaked,
            "danger_false_permit_rate": round(leaked / n, 5) if n else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values("danger_false_permit_rate", ascending=False)
    return out.reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-beat", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", default=None, help="filter per_beat.csv by mode (optional)")
    p.add_argument("--feature-set", default=None, help="filter by feature_set (optional)")
    p.add_argument("--split", default="test", help="split to evaluate (default: test)")
    p.add_argument("--danger-groups", nargs="+", default=["DANGEROUS"],
                   help="safety_group values counted as dangerous (default: DANGEROUS)")
    p.add_argument("--danger-budget", type=float, default=0.01,
                   help="max tolerated dangerous false-permit rate (Type I bound)")
    p.add_argument("--n-grid", type=int, default=200)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.per_beat, low_memory=False)

    if args.split and "split" in df.columns:
        df = df[df["split"] == args.split].copy()
    if args.mode and "mode" in df.columns:
        df = df[df["mode"] == args.mode].copy()
    if args.feature_set and "feature_set" in df.columns:
        df = df[df["feature_set"] == args.feature_set].copy()
    if len(df) == 0:
        raise SystemExit("No rows left after filtering; check --split/--mode/--feature-set.")

    score_col = _score_column(df)
    frontier = build_frontier(df, score_col, args.danger_groups, args.n_grid)
    op = select_np_operating_point(frontier, args.danger_budget)
    worst = worst_record_report(df, score_col, op["selected_threshold"], args.danger_groups)

    frontier.to_csv(args.out_dir / "np_frontier.csv", index=False)
    pd.DataFrame([op]).to_csv(args.out_dir / "np_operating_point.csv", index=False)
    worst.to_csv(args.out_dir / "worst_record_danger.csv", index=False)

    print(f"score column          : {score_col}")
    print(f"NP status             : {op['status']}")
    print(f"danger budget (Type I): {op['danger_budget']:.4f}")
    print(f"selected threshold    : {op['selected_threshold']:.4f}")
    print(f"danger false-permit   : {op['danger_false_permit']:.4f}")
    print(f"normal false-inhibit  : {op['normal_false_inhibit']:.4f}")
    if len(worst):
        w = worst.iloc[0]
        print(f"worst record          : {w['record']} "
              f"({w['danger_false_permit_rate']:.4f} over {int(w['n_danger_beats'])} danger beats)")
    print(f"Wrote: {args.out_dir/'np_frontier.csv'}, np_operating_point.csv, worst_record_danger.csv")


if __name__ == "__main__":
    main()
