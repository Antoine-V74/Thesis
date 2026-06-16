#!/usr/bin/env python3
"""
Compare existing Layer 2 beat-sync decisions with Layer 3 beat-sync decisions.

The combined safety rule is a veto AND:
    combined permits only if Layer 2 permits AND Layer 3 permits.
Everything else inhibits.

This script does NOT command stimulation. It only summarizes offline decisions
from the two upstream gate validation outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from layer3_validation_utils import (  # noqa: E402
    SAFETY_DISCLAIMER,
    decision_metrics,
    get_logger,
)

LOG = get_logger("compare_layer2_layer3")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_first_csv(directory: str | Path, preferred: List[str]) -> Optional[Path]:
    d = Path(directory)
    if not d.exists():
        return None
    for name in preferred:
        p = d / name
        if p.exists():
            return p
    csvs = sorted(d.glob("*.csv"))
    return csvs[0] if csvs else None


# ---------------------------------------------------------------------------
# Decision and label normalization
# ---------------------------------------------------------------------------

PERMIT_TOKENS = {"permit", "permitted", "allow", "allowed", "true", "1", "accept", "accepted"}


def _coerce_decision_column(s: pd.Series) -> pd.Series:
    """Coerce an arbitrary decision-like column into 'permit' / 'inhibit'."""
    if s.dtype == bool:
        return pd.Series(np.where(s, "permit", "inhibit"), index=s.index)
    text = s.astype(str).str.strip().str.lower()
    # Any non-permit token (including 'calibration_no_stim', 'unknown', NaN, blanks)
    # becomes 'inhibit' for combined safety. This is intentionally conservative.
    return pd.Series(np.where(text.isin(PERMIT_TOKENS), "permit", "inhibit"), index=s.index)


def normalize_decision_series(df: pd.DataFrame, layer: str) -> pd.Series:
    candidates = [
        "decision",
        f"{layer}_decision",
        "final_decision",
        "gate_decision",
        "permit",
        "is_permitted",
        "permitted",
        "allow_stim",
    ]
    for col in candidates:
        if col in df.columns:
            return _coerce_decision_column(df[col])
    raise RuntimeError(
        f"Could not infer a decision column for {layer}. Available columns: {list(df.columns)}"
    )


def infer_merge_keys(layer2: pd.DataFrame, layer3: pd.DataFrame) -> List[str]:
    candidates = [
        ["dataset", "record", "beat_sample"],
        ["dataset", "record", "sample"],
        ["record", "beat_sample"],
        ["record", "sample"],
        ["record_key", "beat_sample"],
        ["record_key", "sample"],
        ["beat_id"],
        ["window_id"],
    ]
    for keys in candidates:
        if all(k in layer2.columns for k in keys) and all(k in layer3.columns for k in keys):
            return keys
    raise RuntimeError(
        "Could not infer merge keys. Need common columns such as dataset, record, beat_sample. "
        f"Layer2 columns={list(layer2.columns)}; Layer3 columns={list(layer3.columns)}"
    )


def infer_nearest_merge_components(layer2: pd.DataFrame, layer3: pd.DataFrame) -> Tuple[List[str], str, str]:
    """Infer record grouping columns and sample columns for tolerant beat matching."""
    by_candidates = [
        ["dataset", "record"],
        ["record_key"],
        ["record"],
    ]
    sample_candidates = ["beat_sample", "sample"]
    for by in by_candidates:
        if not all(c in layer2.columns and c in layer3.columns for c in by):
            continue
        for l2_sample in sample_candidates:
            if l2_sample not in layer2.columns:
                continue
            for l3_sample in sample_candidates:
                if l3_sample in layer3.columns:
                    return by, l2_sample, l3_sample
    raise RuntimeError(
        "Nearest-neighbor merge needs common record identity columns and sample columns "
        "(for example dataset, record, beat_sample). "
        f"Layer2 columns={list(layer2.columns)}; Layer3 columns={list(layer3.columns)}"
    )


def infer_sampling_rate(layer2: pd.DataFrame, layer3: pd.DataFrame) -> float:
    for df in (layer3, layer2):
        for col in ["fs", "sampling_rate", "sample_rate"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if not vals.empty:
                    return float(vals.median())
    raise RuntimeError(
        "--merge-tolerance-s requires an `fs`, `sampling_rate`, or `sample_rate` column. "
        "Use --merge-tolerance-samples if sampling rate is not available."
    )


def nearest_neighbor_merge(
    layer2: pd.DataFrame,
    layer3: pd.DataFrame,
    tolerance_samples: int,
    l2_extras: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    by, l2_sample_col, l3_sample_col = infer_nearest_merge_components(layer2, layer3)
    left = layer3.copy()
    right_cols = by + [l2_sample_col, "layer2_decision"] + l2_extras
    right = layer2[right_cols].copy()

    left["__merge_sample"] = pd.to_numeric(left[l3_sample_col], errors="coerce")
    right["__merge_sample"] = pd.to_numeric(right[l2_sample_col], errors="coerce")
    right["layer2_match_sample"] = right["__merge_sample"]
    left = left.dropna(subset=["__merge_sample"]).copy()
    right = right.dropna(subset=["__merge_sample"]).copy()
    left["__merge_sample"] = left["__merge_sample"].astype("int64")
    right["__merge_sample"] = right["__merge_sample"].astype("int64")

    merged_parts = []
    for key, left_g in left.groupby(by, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        right_g = right
        for col, value in zip(by, key):
            right_g = right_g[right_g[col].eq(value)]
        if right_g.empty:
            continue
        left_g = left_g.sort_values("__merge_sample").reset_index(drop=True)
        right_g = right_g.sort_values("__merge_sample").reset_index(drop=True)
        merged_parts.append(
            pd.merge_asof(
                left_g,
                right_g,
                on="__merge_sample",
                direction="nearest",
                tolerance=int(tolerance_samples),
                suffixes=("_layer3", "_layer2"),
            )
        )
    if merged_parts:
        merged = pd.concat(merged_parts, ignore_index=True)
    else:
        merged = pd.DataFrame()
    merged = merged[merged["layer2_decision"].notna()].copy()
    merged["merge_abs_sample_delta"] = (merged["__merge_sample"] - merged["layer2_match_sample"]).abs()
    merged["merge_tolerance_samples"] = int(tolerance_samples)
    merged["merge_method"] = "nearest_neighbor"
    return merged, by + [f"{l3_sample_col}~{l2_sample_col}"]


def infer_healthy_suffix_aware(df: pd.DataFrame) -> pd.Series:
    """Find a healthy/abnormal label column in df, handling pandas merge suffixes.

    After merging Layer 2 and Layer 3 with suffixes=('_layer3', '_layer2'), shared
    columns like 'is_healthy_beat' or 'beat_symbol' end up as 'is_healthy_beat_layer3'
    and 'is_healthy_beat_layer2'. We prefer the Layer 3 side (the left frame in the
    merge) but accept either, and fall back to inferring from beat-symbol columns.
    """
    bool_candidates = [
        "is_healthy",
        "is_healthy_beat",
        "is_healthy_window",
    ]
    suffixes = ["", "_layer3", "_layer2"]
    for base in bool_candidates:
        for suffix in suffixes:
            col = f"{base}{suffix}"
            if col in df.columns:
                return df[col].astype(bool)

    label_candidates = ["beat_symbol", "dominant_label", "label"]
    for base in label_candidates:
        for suffix in suffixes:
            col = f"{base}{suffix}"
            if col in df.columns:
                # Conservative: only the literal 'N' (normal sinus beat) counts as healthy.
                return df[col].astype(str).str.strip().eq("N")

    raise RuntimeError(
        "Could not infer healthy/abnormal labels from merged table. "
        f"Available columns: {list(df.columns)}"
    )


def summarize_decisions(df: pd.DataFrame, decision_col: str, name: str) -> Dict[str, object]:
    tmp = df.copy()
    tmp["decision_eval"] = tmp[decision_col]
    tmp["is_healthy_eval"] = infer_healthy_suffix_aware(tmp)
    row = {"method": name}
    row.update(decision_metrics(tmp, decision_col="decision_eval", healthy_col="is_healthy_eval"))
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare Layer 2 and Layer 3 beat-sync validation outputs."
    )
    p.add_argument("--layer2-dir", required=True)
    p.add_argument("--layer3-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer2-csv", default=None, help="Optional explicit Layer 2 CSV path")
    p.add_argument("--layer3-csv", default=None, help="Optional explicit Layer 3 CSV path")
    p.add_argument("--merge-tolerance-s", type=float, default=0.0,
                   help="Optional nearest-neighbor beat merge tolerance in seconds. Default 0 uses exact-key merge.")
    p.add_argument("--merge-tolerance-samples", type=int, default=None,
                   help="Optional nearest-neighbor beat merge tolerance in samples. Overrides --merge-tolerance-s.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer2_csv = (
        Path(args.layer2_csv)
        if args.layer2_csv
        else find_first_csv(args.layer2_dir, ["per_beat.csv", "beat_sync_decisions.csv", "decisions.csv", "per_window.csv"])
    )
    layer3_csv = (
        Path(args.layer3_csv)
        if args.layer3_csv
        else find_first_csv(args.layer3_dir, ["per_beat.csv", "per_window.csv", "embedding_scores.csv"])
    )
    if layer2_csv is None or not layer2_csv.exists():
        raise RuntimeError(f"Could not find Layer 2 CSV in {args.layer2_dir}")
    if layer3_csv is None or not layer3_csv.exists():
        raise RuntimeError(f"Could not find Layer 3 CSV in {args.layer3_dir}")

    LOG.info("Layer 2 CSV: %s", layer2_csv)
    LOG.info("Layer 3 CSV: %s", layer3_csv)

    l2 = pd.read_csv(layer2_csv)
    l3 = pd.read_csv(layer3_csv)
    LOG.info("Loaded Layer 2 (%d rows) and Layer 3 (%d rows)", len(l2), len(l3))

    l2["layer2_decision"] = normalize_decision_series(l2, "layer2")
    l3["layer3_decision"] = normalize_decision_series(l3, "layer3")

    # Carry through useful Layer 2 reason/diagnostic columns where available so
    # the combined per-beat CSV is debuggable.
    l2_extras = [c for c in ["layer2_reason", "reason", "diagnostic", "mode"] if c in l2.columns]
    tolerance_samples = args.merge_tolerance_samples
    if tolerance_samples is None and float(args.merge_tolerance_s) > 0.0:
        tolerance_samples = int(round(float(args.merge_tolerance_s) * infer_sampling_rate(l2, l3)))

    if tolerance_samples is not None and int(tolerance_samples) > 0:
        merged, keys = nearest_neighbor_merge(l2, l3, int(tolerance_samples), l2_extras)
        merge_method = "nearest_neighbor"
        LOG.info("Nearest-neighbor merge on %s with tolerance %d samples", keys, int(tolerance_samples))
    else:
        keys = infer_merge_keys(l2, l3)
        LOG.info("Exact merge on keys: %s", keys)
        merged = l3.merge(
            l2[keys + ["layer2_decision"] + l2_extras],
            on=keys,
            how="inner",
            suffixes=("_layer3", "_layer2"),
        )
        merged["merge_method"] = "exact"
        merge_method = "exact"
    if merged.empty:
        raise RuntimeError(f"Merge on keys {keys} produced zero rows")
    LOG.info("Merged rows: %d", len(merged))

    # Combined veto rule: permit only if BOTH layers permit. Anything else inhibits.
    merged["combined_layer2_and_layer3_veto_decision"] = np.where(
        (merged["layer2_decision"] == "permit") & (merged["layer3_decision"] == "permit"),
        "permit",
        "inhibit",
    )

    # Resolve a single is_healthy column once on the merged frame.
    merged["is_healthy"] = infer_healthy_suffix_aware(merged)
    merged["is_abnormal"] = ~merged["is_healthy"]

    rows = [
        summarize_decisions(merged, "layer2_decision", "layer2_only"),
        summarize_decisions(merged, "layer3_decision", "layer3_mahal_only"),
        summarize_decisions(merged, "combined_layer2_and_layer3_veto_decision", "layer2_and_layer3_veto"),
    ]
    comparison = pd.DataFrame(rows)

    # Outputs
    merged.to_csv(out_dir / "combined_per_beat.csv", index=False)
    comparison.to_csv(out_dir / "comparison_layer2_layer3.csv", index=False)

    # Short headline table for the thesis: one row per method, only the four
    # safety-gate rates plus n. Easier to drop into a thesis table.
    headline_cols = [
        "method",
        "n",
        "healthy_n",
        "abnormal_n",
        "healthy_permit_rate",
        "healthy_false_inhibit_rate",
        "abnormal_inhibit_rate",
        "abnormal_false_permit_rate",
    ]
    available = [c for c in headline_cols if c in comparison.columns]
    comparison[available].to_csv(out_dir / "final_comparison_table.csv", index=False)

    false_permits = merged[
        (merged["is_abnormal"])
        & (merged["combined_layer2_and_layer3_veto_decision"] == "permit")
    ].copy()
    false_permits["reason"] = "combined_layer2_and_layer3_permitted_abnormal_beat"
    false_permits.to_csv(out_dir / "false_permits_detail.csv", index=False)

    # Markdown summary
    with (out_dir / "FINAL_COMPARISON_SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("# Layer 2 vs Layer 3 comparison\n\n")
        f.write(SAFETY_DISCLAIMER + "\n\n")
        f.write(f"- Layer 2 CSV: `{layer2_csv}`\n")
        f.write(f"- Layer 3 CSV: `{layer3_csv}`\n")
        f.write(f"- Merge method: `{merge_method}`\n")
        f.write(f"- Merge keys: `{keys}`\n")
        if merge_method == "nearest_neighbor":
            f.write(f"- Merge tolerance: `{int(tolerance_samples)}` samples")
            if float(args.merge_tolerance_s) > 0:
                f.write(f" (`{float(args.merge_tolerance_s):.3f}` s requested)")
            f.write("\n")
        else:
            f.write("- Exact-key merge assumes Layer 2 and Layer 3 use identical beat/sample positions. "
                    "If Layer 2 uses accepted Layer 1 triggers with jitter, rerun with "
                    "`--merge-tolerance-s 0.10` or `--merge-tolerance-samples`.\n")
        f.write(f"- Merged rows: {len(merged)}\n")
        f.write(f"- Healthy beats: {int(merged['is_healthy'].sum())}\n")
        f.write(f"- Abnormal beats: {int(merged['is_abnormal'].sum())}\n\n")
        f.write("## Combined rule\n\n")
        f.write("`permit` only if Layer 2 permits AND Layer 3 permits. Otherwise `inhibit`.\n")
        f.write("Any decision token other than `permit` (including `calibration_no_stim`, NaN, "
                "or unknown) is treated as `inhibit` for combined-safety reporting.\n\n")
        f.write("## Results\n\n")
        f.write("See `final_comparison_table.csv` and `comparison_layer2_layer3.csv`.\n\n")
        # Inline a quick markdown table for the headline rates.
        f.write("| method | n | healthy_permit | false_inhibit | abnormal_inhibit | false_permit |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in comparison.iterrows():
            def _fmt(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)
            f.write(
                f"| {r.get('method','')} | {int(r.get('n',0))} | "
                f"{_fmt(r.get('healthy_permit_rate'))} | "
                f"{_fmt(r.get('healthy_false_inhibit_rate'))} | "
                f"{_fmt(r.get('abnormal_inhibit_rate'))} | "
                f"{_fmt(r.get('abnormal_false_permit_rate'))} |\n"
            )
        f.write("\n## Caveats\n\n")
        f.write("- Human MIT-BIH validation is proxy validation only. Animal deployment requires "
                "prospective per-session animal calibration and validation.\n")
        f.write("- Runtime systems must NOT use oracle annotations to gate stimulation.\n")
        f.write("- Calibration windows never trigger stimulation, regardless of model output.\n")

    LOG.info("Wrote comparison outputs to %s", out_dir)
    print(f"[DONE] wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
