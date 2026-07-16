#!/usr/bin/env python3
"""
Go/no-go analysis: how many within-subject sinus -> danger transition records
exist across the downloaded datasets.

Why
---
The Layer 3 anomaly veto is *personalized*: calibrate on this subject's healthy
baseline, then dangerous rhythm may emerge in the SAME subject. The only
deployment-faithful test of "personalized danger detection" therefore needs
records that contain BOTH:

    (1) a sufficient healthy baseline segment occurring BEFORE
    (2) a dangerous span, in the same record.

A pure-VF record with no sinus segment cannot test that question (no within-
subject baseline to personalize against) and is out-of-distribution for the use
case. This tool counts the eligible records and the scoreable DANGEROUS beats in
them, so the thesis claim can be framed honestly (and so we know whether the
DANGEROUS evaluation is statistically powered).

This script is READ-ONLY. It reads WFDB annotations only (no waveforms), makes
no safety decisions, and writes a per-record table + per-dataset summary.

Usage
-----
    C:\\Users\\antoi\\.venvs\\ecg\\Scripts\\python.exe ^
        Layer3\\tools\\count_transition_records.py --data-dir data ^
        --out-dir Results\\layer3\\transition_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
for path in (PROJECT_ROOT, LAYER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Layer3._bootstrap import setup_layer3_paths  # noqa: E402

setup_layer3_paths()

from label_grouping import (  # noqa: E402
    AF_CONTEXT,
    BENIGN_ABNORMAL,
    DANGEROUS,
    NORMAL,
    build_rhythm_spans,
    group_for_beat,
    symbol_group,
)

DATASETS_DEFAULT = [
    "mit_bih_arrhythmia",
    "normal_sinus_rhythm",
    "supraventricular_arrhythmia",
    "long_term_atrial_fibrillation",
    "noise_stress_test",
    "st_petersburg_12lead",
    "atrial_fibrillation",
    "malignant_ventricular_arrhythmia",
    "creighton_vfib",
]

# Strict baseline = sinus only (what the current per-record calibration uses).
# Relaxed baseline = any stable, non-dangerous, non-noise rhythm (covers chronic
# AF subjects whose "healthy baseline" is AF, not sinus).
STRICT_BASELINE = frozenset({NORMAL})
RELAXED_BASELINE = frozenset({NORMAL, AF_CONTEXT, BENIGN_ABNORMAL})

HEALTHY_THRESHOLDS = (30, 100, 300)   # min baseline beats before first danger
BASELINE_SECONDS_THRESHOLDS = (30.0, 60.0, 120.0)  # for rhythm-only datasets


def analyze_record(
    rec_path: str,
    dataset: str,
    ann_ext: str,
) -> Optional[Dict[str, object]]:
    import wfdb
    try:
        ann = wfdb.rdann(rec_path, ann_ext)
        hdr = wfdb.rdheader(rec_path)
    except Exception:
        return None
    fs = float(hdr.fs)
    sig_len = int(hdr.sig_len)
    samples = np.asarray(ann.sample, dtype=np.int64)
    symbols = [str(s) for s in ann.symbol]
    aux = [str(a) for a in (ann.aux_note if getattr(ann, "aux_note", None) is not None else [""] * len(symbols))]
    order = np.argsort(samples, kind="mergesort")
    samples = samples[order]
    symbols = [symbols[i] for i in order]
    aux = [aux[i] for i in order]

    spans = build_rhythm_spans(samples, symbols, aux, sig_len)
    danger_spans = [s for s in spans if s.group == DANGEROUS]
    first_danger_span = min((s.start for s in danger_spans), default=None)

    # Beat-based view (datasets that have beat symbols).
    beats = []
    for i, sym in enumerate(symbols):
        if symbol_group(sym) is None:
            continue
        g = group_for_beat(int(samples[i]), sym, spans, dataset=dataset)
        beats.append((int(samples[i]), g))
    danger_beat_samples = [s for s, g in beats if g == DANGEROUS]
    first_danger_beat = min(danger_beat_samples) if danger_beat_samples else None

    candidates = [v for v in (first_danger_span, first_danger_beat) if v is not None]
    first_danger = min(candidates) if candidates else None
    n_danger_beats = len(danger_beat_samples)

    def baseline_count(baseline_groups) -> int:
        if first_danger is None:
            return sum(1 for s, g in beats if g in baseline_groups)
        return sum(1 for s, g in beats if s < first_danger and g in baseline_groups)

    n_baseline_strict = baseline_count(STRICT_BASELINE)
    n_baseline_relaxed = baseline_count(RELAXED_BASELINE)

    # Span-based baseline seconds before first danger (for rhythm-only datasets).
    def baseline_seconds(baseline_groups) -> float:
        if first_danger_span is None:
            total = sum((s.end - s.start) for s in spans if s.group in baseline_groups)
            return total / fs
        total = 0
        for s in spans:
            if s.group in baseline_groups and s.start < first_danger_span:
                total += min(s.end, first_danger_span) - s.start
        return total / fs

    danger_seconds = sum((s.end - s.start) for s in danger_spans) / fs

    return {
        "dataset": dataset,
        "record": Path(rec_path).name,
        "fs": fs,
        "duration_s": sig_len / fs,
        "n_beats": len(beats),
        "has_beats": len(beats) > 0,
        "has_danger": first_danger is not None,
        "first_danger_sample": -1 if first_danger is None else int(first_danger),
        "first_danger_s": -1.0 if first_danger is None else float(first_danger / fs),
        "n_danger_beats": int(n_danger_beats),
        "n_baseline_beats_strict": int(n_baseline_strict),
        "n_baseline_beats_relaxed": int(n_baseline_relaxed),
        "baseline_seconds_strict": float(baseline_seconds(STRICT_BASELINE)),
        "baseline_seconds_relaxed": float(baseline_seconds(RELAXED_BASELINE)),
        "danger_span_seconds": float(danger_seconds),
        "n_danger_spans": len(danger_spans),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Count within-subject sinus->danger transition records (read-only).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="Results/layer3/transition_analysis")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--ann-ext", default="atr")
    args = p.parse_args()

    import pandas as pd

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DATASETS_DEFAULT

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        ds_dir = data_dir / ds
        if not ds_dir.exists():
            print(f"[WARN] missing dataset: {ds_dir}", file=sys.stderr)
            continue
        for hea in sorted(ds_dir.rglob("*.hea")):
            res = analyze_record(str(hea.with_suffix("")), ds, args.ann_ext)
            if res is not None:
                rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "transition_records_by_record.csv", index=False)

    # Per-dataset summary.
    summary: List[Dict[str, object]] = []
    for ds in datasets:
        sub = df[df["dataset"] == ds] if not df.empty else df
        if sub.empty:
            continue
        with_danger = sub[sub["has_danger"]]
        row: Dict[str, object] = {
            "dataset": ds,
            "n_records": int(len(sub)),
            "n_with_danger": int(len(with_danger)),
            "total_danger_beats": int(sub["n_danger_beats"].sum()),
            "total_danger_span_s": float(sub["danger_span_seconds"].sum()),
        }
        for thr in HEALTHY_THRESHOLDS:
            elig = with_danger[with_danger["n_baseline_beats_strict"] >= thr]
            elig_relaxed = with_danger[with_danger["n_baseline_beats_relaxed"] >= thr]
            row[f"eligible_strict_ge{thr}beats"] = int(len(elig))
            row[f"eligible_relaxed_ge{thr}beats"] = int(len(elig_relaxed))
            row[f"scoreable_danger_beats_strict_ge{thr}"] = int(elig["n_danger_beats"].sum())
        # Rhythm-only fallback: records with no beats but danger spans + baseline secs.
        rhythm_only = with_danger[~with_danger["has_beats"]]
        for thr in BASELINE_SECONDS_THRESHOLDS:
            elig_s = rhythm_only[rhythm_only["baseline_seconds_strict"] >= thr]
            row[f"rhythmonly_eligible_strict_ge{int(thr)}s"] = int(len(elig_s))
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "transition_summary_by_dataset.csv", index=False)

    # Console headline.
    print("\n=== Within-subject sinus->danger eligibility (strict NORMAL baseline) ===")
    cols = ["dataset", "n_records", "n_with_danger",
            "eligible_strict_ge30beats", "eligible_strict_ge100beats", "eligible_strict_ge300beats",
            "scoreable_danger_beats_strict_ge30",
            "rhythmonly_eligible_strict_ge30s"]
    avail = [c for c in cols if c in summary_df.columns]
    print(summary_df[avail].to_string(index=False) if not summary_df.empty else "(no records)")

    print("\n=== Eligibility with RELAXED baseline (NORMAL+AF+benign before danger) ===")
    cols2 = ["dataset", "n_with_danger",
             "eligible_relaxed_ge30beats", "eligible_relaxed_ge100beats", "eligible_relaxed_ge300beats"]
    avail2 = [c for c in cols2 if c in summary_df.columns]
    print(summary_df[avail2].to_string(index=False) if not summary_df.empty else "(no records)")

    # Eligible record lists (strict, >=30 beats) for the gold-data set.
    if not df.empty:
        gold = df[(df["has_danger"]) & (df["has_beats"]) & (df["n_baseline_beats_strict"] >= 30)]
        gold_sorted = gold.sort_values(["dataset", "n_baseline_beats_strict"], ascending=[True, False])
        gold_sorted.to_csv(out_dir / "gold_transition_records_strict_ge30.csv", index=False)
        total_gold = len(gold_sorted)
        total_scoreable = int(gold_sorted["n_danger_beats"].sum())
        print(f"\nGOLD within-subject transition records (strict, >=30 sinus beats before danger): {total_gold}")
        print(f"Total scoreable DANGEROUS beats in those records: {total_scoreable}")
        print("\nPer-record gold list:")
        show = gold_sorted[["dataset", "record", "n_baseline_beats_strict", "n_danger_beats", "first_danger_s", "duration_s"]]
        print(show.to_string(index=False))

        # Pilot allowlists (SOFT GO: MIT-BIH primary; others secondary).
        role_map = {
            "mit_bih_arrhythmia": "primary",
            "creighton_vfib": "robustness_vf",
            "long_term_atrial_fibrillation": "secondary_af",
        }
        role_rows: List[Dict[str, object]] = []
        for ds, role in role_map.items():
            sub = gold_sorted[gold_sorted["dataset"] == ds]
            role_rows.append({
                "dataset": ds,
                "pilot_role": role,
                "n_gold_records": int(len(sub)),
                "n_danger_beats": int(sub["n_danger_beats"].sum()),
            })
            out_name = {
                "primary": "pilot_primary_mitbih_gold.csv",
                "robustness_vf": "pilot_secondary_creighton_gold.csv",
                "secondary_af": "pilot_secondary_ltafdb_gold.csv",
            }[role]
            sub.to_csv(out_dir / out_name, index=False)
            print(f"[INFO] wrote {out_name} ({len(sub)} records, role={role})")
        pd.DataFrame(role_rows).to_csv(out_dir / "pilot_dataset_roles.csv", index=False)
        print("[INFO] wrote pilot_dataset_roles.csv")

    print(f"\n[DONE] wrote transition analysis to {out_dir}")


if __name__ == "__main__":
    main()
