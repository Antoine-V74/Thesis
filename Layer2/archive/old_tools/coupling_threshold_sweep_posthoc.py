"""
Post-hoc sweep of rr__beat_coupling_ratio hard-rule thresholds.

This script answers one specific question:

    If we force the coupling hard rule to remain active and set its lower
    threshold to 0.75, 0.80, or 0.85, how do HP / AI / FP / SVT inhibit change
    across datasets?

It starts from an existing per_beat.csv, recomputes rr__beat_coupling_ratio from
the WFDB annotations, and applies an additional inhibit rule:

    permit_new = permit_existing AND NOT(coupling_ratio < threshold)

That isolates the effect of the coupling rule without re-running full Layer 2
feature extraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    import wfdb
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install wfdb")

from run_cross_dataset_benchmark import (  # noqa: E402
    DATASET_ABNORMAL,
    DATASET_ANN_EXT,
    DATASET_NORMAL,
)


def _record_stem(data_dir: Path, dataset: str, record: str) -> Path:
    return data_dir / dataset / str(record)


def _annotation_peak_times(data_dir: Path, dataset: str, record: str) -> np.ndarray:
    """Return normal+ventricular annotation times used by Layer 2 RR coupling."""
    stem = _record_stem(data_dir, dataset, record)
    rec = wfdb.rdrecord(str(stem), sampto=1)
    ann = wfdb.rdann(str(stem), DATASET_ANN_EXT[dataset])
    keep_symbols = DATASET_NORMAL[dataset] | DATASET_ABNORMAL[dataset]
    return np.array(
        [s / float(rec.fs) for s, sym in zip(ann.sample, ann.symbol) if sym in keep_symbols],
        dtype=float,
    )


def _coupling_ratio(beat_time_s: float, peaks_s: np.ndarray) -> float:
    """Match run_cross_dataset_benchmark.extract_beat_features coupling logic."""
    # per_beat.csv stores beat_time_s rounded to milliseconds.  If the rounded
    # value is slightly after the annotation, a strict "< beat_time_s" comparison
    # can accidentally include the current beat as the previous peak and produce
    # a near-zero RR.  Exclude peaks within 3 ms of the row's beat time.
    prev_peaks = peaks_s[peaks_s < (beat_time_s - 0.003)]
    if len(prev_peaks) < 4:
        return float("nan")
    current_rr_s = float(beat_time_s - prev_peaks[-1])
    recent_rrs = np.diff(prev_peaks[-min(20, len(prev_peaks)):])
    valid_rrs = recent_rrs[(recent_rrs > 0.20) & (recent_rrs < 3.0)]
    if len(valid_rrs) < 3 or current_rr_s <= 0.20:
        return float("nan")
    median_rr = float(np.median(valid_rrs))
    if median_rr <= 0.20:
        return float("nan")
    return current_rr_s / median_rr


def add_coupling_ratios(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["rr__beat_coupling_ratio_posthoc"] = np.nan

    cache: Dict[Tuple[str, str], np.ndarray] = {}
    for (dataset, record), idx in out.groupby(["dataset", "record"]).groups.items():
        key = (str(dataset), str(record))
        if key not in cache:
            cache[key] = _annotation_peak_times(data_dir, key[0], key[1])
        peaks_s = cache[key]
        times = out.loc[idx, "beat_time_s"].astype(float).to_numpy()
        out.loc[idx, "rr__beat_coupling_ratio_posthoc"] = [
            _coupling_ratio(float(t), peaks_s) for t in times
        ]
    return out


def _metrics_for(sub: pd.DataFrame, permit_col: str) -> Dict[str, float]:
    healthy = sub[sub["label"] == "healthy"]
    abnormal = sub[sub["label"] == "abnormal_v"]
    svt = sub[sub["label"] == "svt"]

    return {
        "n_healthy": len(healthy),
        "n_abnormal": len(abnormal),
        "n_svt": len(svt),
        "healthy_permit": float(healthy[permit_col].mean()) if len(healthy) else np.nan,
        "abnormal_inhibit": float((~abnormal[permit_col]).mean()) if len(abnormal) else np.nan,
        "false_permit": float(abnormal[permit_col].mean()) if len(abnormal) else np.nan,
        "svt_inhibit": float((~svt[permit_col]).mean()) if len(svt) else np.nan,
    }


def sweep(
    df: pd.DataFrame,
    thresholds: Iterable[float],
    eval_mode: str,
    benchmark_mode: str,
    feature_set: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df[
        (df["eval_mode"] == eval_mode)
        & (df["benchmark_mode"] == benchmark_mode)
        & (df["feature_set"] == feature_set)
    ].copy()

    rows: List[Dict[str, object]] = []
    record_rows: List[Dict[str, object]] = []

    for threshold in [None, *thresholds]:
        if threshold is None:
            permit_col = "permit"
            label = "baseline"
        else:
            permit_col = f"permit_coupling_{threshold:.2f}"
            label = f"coupling_{threshold:.2f}"
            c = base["rr__beat_coupling_ratio_posthoc"]
            extra_inhibit = c.notna() & (c < float(threshold))
            base[permit_col] = base["permit"].astype(bool) & ~extra_inhibit

        for dataset, sub in base.groupby("dataset"):
            m = _metrics_for(sub, permit_col)
            rows.append({"dataset": dataset, "strategy": label, **m})

        for (dataset, record), sub in base.groupby(["dataset", "record"]):
            m = _metrics_for(sub, permit_col)
            record_rows.append({"dataset": dataset, "record": record, "strategy": label, **m})

    return pd.DataFrame(rows), pd.DataFrame(record_rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-beat", type=Path, default=Path("Results/layer2/cross_dataset/per_beat.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/analysis/coupling_sweep"))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.75, 0.80, 0.85])
    p.add_argument("--eval-mode", default="oracle")
    p.add_argument("--benchmark-mode", default="zero_shot")
    p.add_argument("--feature-set", default="all")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_beat, low_memory=False)
    df = add_coupling_ratios(df, args.data_dir)

    annotated = args.out_dir / "per_beat_with_coupling.csv"
    df.to_csv(annotated, index=False)

    overall, by_record = sweep(
        df,
        thresholds=args.thresholds,
        eval_mode=args.eval_mode,
        benchmark_mode=args.benchmark_mode,
        feature_set=args.feature_set,
    )
    overall.to_csv(args.out_dir / "coupling_sweep_overall.csv", index=False)
    by_record.to_csv(args.out_dir / "coupling_sweep_by_record.csv", index=False)

    display = overall.copy()
    for col in ["healthy_permit", "abnormal_inhibit", "false_permit", "svt_inhibit"]:
        display[col] = (100.0 * display[col]).round(1)
    print(display.to_string(index=False))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()
