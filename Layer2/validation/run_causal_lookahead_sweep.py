"""
Sweep causal Layer 2 post-R lookahead for the fast causal threshold detector.

Each sweep point uses:
  - causal Layer 2 features: [R - morphology_window_s, R + lookahead]
  - oracle eval mode as the upper-bound trigger source
  - fast_causal_gated / adaptive_candidate_fast_context / nextbeat / stateful
    / prospective 1-in-8 cadence as deployment modes

Outputs are written under one folder per lookahead plus combined CSV summaries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_HERE = Path(__file__).parent
_L2 = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_L2 / "tools"))

from run_cross_dataset_validation import run_benchmark  # noqa: E402
from compare_oracle_adaptive import nstdb_snr, summarize  # noqa: E402


def _comparison_from_overall(overall: pd.DataFrame, lookahead_ms: int) -> pd.DataFrame:
    rows: List[Dict] = []
    metrics = ["healthy_permit", "false_inhibit", "abnormal_inhibit", "false_permit", "svt_inhibit"]
    candidate_modes = [
        "fast_causal_gated",
        "adaptive_candidate_fast_context",
        "fast_causal_nextbeat",
        "fast_causal_stateful",
        "fast_causal_cadence_1of8",
    ]
    overall = overall[
        (overall["benchmark_mode"] == "zero_shot")
        & (overall["feature_set"] == "all")
        & (overall["eval_mode"].isin(["oracle", *candidate_modes]))
    ]
    for (dataset, candidate_mode), sub in overall.groupby(["dataset", "eval_mode"]):
        if candidate_mode == "oracle":
            continue
        ds_all = overall[overall["dataset"] == dataset]
        oracle = ds_all[ds_all["eval_mode"] == "oracle"]
        candidate = sub
        if oracle.empty or candidate.empty:
            continue
        o = oracle.iloc[0]
        c = candidate.iloc[0]
        row = {
            "lookahead_ms": lookahead_ms,
            "dataset": dataset,
            "candidate_mode": candidate_mode,
            "candidate_n_healthy": int(c["n_healthy"]),
            "candidate_n_abnormal": int(c["n_abnormal"]),
            "oracle_n_healthy": int(o["n_healthy"]),
            "oracle_n_abnormal": int(o["n_abnormal"]),
        }
        for metric in metrics:
            row[f"candidate_{metric}"] = c[metric]
            row[f"oracle_{metric}"] = o[metric]
            row[f"candidate_minus_oracle_{metric}_pp"] = round(100.0 * (c[metric] - o[metric]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _nstdb_filtered(per_beat_path: Path, lookahead_ms: int, min_snr: int) -> pd.DataFrame:
    df = pd.read_csv(per_beat_path, dtype={"record": str}, low_memory=False)
    if df["permit"].dtype != bool:
        df["permit"] = df["permit"].astype(str).str.lower().map({"true": True, "false": False})
    df = df[
        (df["dataset"] == "nstdb")
        & (df["benchmark_mode"] == "zero_shot")
        & (df["feature_set"] == "all")
        & (df["eval_mode"].isin([
            "oracle",
            "fast_causal_gated",
            "adaptive_candidate_fast_context",
            "fast_causal_nextbeat",
            "fast_causal_stateful",
            "fast_causal_cadence_1of8",
        ]))
    ].copy()
    df["snr_db"] = df["record"].map(nstdb_snr)
    df = df[df["snr_db"] >= min_snr].copy()
    out = summarize(df, ["dataset", "eval_mode"])
    if out.empty:
        out = pd.DataFrame(columns=[
            "lookahead_ms", "dataset", "snr_filter", "eval_mode",
            "n_healthy", "n_abnormal", "n_svt", "healthy_permit",
            "false_inhibit", "abnormal_inhibit", "false_permit", "svt_inhibit",
        ])
        return out
    out.insert(0, "lookahead_ms", lookahead_ms)
    out.insert(min(2, len(out.columns)), "snr_filter", f">={min_snr} dB")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/causal_lookahead_sweep"))
    p.add_argument("--datasets", nargs="+", default=["mitdb", "nstdb", "svdb", "incartdb"])
    p.add_argument("--lookahead-ms", nargs="+", type=int, default=[30, 40, 50, 80, 100, 150, 200])
    p.add_argument("--feature-sets", nargs="+", default=["all"])
    p.add_argument("--cal-frac", type=float, default=0.6)
    p.add_argument("--threshold-quantile", type=float, default=0.999)
    p.add_argument("--morphology-window-s", type=float, default=5.0)
    p.add_argument("--rr-lookback-s", type=float, default=30.0)
    p.add_argument("--species", default="human")
    p.add_argument("--window-limit", type=int, default=None)
    p.add_argument("--max-records-per-dataset", type=int, default=0)
    p.add_argument("--nstdb-min-snr", type=int, default=12)
    p.add_argument("--include-adaptive", action="store_true", help="Also score layer1_adaptive_gated.")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    nstdb_filtered = []

    for look_ms in args.lookahead_ms:
        run_dir = args.out_dir / f"lookahead_{look_ms:03d}ms"
        run_benchmark(
            data_dir=args.data_dir,
            out_dir=run_dir,
            datasets=args.datasets,
            benchmark_mode="zero_shot",
            feature_sets=args.feature_sets,
            cal_frac=args.cal_frac,
            threshold_quantile=args.threshold_quantile,
            morphology_window_s=args.morphology_window_s,
            rr_lookback_s=args.rr_lookback_s,
            species=args.species,
            abnormal_target=0.95,
            max_healthy_fi=0.18,
            window_limit=args.window_limit,
            include_adaptive=args.include_adaptive,
            feature_window_mode="causal",
            post_r_lookahead_s=look_ms / 1000.0,
            max_records_per_dataset=args.max_records_per_dataset,
            cadence_observation_lookahead_s=0.40,
            cadence_min_safe_observations=6,
            cadence_require_last_observation_safe=True,
        )
        overall = pd.read_csv(run_dir / "overall_summary.csv")
        comparisons.append(_comparison_from_overall(overall, look_ms))
        if (run_dir / "per_beat.csv").exists():
            nstdb_filtered.append(_nstdb_filtered(run_dir / "per_beat.csv", look_ms, args.nstdb_min_snr))

    combined = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    combined.to_csv(args.out_dir / "lookahead_fast_causal_vs_oracle.csv", index=False)

    if nstdb_filtered:
        pd.concat(nstdb_filtered, ignore_index=True).to_csv(
            args.out_dir / f"nstdb_snr_ge{args.nstdb_min_snr}_lookahead_summary.csv",
            index=False,
        )

    print(f"Wrote {args.out_dir / 'lookahead_fast_causal_vs_oracle.csv'}")
    if nstdb_filtered:
        print(f"Wrote {args.out_dir / f'nstdb_snr_ge{args.nstdb_min_snr}_lookahead_summary.csv'}")


if __name__ == "__main__":
    main()
