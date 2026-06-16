"""
Compare Layer 2 oracle decisions with causal adaptive Layer 1 gated decisions.

Inputs are the per-beat rows produced by run_cross_dataset_benchmark.py.
The deployment-like path is eval_mode="layer1_adaptive_gated": causal adaptive
Layer 1 accepted triggers feeding the frozen Layer 2 gate.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def nstdb_snr(record: str) -> int | None:
    """Parse NSTDB record names such as 118e_6, 118e00, 118e06, 118e12."""
    name = str(record)
    if name.endswith("e_6"):
        return -6
    m = re.search(r"e(\d{2})$", name)
    if not m:
        return None
    return int(m.group(1))


def summarize(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        h = sub[sub["label"] == "healthy"]
        ab = sub[sub["label"] == "abnormal_v"]
        svt = sub[sub["label"] == "svt"]
        row.update({
            "n_healthy": int(len(h)),
            "n_abnormal": int(len(ab)),
            "n_svt": int(len(svt)),
            "healthy_permit": round(float(h["permit"].mean()), 4) if len(h) else float("nan"),
            "false_inhibit": round(float((~h["permit"]).mean()), 4) if len(h) else float("nan"),
            "abnormal_inhibit": round(float((~ab["permit"]).mean()), 4) if len(ab) else float("nan"),
            "false_permit": round(float(ab["permit"].mean()), 4) if len(ab) else float("nan"),
            "svt_inhibit": round(float((~svt["permit"]).mean()), 4) if len(svt) else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison(summary: pd.DataFrame, candidate_mode: str) -> pd.DataFrame:
    rows = []
    metrics = ["healthy_permit", "false_inhibit", "abnormal_inhibit", "false_permit", "svt_inhibit"]
    for dataset, sub in summary.groupby("dataset"):
        oracle = sub[sub["eval_mode"] == "oracle"]
        candidate = sub[sub["eval_mode"] == candidate_mode]
        if oracle.empty or candidate.empty:
            continue
        o = oracle.iloc[0]
        c = candidate.iloc[0]
        row = {
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
    return pd.DataFrame(rows).sort_values("dataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-beat", type=Path, default=Path("Results/layer2/cross_dataset/per_beat.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/cross_dataset"))
    p.add_argument("--candidate-mode", default="layer1_adaptive_gated")
    p.add_argument("--prefix", default=None)
    p.add_argument(
        "--nstdb-min-snr",
        type=int,
        default=12,
        help="NSTDB records below this SNR are removed from the filtered table.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.per_beat, dtype={"record": str}, low_memory=False)
    if df["permit"].dtype != bool:
        df["permit"] = df["permit"].astype(str).str.lower().map({"true": True, "false": False})
    df = df[
        (df["benchmark_mode"] == "zero_shot")
        & (df["feature_set"] == "all")
        & (df["eval_mode"].isin(("oracle", args.candidate_mode)))
    ].copy()

    summary = summarize(df, ["dataset", "eval_mode"])
    comparison = build_comparison(summary, args.candidate_mode)

    nstdb = df[df["dataset"] == "nstdb"].copy()
    nstdb["snr_db"] = nstdb["record"].map(nstdb_snr)
    nstdb_filtered = nstdb[nstdb["snr_db"] >= args.nstdb_min_snr].copy()
    nstdb_filtered_summary = summarize(nstdb_filtered, ["dataset", "eval_mode"])
    nstdb_filtered_summary.insert(1, "snr_filter", f">={args.nstdb_min_snr} dB")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"oracle_vs_{args.candidate_mode}"
    summary.to_csv(args.out_dir / f"{prefix}_summary.csv", index=False)
    comparison.to_csv(args.out_dir / f"{prefix}_comparison.csv", index=False)
    nstdb_filtered_summary.to_csv(
        args.out_dir / f"nstdb_snr_ge{args.nstdb_min_snr}_{prefix}.csv",
        index=False,
    )

    print("Wrote:")
    print(args.out_dir / f"{prefix}_summary.csv")
    print(args.out_dir / f"{prefix}_comparison.csv")
    print(args.out_dir / f"nstdb_snr_ge{args.nstdb_min_snr}_{prefix}.csv")


if __name__ == "__main__":
    main()
