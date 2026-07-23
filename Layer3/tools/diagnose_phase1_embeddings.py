#!/usr/bin/env python3
"""Diagnose frozen Phase 1 representations without changing the locked protocol.

The tool reads existing validation directories. It never fits a threshold or
uses DANGEROUS labels to alter a decision; it only summarizes already-written
outputs and embedding geometry for post-hoc diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _effective_rank(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 2 or x.shape[1] < 1:
        return {"embedding_dim": float("nan"), "effective_rank": float("nan"), "top1_variance_frac": float("nan")}
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 0:
        return {"embedding_dim": float(x.shape[1]), "effective_rank": 0.0, "top1_variance_frac": 1.0}
    probs = variance / total
    positive = probs[probs > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "embedding_dim": float(x.shape[1]),
        "effective_rank": float(np.exp(entropy)),
        "top1_variance_frac": float(probs[0]),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def summarize_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = _read_csv(run_dir / "phase1_metrics_overall.csv")
    by_record = _read_csv(run_dir / "phase1_metrics_by_record.csv")
    by_subtype = _read_csv(run_dir / "phase1_metrics_by_danger_subtype.csv")
    encoder_info_path = run_dir / "encoder_info.json"
    encoder_info = json.loads(encoder_info_path.read_text(encoding="utf-8")) if encoder_info_path.exists() else {}

    rows: List[Dict[str, object]] = []
    if not overall.empty:
        selected = overall[
            overall["threshold_method"].astype(str).eq("conformal")
            & overall["arm"].astype(str).eq("layer3")
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "run": run_dir.name,
                    "scorer": row.get("scorer"),
                    "checkpoint_loaded": encoder_info.get("checkpoint_loaded"),
                    "n_NORMAL": row.get("n_NORMAL"),
                    "n_DANGEROUS": row.get("n_DANGEROUS"),
                    "false_permit_DANGEROUS": row.get("false_permit_DANGEROUS"),
                    "false_inhibit_NORMAL": row.get("false_inhibit_NORMAL"),
                    "auroc_NORMAL_vs_DANGEROUS": row.get("auroc_NORMAL_vs_DANGEROUS"),
                }
            )

    emb_path = run_dir / "embeddings.npy"
    if emb_path.exists():
        geometry = _effective_rank(np.load(emb_path, mmap_mode="r"))
        for row in rows:
            row.update(geometry)

    if not by_record.empty:
        by_record = by_record.copy()
        by_record.insert(0, "run", run_dir.name)
    if not by_subtype.empty:
        by_subtype = by_subtype.copy()
        by_subtype.insert(0, "run", run_dir.name)
    return pd.DataFrame(rows), by_record, by_subtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Phase 1 validation output directories.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries: List[pd.DataFrame] = []
    records: List[pd.DataFrame] = []
    subtypes: List[pd.DataFrame] = []
    for run_dir in args.run_dirs:
        summary, by_record, by_subtype = summarize_run(run_dir)
        summaries.append(summary)
        records.append(by_record)
        subtypes.append(by_subtype)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    record_df = pd.concat([d for d in records if not d.empty], ignore_index=True) if any(not d.empty for d in records) else pd.DataFrame()
    subtype_df = pd.concat([d for d in subtypes if not d.empty], ignore_index=True) if any(not d.empty for d in subtypes) else pd.DataFrame()
    summary_df.to_csv(args.out_dir / "representation_summary.csv", index=False)
    record_df.to_csv(args.out_dir / "metrics_by_record_all_runs.csv", index=False)
    subtype_df.to_csv(args.out_dir / "metrics_by_danger_subtype_all_runs.csv", index=False)

    lines = [
        "# Phase 1 representation diagnosis",
        "",
        "Post-hoc diagnostic only. Locked conformal alpha and primary results are unchanged.",
        "",
        "```text",
        summary_df.to_string(index=False) if not summary_df.empty else "No conformal Layer 3 rows found.",
        "```",
        "",
        "Interpret low effective rank as possible embedding collapse; inspect record/subtype CSVs before choosing an ablation.",
    ]
    (args.out_dir / "DIAGNOSTIC_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Diagnostics -> {args.out_dir}")


if __name__ == "__main__":
    main()
