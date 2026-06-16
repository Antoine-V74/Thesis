"""
Dataset-level performance figures for Layer 2 validation results.

Outputs:
  dataset_summary.png       — healthy permit vs abnormal inhibit per dataset
  arrhythmia_breakdown.png  — inhibit rate by beat class and arrhythmia symbol
  worst_records.png         — records with lowest abnormal inhibit (false permits)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    DS_COLOR,
    DS_LABEL,
    LABEL_COLOR,
    LABEL_NICE,
    RED,
    apply_style,
    style_axes,
)


def _load(per_beat: Path, mode: str | None, feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(per_beat, low_memory=False)
    if "feature_set" in df.columns:
        df = df[df["feature_set"] == feature_set]
    if mode and "mode" in df.columns:
        df = df[df["mode"] == mode]
    elif mode and "eval_mode" in df.columns:
        df = df[df["eval_mode"] == mode]
    if "benchmark_mode" in df.columns:
        df = df[df["benchmark_mode"] == "zero_shot"]
    return df


def plot_dataset_summary(df: pd.DataFrame, out: Path) -> None:
    if "dataset" not in df.columns:
        print("  skip dataset_summary (no dataset column)")
        return

    rows = []
    for ds, grp in df.groupby("dataset"):
        h = grp[grp["label"] == "healthy"]
        ab = grp[grp["label"].isin(["abnormal_v", "abnormal"])]
        if len(h) == 0 and len(ab) == 0:
            continue
        rows.append({
            "dataset": ds,
            "healthy_permit": h["permit"].mean() if len(h) else float("nan"),
            "abnormal_inhibit": (~ab["permit"]).mean() if len(ab) else float("nan"),
            "n_healthy": len(h),
            "n_abnormal": len(ab),
        })
    if not rows:
        return

    stats = pd.DataFrame(rows).sort_values("abnormal_inhibit", ascending=True)
    x = np.arange(len(stats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, stats["healthy_permit"], w, label="Healthy permit",
           color="#4C9BE8", edgecolor="white")
    ax.bar(x + w / 2, stats["abnormal_inhibit"], w, label="Abnormal inhibit",
           color="#E8614C", edgecolor="white")
    ax.axhline(0.95, color="green", ls="--", lw=1, alpha=0.7, label="95% abnormal target")
    ax.axhline(0.82, color="#E8614C", ls=":", lw=1, alpha=0.5, label="82% healthy ref")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL.get(d, d) for d in stats["dataset"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Layer 2 gate performance by dataset", fontweight="bold")
    ax.legend(loc="lower right")
    style_axes(ax)

    for i, row in stats.iterrows():
        idx = list(stats.index).index(i)
        ax.text(idx - w / 2, row["healthy_permit"] + 0.02, f"{row['healthy_permit']:.0%}",
                ha="center", fontsize=8)
        ax.text(idx + w / 2, row["abnormal_inhibit"] + 0.02, f"{row['abnormal_inhibit']:.0%}",
                ha="center", fontsize=8)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_arrhythmia_breakdown(df: pd.DataFrame, out: Path) -> None:
    labels = ["healthy", "svt", "abnormal_v"]
    present = [l for l in labels if l in df["label"].values]
    if not present:
        print("  skip arrhythmia_breakdown (no labels)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: inhibit rate by class
    ax = axes[0]
    rates = [1 - df[df["label"] == l]["permit"].mean() for l in present]
    bars = ax.bar([LABEL_NICE.get(l, l) for l in present], rates,
                  color=[LABEL_COLOR.get(l, "gray") for l in present], edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", fontweight="bold")
    ax.axhline(0.95, color="green", ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Inhibit rate")
    ax.set_title("Inhibit rate by arrhythmia class", fontweight="bold")
    style_axes(ax)

    # Panel 2: top beat symbols driving false permits on abnormal beats
    ax = axes[1]
    ab = df[df["label"].isin(["abnormal_v", "abnormal"])].copy()
    if "beat_symbol" in ab.columns and len(ab):
        fp = ab[ab["permit"] == True]  # noqa: E712
        if len(fp):
            sym_counts = fp["beat_symbol"].value_counts().head(8)
            y = np.arange(len(sym_counts))
            ax.barh(y, sym_counts.values, color=RED, alpha=0.75)
            ax.set_yticks(y)
            ax.set_yticklabels(sym_counts.index)
            ax.invert_yaxis()
            ax.set_xlabel("False permit count")
            ax.set_title("Beat symbols most often permitted (abnormal)", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No false permits", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No beat_symbol column", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax)

    fig.suptitle("Arrhythmia-level gate analysis", fontweight="bold", y=1.02)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_worst_records(df: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    if "record" not in df.columns:
        print("  skip worst_records (no record column)")
        return

    ab = df[df["label"].isin(["abnormal_v", "abnormal"])].copy()
    if ab.empty:
        return

    rec_stats = (
        ab.groupby(["dataset", "record"])
        .agg(
            n=("permit", "count"),
            abnormal_inhibit=("permit", lambda s: (~s).mean()),
            false_permits=("permit", lambda s: s.sum()),
        )
        .reset_index()
    )
    rec_stats = rec_stats[rec_stats["n"] >= 3].sort_values("abnormal_inhibit").head(top_n)
    if rec_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rec_stats))))
    y = np.arange(len(rec_stats))
    colors = [DS_COLOR.get(d, "gray") for d in rec_stats["dataset"]]
    ax.barh(y, rec_stats["abnormal_inhibit"], color=colors, edgecolor="white")
    ax.axvline(0.95, color="green", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{DS_LABEL.get(r.dataset, r.dataset)} / {r.record}  (n={int(r.n)})"
         for r in rec_stats.itertuples()],
        fontsize=9,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Abnormal inhibit rate")
    ax.set_title("Worst records — lowest abnormal inhibit (most false permits)", fontweight="bold")
    ax.invert_yaxis()
    style_axes(ax)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-beat", type=Path,
                   default=Path("Results/layer2/cross_dataset_causal_100ms/per_beat.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/viz"))
    p.add_argument("--mode", default=None, help="Filter mode/eval_mode column")
    p.add_argument("--feature-set", default="all")
    args = p.parse_args(argv)

    if not args.per_beat.exists():
        raise SystemExit(f"Missing input: {args.per_beat}")

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(args.per_beat, args.mode, args.feature_set)
    print(f"Loaded {len(df):,} beats from {args.per_beat.name}")

    plot_dataset_summary(df, args.out_dir / "dataset_summary.png")
    plot_arrhythmia_breakdown(df, args.out_dir / "arrhythmia_breakdown.png")
    plot_worst_records(df, args.out_dir / "worst_records.png")
    print(f"\nDone. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
