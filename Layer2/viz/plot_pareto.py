"""
Pareto / operating-point figures for Layer 2 safety trade-offs.

Outputs:
  pareto_operating_curves.png  — healthy permit vs abnormal inhibit sweep
  pareto_frontier.png          — frontier from posthoc CSV (if provided)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import DS_COLOR, DS_LABEL, apply_style, style_axes


def _sweep_mahal(sub: pd.DataFrame, n: int = 200):
    h = sub[~sub["is_abnormal"]]
    a = sub[sub["is_abnormal"]]
    hr = h["mahalanobis"] / h["mahalanobis_threshold"].replace(0, np.nan)
    ar = a["mahalanobis"] / a["mahalanobis_threshold"].replace(0, np.nan)
    scales = np.linspace(0.05, 5.0, n)
    hp = np.array([(hr < t).mean() for t in scales])
    ai = np.array([(ar >= t).mean() for t in scales])
    return scales, hp, ai


def plot_operating_curves(per_beat: Path, out: Path, datasets: list[str] | None = None) -> None:
    use = ["dataset", "label", "mahalanobis", "mahalanobis_threshold"]
    chunks = []
    for chunk in pd.read_csv(per_beat, usecols=use, chunksize=300_000, low_memory=False):
        chunk = chunk[chunk["label"].isin(["healthy", "abnormal", "abnormal_v", "abnormal_s"])]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["is_abnormal"] = df["label"].str.startswith("abnormal")

    if datasets:
        df = df[df["dataset"].isin(datasets)]

    ds_list = sorted(df["dataset"].unique())
    if not ds_list:
        print("  skip operating curves (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for ds in ds_list:
        sub = df[df["dataset"] == ds]
        _, hp, ai = _sweep_mahal(sub)
        ax.plot(hp, ai, lw=2, color=DS_COLOR.get(ds, "gray"), label=DS_LABEL.get(ds, ds))
    ax.axhline(0.95, color="green", ls="--", lw=1, alpha=0.7)
    ax.axvline(0.82, color="#4C9BE8", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Healthy permit rate")
    ax.set_ylabel("Abnormal inhibit rate")
    ax.set_title("Operating curve (Mahalanobis scale sweep)", fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    style_axes(ax)

    ax = axes[1]
    for ds in ds_list:
        sub = df[df["dataset"] == ds]
        _, hp, ai = _sweep_mahal(sub)
        fp = 1.0 - ai
        ax.plot(fp, hp, lw=2, color=DS_COLOR.get(ds, "gray"), label=DS_LABEL.get(ds, ds))
    ax.axvline(0.05, color="red", ls="--", lw=1, alpha=0.7, label="5% false permit")
    ax.set_xlabel("False permit rate")
    ax.set_ylabel("Healthy permit rate")
    ax.set_title("Safety trade-off (Pareto view)", fontweight="bold")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.suptitle("Layer 2 operating points — threshold sweep on saved scores", fontweight="bold", y=1.02)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_posthoc_frontier(posthoc_csv: Path, out: Path) -> None:
    if not posthoc_csv.exists():
        print(f"  skip pareto_frontier ({posthoc_csv.name} not found)")
        return

    df = pd.read_csv(posthoc_csv)
    fig, ax = plt.subplots(figsize=(8, 6))
    for ignore_morph, marker in [(False, "o"), (True, "^")]:
        sub = df[df["ignore_morph_hard_rules"] == ignore_morph]
        ax.scatter(
            sub["healthy_permit"], sub["abnormal_inhibit"],
            c=sub["false_permit"], cmap="RdYlGn_r", s=40, alpha=0.7,
            marker=marker,
            label=f"morph hard rules {'ignored' if ignore_morph else 'kept'}",
            edgecolors="white", linewidths=0.3,
        )
    ax.axhline(0.95, color="green", ls="--", lw=1)
    ax.axvline(0.82, color="#4C9BE8", ls=":", lw=1)
    ax.set_xlabel("Healthy permit rate")
    ax.set_ylabel("Abnormal inhibit rate")
    ax.set_title("Post-hoc Pareto sweep", fontweight="bold")
    ax.legend()
    style_axes(ax)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-beat", type=Path,
                   default=Path("Results/layer2/cross_dataset_causal_100ms/per_beat.csv"))
    p.add_argument("--posthoc-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/viz"))
    args = p.parse_args(argv)

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_beat.exists():
        plot_operating_curves(args.per_beat, args.out_dir / "pareto_operating_curves.png")
    else:
        print(f"Missing: {args.per_beat}")

    posthoc = args.posthoc_csv or Path("Results/final_mitbih_validation/pareto_posthoc/pareto_posthoc.csv")
    plot_posthoc_frontier(posthoc, args.out_dir / "pareto_frontier.png")
    print(f"\nDone. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
