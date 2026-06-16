"""
Feature AUROC and separability figures for Layer 2.

Outputs:
  feature_auroc.png          — ranked AUROC bar chart
  feature_group_auroc.png    — mean AUROC per clinical group
  top_deviators.png          — features most often triggering inhibit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from plot_style import apply_style, style_axes

GROUP_COLORS = {
    "Timing": "#e74c3c",
    "Rhythm": "#e67e22",
    "Morphology": "#2980b9",
    "Signal": "#27ae60",
    "Wavelet": "#1abc9c",
    "Other": "#95a5a6",
}

FEATURE_PREFIX_GROUP = {
    "rr__": "Rhythm",
    "morph__": "Morphology",
    "signal__wave_": "Wavelet",
    "signal__": "Signal",
}


def _feature_group(name: str) -> str:
    for prefix, group in FEATURE_PREFIX_GROUP.items():
        if name.startswith(prefix):
            return group
    return "Other"


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    m = np.isfinite(s)
    if m.sum() < 20:
        return float("nan")
    auc = roc_auc_score(y[m], s[m])
    return max(auc, 1.0 - auc)


def compute_auroc(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    h_all = df[df["label"] == "healthy"]
    ab_all = df[df["label"].isin(["abnormal_v", "abnormal"])]
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        h = h_all[feat].dropna().values
        ab = ab_all[feat].dropna().values
        if len(h) < 20 or len(ab) < 20:
            continue
        rows.append({
            "feature": feat,
            "group": _feature_group(feat),
            "auroc": round(_auroc(ab, h), 3),
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "group", "auroc"])
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def plot_auroc_bar(auroc_df: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    sub = auroc_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(sub))))
    colors = [GROUP_COLORS.get(g, "#95a5a6") for g in sub["group"]]
    y = np.arange(len(sub))
    ax.barh(y, sub["auroc"], color=colors, edgecolor="white")
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axvline(0.7, color="orange", ls=":", lw=1, alpha=0.8)
    ax.axvline(0.8, color="red", ls=":", lw=1, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=8)
    ax.set_xlabel("AUROC (healthy vs ventricular)")
    ax.set_title("Feature separability — top features", fontweight="bold")
    ax.set_xlim(0.45, 1.02)
    ax.invert_yaxis()
    style_axes(ax)

    seen = {}
    for g, c in GROUP_COLORS.items():
        if g in sub["group"].values:
            seen[g] = mpatches.Patch(facecolor=c, label=g)
    ax.legend(handles=list(seen.values()), loc="lower right", fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_group_summary(auroc_df: pd.DataFrame, out: Path) -> None:
    grp = auroc_df.groupby("group")["auroc"].agg(["mean", "count"]).reset_index()
    grp = grp.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [GROUP_COLORS.get(g, "#95a5a6") for g in grp["group"]]
    ax.bar(range(len(grp)), grp["mean"], color=colors, edgecolor="white")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([f"{g}\n(n={int(n)})" for g, n in zip(grp["group"], grp["count"])],
                       fontsize=9)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Feature group separability", fontweight="bold")
    ax.set_ylim(0.45, 1.0)
    style_axes(ax)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_top_deviators(df: pd.DataFrame, out: Path) -> None:
    cols = [c for c in ["top1_feature", "top2_feature", "top3_feature"] if c in df.columns]
    if not cols:
        print("  skip top_deviators (no topN_feature columns)")
        return

    sub = df[
        df["label"].isin(["abnormal_v", "abnormal"]) & (df["permit"] == False)  # noqa: E712
    ]
    if sub.empty:
        return

    all_feats = pd.concat([sub[c] for c in cols]).dropna().astype(str)
    all_feats = all_feats[all_feats != ""]
    counts = all_feats.value_counts().head(15)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(counts))
    colors = [GROUP_COLORS.get(_feature_group(f), "#95a5a6") for f in counts.index]
    ax.barh(y, counts.values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Count as top deviating feature")
    ax.set_title("Features driving inhibit on abnormal beats", fontweight="bold")
    ax.invert_yaxis()
    style_axes(ax)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-beat", type=Path,
                   default=Path("Results/layer2/cross_dataset_causal_100ms/per_beat.csv"))
    p.add_argument("--beat-features", type=Path, default=None,
                   help="Optional CSV with raw feature columns (e.g. INCART analysis)")
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/viz"))
    args = p.parse_args(argv)

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = args.beat_features if args.beat_features and args.beat_features.exists() else args.per_beat
    if not source.exists():
        raise SystemExit(f"Missing input: {source}")

    df = pd.read_csv(source, low_memory=False)
    feature_cols = [c for c in df.columns if c.startswith(("rr__", "morph__", "signal__"))]
    print(f"Loaded {len(df):,} rows, {len(feature_cols)} feature columns")

    auroc_df = compute_auroc(df, feature_cols)
    if auroc_df.empty:
        print("No AUROC computed — per_beat.csv has no raw feature columns.")
        print("  Pass --beat-features with a CSV that includes rr__/morph__/signal__ columns.")
    else:
        auroc_df.to_csv(args.out_dir / "auroc_per_feature.csv", index=False)
        plot_auroc_bar(auroc_df, args.out_dir / "feature_auroc.png")
        plot_group_summary(auroc_df, args.out_dir / "feature_group_auroc.png")

    if args.per_beat.exists():
        pb = pd.read_csv(args.per_beat, low_memory=False)
        plot_top_deviators(pb, args.out_dir / "top_deviators.png")

    print(f"\nDone. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
