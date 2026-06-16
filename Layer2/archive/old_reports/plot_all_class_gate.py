"""
Plot gate performance across all beat classes (healthy, ventricular, SVT).
Output: Results/feature_importance/all_class_gate_analysis.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "Results" / "layer2" / "analysis" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "Results" / "cross_dataset_full" / "per_beat.csv", low_memory=False)
df = df[(df.benchmark_mode == "zero_shot") & (df.feature_set == "all")].copy()

CLR  = {"healthy": "#4C9BE8", "abnormal_v": "#E8614C", "svt": "#F5A623"}
NICE = {"healthy": "Healthy (N)", "abnormal_v": "Ventricular (V/PVC)", "svt": "SVT / PAC (S/A)"}

def auroc_vs_healthy(pos_label: str, col: str) -> float:
    h  = df[df.label == "healthy"][col].dropna()
    p  = df[df.label == pos_label][col].dropna()
    y  = np.concatenate([np.ones(len(p)), np.zeros(len(h))])
    sc = np.concatenate([p.values, h.values])
    v  = np.isfinite(sc)
    auc = roc_auc_score(y[v], sc[v])
    return max(auc, 1 - auc)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Gate performance across all beat classes — zero-shot, oracle L1",
    fontsize=13, fontweight="bold",
)

# ── Panel 1: inhibit rate per class ──────────────────────────────────────────
ax = axes[0]
labels_ord = ["healthy", "svt", "abnormal_v"]
inhibit_rates = [1 - df[df.label == l]["permit"].mean() for l in labels_ord]
bars = ax.bar(
    [NICE[l] for l in labels_ord], inhibit_rates,
    color=[CLR[l] for l in labels_ord], edgecolor="white", linewidth=0.5,
)
for bar, rate in zip(bars, inhibit_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
        f"{rate:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
ax.set_ylim(0, 1.05)
ax.set_ylabel("Inhibit rate", fontsize=11)
ax.set_title("Inhibit rate by beat class\n(higher = gate catches more)", fontsize=10)
ax.axhline(0.95, color="green", ls="--", lw=1, label="95% target (V)")
ax.axhline(0.08, color="red",   ls="--", lw=1, label="8% budget (healthy)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ── Panel 2: Mahalanobis distributions ───────────────────────────────────────
ax = axes[1]
hi_clip = float(df["mahalanobis"].quantile(0.98))
bins = np.linspace(0, hi_clip, 70)
for lbl in ["healthy", "svt", "abnormal_v"]:
    vals = df[df.label == lbl]["mahalanobis"].dropna().clip(0, hi_clip)
    ax.hist(vals, bins=bins, color=CLR[lbl], alpha=0.55,
            density=True, label=NICE[lbl], edgecolor="none")

thr = float(df[df.label == "healthy"]["mahalanobis_threshold"].dropna().median())
ax.axvline(thr, color="black", ls="--", lw=1.5, label=f"Threshold ({thr:.1f})")
auc_v = auroc_vs_healthy("abnormal_v", "mahalanobis")
auc_s = auroc_vs_healthy("svt",        "mahalanobis")
ax.set_xlabel("Mahalanobis distance", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    f"Mahalanobis — main continuous gate\nAUROC vs V: {auc_v:.2f}   AUROC vs SVT: {auc_s:.2f}",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ── Panel 3: what inhibits each unhealthy class ───────────────────────────────
ax = axes[2]
breakdown: dict[str, dict[str, float]] = {}
for lbl in ["abnormal_v", "svt"]:
    sub      = df[df.label == lbl]
    tot      = len(sub)
    inhibited = sub[~sub["permit"].astype(bool)]
    hard     = (inhibited["inhibit_class"] == "hard_rule").sum()
    thresh   = (inhibited["inhibit_class"] == "threshold_other").sum()
    other    = len(inhibited) - hard - thresh
    missed   = int(sub["permit"].sum())
    breakdown[lbl] = {
        "Hard rule\n(coupling/SQI)": hard / tot,
        "Threshold\n(Mahal/Z-score)": thresh / tot,
        "Other": other / tot,
        "Permitted\n(missed)": missed / tot,
    }

cats  = list(list(breakdown.values())[0].keys())
x     = np.arange(len(cats))
width = 0.35
for i, (lbl, vals) in enumerate(breakdown.items()):
    rates = [vals[c] for c in cats]
    offset = (i - 0.5) * width
    bars = ax.bar(
        x + offset, rates, width,
        label=NICE[lbl], color=CLR[lbl],
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    for bar, rate in zip(bars, rates):
        if rate > 0.03:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=8,
            )

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("Fraction of total beats", fontsize=11)
ax.set_title("What inhibits each class?\n(as fraction of all beats of that type)", fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.85)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_path = OUT / "all_class_gate_analysis.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
