"""
Quick focused plot: top-5 feature pairplot + marginal histograms.
Shows healthy vs ventricular beat separation on INCART data.

Output: Results/feature_importance/top5_pairplot.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "Results" / "layer2" / "analysis" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)

CSV  = ROOT / "Results" / "incart_pvc_analysis" / "beat_features.csv"

# Top 5 by AUROC (from previous analysis)
TOP5 = [
    ("rr__beat_coupling_ratio",   "Beat coupling\nratio",    "AUROC 0.95"),
    ("morph__neighbor_corr",      "Neighbor beat\ncorr.",    "AUROC 0.93"),
    ("morph__template_corr",      "Template\ncorr.",         "AUROC 0.90"),
    ("morph__qrs_width_ms",       "QRS width\n(ms)",         "AUROC 0.80"),
    ("rr__rr_cv",                 "RR coeff.\nof variation", "AUROC 0.79"),
]

COLS   = [c for c, _, _ in TOP5]
LABELS = [l for _, l, _ in TOP5]
AUROCS = [a for _, _, a in TOP5]

CLR = {"Healthy": "#4C9BE8", "Ventricular": "#E8614C"}
ALPHA = 0.18
EDGE  = 0.55
MS    = 3

df = pd.read_csv(CSV, low_memory=False)
df = df[df["label"].isin(["healthy", "abnormal_v"])].copy()
df["class"] = df["label"].map({"healthy": "Healthy", "abnormal_v": "Ventricular"})

N = len(TOP5)
fig, axes = plt.subplots(N, N, figsize=(14, 13))
fig.suptitle(
    "Feature pairplot — Healthy vs Ventricular beats (INCART)\n"
    "Blue = healthy  ·  Red = ventricular",
    fontsize=13, fontweight="bold", y=0.995,
)

for i, (ci, li, ai) in enumerate(TOP5):
    for j, (cj, lj, aj) in enumerate(TOP5):
        ax = axes[i, j]

        if i == j:
            # Diagonal: marginal distribution
            for cls, clr in CLR.items():
                sub = df[df["class"] == cls][ci].dropna()
                # Use percentile clipping for readability
                lo, hi = np.nanpercentile(df[ci].dropna(), [1, 99])
                sub = sub.clip(lo, hi)
                ax.hist(sub, bins=50, color=clr, alpha=0.55, density=True,
                        edgecolor="none")
            ax.set_xlabel(li, fontsize=8, labelpad=2)
            ax.set_ylabel("Density", fontsize=7)
            ax.set_title(ai, fontsize=8, color="#444444", pad=2)
            ax.tick_params(labelsize=7)

        else:
            # Off-diagonal: scatter
            for cls, clr in CLR.items():
                sub = df[df["class"] == cls][[cj, ci]].dropna()
                # clip to 1st–99th percentile for both axes
                xlo, xhi = np.nanpercentile(df[cj].dropna(), [1, 99])
                ylo, yhi = np.nanpercentile(df[ci].dropna(), [1, 99])
                sub = sub[
                    sub[cj].between(xlo, xhi) & sub[ci].between(ylo, yhi)
                ]
                # subsample if too many points
                if len(sub) > 3000:
                    sub = sub.sample(3000, random_state=42)
                ax.scatter(
                    sub[cj], sub[ci],
                    c=clr, alpha=ALPHA, s=MS,
                    linewidths=0, rasterized=True,
                )
            ax.tick_params(labelsize=6)

        # Column labels on top row, row labels on left col
        if i == 0 and j != 0:
            ax.set_title(lj, fontsize=8, pad=4)
        if j == 0 and i != 0:
            ax.set_ylabel(li, fontsize=8, labelpad=2)
        if i == 0 and j == 0:
            ax.set_title(li, fontsize=8, pad=4)
            ax.set_ylabel(li, fontsize=8, labelpad=2)

        ax.spines[["top","right"]].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=v, label=k, alpha=0.75) for k, v in CLR.items()]
fig.legend(handles=legend_handles, loc="lower right",
           bbox_to_anchor=(0.99, 0.01), fontsize=10, frameon=True)

plt.tight_layout(rect=[0, 0.01, 1, 0.99])
out_path = OUT / "top5_pairplot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
