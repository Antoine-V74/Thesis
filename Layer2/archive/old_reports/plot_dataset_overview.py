"""
One-figure summary: beat composition + gate performance per dataset x label.
Output: Results/feature_importance/dataset_overview.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "Results" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "Results" / "layer2" / "cross_dataset" / "per_beat.csv", low_memory=False)
df = df[(df.benchmark_mode == "zero_shot") & (df.feature_set == "all")].copy()

DS_ORDER  = ["mitdb", "incartdb", "nstdb", "svdb"]
DS_LABELS = ["MIT-BIH", "INCART", "NSTDB", "SVDB"]
LAB_ORDER = ["healthy", "svt", "abnormal_v"]
LAB_NICE  = {"healthy": "Healthy (N)", "svt": "SVT / PAC", "abnormal_v": "Ventricular (V/PVC)"}
LAB_CLR   = {"healthy": "#4C9BE8", "svt": "#F5A623", "abnormal_v": "#E8614C"}

# ── Symbol legend (what each WFDB symbol means) ──────────────────────────────
SYM_DESC = {
    "N": "Normal", "L": "LBBB", "R": "RBBB", "e": "Atrial escape",
    "j": "Junctional escape",
    "A": "PAC", "a": "Aberrant PAC", "J": "Junctional premature", "S": "SVT beat",
    "V": "PVC", "E": "Ventricular escape", "F": "Fusion", "f": "Fusion (paced)",
    "!": "Ventricular flutter", "/": "Paced beat",
}

# Build per-(dataset, label) stats
rows = []
for ds in DS_ORDER:
    grp = df[df.dataset == ds]
    tot = len(grp)
    for lbl in LAB_ORDER:
        sub = grp[grp.label == lbl]
        if len(sub) == 0:
            continue
        syms = sorted(sub.beat_symbol.unique())
        rows.append({
            "dataset": ds,
            "label": lbl,
            "n": len(sub),
            "pct": len(sub) / tot * 100,
            "permit": sub.permit.mean() * 100,
            "inhibit": (1 - sub.permit.mean()) * 100,
            "symbols": ", ".join(syms),
        })
stats = pd.DataFrame(rows)

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Dataset composition and Layer 2 gate performance — zero-shot, oracle L1",
    fontsize=14, fontweight="bold", y=0.98,
)

gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35,
                      top=0.90, bottom=0.18, left=0.06, right=0.97)

# ── Row 0: stacked bar — beat composition ─────────────────────────────────────
for col_i, (ds, ds_lbl) in enumerate(zip(DS_ORDER, DS_LABELS)):
    ax = fig.add_subplot(gs[0, col_i])
    sub = stats[stats.dataset == ds]
    bottom = 0.0
    for lbl in LAB_ORDER:
        row = sub[sub.label == lbl]
        if row.empty:
            continue
        pct = float(row.pct.iloc[0])
        bar = ax.bar(0, pct, bottom=bottom, color=LAB_CLR[lbl], width=0.6, edgecolor="white")
        if pct > 3:
            ax.text(0, bottom + pct / 2, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        bottom += pct

    # total beats annotation
    tot = int(stats[stats.dataset == ds]["n"].sum())
    ax.set_title(f"{ds_lbl}\n(n={tot:,})", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of all beats" if col_i == 0 else "", fontsize=9)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)

# ── Row 1: inhibit rate by beat class ─────────────────────────────────────────
for col_i, (ds, ds_lbl) in enumerate(zip(DS_ORDER, DS_LABELS)):
    ax = fig.add_subplot(gs[1, col_i])
    sub = stats[stats.dataset == ds]

    x     = np.arange(len(LAB_ORDER))
    vals  = []
    clrs  = []
    syms_list = []
    for lbl in LAB_ORDER:
        row = sub[sub.label == lbl]
        vals.append(float(row.inhibit.iloc[0]) if not row.empty else 0.0)
        clrs.append(LAB_CLR[lbl])
        syms_list.append(row.symbols.values[0] if not row.empty else "—")

    bars = ax.bar(x, vals, color=clrs, edgecolor="white", linewidth=0.5)
    for bar, v, sym in zip(bars, vals, syms_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, -7,
                f"[{sym}]", ha="center", va="top", fontsize=7, color="#555555")

    ax.axhline(95, color="green", ls="--", lw=1, alpha=0.7)
    ax.axhline(10, color="red",   ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Inhibit rate (%)" if col_i == 0 else "", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Healthy", "SVT/PAC", "Ventricular"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

# ── Legend + symbol key ───────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=LAB_CLR[l], label=LAB_NICE[l]) for l in LAB_ORDER]
fig.legend(handles=legend_patches, loc="lower left", bbox_to_anchor=(0.01, 0.01),
           fontsize=9, frameon=True, title="Beat class", title_fontsize=9)

# Symbol description box
sym_text = "WFDB symbols:  " + "  |  ".join(
    f"{s}={d}" for s, d in SYM_DESC.items()
)
fig.text(0.30, 0.03, sym_text, fontsize=7.5, color="#333333",
         ha="left", va="bottom", wrap=True,
         bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.5))

# Row labels
fig.text(0.005, 0.73, "Beat\ncomposition", fontsize=9, va="center",
         rotation=90, color="#444444", fontweight="bold")
fig.text(0.005, 0.37, "Gate\nperformance", fontsize=9, va="center",
         rotation=90, color="#444444", fontweight="bold")

out_path = OUT / "dataset_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
