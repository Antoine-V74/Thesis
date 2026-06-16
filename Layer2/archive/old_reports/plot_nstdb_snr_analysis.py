"""
Two-panel figure:
  Left  — Beat class definitions table
  Right — NSTDB gate performance vs SNR level (zero-shot, all features, oracle)
Output: Results/feature_importance/nstdb_snr_analysis.png
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
OUT  = ROOT / "Results" / "layer2" / "analysis" / "feature_importance"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load NSTDB SNR data ───────────────────────────────────────────────────────
snr_df = pd.read_csv(ROOT / "Results" / "layer2" / "cross_dataset" / "nstdb_snr_breakdown.csv")
snr_df = snr_df[
    (snr_df.benchmark_mode == "zero_shot") &
    (snr_df.feature_set    == "all") &
    (snr_df.eval_mode      == "oracle")
].copy()

# Parse SNR level from dataset name  e.g. "nstdb_snr+12" → 12
def parse_snr(s: str) -> int:
    return int(s.replace("nstdb_snr", "").replace("+", ""))

snr_df["snr_db"] = snr_df["dataset"].apply(parse_snr)
snr_df = snr_df.sort_values("snr_db").reset_index(drop=True)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.suptitle(
    "Beat class definitions  ·  NSTDB noise-stress analysis",
    fontsize=14, fontweight="bold", y=0.98,
)
gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.01, right=0.98,
                      top=0.88, bottom=0.10)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — definitions table
# ══════════════════════════════════════════════════════════════════════════════
ax_tbl = fig.add_subplot(gs[0])
ax_tbl.axis("off")

# Table data
col_labels = ["Class", "WFDB\nsymbols", "Clinical meaning", "Dangerous for\nstimulation?", "Gate target"]
rows_data = [
    ["Healthy\n(N)", "N, L, R,\ne, j",
     "Normal sinus beat\nLBBB / RBBB\nJunctional / atrial escape",
     "No — normal rhythm,\nsafe to stimulate",
     "PERMIT\n(high HP target)"],
    ["SVT / PAC\n(S/A)", "A, a, S,\nJ, j",
     "Premature atrial contraction (PAC)\nJunctional premature\nSupraventricular tachycardia beat",
     "Usually no — morphology\nnormal, only timing early\n(policy-dependent)",
     "INHIBIT or PERMIT\n(ambiguous — conservative\npolicy → inhibit)"],
    ["Ventricular\n(V/PVC)", "V, E, F, f,\n!, /",
     "Premature ventricular contraction (PVC)\nVentricular escape beat\nFusion beat (QRS overlap)\nVentricular flutter / paced",
     "YES — abnormal QRS\nmorphology, risk of\ninducing arrhythmia",
     "INHIBIT\n(primary safety target,\n≥ 95% AI goal)"],
]

CLR_ROW = ["#D6EAF8", "#FEF9E7", "#FADBD8"]

tbl = ax_tbl.table(
    cellText=rows_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 3.8)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("#1B2A4A")
    cell.set_text_props(color="white", fontweight="bold")

# Style rows
for i, clr in enumerate(CLR_ROW, start=1):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(clr)
        tbl[i, j].set_edgecolor("#cccccc")

ax_tbl.set_title("Beat class definitions (WFDB annotation standard)",
                 fontsize=11, fontweight="bold", pad=8)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — NSTDB SNR performance
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1])

snr_vals    = snr_df["snr_db"].values
hp          = snr_df["healthy_permit"].values * 100
ai          = snr_df["abnormal_inhibit"].values * 100
fi          = snr_df["false_inhibit"].values * 100
fp          = snr_df["false_permit"].values * 100
svt_inh     = snr_df["svt_inhibit"].values * 100

x = np.arange(len(snr_vals))
width = 0.18

# Grouped bars
b1 = ax.bar(x - 1.5*width, hp,      width, label="Healthy permit (HP)",      color="#4C9BE8", alpha=0.85)
b2 = ax.bar(x - 0.5*width, ai,      width, label="Ventricular inhibit (AI)", color="#E8614C", alpha=0.85)
b3 = ax.bar(x + 0.5*width, fp*100/100, width, label="False permit (FP)",     color="#E67E22", alpha=0.85)
b4 = ax.bar(x + 1.5*width, fi,      width, label="False inhibit (FI)",       color="#A569BD", alpha=0.85)

# Value labels on top of bars
for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)

# Reference lines
ax.axhline(95, color="#E8614C", ls="--", lw=1.2, alpha=0.6, label="95% AI target")
ax.axhline(10, color="#4C9BE8", ls="--", lw=1.2, alpha=0.6, label="10% FI budget")

# Shade the danger zone (SNR < 6 dB)
threshold_idx = next(i for i, v in enumerate(snr_vals) if v >= 6)
ax.axvspan(-0.5, threshold_idx - 0.5, alpha=0.07, color="red")
ax.text(threshold_idx/2 - 0.5, 103, "Below\noperational\nthreshold",
        ha="center", va="top", fontsize=8, color="#cc0000", fontstyle="italic")
ax.axvline(threshold_idx - 0.5, color="red", ls=":", lw=1.5, alpha=0.7)
ax.text(threshold_idx - 0.5, 55, "SNR = +6 dB\nboundary",
        ha="center", va="center", fontsize=7.5, color="#cc0000",
        bbox=dict(fc="white", ec="#cc0000", lw=0.5, pad=2))

ax.set_xticks(x)
ax.set_xticklabels([f"{v:+d} dB" for v in snr_vals], fontsize=10)
ax.set_xlabel("NSTDB noise level (SNR)", fontsize=11)
ax.set_ylabel("Rate (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_title(
    "NSTDB gate performance vs noise level\n(zero-shot · all features · oracle L1)",
    fontsize=11, fontweight="bold",
)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)

# Annotation box
note = (
    "NSTDB noise type: electrode motion (1–10 Hz, low-freq)\n"
    "→ corrupts RR features, bypasses HF-SQI gate\n"
    "→ caught by max_zscore / Mahalanobis at SNR ≥ 12 dB\n"
    "Operational envelope: SNR ≥ 12 dB recommended"
)
ax.text(0.99, 0.03, note, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8, color="#333",
        bbox=dict(fc="#f9f9f9", ec="#cccccc", lw=0.5, pad=5))

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = OUT / "nstdb_snr_analysis.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
