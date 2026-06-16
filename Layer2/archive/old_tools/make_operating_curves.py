"""
HP vs AI operating curves for Layer 2.

Two threshold knobs are swept:
  1. Mahalanobis scale  — uses per_beat.csv (mahalanobis / mahalanobis_threshold)
  2. Coupling threshold — uses per_beat_with_coupling.csv (rr__beat_coupling_ratio_posthoc)

All data is oracle-mode, zero_shot, feature_set=all.

Outputs (all in Results/slides/):
  operating_curves_explanation.png      — conceptual ROC vs HP/AI
  operating_curves_mahal.png            — Mahalanobis sweep (all datasets)
  operating_curves_coupling.png         — Coupling threshold sweep
  operating_curves_combined.png         — Both knobs together per dataset (2x2)

Run from ECG Processing root:
    .venv\\Scripts\\python Layer2\\make_operating_curves.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
RES     = ROOT / "Results"
OUT     = RES / "slides"
OUT.mkdir(parents=True, exist_ok=True)

PB_CSV  = RES / "layer2" / "cross_dataset_causal_100ms" / "per_beat.csv"
COU_CSV = RES / "layer2" / "analysis" / "coupling_sweep" / "per_beat_with_coupling.csv"

NAVY  = (26/255, 35/255, 78/255)
DARK  = (16/255, 24/255, 53/255)
BLUE  = (39/255, 128/255, 185/255)
RED   = (231/255, 76/255, 60/255)
GREEN = (39/255, 174/255, 96/255)
ORAN  = (230/255, 126/255, 34/255)
PURP  = (142/255, 68/255, 173/255)
WHITE = (1.0, 1.0, 1.0)
LGRAY = (0.94, 0.94, 0.94)

DS_COL = {"mitdb": BLUE, "svdb": GREEN, "incartdb": ORAN, "nstdb": PURP}
DS_LBL = {
    "mitdb":    "MIT-BIH  (48 records)",
    "svdb":     "SVDB  (78 records)",
    "incartdb": "INCART  (75 records)",
    "nstdb":    "NSTDB  (15 records)",
}


# ─── data loading ─────────────────────────────────────────────────────────────

def load_mahal() -> pd.DataFrame:
    """Oracle rows: mahalanobis score + threshold + label."""
    print("Loading per_beat.csv (oracle only)…")
    wanted_cols = ["dataset", "label", "mahalanobis", "mahalanobis_threshold",
                   "max_zscore", "zscore_threshold"]
    chunks = []
    for chunk in pd.read_csv(PB_CSV, chunksize=200_000,
                             usecols=wanted_cols, low_memory=False):
        mask = chunk["dataset"].isin(DS_COL.keys())
        chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    # Keep only healthy / abnormal (drop svt for the pure gate analysis)
    df = df[df["label"].isin(["healthy", "abnormal", "abnormal_v", "abnormal_s"])]
    df["is_abnormal"] = df["label"].str.startswith("abnormal")
    print(f"  {len(df):,} rows  ({df['is_abnormal'].sum():,} abnormal)")
    return df


def load_coupling() -> pd.DataFrame:
    """Oracle rows with post-hoc coupling ratio."""
    print("Loading per_beat_with_coupling.csv…")
    wanted = ["dataset", "label", "mahalanobis", "mahalanobis_threshold",
              "rr__beat_coupling_ratio_posthoc"]
    df = pd.read_csv(COU_CSV, usecols=wanted, low_memory=False)
    df = df[df["dataset"].isin(DS_COL.keys())]
    df = df[df["label"].isin(["healthy", "abnormal", "abnormal_v", "abnormal_s"])]
    df["is_abnormal"] = df["label"].str.startswith("abnormal")
    df["coupling"] = df["rr__beat_coupling_ratio_posthoc"].clip(0, 1.5)
    print(f"  {len(df):,} rows  ({df['is_abnormal'].sum():,} abnormal)")
    return df


# ─── sweep functions ──────────────────────────────────────────────────────────

def sweep_mahal(sub: pd.DataFrame, n: int = 300):
    """Sweep Mahalanobis threshold scale from very strict to very loose."""
    h = sub[~sub["is_abnormal"]]
    a = sub[ sub["is_abnormal"]]
    if len(h) == 0 or len(a) == 0:
        return np.array([]), np.array([]), np.array([])
    hr = h["mahalanobis"] / h["mahalanobis_threshold"].replace(0, np.nan)
    ar = a["mahalanobis"] / a["mahalanobis_threshold"].replace(0, np.nan)
    scales = np.linspace(0.05, 5.0, n)
    hp = np.array([(hr < s).mean() for s in scales])
    ai = np.array([(ar >= s).mean() for s in scales])
    return scales, hp, ai


def sweep_coupling(sub: pd.DataFrame, n: int = 300):
    """
    Sweep the coupling hard-rule threshold from 0.60 to 0.99.
    Inhibit  if coupling < threshold  OR  Mahalanobis >= threshold.
    """
    h = sub[~sub["is_abnormal"]]
    a = sub[ sub["is_abnormal"]]
    if len(h) == 0 or len(a) == 0:
        return np.array([]), np.array([]), np.array([])
    h_coup = h["coupling"].fillna(1.0)
    a_coup = a["coupling"].fillna(1.0)
    h_mok  = h["mahalanobis"] < h["mahalanobis_threshold"]
    a_mok  = a["mahalanobis"] < a["mahalanobis_threshold"]
    thresholds = np.linspace(0.60, 0.99, n)
    hp = np.array([((h_coup >= t) & h_mok).mean() for t in thresholds])
    ai = np.array([((a_coup <  t) | ~a_mok).mean() for t in thresholds])
    return thresholds, hp, ai


def default_point_mahal(sub: pd.DataFrame):
    h = sub[~sub["is_abnormal"]]
    a = sub[ sub["is_abnormal"]]
    if len(h) == 0 or len(a) == 0:
        return None, None
    hp = (h["mahalanobis"] < h["mahalanobis_threshold"]).mean()
    ai = (a["mahalanobis"] >= a["mahalanobis_threshold"]).mean()
    return float(hp)*100, float(ai)*100


# ─── figure helpers ───────────────────────────────────────────────────────────

def ax_style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor((0.25, 0.31, 0.44))
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=10)
    if title:  ax.set_title(title, color="white", fontsize=11, pad=5)

def iso_fp_lines(ax, fp_list=((1,0.75),(3,0.55),(5,0.4),(10,0.3))):
    for fp_pct, alpha in fp_list:
        ai_val = 100 - fp_pct
        ax.axhline(ai_val, color="white", lw=0.9, ls="--", alpha=alpha)
        ax.text(1.5, ai_val + 0.5, f"FP = {fp_pct}%",
                color="white", fontsize=8, alpha=min(1, alpha+0.2))

def add_legend(ax, **kwargs):
    h, l = ax.get_legend_handles_labels()
    visible = [(hh, ll) for hh, ll in zip(h, l) if not ll.startswith("_")]
    if visible:
        ax.legend(*zip(*visible), facecolor=NAVY, labelcolor="white",
                  framealpha=0.85, **kwargs)


# ─── FIGURE 1: conceptual explanation ─────────────────────────────────────────

def fig_explanation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(NAVY)
    fig.suptitle("ROC Curve vs HP / AI Operating Curve — What is the difference?",
                 color="white", fontsize=14, y=1.01)

    # ── Left: classic ROC ─────────────────────────────────────────────────────
    ax_style(ax1, "False Positive Rate (FPR) %\n= False Permit / all healthy beats",
             "True Positive Rate (TPR) %\n= Abnormal Inhibit / all abnormal beats",
             "Standard ROC Curve")

    fpr = np.linspace(0, 1, 300)
    tpr_good = 1 - (1 - fpr)**4         # high-AUROC gate
    tpr_poor = 1 - (1 - fpr)**1.3       # low-AUROC gate
    ax1.plot(fpr*100, tpr_good*100, color=BLUE, lw=2.5, label="Gate A  (AUROC = 0.94)")
    ax1.plot(fpr*100, tpr_poor*100, color=RED,  lw=2.0, ls="--", label="Gate B  (AUROC = 0.72)")
    ax1.plot([0,100],[0,100], color="white", lw=0.7, ls=":", alpha=0.4, label="Random (AUROC = 0.50)")
    ax1.fill_between(fpr*100, tpr_good*100, alpha=0.12, color=BLUE)

    # operating point
    idx = np.argmin(np.abs(fpr - 0.05))
    ax1.scatter([fpr[idx]*100], [tpr_good[idx]*100], color="white", s=120,
                zorder=10, marker="*")
    ax1.annotate(f"  threshold=0.5\n  FPR={fpr[idx]*100:.0f}% TPR={tpr_good[idx]*100:.0f}%",
                 (fpr[idx]*100, tpr_good[idx]*100), color="white", fontsize=8,
                 xytext=(12,-20), textcoords="offset points")

    ax1.text(55, 15, "AUROC = 0.94\n(area under\nblue curve)", color=BLUE,
             fontsize=11, fontweight="bold", ha="center")
    ax1.set_xlim(0,100); ax1.set_ylim(0,105)
    add_legend(ax1, fontsize=9, loc="lower right")

    # ── Right: HP vs AI ───────────────────────────────────────────────────────
    ax_style(ax2, "Healthy Permit rate (HP) %\n= permit / all healthy beats",
             "Abnormal Inhibit rate (AI) %\n= inhibit / all abnormal beats",
             "HP / AI Operating Curve  (Layer 2 Mahalanobis scale swept)")

    scale = np.linspace(0.05, 5, 300)
    hp_sim = 100 / (1 + np.exp(-3.2*(scale - 1.05)))
    ai_sim = 100 / (1 + np.exp(+2.5*(scale - 1.20)))

    ax2.plot(hp_sim, ai_sim, color=ORAN, lw=3, label="Layer 2 gate")

    # Pareto-optimal front (neither metric can improve without hurting the other)
    pareto = (hp_sim >= 55) & (ai_sim >= 75)
    if pareto.any():
        ax2.plot(hp_sim[pareto], ai_sim[pareto], color=GREEN, lw=5,
                 alpha=0.6, label="Pareto-optimal segment")

    # mark default operating point
    ax2.scatter([82], [95], color="white", s=160, zorder=12, marker="*")
    ax2.annotate("  Default\n  HP=82%, AI=95%\n  FP=5%",
                 (82, 95), color="white", fontsize=9,
                 xytext=(4, -25), textcoords="offset points")

    # annotations for sweep direction
    ax2.annotate("", xy=(30, 98), xytext=(70, 88),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax2.text(35, 99, "stricter gate\n(lower HP, higher AI)", color=RED, fontsize=8)
    ax2.annotate("", xy=(95, 65), xytext=(85, 85),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax2.text(86, 60, "looser gate\n(higher HP, lower AI)", color=GREEN, fontsize=8)

    iso_fp_lines(ax2, [(1,0.7),(3,0.55),(5,0.45),(10,0.3)])
    ax2.fill_between([75,101],[92,92],[101,101], color=GREEN, alpha=0.10)
    ax2.text(76, 93, "Target zone\nHP>75%, AI>92%", color=GREEN, fontsize=8, alpha=0.75)

    ax2.set_xlim(0, 101); ax2.set_ylim(50, 101)
    add_legend(ax2, fontsize=9, loc="lower right")

    # bottom text boxes
    for ax, col, txt in [
        (ax1, BLUE, "ROC tells you the classifier's separability.\nAUROC = single number comparing gate variants."),
        (ax2, ORAN, "HP/AI shows you WHERE to set the threshold.\nRead off: 'FP \u2264 5%' gives HP = XX%."),
    ]:
        ax.text(0.5, -0.22, txt, transform=ax.transAxes,
                ha="center", fontsize=9, color=LGRAY,
                bbox=dict(facecolor=NAVY, edgecolor=col, boxstyle="round,pad=0.4"))

    fig.tight_layout(pad=1.5)
    return fig


# ─── FIGURE 2: Mahalanobis sweep, all datasets ────────────────────────────────

def fig_mahal(df_mahal: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(NAVY)
    ax_style(ax,
             "Healthy Permit rate (HP) %",
             "Abnormal Inhibit rate (AI) %",
             "HP vs AI — Mahalanobis Threshold Sweep\n"
             "(oracle gate, causal 100 ms, zero-shot calibration)")

    for ds, col in DS_COL.items():
        sub = df_mahal[df_mahal["dataset"] == ds]
        scales, hp, ai = sweep_mahal(sub)
        if len(hp) == 0:
            continue
        ax.plot(hp*100, ai*100, color=col, lw=2.8, alpha=0.9, label=DS_LBL[ds])

        # arrow showing sweep direction (left=strict → right=loose)
        mid = len(hp) // 2
        dx = hp[mid+5] - hp[mid-5]
        dy = ai[mid+5] - ai[mid-5]
        ax.annotate("", xy=(hp[mid+5]*100, ai[mid+5]*100),
                    xytext=(hp[mid-5]*100, ai[mid-5]*100),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))

        # default operating point
        hp0, ai0 = default_point_mahal(sub)
        if hp0:
            ax.scatter([hp0], [ai0], color=col, s=180, zorder=12,
                       marker="o", edgecolors="white", linewidths=1.5)
            ax.annotate(f"  {ds.upper()}\n  HP={hp0:.0f}%, AI={ai0:.0f}%",
                        (hp0, ai0), color=col, fontsize=9,
                        xytext=(7, -12), textcoords="offset points")

    iso_fp_lines(ax, [(1,0.75),(3,0.55),(5,0.45),(10,0.3)])
    ax.fill_between([70,101],[92,92],[101,101], color=GREEN, alpha=0.10)
    ax.text(71, 92.8, "HP>70%, AI>92%", color=GREEN, fontsize=9, alpha=0.7)

    ax.set_xlim(0, 101); ax.set_ylim(50, 101)
    ax.text(30, 52,
            "Left end of curve = very strict threshold (FP\u22480%, but HP very low)\n"
            "Right end = very loose threshold (HP\u2248100%, but AI drops)\n"
            "Circles = calibrated default (99.9th pct of healthy calibration beats)",
            color=LGRAY, fontsize=9,
            bbox=dict(facecolor=DARK, edgecolor=(0.25,0.31,0.44),
                      boxstyle="round,pad=0.4"))
    add_legend(ax, fontsize=11, loc="lower right")
    fig.tight_layout()
    return fig


# ─── FIGURE 3: Coupling threshold sweep, all datasets ─────────────────────────

def fig_coupling(df_coup: pd.DataFrame):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(NAVY)
    fig.suptitle("HP vs AI — Coupling Hard-Rule Threshold Sweep\n"
                 "(oracle gate + coupling veto, causal 100 ms)",
                 color="white", fontsize=13, y=1.01)

    # ── Left: HP/AI curve ─────────────────────────────────────────────────────
    ax_style(ax, "Healthy Permit rate (HP) %",
             "Abnormal Inhibit rate (AI) %",
             "Sweep coupling threshold (0.60 → 0.99)")

    key_thr = [0.70, 0.75, 0.80, 0.85, 0.90]
    bar_hp = {ds: [] for ds in DS_COL}
    bar_ai = {ds: [] for ds in DS_COL}

    for ds, col in DS_COL.items():
        sub = df_coup[df_coup["dataset"] == ds]
        thr, hp, ai = sweep_coupling(sub)
        if len(hp) == 0:
            continue
        ax.plot(hp*100, ai*100, color=col, lw=2.5, alpha=0.9, label=DS_LBL[ds])

        # annotate key threshold values
        for kt in key_thr:
            idx = np.argmin(np.abs(thr - kt))
            ax.scatter([hp[idx]*100], [ai[idx]*100], color=col,
                       s=50, zorder=10, edgecolors="white", linewidths=0.8)
            if ds == "mitdb":
                ax.annotate(f"{kt}", (hp[idx]*100, ai[idx]*100),
                            color="white", fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            bar_hp[ds].append(hp[idx]*100)
            bar_ai[ds].append(ai[idx]*100)

        # arrow direction
        mid = len(hp) // 2
        ax.annotate("", xy=(hp[mid+5]*100, ai[mid+5]*100),
                    xytext=(hp[mid-5]*100, ai[mid-5]*100),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    iso_fp_lines(ax, [(1,0.75),(3,0.55),(5,0.4)])
    ax.set_xlim(0, 101); ax.set_ylim(50, 101)
    add_legend(ax, fontsize=9, loc="lower right")
    ax.text(1, 51.5,
            "Right end (threshold=0.60): few beats vetoed by coupling alone\n"
            "Left end  (threshold=0.99): almost all beats vetoed → very high AI, very low HP",
            color=LGRAY, fontsize=8,
            bbox=dict(facecolor=DARK, edgecolor=(0.25,0.31,0.44),
                      boxstyle="round,pad=0.3"))

    # ── Right: grouped bar chart at key thresholds ────────────────────────────
    ax_style(ax2, "Coupling threshold",
             "Rate (%)",
             "HP (solid) and AI (hatched) at key coupling values")

    x = np.arange(len(key_thr))
    bw = 0.18
    for i, (ds, col) in enumerate(DS_COL.items()):
        hp_v = bar_hp[ds]
        ai_v = bar_ai[ds]
        if len(hp_v) != len(key_thr):
            continue
        ax2.bar(x + (i-1.5)*bw, hp_v, bw, color=col, alpha=0.8,
                label=ds.upper())
        ax2.bar(x + (i-1.5)*bw, ai_v, bw, color=col, alpha=0.35,
                bottom=[0]*len(ai_v), hatch="///", zorder=0)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in key_thr], color="white", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.axhline(90, color="white", lw=0.8, ls="--", alpha=0.4)
    ax2.text(len(key_thr)-0.5, 91, "90%", color="white", fontsize=8, alpha=0.5)

    patches = [mpatches.Patch(color=DS_COL[ds], label=f"{ds.upper()} HP (solid) / AI (hatch)")
               for ds in DS_COL]
    ax2.legend(handles=patches, fontsize=8, facecolor=NAVY, labelcolor="white",
               framealpha=0.85, loc="lower right")
    ax2.tick_params(colors="white")

    fig.tight_layout(pad=1.5)
    return fig


# ─── FIGURE 4: Both knobs, per-dataset 2×2 ────────────────────────────────────

def fig_combined(df_mahal: pd.DataFrame, df_coup: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor(NAVY)
    fig.suptitle(
        "HP vs AI Operating Curves per Dataset — Mahalanobis Scale vs Coupling Threshold\n"
        "Circles = calibrated default  |  Numbers on curves = coupling threshold value",
        color="white", fontsize=13, y=1.00)

    datasets = ["mitdb", "svdb", "incartdb", "nstdb"]
    key_thr  = [0.70, 0.80, 0.90]

    for ax, ds in zip(axes.flat, datasets):
        col = DS_COL[ds]
        ax_style(ax, "HP (%)", "AI (%)", DS_LBL[ds])

        # Mahalanobis sweep
        sub_m = df_mahal[df_mahal["dataset"] == ds]
        sc, hp_m, ai_m = sweep_mahal(sub_m)
        if len(hp_m):
            ax.plot(hp_m*100, ai_m*100, color=BLUE, lw=2.5,
                    label="Mahalanobis scale sweep")
            mid = len(hp_m) // 2
            ax.annotate("", xy=(hp_m[mid+5]*100, ai_m[mid+5]*100),
                        xytext=(hp_m[mid-5]*100, ai_m[mid-5]*100),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
            hp0, ai0 = default_point_mahal(sub_m)
            if hp0:
                ax.scatter([hp0], [ai0], color="white", s=150, zorder=12,
                           marker="*")
                ax.annotate(f"  default\n  {hp0:.0f}/{ai0:.0f}%",
                            (hp0, ai0), color="white", fontsize=8,
                            xytext=(4,-18), textcoords="offset points")

        # Coupling sweep
        sub_c = df_coup[df_coup["dataset"] == ds]
        thr, hp_c, ai_c = sweep_coupling(sub_c)
        if len(hp_c):
            ax.plot(hp_c*100, ai_c*100, color=ORAN, lw=2.5, ls="--",
                    label="Coupling threshold sweep")
            for kt in key_thr:
                idx = np.argmin(np.abs(thr - kt))
                ax.scatter([hp_c[idx]*100], [ai_c[idx]*100],
                           color=ORAN, s=60, zorder=10,
                           edgecolors="white", linewidths=0.8)
                ax.annotate(f"{kt}", (hp_c[idx]*100, ai_c[idx]*100),
                            color=ORAN, fontsize=8,
                            xytext=(3, 3), textcoords="offset points")

        iso_fp_lines(ax, [(1,0.65),(5,0.4),(10,0.25)])
        ax.fill_between([70,101],[90,90],[101,101], color=GREEN, alpha=0.10)
        ax.set_xlim(0,101); ax.set_ylim(50,101)
        add_legend(ax, fontsize=8, loc="lower right")

    fig.tight_layout(pad=0.8)
    return fig


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    df_mahal = load_mahal()
    df_coup  = load_coupling()

    figs = [
        ("operating_curves_explanation.png", fig_explanation()),
        ("operating_curves_mahal.png",        fig_mahal(df_mahal)),
        ("operating_curves_coupling.png",     fig_coupling(df_coup)),
        ("operating_curves_combined.png",     fig_combined(df_mahal, df_coup)),
    ]
    for fname, fig in figs:
        out = OUT / fname
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
