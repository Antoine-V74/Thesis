"""
Feature importance and PCA analysis for Layer 2.

Outputs (in Results/feature_importance/):
  1. auroc_per_feature.csv   — per-feature AUROC (healthy vs abnormal_v) per dataset
  2. feature_importance.png  — ranked AUROC bar chart, coloured by feature group
  3. pca_2d.png              — PCA scatter (healthy vs abnormal, all datasets)
  4. pca_loadings.png        — PCA component loadings (which features drive each axis)
  5. feature_distributions.png — violin plots for top separating features
  6. feature_group_summary.png — mean AUROC per clinical group, per dataset
  7. top_deviator_counts.png  — how often each feature is the top deviator on inhibited beats
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Layer2"))

BEAT_CSV   = ROOT / "Results" / "incart_pvc_analysis" / "beat_features.csv"
PBEAT_CSV  = ROOT / "Results" / "layer2" / "cross_dataset" / "per_beat.csv"
OUT_DIR    = ROOT / "Results" / "layer2" / "analysis" / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    # --- RR / timing ---
    "rr__beat_coupling_ratio",
    "rr__rr_mean_ms", "rr__hr_bpm",
    "rr__rr_sdnn_ms", "rr__rr_rmssd_ms", "rr__rr_cv",
    "rr__rr_min_ms",  "rr__rr_max_ms",  "rr__rr_range_ms",
    "rr__short_rr_fraction", "rr__long_rr_fraction",
    "rr__rr_diff_abs_mean_ms", "rr__rr_diff_abs_max_ms",
    # --- Morphology ---
    "morph__template_corr", "morph__neighbor_corr",
    "morph__qrs_width_ms",  "morph__beat_amp",
    "morph__amp_vs_median", "morph__post_pre_area_ratio",
    # --- Signal quality / energy ---
    "signal__rms", "signal__peak_to_peak",
    "signal__abs_p95", "signal__abs_p99",
    "signal__line_length", "signal__energy",
    "signal__zero_cross_rate",
    "signal__hf_noise_ratio", "signal__lf_wander_ratio",
    # --- Wavelet energy ---
    "signal__wave_D1_log_energy", "signal__wave_D2_log_energy",
    "signal__wave_D3_log_energy", "signal__wave_D4_log_energy",
    "signal__wave_A_log_energy",
    # --- Wavelet entropy ---
    "signal__wave_D1_shannon_ent", "signal__wave_D2_shannon_ent",
    "signal__wave_D3_shannon_ent", "signal__wave_D4_shannon_ent",
    "signal__wave_A_shannon_ent",
]

# Clinical grouping
GROUP_LABELS = {
    "rr__beat_coupling_ratio":     "Timing",
    "rr__rr_mean_ms":              "Timing",
    "rr__hr_bpm":                  "Timing",
    "rr__rr_sdnn_ms":              "Rhythm variability",
    "rr__rr_rmssd_ms":             "Rhythm variability",
    "rr__rr_cv":                   "Rhythm variability",
    "rr__rr_min_ms":               "Rhythm variability",
    "rr__rr_max_ms":               "Rhythm variability",
    "rr__rr_range_ms":             "Rhythm variability",
    "rr__short_rr_fraction":       "Rhythm variability",
    "rr__long_rr_fraction":        "Rhythm variability",
    "rr__rr_diff_abs_mean_ms":     "Rhythm variability",
    "rr__rr_diff_abs_max_ms":      "Rhythm variability",
    "morph__template_corr":        "Morphology",
    "morph__neighbor_corr":        "Morphology",
    "morph__qrs_width_ms":         "Morphology",
    "morph__beat_amp":             "Morphology",
    "morph__amp_vs_median":        "Morphology",
    "morph__post_pre_area_ratio":  "Morphology",
    "signal__rms":                 "Signal energy",
    "signal__peak_to_peak":        "Signal energy",
    "signal__abs_p95":             "Signal energy",
    "signal__abs_p99":             "Signal energy",
    "signal__line_length":         "Signal energy",
    "signal__energy":              "Signal energy",
    "signal__zero_cross_rate":     "Signal quality",
    "signal__hf_noise_ratio":      "Signal quality",
    "signal__lf_wander_ratio":     "Signal quality",
    "signal__wave_D1_log_energy":  "Wavelet (HF)",
    "signal__wave_D2_log_energy":  "Wavelet (HF)",
    "signal__wave_D3_log_energy":  "Wavelet (MF)",
    "signal__wave_D4_log_energy":  "Wavelet (LF)",
    "signal__wave_A_log_energy":   "Wavelet (LF)",
    "signal__wave_D1_shannon_ent": "Wavelet entropy",
    "signal__wave_D2_shannon_ent": "Wavelet entropy",
    "signal__wave_D3_shannon_ent": "Wavelet entropy",
    "signal__wave_D4_shannon_ent": "Wavelet entropy",
    "signal__wave_A_shannon_ent":  "Wavelet entropy",
}

GROUP_COLORS = {
    "Timing":            "#e74c3c",
    "Rhythm variability":"#e67e22",
    "Morphology":        "#2980b9",
    "Signal energy":     "#27ae60",
    "Signal quality":    "#8e44ad",
    "Wavelet (HF)":      "#1abc9c",
    "Wavelet (MF)":      "#16a085",
    "Wavelet (LF)":      "#2ecc71",
    "Wavelet entropy":   "#3498db",
}

HUMAN_NAMES = {
    "rr__beat_coupling_ratio":     "Beat coupling ratio",
    "rr__rr_mean_ms":              "Mean RR interval",
    "rr__hr_bpm":                  "Heart rate",
    "rr__rr_sdnn_ms":              "RR SDNN",
    "rr__rr_rmssd_ms":             "RR RMSSD",
    "rr__rr_cv":                   "RR coefficient of variation",
    "rr__rr_min_ms":               "Min RR",
    "rr__rr_max_ms":               "Max RR",
    "rr__rr_range_ms":             "RR range",
    "rr__short_rr_fraction":       "Short RR fraction",
    "rr__long_rr_fraction":        "Long RR fraction",
    "rr__rr_diff_abs_mean_ms":     "Mean RR diff",
    "rr__rr_diff_abs_max_ms":      "Max RR diff",
    "morph__template_corr":        "Template correlation",
    "morph__neighbor_corr":        "Neighbor beat correlation",
    "morph__qrs_width_ms":         "QRS width",
    "morph__beat_amp":             "Beat amplitude",
    "morph__amp_vs_median":        "Amplitude vs median",
    "morph__post_pre_area_ratio":  "Post/pre-QRS area ratio",
    "signal__rms":                 "Signal RMS",
    "signal__peak_to_peak":        "Peak-to-peak amplitude",
    "signal__abs_p95":             "95th pct amplitude",
    "signal__abs_p99":             "99th pct amplitude",
    "signal__line_length":         "Signal line length",
    "signal__energy":              "Signal energy",
    "signal__zero_cross_rate":     "Zero-crossing rate",
    "signal__hf_noise_ratio":      "HF noise ratio (filtered)",
    "signal__lf_wander_ratio":     "LF wander ratio",
    "signal__wave_D1_log_energy":  "Wavelet D1 energy (HF)",
    "signal__wave_D2_log_energy":  "Wavelet D2 energy",
    "signal__wave_D3_log_energy":  "Wavelet D3 energy",
    "signal__wave_D4_log_energy":  "Wavelet D4 energy (LF)",
    "signal__wave_A_log_energy":   "Wavelet A energy (baseline)",
    "signal__wave_D1_shannon_ent": "Wavelet D1 entropy",
    "signal__wave_D2_shannon_ent": "Wavelet D2 entropy",
    "signal__wave_D3_shannon_ent": "Wavelet D3 entropy",
    "signal__wave_D4_shannon_ent": "Wavelet D4 entropy",
    "signal__wave_A_shannon_ent":  "Wavelet A entropy",
}

DATASETS = ["mitdb", "incartdb"]  # two main datasets for PCA; use all for AUROC


# ──────────────────────────────────────────────────────────────────────────────
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    m = np.isfinite(s)
    if m.sum() < 20:
        return float("nan")
    return roc_auc_score(y[m], s[m])


def load_incart() -> pd.DataFrame:
    df = pd.read_csv(BEAT_CSV, low_memory=False)
    df["dataset"] = "incartdb"
    return df


def load_mit() -> pd.DataFrame:
    """Re-use per_beat.csv for MIT-BIH; it has per-record scores but not raw features.
    We need a dataset with raw features for MIT-BIH.  Use incart_pvc_analysis cache
    only for INCART; for MIT-BIH we use the top1/top2/top3_feature columns from
    per_beat.csv only for the top-deviator plot."""
    return pd.read_csv(PBEAT_CSV, low_memory=False)


# ──────────────────────────────────────────────────────────────────────────────
# 1. AUROC per feature (INCART has raw features; used as proxy for all)
# ──────────────────────────────────────────────────────────────────────────────
def compute_auroc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue
        h  = df[df["label"] == "healthy"][feat].dropna().values
        ab = df[df["label"] == "abnormal_v"][feat].dropna().values
        if len(h) < 20 or len(ab) < 20:
            continue
        auc = auroc(ab, h)
        # Take the side that matters (>0.5: high value = abnormal; <0.5: low value = abnormal)
        auc = max(auc, 1.0 - auc)
        rows.append({
            "feature":    feat,
            "human_name": HUMAN_NAMES.get(feat, feat),
            "group":      GROUP_LABELS.get(feat, "Other"),
            "auroc":      round(auc, 3),
        })
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Ranked AUROC bar chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_auroc_bar(auroc_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors  = [GROUP_COLORS.get(g, "#95a5a6") for g in auroc_df["group"]]
    y_pos   = np.arange(len(auroc_df))

    bars = ax.barh(y_pos, auroc_df["auroc"], color=colors, height=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Random (0.5)")
    ax.axvline(0.7, color="orange", linestyle=":", linewidth=1.0, alpha=0.8, label="Good (0.7)")
    ax.axvline(0.8, color="red",    linestyle=":", linewidth=1.0, alpha=0.8, label="Strong (0.8)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(auroc_df["human_name"], fontsize=9)
    ax.set_xlabel("AUROC (healthy vs abnormal_v)  |  0.5 = random, 1.0 = perfect", fontsize=10)
    ax.set_title("Feature separability: healthy vs ventricular beats\n(INCART dataset, oracle mode)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.4, 1.02)
    ax.invert_yaxis()

    # Legend for groups
    seen = {}
    for g, c in GROUP_COLORS.items():
        if g in auroc_df["group"].values:
            seen[g] = mpatches.Patch(facecolor=c, label=g)
    ax.legend(handles=list(seen.values()), loc="lower right", fontsize=8, title="Feature group")

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. PCA 2D scatter
# ──────────────────────────────────────────────────────────────────────────────
def plot_pca(df: pd.DataFrame, out_scatter: Path, out_loadings: Path) -> None:
    present = [f for f in FEATURE_COLS if f in df.columns]
    sub = df[df["label"].isin(["healthy", "abnormal_v"])].copy()
    X   = sub[present].copy()
    X   = X.apply(lambda c: c.fillna(c.median()))
    mask = np.all(np.isfinite(X.values), axis=1)
    X   = X[mask]
    labels  = sub["label"].values[mask]
    dataset = sub["dataset"].values[mask] if "dataset" in sub.columns else np.array(["incartdb"] * mask.sum())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(Xs)

    # -- Scatter --
    fig, ax = plt.subplots(figsize=(9, 7))
    col_map = {"healthy": "#2980b9", "abnormal_v": "#e74c3c"}
    alpha_map = {"healthy": 0.15, "abnormal_v": 0.35}

    for lbl in ["healthy", "abnormal_v"]:
        m = labels == lbl
        ax.scatter(Z[m, 0], Z[m, 1],
                   c=col_map[lbl], alpha=alpha_map[lbl],
                   s=8, label=lbl.replace("_", " ").title(), rasterized=True)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA of Layer 2 features\nHealthy vs Ventricular beats (INCART)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_scatter, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_scatter.name}")

    # -- Loadings --
    loadings = pd.DataFrame(
        pca.components_.T,
        index=present,
        columns=["PC1", "PC2"]
    )
    loadings["abs_pc1"] = loadings["PC1"].abs()
    loadings["abs_pc2"] = loadings["PC2"].abs()
    top_pc1 = loadings.nlargest(12, "abs_pc1")
    top_pc2 = loadings.nlargest(12, "abs_pc2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, top, pc, title in zip(
        axes,
        [top_pc1, top_pc2],
        ["PC1", "PC2"],
        [f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
         f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"]
    ):
        names  = [HUMAN_NAMES.get(f, f) for f in top.index]
        values = top[pc].values
        colors = [GROUP_COLORS.get(GROUP_LABELS.get(f, ""), "#95a5a6") for f in top.index]
        y_pos  = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Loading")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    # Shared group legend
    seen = {}
    for g, c in GROUP_COLORS.items():
        seen[g] = mpatches.Patch(facecolor=c, label=g)
    fig.legend(handles=list(seen.values()), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("PCA loadings — which features drive each component", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_loadings, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_loadings.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Violin plots for top-separating features
# ──────────────────────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame, auroc_df: pd.DataFrame, out: Path) -> None:
    top_feats = auroc_df.head(9)["feature"].tolist()
    sub = df[df["label"].isin(["healthy", "abnormal_v"])]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        if feat not in sub.columns:
            ax.set_visible(False)
            continue
        h  = sub[sub["label"] == "healthy"][feat].dropna().values
        ab = sub[sub["label"] == "abnormal_v"][feat].dropna().values

        # Clip extremes for display
        lo = np.nanpercentile(np.concatenate([h, ab]), 1)
        hi = np.nanpercentile(np.concatenate([h, ab]), 99)
        h  = np.clip(h,  lo, hi)
        ab = np.clip(ab, lo, hi)

        parts = ax.violinplot([h, ab], positions=[0, 1], showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ["#2980b9", "#e74c3c"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Healthy", "Ventricular"], fontsize=9)
        auc = auroc_df[auroc_df["feature"] == feat]["auroc"].values[0]
        ax.set_title(f"{HUMAN_NAMES.get(feat, feat)}\nAUROC={auc:.2f}",
                     fontsize=9, fontweight="bold",
                     color=GROUP_COLORS.get(GROUP_LABELS.get(feat, ""), "black"))

    fig.suptitle("Distribution of top separating features\n(INCART, healthy vs ventricular beats)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Group-level mean AUROC bar chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_group_summary(auroc_df: pd.DataFrame, out: Path) -> None:
    grp = (
        auroc_df.groupby("group")["auroc"]
        .agg(mean="mean", count="count")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GROUP_COLORS.get(g, "#95a5a6") for g in grp["group"]]
    bars = ax.bar(range(len(grp)), grp["mean"], color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.7, color="orange", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(grp["group"], grp["count"])],
                       fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Mean AUROC (healthy vs ventricular)", fontsize=10)
    ax.set_title("Clinical feature group separability\n(INCART, oracle mode)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.45, 1.0)
    for bar, val in zip(bars, grp["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Top-deviator counts from per_beat.csv (all datasets)
# ──────────────────────────────────────────────────────────────────────────────
def plot_top_deviator_counts(out: Path) -> None:
    pb = pd.read_csv(PBEAT_CSV, low_memory=False)
    sub = pb[
        (pb["eval_mode"] == "oracle")
        & (pb["benchmark_mode"] == "zero_shot")
        & (pb["feature_set"] == "all")
        & (pb["label"] == "abnormal_v")
        & (pb["permit"] == False)
    ].copy()

    all_feats = pd.concat([
        sub["top1_feature"], sub["top2_feature"], sub["top3_feature"]
    ]).dropna().astype(str)
    all_feats = all_feats[all_feats != ""]

    counts = all_feats.value_counts().head(20)
    feats_list = counts.index.tolist()
    colors = [GROUP_COLORS.get(GROUP_LABELS.get(f, ""), "#95a5a6") for f in feats_list]
    names  = [HUMAN_NAMES.get(f, f) for f in feats_list]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(feats_list))
    ax.barh(y_pos, counts.values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Count as top-1/2/3 deviating feature (all inhibited abnormal beats, all datasets)", fontsize=9)
    ax.set_title("Which features most often trigger the inhibit decision\n(top-3 deviating features on inhibited ventricular beats)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    seen = {}
    for g, c in GROUP_COLORS.items():
        if any(GROUP_LABELS.get(f, "") == g for f in feats_list):
            seen[g] = mpatches.Patch(facecolor=c, label=g)
    ax.legend(handles=list(seen.values()), loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading INCART beat features...")
    inc = load_incart()
    feat_cols_present = [f for f in FEATURE_COLS if f in inc.columns]
    print(f"  {len(inc):,} beats  |  {len(feat_cols_present)} feature columns")

    print("\n1. Computing AUROC per feature...")
    auroc_df = compute_auroc_table(inc)
    auroc_df.to_csv(OUT_DIR / "auroc_per_feature.csv", index=False)
    print(auroc_df[["human_name","group","auroc"]].head(15).to_string(index=False))

    print("\n2. Feature importance bar chart...")
    plot_auroc_bar(auroc_df, OUT_DIR / "feature_importance.png")

    print("\n3. PCA scatter + loadings...")
    plot_pca(inc, OUT_DIR / "pca_2d.png", OUT_DIR / "pca_loadings.png")

    print("\n4. Distribution violins (top-9 features)...")
    plot_distributions(inc, auroc_df, OUT_DIR / "feature_distributions.png")

    print("\n5. Clinical group summary...")
    plot_group_summary(auroc_df, OUT_DIR / "feature_group_summary.png")

    print("\n6. Top-deviator counts (all datasets, per_beat.csv)...")
    plot_top_deviator_counts(OUT_DIR / "top_deviator_counts.png")

    print(f"\nDone. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
