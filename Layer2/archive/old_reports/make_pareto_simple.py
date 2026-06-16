from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\antoi\OneDrive\Bureau\Master Thesis\Code Base\ECG Processing")
RES = ROOT / "Results"
OUT = RES / "slides"
OUT.mkdir(parents=True, exist_ok=True)

PB = RES / "layer2" / "cross_dataset_causal_100ms" / "per_beat.csv"
COU = RES / "layer2" / "analysis" / "coupling_sweep" / "per_beat_with_coupling.csv"

# simple presentation palette (no blue)
RED = "#C00000"
DARK = "#2F2F2F"
GRAY = "#9A9A9A"
LIGHT = "#F3F3F3"
GREEN = "#4D7F4D"

DS = ["mitdb", "svdb", "incartdb", "nstdb"]


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=LIGHT, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=DARK, labelsize=10)


def load_mahal():
    use = ["dataset","label","mahalanobis","mahalanobis_threshold"]
    chunks = []
    for c in pd.read_csv(PB, usecols=use, chunksize=300_000, low_memory=False):
        c = c[c["dataset"].isin(DS)]
        c = c[c["label"].isin(["healthy","abnormal","abnormal_v","abnormal_s"])]
        chunks.append(c)
    df = pd.concat(chunks, ignore_index=True)
    df["is_abnormal"] = df["label"].str.startswith("abnormal")
    return df


def load_coupling():
    use = ["dataset","label","mahalanobis","mahalanobis_threshold","rr__beat_coupling_ratio_posthoc"]
    df = pd.read_csv(COU, usecols=use, low_memory=False)
    df = df[df["dataset"].isin(DS)]
    df = df[df["label"].isin(["healthy","abnormal","abnormal_v","abnormal_s"])]
    df["is_abnormal"] = df["label"].str.startswith("abnormal")
    df["coupling"] = df["rr__beat_coupling_ratio_posthoc"].clip(0,1.5)
    return df


def sweep_mahal(sub, n=220):
    h = sub[~sub["is_abnormal"]]
    a = sub[sub["is_abnormal"]]
    hr = h["mahalanobis"] / h["mahalanobis_threshold"].replace(0, np.nan)
    ar = a["mahalanobis"] / a["mahalanobis_threshold"].replace(0, np.nan)
    s = np.linspace(0.05, 5.0, n)
    hp = np.array([(hr < t).mean() for t in s])
    ai = np.array([(ar >= t).mean() for t in s])
    fp = 1.0 - ai
    return s, hp, ai, fp


def sweep_coupling(sub, n=220):
    h = sub[~sub["is_abnormal"]]
    a = sub[sub["is_abnormal"]]
    h_c = h["coupling"].fillna(1.0)
    a_c = a["coupling"].fillna(1.0)
    h_m = h["mahalanobis"] < h["mahalanobis_threshold"]
    a_m = a["mahalanobis"] < a["mahalanobis_threshold"]
    t = np.linspace(0.60, 0.99, n)
    hp = np.array([((h_c >= x) & h_m).mean() for x in t])
    ai = np.array([((a_c < x) | (~a_m)).mean() for x in t])
    fp = 1.0 - ai
    return t, hp, ai, fp


def default_point_mahal(sub):
    h = sub[~sub["is_abnormal"]]
    a = sub[sub["is_abnormal"]]
    hp = (h["mahalanobis"] < h["mahalanobis_threshold"]).mean()
    ai = (a["mahalanobis"] >= a["mahalanobis_threshold"]).mean()
    return hp, ai, 1.0 - ai


def pareto_mask(hp, y, maximize_y=True):
    # for curves parameterized by threshold, keep non-dominated points in HP-y plane
    pts = np.column_stack([hp, y])
    m = np.zeros(len(pts), dtype=bool)
    for i, (x_i, y_i) in enumerate(pts):
        dominated = False
        for j, (x_j, y_j) in enumerate(pts):
            if j == i:
                continue
            if maximize_y:
                if (x_j >= x_i and y_j >= y_i) and (x_j > x_i or y_j > y_i):
                    dominated = True
                    break
            else:
                if (x_j >= x_i and y_j <= y_i) and (x_j > x_i or y_j < y_i):
                    dominated = True
                    break
        m[i] = not dominated
    return m


m = load_mahal()
c = load_coupling()

# ---------- Figure 1: HP vs AI (Mahalanobis sweep) ----------
fig, ax = plt.subplots(figsize=(9.6,5.4), dpi=170)
all_hp, all_ai = [], []
for ds in DS:
    sub = m[m.dataset == ds]
    _, hp, ai, _ = sweep_mahal(sub)
    ax.plot(hp*100, ai*100, color=GRAY, linewidth=1.5, alpha=0.9)
    all_hp.append(hp)
    all_ai.append(ai)

hp_mean = np.mean(np.vstack(all_hp), axis=0)
ai_mean = np.mean(np.vstack(all_ai), axis=0)
ax.plot(hp_mean*100, ai_mean*100, color=RED, linewidth=3, label="Mean operating curve")

pm = pareto_mask(hp_mean, ai_mean, maximize_y=True)
ax.plot(hp_mean[pm]*100, ai_mean[pm]*100, color=GREEN, linewidth=3, alpha=0.95, label="Pareto segment")

# default dot = dataset-weighted mean default
def_rows = []
for ds in DS:
    hp0, ai0, _ = default_point_mahal(m[m.dataset == ds])
    w = len(m[m.dataset == ds])
    def_rows.append((hp0, ai0, w))
W = sum(w for _,_,w in def_rows)
hp0 = sum(h*w for h,a,w in def_rows)/W
ai0 = sum(a*w for h,a,w in def_rows)/W
ax.scatter([hp0*100], [ai0*100], s=90, color=DARK, zorder=5, label="Calibrated default")

ax.set_title("Layer 2 Pareto: Healthy Permit vs Abnormal Inhibit", color=DARK, fontsize=14, fontweight="bold")
ax.set_xlabel("Healthy permit (%)  - higher is better", color=DARK)
ax.set_ylabel("Abnormal inhibit (%)  - higher is better", color=DARK)
ax.set_xlim(0,100); ax.set_ylim(0,100)
style_axes(ax)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "pareto_hp_vs_ai_simple.png", facecolor="white")
plt.close(fig)

# ---------- Figure 2: HP vs FP (Mahalanobis sweep) ----------
fig, ax = plt.subplots(figsize=(9.6,5.4), dpi=170)
all_hp, all_fp = [], []
for ds in DS:
    sub = m[m.dataset == ds]
    _, hp, _, fp = sweep_mahal(sub)
    ax.plot(hp*100, fp*100, color=GRAY, linewidth=1.5, alpha=0.85)
    all_hp.append(hp)
    all_fp.append(fp)

hp_mean = np.mean(np.vstack(all_hp), axis=0)
fp_mean = np.mean(np.vstack(all_fp), axis=0)
ax.plot(hp_mean*100, fp_mean*100, color=RED, linewidth=3, label="Mean operating curve")

pm = pareto_mask(hp_mean, fp_mean, maximize_y=False)
ax.plot(hp_mean[pm]*100, fp_mean[pm]*100, color=GREEN, linewidth=3, alpha=0.95, label="Pareto segment")

# default dot
def_rows = []
for ds in DS:
    hp0, _, fp0 = default_point_mahal(m[m.dataset == ds])
    w = len(m[m.dataset == ds])
    def_rows.append((hp0, fp0, w))
W = sum(w for _,_,w in def_rows)
hp0 = sum(h*w for h,f,w in def_rows)/W
fp0 = sum(f*w for h,f,w in def_rows)/W
ax.scatter([hp0*100], [fp0*100], s=90, color=DARK, zorder=5, label="Calibrated default")

# safety budget lines
ax.axhline(1.0, color=DARK, linestyle="--", linewidth=1.2)
ax.text(2, 1.25, "FP budget 1%", color=DARK, fontsize=9)
ax.axhline(0.5, color=DARK, linestyle=":", linewidth=1.2)
ax.text(2, 0.68, "FP budget 0.5%", color=DARK, fontsize=9)

ax.set_title("Safety Pareto: Healthy Permit vs False Permit", color=DARK, fontsize=14, fontweight="bold")
ax.set_xlabel("Healthy permit (%)  - higher is better", color=DARK)
ax.set_ylabel("False permit (%)  - lower is safer", color=DARK)
ax.set_xlim(0,100); ax.set_ylim(0,20)
style_axes(ax)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "pareto_hp_vs_fp_simple.png", facecolor="white")
plt.close(fig)

# ---------- Figure 3: HP vs FP (Coupling threshold sweep) ----------
fig, ax = plt.subplots(figsize=(9.6,5.4), dpi=170)
all_hp, all_fp = [], []
for ds in DS:
    sub = c[c.dataset == ds]
    _, hp, _, fp = sweep_coupling(sub)
    ax.plot(hp*100, fp*100, color=GRAY, linewidth=1.5, alpha=0.85)
    all_hp.append(hp)
    all_fp.append(fp)

hp_mean = np.mean(np.vstack(all_hp), axis=0)
fp_mean = np.mean(np.vstack(all_fp), axis=0)
ax.plot(hp_mean*100, fp_mean*100, color=RED, linewidth=3, label="Mean coupling-sweep curve")

pm = pareto_mask(hp_mean, fp_mean, maximize_y=False)
ax.plot(hp_mean[pm]*100, fp_mean[pm]*100, color=GREEN, linewidth=3, alpha=0.95, label="Pareto segment")

# reference default point from Mahalanobis default for interpretability
ax.scatter([hp0*100], [fp0*100], s=90, color=DARK, zorder=5, label="Calibrated default")
ax.axhline(1.0, color=DARK, linestyle="--", linewidth=1.2)
ax.text(2, 1.25, "FP budget 1%", color=DARK, fontsize=9)

ax.set_title("Safety Pareto: Healthy Permit vs False Permit (Coupling Sweep)", color=DARK, fontsize=14, fontweight="bold")
ax.set_xlabel("Healthy permit (%)  - higher is better", color=DARK)
ax.set_ylabel("False permit (%)  - lower is safer", color=DARK)
ax.set_xlim(0,100); ax.set_ylim(0,25)
style_axes(ax)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "pareto_hp_vs_fp_coupling_simple.png", facecolor="white")
plt.close(fig)

print("Saved:")
print(OUT / "pareto_hp_vs_ai_simple.png")
print(OUT / "pareto_hp_vs_fp_simple.png")
print(OUT / "pareto_hp_vs_fp_coupling_simple.png")
