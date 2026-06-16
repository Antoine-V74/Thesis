"""Quick Pareto analysis on INCART per-beat scores."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

CSV = pathlib.Path("Results/cross_dataset_v4/per_beat.csv")
df = pd.read_csv(CSV, low_memory=False)

inc = df[
    (df["dataset"] == "incartdb")
    & (df["eval_mode"] == "oracle")
    & (df["feature_set"] == "all")
    & (df["benchmark_mode"] == "zero_shot")
].copy()

h   = inc[inc["label"] == "healthy"]
abn = inc[inc["label"] == "abnormal_v"]
print(f"INCART  healthy={len(h):,}  abnormal={len(abn):,}")

# Detect max_zscore column name
max_z_col = "max_abs_zscore" if "max_abs_zscore" in inc.columns else "max_zscore"
print(f"max-z column: {max_z_col}")

# Distributions
for col, label in [("mahalanobis", "Mahalanobis"),
                   (max_z_col, "max_zscore"),
                   ("signal_mahal_proxy", "signal_proxy")]:
    if col not in inc.columns:
        continue
    hv = h[col].dropna().values
    av = abn[col].dropna().values
    y = np.concatenate([np.ones(len(av)), np.zeros(len(hv))])
    sc = np.concatenate([av, hv])
    mask = np.isfinite(sc)
    auc = roc_auc_score(y[mask], sc[mask])
    print(f"  {label:20s}: AUROC={auc:.3f}  "
          f"h p50={np.nanpercentile(hv,50):.1f} p90={np.nanpercentile(hv,90):.1f}  "
          f"a p50={np.nanpercentile(av,50):.1f} p90={np.nanpercentile(av,90):.1f}")

# State of current gate
print(f"\nCurrent gate (zero_shot zscore=0.90):")
print(f"  AI = {1-abn['permit'].mean():.1%}")
print(f"  HP = {h['permit'].mean():.1%}")

# How many more abnormal beats need to be caught to reach 95% AI
n_abn = len(abn)
n_already_blocked = int((abn["permit"] == False).sum())
n_needed = max(0, int(np.ceil(0.95 * n_abn)) - n_already_blocked)
abn_pass = abn[abn["permit"] == True]
h_pass   = h[h["permit"] == True]
print(f"  Abnormals passing current gate: {len(abn_pass):,}")
print(f"  Additional to block for 95%: {n_needed} = {n_needed/max(1,len(abn_pass)):.1%} of remaining")

# Pareto for each score: what HP at various AI levels?
print("\nPareto (sweep threshold, report HP at target AI):")
targets = [0.85, 0.90, 0.95, 0.97]
for col in ["mahalanobis", max_z_col]:
    if col not in inc.columns:
        continue
    print(f"  {col}:")
    all_vals = np.concatenate([h[col].dropna().values, abn[col].dropna().values])
    thresholds = np.unique(np.percentile(all_vals[np.isfinite(all_vals)],
                                          np.linspace(0, 100, 500)))
    for target_ai in targets:
        best_hp = 0.0
        for thr in thresholds:
            ai = (abn[col] > thr).mean()
            hp = (h[col] <= thr).mean()
            if ai >= target_ai and hp > best_hp:
                best_hp = hp
        print(f"    AI>={target_ai:.0%}: HP = {best_hp:.1%}")

# Combined: already-inhibited + additional zscore tightening
print("\nCombined gate (current hard rules fixed, tighten zscore only):")
if max_z_col in abn_pass.columns and max_z_col in h_pass.columns:
    avp = abn_pass[max_z_col].dropna().values
    hvp = h_pass[max_z_col].dropna().values
    for target_ai in [0.90, 0.95]:
        extra_needed = max(0, int(np.ceil(target_ai * n_abn)) - n_already_blocked)
        if extra_needed > len(avp):
            print(f"  AI>={target_ai:.0%}: NOT achievable (only {len(avp)} remaining)")
            continue
        # Threshold to block extra_needed of remaining abnormals
        frac_to_miss = 1.0 - extra_needed / len(avp)
        thr = float(np.nanpercentile(avp, 100 * frac_to_miss))
        extra_h_blocked = int((hvp > thr).sum())
        new_total_h_blocked = int((h["permit"] == False).sum()) + extra_h_blocked
        hp_new = 1.0 - new_total_h_blocked / len(h)
        print(f"  AI>={target_ai:.0%}: zscore thr={thr:.1f}  "
              f"extra healthy blocked={extra_h_blocked:,} ({extra_h_blocked/len(h_pass):.1%} of passing)  "
              f"resulting HP={hp_new:.1%}")
