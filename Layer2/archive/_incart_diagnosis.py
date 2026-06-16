"""Diagnose why INCART needs large HP sacrifice for 95% AI."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

CSV = pathlib.Path("Results/cross_dataset_v4/per_beat.csv")
df = pd.read_csv(CSV, low_memory=False)

def subset(dataset, mode="oracle", bench="zero_shot", fs="all"):
    return df[
        (df["dataset"] == dataset)
        & (df["eval_mode"] == mode)
        & (df["benchmark_mode"] == bench)
        & (df["feature_set"] == fs)
    ].copy()

def auroc(pos, neg):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    m = np.isfinite(s)
    if m.sum() < 20:
        return float("nan")
    return roc_auc_score(y[m], s[m])

for ds in ["mitdb", "incartdb"]:
    sub = subset(ds)
    h = sub[sub["label"] == "healthy"]
    ab = sub[sub["label"] == "abnormal_v"]
    print(f"\n{'='*60}\n{ds.upper()}  healthy={len(h):,}  abnormal={len(ab):,}")
    print(f"  HP={h['permit'].mean():.1%}  AI={1-ab['permit'].mean():.1%}  FP={ab['permit'].mean():.1%}")

    # Symbol breakdown for abnormals
    if "symbol" in ab.columns:
        print("\n  Abnormal beat symbols:")
        for sym, g in ab.groupby("symbol"):
            n = len(g)
            ai = 1 - g["permit"].mean()
            print(f"    {sym!r:6s} n={n:6,}  AI={ai:.1%}")

    # AUROC per score
    print("\n  AUROC:")
    for col in ["mahalanobis", "max_zscore", "signal_mahal_proxy", "rr_mahal_proxy",
                "morph__template_corr", "morph__neighbor_corr", "rr__beat_coupling_ratio",
                "signal__raw_hf_noise_ratio"]:
        if col not in sub.columns:
            continue
        a = auroc(ab[col].dropna().values, h[col].dropna().values)
        print(f"    {col:35s} {a:.3f}")

    # Overlap: abnormals below healthy median
    for col in ["mahalanobis", "max_zscore"]:
        if col not in sub.columns:
            continue
        med_h = h[col].median()
        frac = (ab[col] < med_h).mean()
        print(f"  {col}: {frac:.1%} of abnormals BELOW healthy median ({med_h:.1f})")

    # False permits: what do they look like?
    fp = ab[ab["permit"] == True]
    print(f"\n  False permits: {len(fp):,} ({len(fp)/len(ab):.1%})")
    if "reason" in fp.columns:
        print("  Top reasons for FP:")
        for r, c in fp["reason"].value_counts().head(5).items():
            print(f"    {c:5,}  {r[:80]}")

    # Hard rule catch rate
    if "reason" in ab.columns:
        hr = ab["reason"].astype(str).str.contains("hard_rule|template|coupling|noise", case=False, regex=True)
        print(f"  Hard-rule inhibit rate on abnormals: {hr.mean():.1%}")

inc = subset("incartdb")
ab = inc[inc["label"] == "abnormal_v"]
h = inc[inc["label"] == "healthy"]
fp = ab[ab["permit"] == True]

# zscore overlap detail
print("\n" + "="*60)
print("INCART max_zscore overlap (main bottleneck)")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    thr = h["max_zscore"].quantile(q)
    ai = (ab["max_zscore"] > thr).mean()
    hp = (h["max_zscore"] <= thr).mean()
    print(f"  healthy p{int(q*100)} thr={thr:8.1f}  -> AI={ai:.1%}  HP={hp:.1%}")

# Which features drive max_zscore on FP vs healthy?
for col in ["top1_zscore", "top2_zscore", "top3_zscore"]:
    if col not in fp.columns:
        continue
    print(f"\n  FP top feature in {col}: (need reason parsing)")

# Parse top deviator from reason if present
def top_feat(reason):
    if not isinstance(reason, str):
        return None
    if "max_zscore" in reason:
        return "max_zscore"
    if "mahalanobis" in reason:
        return "mahalanobis"
    if "hard_rule" in reason:
        return "hard_rule"
    return "other"

fp = fp.copy()
fp["gate"] = fp["reason"].map(top_feat)
print("\n  FP gate breakdown:")
print(fp["gate"].value_counts())

# Per-record worst performers
print("\n  Worst 10 records by abnormal AI:")
rec_stats = []
for rec, g in ab.groupby("record"):
    rec_stats.append((rec, len(g), 1 - g["permit"].mean()))
rec_stats.sort(key=lambda x: x[2])
for rec, n, ai in rec_stats[:10]:
    print(f"    {rec}: n={n:4d}  AI={ai:.1%}")

# Compare zscore threshold used vs distribution
if "zscore_threshold" in inc.columns:
    thr = inc["zscore_threshold"].dropna().median()
    print(f"\n  Calibrated zscore_threshold (median across records): {thr:.1f}")
    print(f"  Healthy beats above threshold: {(h['max_zscore'] > thr).mean():.1%}")
    print(f"  Abnormal beats above threshold: {(ab['max_zscore'] > thr).mean():.1%}")
    fp_above = fp["max_zscore"].describe()
    print(f"  FP max_zscore: p50={fp['max_zscore'].median():.1f}  p90={fp['max_zscore'].quantile(.9):.1f}")

print("\nDone.")
