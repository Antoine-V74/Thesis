"""Compare live coupling-0.80 benchmark vs historical post-hoc sweep."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

live = pd.read_csv(ROOT / "Results/layer2/cross_dataset/overall_summary.csv")
live = live[
    (live.benchmark_mode == "zero_shot")
    & (live.feature_set == "all")
    & (live.eval_mode == "oracle")
]

posthoc_path = ROOT / "Results/layer2/analysis/coupling_sweep/coupling_sweep_overall.csv"
if not posthoc_path.exists():
    posthoc_path = ROOT / "Results/archive/cross_dataset_v4/../layer2/analysis/coupling_sweep/coupling_sweep_overall.csv"
posthoc = pd.read_csv(posthoc_path)
post80 = posthoc[posthoc.strategy == "coupling_0.80"]
base = posthoc[posthoc.strategy == "baseline"]

print("LIVE gate (coupling 0.80 enforced in decision/config.py)")
for _, r in live.sort_values("dataset").iterrows():
    print(
        f"  {r['dataset']:8s}  HP={r['healthy_permit']:.1%}  "
        f"AI={r['abnormal_inhibit']:.1%}  FP={r['false_permit']:.1%}  "
        f"SVT={r['svt_inhibit']:.1%}"
    )

print("\nPOST-HOC coupling 0.80 on v4 baseline (historical)")
for _, r in post80.sort_values("dataset").iterrows():
    print(
        f"  {r['dataset']:8s}  HP={r['healthy_permit']:.1%}  "
        f"AI={r['abnormal_inhibit']:.1%}  FP={r['false_permit']:.1%}  "
        f"SVT={r['svt_inhibit']:.1%}"
    )

print("\nDelta live - posthoc (pp)")
m = live.merge(post80, on="dataset", suffixes=("_live", "_post"))
for _, r in m.sort_values("dataset").iterrows():
    dhp = (r["healthy_permit_live"] - r["healthy_permit_post"]) * 100
    dai = (r["abnormal_inhibit_live"] - r["abnormal_inhibit_post"]) * 100
    print(f"  {r['dataset']:8s}  dHP={dhp:+.2f}pp  dAI={dai:+.2f}pp")

out = ROOT / "Results/layer2/cross_dataset/comparison_vs_posthoc.csv"
rows = []
for ds in sorted(live.dataset.unique()):
    l = live[live.dataset == ds].iloc[0]
    p = post80[post80.dataset == ds].iloc[0]
    b = base[base.dataset == ds].iloc[0]
    rows.append({
        "dataset": ds,
        "baseline_ai": b["abnormal_inhibit"],
        "posthoc_080_ai": p["abnormal_inhibit"],
        "live_080_ai": l["abnormal_inhibit"],
        "live_minus_posthoc_ai_pp": l["abnormal_inhibit"] - p["abnormal_inhibit"],
        "live_hp": l["healthy_permit"],
        "posthoc_hp": p["healthy_permit"],
    })
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nWrote {out}")
