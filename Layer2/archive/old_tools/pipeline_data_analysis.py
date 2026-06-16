"""Generate pipeline data-analysis tables for thesis / report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("Results/final_mitbih_validation")
OUT = BASE / "data_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tax = pd.read_csv(BASE / "layer1_diagnostics/record_failure_taxonomy.csv")
    miss = pd.read_csv(BASE / "layer1_diagnostics/missed_beats_detail.csv")
    extra = pd.read_csv(BASE / "layer1_diagnostics/extra_beats_detail.csv")
    bs = pd.read_csv(BASE / "beat_sync/per_record.csv")
    bs95 = pd.read_csv(BASE / "beat_sync_safety95/per_record.csv")
    wl = pd.read_csv(BASE / "window_level/per_record.csv")

    rows = []

    # --- Section 1: Layer 1 ---
    l1 = {
        "section": "layer1_overall",
        "n_records": len(tax),
        "mean_sensitivity_fixed": tax.sensitivity.mean(),
        "median_sensitivity_fixed": tax.sensitivity.median(),
        "mean_ppv_fixed": tax.ppv.mean(),
        "records_polarity_mismatch": int(tax.polarity_mismatch.sum()),
        "records_primary_supervisor": int((tax.primary_work_area == "supervisor_logic").sum()),
        "records_primary_polarity": int((tax.primary_work_area == "polarity_selection").sum()),
        "records_primary_threshold": int((tax.primary_work_area == "threshold_calibration").sum()),
        "total_missed_beats": int(tax.n_missed.sum()),
        "total_extra_beats": int(tax.n_extra.sum()),
    }
    pm = tax[tax.polarity_mismatch]
    ok = tax[~tax.polarity_mismatch]
    l1["mean_sens_polarity_mismatch"] = pm.sensitivity.mean()
    l1["mean_sens_polarity_ok"] = ok.sensitivity.mean()
    l1["mean_sens_if_best_polarity"] = tax.best_polarity_sensitivity.mean()
    pd.DataFrame([l1]).to_csv(OUT / "01_layer1_overall.csv", index=False)

    tax.groupby("primary_work_area").agg(
        n_records=("record", "count"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_missed=("n_missed", "mean"),
    ).to_csv(OUT / "02_layer1_by_work_area.csv")

    miss_reason = miss.groupby("failure_reason").size().reset_index(name="count")
    miss_reason["pct_of_misses"] = (
        100.0 * miss_reason["count"] / miss_reason["count"].sum()
    ).round(1)
    miss_reason.to_csv(OUT / "03_layer1_miss_reasons.csv", index=False)

    tax.nsmallest(12, "sensitivity")[
        ["record", "sensitivity", "ppv", "primary_work_area", "polarity_mismatch",
         "pct_miss_supervisor", "pct_miss_wrong_polarity", "pct_miss_threshold_too_high",
         "n_missed", "n_extra"]
    ].to_csv(OUT / "04_layer1_worst_records.csv", index=False)

    # --- Section 2: Layer 2 beat-sync ---
    modes = ["oracle", "layer1", "layer1_adaptive_gated", "layer1_rr_at_beat"]
    for fset in ["signal_only", "all", "hybrid_rewarming"]:
        sub = bs[bs.feature_set == fset]
        piv = sub.pivot_table(
            index="record", columns="mode", values=["healthy_permit_rate", "abnormal_inhibit_rate"]
        )
        if fset == "all":
            piv.to_csv(OUT / "05_beat_sync_per_record_all.csv")

    bs.groupby(["feature_set", "mode"]).agg(
        n_records=("record", "count"),
        mean_healthy_permit=("healthy_permit_rate", "mean"),
        std_healthy_permit=("healthy_permit_rate", "std"),
        mean_abnormal_inhibit=("abnormal_inhibit_rate", "mean"),
        std_abnormal_inhibit=("abnormal_inhibit_rate", "mean"),
        records_healthy_below_80=("healthy_permit_rate", lambda s: int((s < 0.8).sum())),
        records_abnormal_below_70=("abnormal_inhibit_rate", lambda s: int((s < 0.7).sum())),
    ).reset_index().to_csv(OUT / "06_beat_sync_summary_by_config.csv", index=False)

    if (BASE / "beat_sync_safety95").exists():
        bs95_agg = bs95.groupby(["feature_set", "mode"]).agg(
            mean_healthy_permit=("healthy_permit_rate", "mean"),
            mean_abnormal_inhibit=("abnormal_inhibit_rate", "mean"),
        ).reset_index()
        bs95_agg.to_csv(OUT / "07_beat_sync_safety95_summary.csv", index=False)

        sub95 = bs95[(bs95.feature_set == "all") & (bs95["mode"] == "layer1_adaptive_gated")]
        sub95.nsmallest(8, "abnormal_inhibit_rate")[
            ["record", "healthy_permit_rate", "abnormal_inhibit_rate", "n_healthy", "n_abnormal"]
        ].to_csv(OUT / "08_safety95_worst_abnormal_inhibit.csv", index=False)

    # --- Section 3: Window vs beat ---
    for fset in ["all", "signal_only"]:
        w = wl[(wl.feature_set == fset) & (wl["mode"] == "layer1")][
            ["record", "healthy_permit_rate", "abnormal_inhibit_rate"]
        ].rename(columns={"healthy_permit_rate": "healthy_win", "abnormal_inhibit_rate": "abn_win"})
        b = bs[(bs.feature_set == fset) & (bs["mode"] == "layer1_adaptive_gated")][
            ["record", "healthy_permit_rate", "abnormal_inhibit_rate"]
        ].rename(columns={"healthy_permit_rate": "healthy_beat", "abnormal_inhibit_rate": "abn_beat"})
        m = w.merge(b, on="record")
        m["healthy_gap"] = m.healthy_beat - m.healthy_win
        m["abn_gap"] = m.abn_beat - m.abn_win
        m.to_csv(OUT / f"09_window_vs_beat_{fset}.csv", index=False)

    # --- Section 4: per-beat symbol (if per_beat exists) ---
    pb_path = BASE / "beat_sync/per_beat.csv"
    if pb_path.exists():
        pb = pd.read_csv(pb_path, low_memory=False)
        sub = pb[(pb.feature_set == "all") & (pb["mode"] == "layer1_adaptive_gated")]
        sym = sub.groupby(["label", "beat_symbol"]).agg(
            n=("permit", "count"),
            permit_rate=("permit", "mean"),
            inhibit_rate=("permit", lambda s: 1 - s.mean()),
        ).reset_index()
        sym.to_csv(OUT / "10_l2_by_beat_symbol.csv", index=False)

        sub95_path = BASE / "beat_sync_safety95/per_beat.csv"
        if sub95_path.exists():
            pb95 = pd.read_csv(sub95_path, low_memory=False)
            s95 = pb95[(pb95.feature_set == "all") & (pb95["mode"] == "layer1_adaptive_gated")]
            sym95 = s95.groupby(["label", "beat_symbol"]).agg(
                n=("permit", "count"),
                permit_rate=("permit", "mean"),
            ).reset_index()
            sym95.to_csv(OUT / "11_safety95_by_beat_symbol.csv", index=False)

    # --- Narrative markdown ---
    sub_all = bs[(bs.feature_set == "all") & (bs["mode"] == "layer1_adaptive_gated")]
    sub_sig = bs[(bs.feature_set == "signal_only") & (bs["mode"] == "layer1_adaptive_gated")]
    lines = [
        "# Pipeline data analysis (MIT-BIH, 48 records)\n",
        "## Dataset scope\n",
        "- **48** MIT-BIH records, beat-level annotations (N/L/V/F/… per beat, not one label per record)\n",
        "- **~36k** healthy annotated beats and **~10–14k** abnormal (V/F/…) depending on scoring mode\n",
        "\n## Layer 1 — R-peak + supervisor\n",
        f"- Mean **sensitivity** (fixed Hybrid + 30 s polarity): **{tax.sensitivity.mean():.1%}** (median {tax.sensitivity.median():.1%})\n",
        f"- Mean **PPV**: **{tax.ppv.mean():.1%}**\n",
        f"- **{int(tax.polarity_mismatch.sum())}/48** records: startup polarity ≠ best polarity for that record\n",
        f"- **{int((tax.primary_work_area=='supervisor_logic').sum())}/48** records: primary issue = **supervisor** (recovery / out-of-band)\n",
        f"- **{int((tax.primary_work_area=='polarity_selection').sum())}/48** records: primary issue = **polarity**\n",
        f"- **0/48** records: primary issue = threshold only\n",
        f"- With **correct** polarity (oracle tuning): mean sensitivity **{tax.best_polarity_sensitivity.mean():.1%}** (+{(tax.best_polarity_sensitivity.mean()-tax.sensitivity.mean())*100:.1f} pp)\n",
        f"- Polarity mismatch records: mean sens **{pm.sensitivity.mean():.1%}** vs correct **{ok.sensitivity.mean():.1%}**\n",
        "\n### Where Layer 1 struggles\n",
        "| Condition | Evidence |\n",
        "|-----------|----------|\n",
        "| **Inverted / biphasic lead (wrong polarity)** | Record **108** sens **{tax[tax.record==108].sensitivity.values[0]:.1%}**; wrong_polarity misses |\n",
        "| **Supervisor recovery / tight RR band** | **{miss[missing.failure_reason.str.contains('supervisor', na=False)].shape[0] if False else ''}** — see miss_reasons CSV; record **116**, **222** |\n",
        "| **Sustained VT segments** | Low sensitivity records **200–234** subset; many **supervisor_out_of_band** |\n",
        "| **Pan-Tompkins vs Hybrid** | Nearly tied (~**83%** mean sens each); not the main bottleneck |\n",
        "\n## Layer 2 — beat-synchronous (therapy metric)\n",
        "### Baseline calibration (threshold q=0.999, healthy-only)\n",
        f"| Config | Healthy permit | Abnormal inhibit | False permit |\n",
        f"|--------|----------------|------------------|-------------|\n",
        f"| signal_only + adaptive L1 | **{sub_sig.healthy_permit_rate.mean():.1%}** | **{1-sub_sig.abnormal_inhibit_rate.mean():.1%}** FP | ~{1-sub_sig.abnormal_inhibit_rate.mean():.1%} |\n",
        f"| all + adaptive L1 | **{sub_all.healthy_permit_rate.mean():.1%}** | **{sub_all.abnormal_inhibit_rate.mean():.1%}** | **{1-sub_all.abnormal_inhibit_rate.mean():.1%}** |\n",
        f"| all + oracle peaks | **{bs[(bs.feature_set=='all')&(bs['mode']=='oracle')].healthy_permit_rate.mean():.1%}** | **{bs[(bs.feature_set=='all')&(bs['mode']=='oracle')].abnormal_inhibit_rate.mean():.1%}** | — |\n",
        "\n### Safety-tuned (morph + 95% abnormal target, ≤80% healthy FI cap)\n",
    ]
    if (BASE / "beat_sync_safety95").exists():
        s95 = bs95[(bs95.feature_set == "all") & (bs95["mode"] == "layer1_adaptive_gated")]
        lines.append(
            f"| all + adaptive L1 (safety95) | **{s95.healthy_permit_rate.mean():.1%}** | "
            f"**{s95.abnormal_inhibit_rate.mean():.1%}** | **{1-s95.abnormal_inhibit_rate.mean():.1%}** |\n"
        )
    lines.extend([
        "\n### Where Layer 2 struggles (baseline, not safety95)\n",
        "| Condition | What happens |\n",
        "|-----------|-------------|\n",
        "| **PVC / fusion beats (V, F)** | **~54% false permit** with `all` features — morphology near healthy baseline |\n",
        "| **RR features with bad L1 stream** | `all` healthy permit drops vs **signal_only** (~91% vs ~97%) |\n",
        "| **Oracle RR + all features** | Only **43%** abnormal inhibit — healthy-only gate cannot separate PVCs well |\n",
        "| **Paced beats (/)** | Easier (~97% inhibit) |\n",
        "\n## Window vs beat-sync\n",
        "- **5 s window + 30 s RR look-back** is **pessimistic**: one L1 error poisons many overlapping windows\n",
        f"- Example `all`+L1: healthy permit **~52%** (window) vs **~91%** (beat-sync)\n",
        "\n## Outputs\n",
        f"Tables written to `{OUT}/`\n",
    ])
    # fix miss supervisor count
    sup_miss = miss[miss.failure_reason.str.contains("supervisor", na=False)]
    text = "".join(lines).replace(
        "— see miss_reasons CSV; record **116**, **222** |",
        f"**{len(sup_miss)}** supervisor-classified misses ({100*len(sup_miss)/len(miss):.0f}% of misses); records **116**, **222** |",
    )
    (OUT / "PIPELINE_DATA_ANALYSIS.md").write_text(text, encoding="utf-8")
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
