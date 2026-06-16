"""
Compact Layer 1 summary figures.

Creates a small set of figures for comparing Layer 1 performance:

1. Dataset-level performance summary
2. Worst-record/outlier analysis
3. MIT-BIH condition-level failure analysis

Usage:
    .venv\\Scripts\\python Layer1\\tools\\make_layer1_summary_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_LABELS = {
    "mit_bih_arrhythmia": "MIT-BIH Arrhythmia",
    "normal_sinus_rhythm": "Normal Sinus",
    "supraventricular_arrhythmia": "Supraventricular",
    "atrial_fibrillation": "Atrial Fibrillation",
    "long_term_atrial_fibrillation": "Long-Term AF",
    "malignant_ventricular_arrhythmia": "Malignant VA",
    "creighton_vfib": "Creighton VFib",
    "noise_stress_test": "Noise Stress",
    "st_petersburg_12lead": "INCART 12-lead",
}

MITBIH_SYMBOL_LABELS = {
    "V": "V: Premature ventricular contraction",
    "F": "F: Fusion of ventricular + normal beat",
    "E": "E: Ventricular escape beat",
    "/": "/: Paced beat",
    "f": "f: Fusion of paced + normal beat",
    "!": "!: Ventricular flutter wave",
}


def _pct(values: pd.Series) -> pd.Series:
    return 100.0 * pd.to_numeric(values, errors="coerce")


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _load(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = pd.read_csv(csv_dir / "per_dataset_summary.csv")
    records = pd.read_csv(csv_dir / "per_record_all.csv", low_memory=False)
    beats = pd.read_csv(csv_dir / "per_beat_all.csv", low_memory=False)
    return datasets, records, beats


def plot_dataset_summary(datasets: pd.DataFrame, out: Path) -> None:
    df = datasets.copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    df["false_trigger_rate"] = df["n_triggers"] / df["n_ref_beats"].replace(0, np.nan)

    metric_rows = []
    for _, row in df.iterrows():
        if np.isfinite(row.get("healthy_trigger_rate_vs_ref", np.nan)):
            metric_rows.append((row["dataset_label"], "Healthy trigger rate", 100.0 * row["healthy_trigger_rate_vs_ref"]))
        if np.isfinite(row.get("arrhythmia_trigger_rate_vs_ref", np.nan)):
            metric_rows.append((row["dataset_label"], "Arrhythmia trigger-through", 100.0 * row["arrhythmia_trigger_rate_vs_ref"]))
        if np.isfinite(row.get("false_trigger_rate", np.nan)):
            metric_rows.append((row["dataset_label"], "False triggers / refs", 100.0 * row["false_trigger_rate"]))

    plot_df = pd.DataFrame(metric_rows, columns=["dataset", "metric", "value_pct"])
    order = df["dataset_label"].tolist()
    metrics = ["Healthy trigger rate", "Arrhythmia trigger-through", "False triggers / refs"]
    colors = {
        "Healthy trigger rate": "#2e86ab",
        "Arrhythmia trigger-through": "#c0392b",
        "False triggers / refs": "#e67e22",
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(order))
    width = 0.24
    for i, metric in enumerate(metrics):
        values = [
            plot_df.loc[(plot_df["dataset"] == dataset) & (plot_df["metric"] == metric), "value_pct"].mean()
            for dataset in order
        ]
        ax.bar(x + (i - 1) * width, values, width=width, label=metric, color=colors[metric])

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Layer 1 performance by dataset")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def plot_worst_records(records: pd.DataFrame, out: Path, top_n: int) -> pd.DataFrame:
    df = records.copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    df["arrhythmia_trigger_pct"] = _pct(df["arrhythmia_trigger_rate_vs_ref"])
    df["healthy_miss_pct"] = 100.0 * (1.0 - pd.to_numeric(df["healthy_trigger_rate_vs_ref"], errors="coerce"))
    df["extra_trigger_pct"] = 100.0 * pd.to_numeric(df["extra_triggers"], errors="coerce") / pd.to_numeric(df["n_ref_beats"], errors="coerce").replace(0, np.nan)

    arr = df[df["n_ref_arrhythmia"] >= 20].copy()
    arr["problem_pct"] = arr["arrhythmia_trigger_pct"]
    arr["problem_type"] = "Arrhythmia trigger-through"

    healthy = df[df["n_ref_healthy"] >= 200].copy()
    healthy["problem_pct"] = healthy["healthy_miss_pct"]
    healthy["problem_type"] = "Healthy missed trigger"

    extra = df[df["n_ref_beats"] >= 200].copy()
    extra["problem_pct"] = extra["extra_trigger_pct"]
    extra["problem_type"] = "Extra triggers"

    worst = pd.concat([arr, healthy, extra], ignore_index=True)
    worst = worst[np.isfinite(worst["problem_pct"])]
    worst = worst.sort_values("problem_pct", ascending=False).head(top_n).copy()
    worst["label"] = worst["dataset_label"] + " / " + worst["record"].astype(str) + "\n" + worst["problem_type"]

    color_map = {
        "Arrhythmia trigger-through": "#c0392b",
        "Healthy missed trigger": "#2e86ab",
        "Extra triggers": "#e67e22",
    }
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(worst))))
    y = np.arange(len(worst))
    ax.barh(y, worst["problem_pct"], color=[color_map[v] for v in worst["problem_type"]])
    ax.set_yticks(y)
    ax.set_yticklabels(worst["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Problem rate (%)")
    ax.set_title(f"Layer 1 worst records: are a few records pulling metrics down? (top {top_n})")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(worst["problem_pct"]):
        ax.text(value, i, f" {value:.1f}%", va="center", fontsize=8)
    _save(fig, out)
    return worst


def plot_mitbih_conditions(beats: pd.DataFrame, out: Path) -> pd.DataFrame:
    arr = beats[(beats["dataset"] == "mit_bih_arrhythmia") & (beats["label"] == "arrhythmia")].copy()
    arr["triggered_bool"] = arr["triggered_matched"].astype(bool)
    summary = (
        arr.groupby("symbol", as_index=False)
        .agg(
            n_beats=("symbol", "size"),
            n_triggered=("triggered_bool", "sum"),
        )
    )
    summary["trigger_through_pct"] = 100.0 * summary["n_triggered"] / summary["n_beats"]
    summary["rejection_pct"] = 100.0 - summary["trigger_through_pct"]
    summary["condition"] = summary["symbol"].map(MITBIH_SYMBOL_LABELS).fillna(summary["symbol"] + ": unknown abnormal beat")
    summary = summary.sort_values("trigger_through_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = np.where(summary["n_beats"] >= 50, "#c0392b", "#95a5a6")
    ax.bar(summary["condition"], summary["trigger_through_pct"], color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Triggered despite abnormal beat label (%)")
    ax.set_xlabel("MIT-BIH condition")
    ax.set_title("MIT-BIH: which annotated conditions pass through Layer 1?")
    ax.tick_params(axis="x", rotation=25)
    for i, row in enumerate(summary.itertuples(index=False)):
        ax.text(i, row.trigger_through_pct + 1.5, f"{row.trigger_through_pct:.1f}%\nn={row.n_beats}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)
    return summary


def write_readme(out_dir: Path) -> None:
    text = """# Layer 1 compact summary figures

This folder contains the reduced plot set for comparing Layer 1 performance.

## Figures

1. `01_dataset_performance_summary.png`
   Dataset-level rates. The most safety-relevant bar is **arrhythmia trigger-through**:
   abnormal reference beats that still became trigger-eligible.

2. `02_worst_records.png`
   Worst records across datasets and failure modes. This checks whether global metrics
   are pulled down by a few difficult records.

3. `03_mitbih_condition_failures.png`
   MIT-BIH abnormal beat conditions with readable labels instead of raw annotation symbols.

## Interpretation caution

Beat annotations are not complete rhythm-state labels. They are still useful for diagnosing
Layer 1 failure modes, but they should not be treated as a complete dangerous-rhythm classifier.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("Results/layer1/current_benchmark_20260611/csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("Results/layer1/summary"))
    parser.add_argument("--top-records", type=int, default=25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "csv").mkdir(parents=True, exist_ok=True)
    datasets, records, beats = _load(args.csv_dir)

    plot_dataset_summary(datasets, args.out_dir / "01_dataset_performance_summary.png")
    worst = plot_worst_records(records, args.out_dir / "02_worst_records.png", args.top_records)
    conditions = plot_mitbih_conditions(beats, args.out_dir / "03_mitbih_condition_failures.png")
    worst.to_csv(args.out_dir / "csv" / "worst_records.csv", index=False)
    conditions.to_csv(args.out_dir / "csv" / "mitbih_condition_failures.csv", index=False)
    write_readme(args.out_dir)
    print(f"Wrote Layer 1 summary figures to {args.out_dir}")


if __name__ == "__main__":
    main()
