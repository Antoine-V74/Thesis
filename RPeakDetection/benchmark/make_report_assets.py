"""Build GitHub-ready R-peak detection report tables and figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    sys.exit("Missing dependency: pip install wfdb")

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DATA))

from dataset_registry import is_beat_symbol, resolve_dataset  # noqa: E402
from RPeakDetection.algorithms import get_detector  # noqa: E402
from RPeakDetection.benchmark.metrics import greedy_match  # noqa: E402


DISPLAY_NAMES = {
    "mit_bih_arrhythmia": "MIT-BIH",
    "normal_sinus_rhythm": "NSRDB",
    "supraventricular_arrhythmia": "SVDB",
    "long_term_atrial_fibrillation": "LTAFDB",
    "noise_stress_test": "NSTDB",
    "st_petersburg_12lead": "INCART",
}

DATASET_NOTES = {
    "mit_bih_arrhythmia": "strong development-set performance",
    "normal_sinus_rhythm": "strong on mostly sinus rhythm",
    "noise_stress_test": "robust at SNR >= 12 dB",
    "st_petersburg_12lead": "domain and lead morphology shift",
    "supraventricular_arrhythmia": "bad-tail failures dominate",
    "long_term_atrial_fibrillation": "high sensitivity, many extra peaks",
}

DATASET_ORDER = [
    "mit_bih_arrhythmia",
    "normal_sinus_rhythm",
    "noise_stress_test",
    "long_term_atrial_fibrillation",
    "supraventricular_arrhythmia",
    "st_petersburg_12lead",
]

PALETTE = {
    "sensitivity": "#2f6fbb",
    "ppv": "#d97706",
    "f1": "#1b8a5a",
    "missed": "#b91c1c",
    "extra": "#7c3aed",
    "signal": "#23395b",
    "reference": "#15803d",
    "detected": "#dc2626",
}


def _display_dataset(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _ordered_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    return df.assign(_order=df["dataset"].map(lambda x: order.get(x, 999))).sort_values("_order")


def _ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def _format_summary_table(per_dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "n_records",
        "n_annotated_beats",
        "sensitivity",
        "ppv",
        "f1",
        "median_confirmed_event_lag_ms",
        "p95_confirmed_event_lag_ms",
        "extra_peaks_per_hour",
    ]
    summary = _ordered_dataset_frame(per_dataset[cols].copy()).drop(columns="_order")
    summary.insert(1, "dataset_label", summary["dataset"].map(_display_dataset))
    summary["main_note"] = summary["dataset"].map(DATASET_NOTES).fillna("")
    return summary


def _failure_mode(row: pd.Series) -> str:
    sens = float(row["sensitivity"])
    ppv = float(row["ppv"])
    if sens < 0.2 and ppv < 0.2:
        return "catastrophic mismatch"
    if sens < 0.8 and ppv < 0.8:
        return "misses and extra peaks"
    if ppv < 0.75 and sens >= 0.95:
        return "extra peaks dominate"
    if sens < 0.9 and ppv >= 0.9:
        return "missed beats dominate"
    return "mixed degradation"


def _write_tables(per_dataset: pd.DataFrame, per_record: pd.DataFrame, out_tables: Path) -> Dict:
    dataset_summary = _format_summary_table(per_dataset)
    dataset_summary.to_csv(out_tables / "dataset_summary.csv", index=False)

    worst = per_record.sort_values("f1", ascending=True).head(20).copy()
    worst.insert(1, "dataset_label", worst["dataset"].map(_display_dataset))
    worst["failure_mode"] = worst.apply(_failure_mode, axis=1)
    worst_cols = [
        "dataset",
        "dataset_label",
        "record",
        "f1",
        "ppv",
        "sensitivity",
        "n_extra",
        "n_missed",
        "n_annotated_beats",
        "failure_mode",
    ]
    worst[worst_cols].to_csv(out_tables / "worst_records.csv", index=False)

    total_extra = int(per_record["n_extra"].sum())
    total_missed = int(per_record["n_missed"].sum())
    rows = []
    sorted_f1 = per_record.sort_values("f1", ascending=True)
    for k in [1, 3, 5, 10, 20, 30, 50]:
        subset = sorted_f1.head(k)
        rows.append({
            "worst_n_records": k,
            "extra_peak_share": float(subset["n_extra"].sum() / total_extra),
            "missed_beat_share": float(subset["n_missed"].sum() / total_missed),
        })
    concentration = pd.DataFrame(rows)
    concentration.to_csv(out_tables / "failure_concentration.csv", index=False)

    threshold_rows = []
    for threshold in [0.95, 0.90, 0.80, 0.70, 0.50]:
        subset = per_record[per_record["f1"] < threshold]
        threshold_rows.append({
            "f1_threshold": threshold,
            "records_below_threshold": int(len(subset)),
            "record_share": float(len(subset) / len(per_record)),
            "extra_peak_share": float(subset["n_extra"].sum() / total_extra),
            "missed_beat_share": float(subset["n_missed"].sum() / total_missed),
        })
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out_tables / "failure_by_f1_threshold.csv", index=False)

    return {
        "dataset_summary": dataset_summary,
        "worst_records": worst[worst_cols],
        "failure_concentration": concentration,
        "failure_by_f1_threshold": thresholds,
    }


def _style_axis(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_metrics(per_dataset: pd.DataFrame, out_figures: Path) -> None:
    df = _ordered_dataset_frame(per_dataset).copy()
    labels = [_display_dataset(x) for x in df["dataset"]]
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x - width, df["sensitivity"], width, label="Sensitivity", color=PALETTE["sensitivity"])
    ax.bar(x, df["ppv"], width, label="PPV", color=PALETTE["ppv"])
    ax.bar(x + width, df["f1"], width, label="F1", color=PALETTE["f1"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.75, 1.01)
    ax.legend(ncol=3, frameon=False, loc="lower left")
    _style_axis(ax, "Adaptive Threshold v2 Performance By Dataset", "Score")
    _save(fig, out_figures / "rpeak_dataset_metrics.png")


def _plot_latency(per_dataset: pd.DataFrame, out_figures: Path) -> None:
    df = _ordered_dataset_frame(per_dataset).copy()
    labels = [_display_dataset(x) for x in df["dataset"]]
    x = np.arange(len(df))
    y = df["median_confirmed_event_lag_ms"].to_numpy(dtype=float)
    yerr = np.maximum(
        df["p95_confirmed_event_lag_ms"].to_numpy(dtype=float) - y,
        0.0,
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x, y, color="#334155", alpha=0.9)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#111827", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_axis(ax, "Live Confirmation Lag By Dataset", "Median lag with p95 whisker (ms)")
    _save(fig, out_figures / "rpeak_live_lag_by_dataset.png")


def _plot_f1_distribution(per_record: pd.DataFrame, out_figures: Path) -> None:
    data = []
    labels = []
    for dataset in DATASET_ORDER:
        vals = per_record.loc[per_record["dataset"] == dataset, "f1"].to_numpy(dtype=float)
        if len(vals):
            data.append(vals)
            labels.append(_display_dataset(dataset))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#2563eb")
        patch.set_alpha(0.75)

    rng = np.random.default_rng(7)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=18,
            color="#334155",
            alpha=0.45,
            linewidth=0,
        )
    ax.axhline(0.90, color="#b91c1c", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(0.55, 0.905, "F1 = 0.90", color="#b91c1c", fontsize=9, va="bottom")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_axis(ax, "Record-Level F1 Distribution", "F1")
    _save(fig, out_figures / "rpeak_record_f1_distribution.png")


def _plot_error_pareto(per_record: pd.DataFrame, out_figures: Path) -> None:
    df = per_record.sort_values("f1", ascending=True).reset_index(drop=True).copy()
    x = np.arange(1, len(df) + 1)
    missed_share = df["n_missed"].cumsum() / max(float(df["n_missed"].sum()), 1.0)
    extra_share = df["n_extra"].cumsum() / max(float(df["n_extra"].sum()), 1.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, missed_share * 100.0, color=PALETTE["missed"], linewidth=2.3, label="Missed beats")
    ax.plot(x, extra_share * 100.0, color=PALETTE["extra"], linewidth=2.3, label="Extra peaks")
    for marker in [10, 20, 50]:
        ax.axvline(marker, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax, "Failure Concentration In Worst Records", "Cumulative share of total errors (%)")
    ax.set_xlabel("Records sorted from worst to best F1")
    _save(fig, out_figures / "rpeak_error_pareto.png")


def _load_record(dataset: str, record: str):
    info = resolve_dataset(dataset)
    stem = _DATA / info.folder / record
    rec = wfdb.rdrecord(str(stem))
    fs = float(rec.fs)
    channel = min(info.channel, rec.p_signal.shape[1] - 1)
    raw = np.asarray(rec.p_signal[:, channel], dtype=float)
    ann = wfdb.rdann(str(stem), info.rpeak_ann_ext)
    ref_s = np.asarray([
        sample / fs
        for sample, symbol in zip(ann.sample, ann.symbol)
        if is_beat_symbol(symbol)
    ], dtype=float)
    det_result = get_detector("adaptive_threshold_v2").detect(raw, fs)
    det_s = det_result.peak_times_s(fs)
    return raw, fs, ref_s, det_s


def _select_failure_window(ref_s: np.ndarray, det_s: np.ndarray, duration_s: float) -> float:
    max_start = max(8.0, min(float(max(ref_s[-1], det_s[-1]) if len(ref_s) and len(det_s) else 120.0) - duration_s, 300.0))
    starts = np.arange(5.0, max_start, 1.0)
    best_start = 5.0
    best_score = -1.0
    for start in starts:
        end = start + duration_s
        ref = ref_s[(ref_s >= start) & (ref_s <= end)]
        det = det_s[(det_s >= start) & (det_s <= end)]
        if len(ref) < 3 and len(det) < 3:
            continue
        matches, missed, extra = greedy_match(ref, det, 0.100)
        denom = max(len(ref) + len(det), 1)
        score = (len(missed) + len(extra)) / denom
        if score > best_score:
            best_score = score
            best_start = float(start)
    return best_start


def _plot_ecg_overlay(
    dataset: str,
    record: str,
    out_path: Path,
    title: str,
    start_s: float | None = None,
    duration_s: float = 8.0,
) -> None:
    raw, fs, ref_s, det_s = _load_record(dataset, record)
    if start_s is None:
        start_s = _select_failure_window(ref_s, det_s, duration_s)
    end_s = start_s + duration_s
    start = max(0, int(round(start_s * fs)))
    end = min(len(raw), int(round(end_s * fs)))
    t = np.arange(start, end) / fs
    y = raw[start:end]
    y_med = float(np.nanmedian(y))
    y_scale = float(np.nanpercentile(np.abs(y - y_med), 95))
    if not math.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0
    y_norm = (y - y_med) / y_scale

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t, y_norm, color=PALETTE["signal"], linewidth=1.2, label="ECG")

    ref_win = ref_s[(ref_s >= start_s) & (ref_s <= end_s)]
    det_win = det_s[(det_s >= start_s) & (det_s <= end_s)]
    for idx, r in enumerate(ref_win):
        ax.axvline(
            r,
            color=PALETTE["reference"],
            linewidth=1.2,
            alpha=0.8,
            label="Reference beat" if idx == 0 else None,
        )
    for idx, d in enumerate(det_win):
        ax.axvline(
            d,
            color=PALETTE["detected"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Detected R peak" if idx == 0 else None,
        )
    ax.set_xlim(start_s, end_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized ECG")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _style_axis(ax, title, "Normalized ECG")
    subtitle = f"{_display_dataset(dataset)} record {record}, {start_s:.1f}-{end_s:.1f} s"
    ax.text(0.01, 0.96, subtitle, transform=ax.transAxes, fontsize=9, color="#475569", va="top")
    _save(fig, out_path)


def _plot_figures(per_dataset: pd.DataFrame, per_record: pd.DataFrame, out_figures: Path) -> None:
    _plot_dataset_metrics(per_dataset, out_figures)
    _plot_latency(per_dataset, out_figures)
    _plot_f1_distribution(per_record, out_figures)
    _plot_error_pareto(per_record, out_figures)
    _plot_ecg_overlay(
        dataset="mit_bih_arrhythmia",
        record="100",
        out_path=out_figures / "rpeak_example_good_mitdb_100.png",
        title="Good Alignment Example",
        start_s=10.0,
    )
    _plot_ecg_overlay(
        dataset="supraventricular_arrhythmia",
        record="868",
        out_path=out_figures / "rpeak_example_failure_svdb_868.png",
        title="Failure Example",
        start_s=None,
    )


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = df[list(columns)].copy()
    for col in selected.columns:
        if pd.api.types.is_float_dtype(selected[col]):
            selected[col] = selected[col].map(lambda x: f"{x:.4f}" if abs(x) <= 1.5 else f"{x:.2f}")
    lines = []
    lines.append("| " + " | ".join(selected.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(selected.columns)) + "|")
    for _, row in selected.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in selected.columns) + " |")
    return "\n".join(lines)


def _write_readme(out_dir: Path, tables: Dict, run_config: Dict) -> None:
    summary = tables["dataset_summary"].copy()
    summary = summary.rename(columns={
        "dataset_label": "Dataset",
        "n_records": "Records",
        "n_annotated_beats": "Beats",
        "sensitivity": "Sensitivity",
        "ppv": "PPV",
        "f1": "F1",
        "median_confirmed_event_lag_ms": "Median Live Lag (ms)",
        "main_note": "Note",
    })
    worst = tables["worst_records"].head(10).copy().rename(columns={
        "dataset_label": "Dataset",
        "record": "Record",
        "f1": "F1",
        "ppv": "PPV",
        "sensitivity": "Sensitivity",
        "n_extra": "Extra",
        "n_missed": "Missed",
        "failure_mode": "Failure Mode",
    })

    text = f"""# R-Peak Detection Report

This report summarizes `adaptive_threshold_v2` on the default beat-annotated
R-peak benchmark datasets.

Source benchmark folder:

```text
{run_config.get("out_dir", "")}
```

## Key Figures

![Dataset metrics](figures/rpeak_dataset_metrics.png)

![Record-level F1 distribution](figures/rpeak_record_f1_distribution.png)

![Failure concentration](figures/rpeak_error_pareto.png)

![Live lag by dataset](figures/rpeak_live_lag_by_dataset.png)

## ECG Examples

![Good MIT-BIH example](figures/rpeak_example_good_mitdb_100.png)

![Failure SVDB example](figures/rpeak_example_failure_svdb_868.png)

## Dataset Summary

{_markdown_table(summary, ["Dataset", "Records", "Beats", "Sensitivity", "PPV", "F1", "Median Live Lag (ms)", "Note"])}

## Worst Records

{_markdown_table(worst, ["Dataset", "Record", "F1", "PPV", "Sensitivity", "Extra", "Missed", "Failure Mode"])}

## Tables

- `tables/dataset_summary.csv`
- `tables/worst_records.csv`
- `tables/failure_concentration.csv`
- `tables/failure_by_f1_threshold.csv`

## External Viewer

Manual waveform inspection can also be done with PhysioNet LightWAVE:
https://physionet.org/lightwave/
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def build_report(results_dir: Path, out_dir: Path) -> None:
    figures, tables = _ensure_dirs(out_dir)
    per_dataset = pd.read_csv(results_dir / "per_algorithm_per_dataset.csv")
    per_record = pd.read_csv(results_dir / "per_record.csv")
    run_config = json.loads((results_dir / "run_config.json").read_text(encoding="utf-8"))

    table_map = _write_tables(per_dataset, per_record, tables)
    _plot_figures(per_dataset, per_record, figures)
    _write_readme(out_dir, table_map, run_config)

    try:
        results_label = str(results_dir.relative_to(_ROOT))
    except ValueError:
        results_label = str(results_dir)
    try:
        out_label = str(out_dir.relative_to(_ROOT))
    except ValueError:
        out_label = str(out_dir)
    manifest = {
        "results_dir": results_label,
        "out_dir": out_label,
        "figures": sorted(p.name for p in figures.glob("*.png")),
        "tables": sorted(p.name for p in tables.glob("*.csv")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "Results" / "rpeak_comparison" / "adaptive_v2_default_datasets_20260618",
        help="Benchmark output folder containing per_record.csv and per_algorithm_per_dataset.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "reports" / "rpeak_detection",
        help="Report folder to write figures, tables, and README.md.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    build_report(args.results_dir.resolve(), args.out_dir.resolve())
    print(f"Wrote report assets to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
