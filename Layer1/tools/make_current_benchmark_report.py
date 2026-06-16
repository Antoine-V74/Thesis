"""
Create readable visual reports from the current Layer 1 benchmark CSVs.

Usage:
    .venv\\Scripts\\python Layer1\\tools\\make_current_benchmark_report.py `
        --result-dir Results\\layer1\\current_benchmark_20260611
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _fmt(x, digits: int = 3) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def _save_table_html(df: pd.DataFrame, path: Path, title: str, max_rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = df.head(max_rows).copy()
    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 5px 7px; text-align: right; }
    th { background: #f3f5f7; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    tr:nth-child(even) { background: #fafafa; }
    .note { color: #555; font-size: 13px; }
    </style>
    """
    body = shown.to_html(index=False, escape=True)
    path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='note'>Showing first {min(len(df), max_rows)} of {len(df)} rows. Full data remains in CSV.</p>"
        f"{body}</body></html>",
        encoding="utf-8",
    )


def _barh(
    df: pd.DataFrame,
    y: str,
    cols: List[str],
    title: str,
    xlabel: str,
    save_path: Path,
    *,
    xlim=None,
) -> None:
    if df.empty:
        return
    plt = _setup_matplotlib()
    plot_df = df.copy()
    labels = plot_df[y].astype(str).tolist()
    y_pos = np.arange(len(plot_df))
    height = 0.8 / max(len(cols), 1)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(plot_df) + 1.5)))
    for i, col in enumerate(cols):
        vals = pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float)
        offset = (i - (len(cols) - 1) / 2.0) * height
        ax.barh(y_pos + offset, vals, height=height, label=col)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_dataset_overview(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    if dataset_df.empty:
        return
    df = dataset_df.sort_values("healthy_accepted_sensitivity", ascending=True)
    _barh(
        df,
        "dataset",
        ["healthy_accepted_sensitivity", "healthy_trigger_rate_vs_ref"],
        "Layer 1 Healthy Beat Performance By Dataset",
        "Rate",
        fig_dir / "dataset_healthy_performance.png",
        xlim=(0, 1),
    )
    _barh(
        df.sort_values("arrhythmia_trigger_rate_vs_ref", ascending=True),
        "dataset",
        ["arrhythmia_accepted_sensitivity", "arrhythmia_trigger_rate_vs_ref"],
        "Layer 1 Arrhythmia-Labeled Beat Pass-Through By Dataset",
        "Rate",
        fig_dir / "dataset_arrhythmia_passthrough.png",
        xlim=(0, 1),
    )
    _barh(
        dataset_df.sort_values("n_recovery_entries", ascending=True),
        "dataset",
        ["n_recovery_entries"],
        "Recovery Entries By Dataset",
        "Count",
        fig_dir / "dataset_recovery_entries.png",
    )
    _barh(
        dataset_df.sort_values("mean_detector_confirmation_delay_ms", ascending=True),
        "dataset",
        ["mean_detector_confirmation_delay_ms", "p95_detector_confirmation_delay_ms"],
        "Detector Confirmation Delay By Dataset",
        "ms",
        fig_dir / "dataset_detector_delay.png",
    )


def plot_label_summary(label_df: pd.DataFrame, fig_dir: Path) -> None:
    if label_df.empty:
        return
    plt = _setup_matplotlib()
    df = label_df.copy()
    df["dataset_label"] = df["dataset"].astype(str) + " / " + df["label"].astype(str)
    df = df.sort_values(["label", "accepted_sensitivity"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["accepted_sensitivity"], height=0.34, label="accepted sensitivity")
    ax.barh(y + 0.18, df["trigger_rate_vs_ref"], height=0.34, label="trigger rate")
    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_title("Performance By Dataset And Beat Label")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "dataset_label_rates.png", dpi=140)
    plt.close(fig)


def plot_record_outliers(record_df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    if record_df.empty:
        return
    cols = [
        "dataset", "record", "n_ref_beats", "n_ref_arrhythmia",
        "healthy_accepted_sensitivity", "healthy_trigger_rate_vs_ref",
        "arrhythmia_accepted_sensitivity", "arrhythmia_trigger_rate_vs_ref",
        "n_recovery_entries", "accepted_ppv_vs_all_ref",
        "mean_detector_confirmation_delay_ms",
    ]
    available = [c for c in cols if c in record_df.columns]

    worst_healthy = record_df.sort_values("healthy_accepted_sensitivity", ascending=True, na_position="last")
    worst_arrhythmia_trigger = record_df.sort_values(
        "arrhythmia_trigger_rate_vs_ref", ascending=False, na_position="last",
    )
    most_recovery = record_df.sort_values("n_recovery_entries", ascending=False)

    _save_table_html(worst_healthy[available], table_dir / "worst_records_healthy.html", "Worst Records: Healthy Accepted Sensitivity")
    _save_table_html(worst_arrhythmia_trigger[available], table_dir / "worst_records_arrhythmia_trigger.html", "Worst Records: Arrhythmia Trigger Rate")
    _save_table_html(most_recovery[available], table_dir / "records_most_recovery.html", "Records With Most Recovery Entries")

    top = worst_healthy.head(35).copy()
    top["record_id"] = top["dataset"].astype(str) + "/" + top["record"].astype(str)
    _barh(
        top.sort_values("healthy_accepted_sensitivity", ascending=True),
        "record_id",
        ["healthy_accepted_sensitivity", "healthy_trigger_rate_vs_ref"],
        "Worst Records For Healthy Beat Availability",
        "Rate",
        fig_dir / "worst_records_healthy.png",
        xlim=(0, 1),
    )

    arr = worst_arrhythmia_trigger[worst_arrhythmia_trigger["n_ref_arrhythmia"].fillna(0) > 0].head(35).copy()
    arr["record_id"] = arr["dataset"].astype(str) + "/" + arr["record"].astype(str)
    _barh(
        arr.sort_values("arrhythmia_trigger_rate_vs_ref", ascending=True),
        "record_id",
        ["arrhythmia_accepted_sensitivity", "arrhythmia_trigger_rate_vs_ref"],
        "Worst Records: Arrhythmia-Labeled Beats Passing Through",
        "Rate",
        fig_dir / "worst_records_arrhythmia_passthrough.png",
        xlim=(0, 1),
    )


def plot_window_views(window_df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    if window_df.empty:
        return
    cols = [
        "dataset", "record", "window_start_s", "window_end_s", "n_ref",
        "n_ref_healthy", "n_ref_arrhythmia", "n_missed_accepted",
        "n_arrhythmia_triggered", "healthy_trigger_rate", "arrhythmia_trigger_rate",
        "algorithm_after_label_ms_mean", "score_badness",
    ]
    available = [c for c in cols if c in window_df.columns]
    worst = window_df.sort_values("score_badness", ascending=False).head(300)
    good = window_df[
        (window_df["n_ref"].fillna(0) >= 5)
        & (window_df["n_missed_accepted"].fillna(0) == 0)
        & (window_df["n_arrhythmia_triggered"].fillna(0) == 0)
    ].sort_values(["n_ref", "algorithm_after_label_ms_mean"], ascending=[False, True]).head(300)
    _save_table_html(worst[available], table_dir / "worst_windows.html", "Worst 10-Second Windows")
    _save_table_html(good[available], table_dir / "good_windows.html", "Good 10-Second Windows")

    plt = _setup_matplotlib()
    top = worst.head(50).copy()
    top["window_id"] = (
        top["dataset"].astype(str) + "/" + top["record"].astype(str)
        + " @" + top["window_start_s"].astype(int).astype(str) + "s"
    )
    fig, ax = plt.subplots(figsize=(13, max(5, 0.28 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, top["score_badness"], color="crimson", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["window_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Badness score = missed accepted beats + 2 * arrhythmia triggered")
    ax.set_title("Worst Local 10-Second Windows")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "worst_windows_badness.png", dpi=140)
    plt.close(fig)


def plot_delay_distributions(beat_df: pd.DataFrame, fig_dir: Path) -> None:
    if beat_df.empty or "algorithm_after_label_ms" not in beat_df.columns:
        return
    plt = _setup_matplotlib()
    df = beat_df[np.isfinite(pd.to_numeric(beat_df["algorithm_after_label_ms"], errors="coerce"))].copy()
    if df.empty:
        return
    df["algorithm_after_label_ms"] = pd.to_numeric(df["algorithm_after_label_ms"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, sub in df.groupby("label"):
        vals = sub["algorithm_after_label_ms"].dropna().clip(-50, 200)
        if len(vals):
            ax.hist(vals, bins=60, alpha=0.45, label=f"{label} (n={len(vals)})")
    ax.set_xlabel("Algorithm confirmation time after labeled R-peak (ms)")
    ax.set_ylabel("Beat count")
    ax.set_title("Timing Delay Distribution By Beat Label")
    ax.legend()
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "delay_distribution_by_label.png", dpi=140)
    plt.close(fig)

    summary = df.groupby(["dataset", "label"]).agg(
        n=("algorithm_after_label_ms", "count"),
        mean_ms=("algorithm_after_label_ms", "mean"),
        p95_ms=("algorithm_after_label_ms", lambda x: np.nanpercentile(x, 95)),
    ).reset_index()
    pivot = summary.pivot_table(index="dataset", columns="label", values="mean_ms", aggfunc="first")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(pivot))))
    im = ax.imshow(pivot.fillna(np.nan).to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Algorithm Delay After Label (ms)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="ms")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "delay_heatmap_dataset_label.png", dpi=140)
    plt.close(fig)


def write_dashboard(
    result_dir: Path,
    fig_dir: Path,
    table_dir: Path,
    dataset_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    report_dir = result_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    if not dataset_df.empty:
        cards.append(("Datasets", int(dataset_df["dataset"].nunique())))
        cards.append(("Records", int(dataset_df["n_records"].sum())))
        cards.append(("Labeled beats", int(dataset_df["n_ref_beats"].sum())))
        cards.append(("Recovery entries", int(dataset_df["n_recovery_entries"].sum())))
    if not errors_df.empty:
        cards.append(("Skipped/error records", len(errors_df)))

    card_html = "".join(
        f"<div class='card'><div class='num'>{value}</div><div>{html.escape(name)}</div></div>"
        for name, value in cards
    )
    fig_names = [
        "dataset_healthy_performance.png",
        "dataset_arrhythmia_passthrough.png",
        "dataset_label_rates.png",
        "worst_records_healthy.png",
        "worst_records_arrhythmia_passthrough.png",
        "worst_windows_badness.png",
        "delay_distribution_by_label.png",
        "delay_heatmap_dataset_label.png",
        "dataset_recovery_entries.png",
        "dataset_detector_delay.png",
    ]
    figs = "\n".join(
        f"<section><h2>{html.escape(name.replace('_', ' ').replace('.png', '').title())}</h2>"
        f"<img src='../figures/{name}'></section>"
        for name in fig_names if (fig_dir / name).exists()
    )
    links = "\n".join(
        f"<li><a href='tables/{p.name}'>{html.escape(p.stem.replace('_', ' ').title())}</a></li>"
        for p in sorted(table_dir.glob("*.html"))
    )
    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #17202a; }
    .cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 24px; }
    .card { background: #f3f6fa; border: 1px solid #d8e0ea; border-radius: 8px; padding: 14px 18px; min-width: 150px; }
    .num { font-size: 26px; font-weight: 700; }
    img { max-width: 100%; border: 1px solid #ddd; margin-bottom: 24px; }
    section { margin-top: 22px; }
    a { color: #0b61a4; }
    </style>
    """
    (report_dir / "index.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Layer 1 Benchmark Report</title>{css}</head>"
        f"<body><h1>Layer 1 Benchmark Report</h1>"
        f"<p>Readable visual layout generated from <code>{html.escape(str(result_dir / 'csv'))}</code>.</p>"
        f"<div class='cards'>{card_html}</div>"
        f"<h2>Compact Tables</h2><ul>{links}</ul>"
        f"{figs}</body></html>",
        encoding="utf-8",
    )


def resolve_result_dir(arg: Path) -> Path:
    if arg.exists():
        return arg
    pointer = Path("Results/layer1/latest_current_benchmark.txt")
    if pointer.exists():
        text = pointer.read_text(encoding="utf-8").strip()
        p = Path(text)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot resolve result dir: {arg}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("Results/layer1/current_benchmark_20260611"))
    args = parser.parse_args(argv)

    result_dir = resolve_result_dir(args.result_dir)
    csv_dir = result_dir / "csv"
    fig_dir = result_dir / "figures"
    table_dir = result_dir / "report" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = _read_csv(csv_dir / "per_dataset_summary.csv")
    label_df = _read_csv(csv_dir / "per_dataset_label_summary.csv")
    record_df = _read_csv(csv_dir / "per_record_all.csv")
    window_df = _read_csv(csv_dir / "per_window_10s_all.csv")
    beat_df = _read_csv(csv_dir / "per_beat_all.csv")
    errors_df = _read_csv(csv_dir / "errors.csv")
    print(
        "Loaded CSV rows: "
        f"datasets={len(dataset_df)}, labels={len(label_df)}, "
        f"records={len(record_df)}, windows={len(window_df)}, beats={len(beat_df)}"
    )

    plot_dataset_overview(dataset_df, fig_dir)
    plot_label_summary(label_df, fig_dir)
    plot_record_outliers(record_df, fig_dir, table_dir)
    plot_window_views(window_df, fig_dir, table_dir)
    plot_delay_distributions(beat_df, fig_dir)

    _save_table_html(dataset_df, table_dir / "dataset_summary.html", "Dataset Summary")
    _save_table_html(label_df, table_dir / "dataset_label_summary.html", "Dataset Label Summary")
    if not errors_df.empty:
        _save_table_html(errors_df, table_dir / "errors.html", "Skipped / Error Records")

    write_dashboard(result_dir, fig_dir, table_dir, dataset_df, errors_df)
    print(f"Figure files: {len(list(fig_dir.glob('*.png')))}")
    print(f"Wrote report: {result_dir / 'report' / 'index.html'}")
    print(f"Wrote figures: {fig_dir}")


if __name__ == "__main__":
    main()
