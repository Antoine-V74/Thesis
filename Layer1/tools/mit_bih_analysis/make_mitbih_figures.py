"""
Focused MIT-BIH Layer 1 figures.

All outputs are written under:
    Results/layer1/mit_bih/

Usage:
    .venv\\Scripts\\python Layer1\\tools\\mit_bih_analysis\\make_mitbih_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET = "mit_bih_arrhythmia"
OUTCOME_COLORS = {
    "Correctly not triggered": "#27ae60",
    "Incorrectly triggered": "#c0392b",
    "No detector match": "#95a5a6",
    "Detected, supervisor rejected": "#f39c12",
    "Accepted, not triggered": "#3498db",
}


def _violin(ax, groups: list[np.ndarray], positions: list[float], labels: list[str], colors: list[str]) -> None:
    parts = ax.violinplot(groups, positions=positions, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _load(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    beats = pd.read_csv(csv_dir / "per_beat_all.csv", low_memory=False)
    records = pd.read_csv(csv_dir / "per_record_all.csv", low_memory=False)
    beats = beats[beats["dataset"] == DATASET].copy()
    records = records[records["dataset"] == DATASET].copy()
    return beats, records


def _arrhythmia_outcomes(beats: pd.DataFrame) -> pd.DataFrame:
    arr = beats[beats["label"] == "arrhythmia"].copy()
    arr["candidate_bool"] = arr["candidate_matched"].astype(bool)
    arr["accepted_bool"] = arr["accepted_matched"].astype(bool)
    arr["triggered_bool"] = arr["triggered_matched"].astype(bool)
    arr["correctly_not_triggered"] = ~arr["triggered_bool"]
    return arr


def plot_real_accepted_delay_violin(beats: pd.DataFrame, out: Path) -> None:
    """Violin: timing delay for accepted peaks that match a real reference R-peak."""
    df = beats[
        beats["accepted_matched"].astype(bool)
        & np.isfinite(pd.to_numeric(beats["algorithm_after_label_ms"], errors="coerce"))
    ].copy()
    df["algorithm_after_label_ms"] = pd.to_numeric(df["algorithm_after_label_ms"])
    df = df[df["label"].isin(["healthy", "arrhythmia"])]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ["healthy", "arrhythmia"]
    groups = [df.loc[df["label"] == label, "algorithm_after_label_ms"].to_numpy() for label in order]
    _violin(ax, groups, [1, 2], order, ["#2e86ab", "#c0392b"])
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Reference beat label")
    ax.set_ylabel("Delay after labeled R-peak (ms)")
    ax.set_title(
        "MIT-BIH: delay for accepted peaks that match a real reference beat\n"
        "(positive = algorithm confirms after the annotated R-peak)"
    )
    for i, label in enumerate(["healthy", "arrhythmia"]):
        sub = df[df["label"] == label]["algorithm_after_label_ms"]
        ax.text(
            i, ax.get_ylim()[1] * 0.95,
            f"n={len(sub)}\nmedian={sub.median():.1f} ms",
            ha="center", va="top", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_false_triggers(records: pd.DataFrame, out: Path) -> None:
    """How many triggers did not correspond to any reference beat?"""
    df = records.copy()
    df["true_triggers"] = df["n_triggers"] - df["extra_triggers"]
    df["false_trigger_rate"] = df["extra_triggers"] / df["n_triggers"].replace(0, np.nan)
    df = df.sort_values("extra_triggers", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    totals = {
        "True triggers\n(matched reference)": int(df["true_triggers"].sum()),
        "False triggers\n(no reference match)": int(df["extra_triggers"].sum()),
    }
    total_triggers = sum(totals.values())
    axes[0].bar(totals.keys(), totals.values(), color=["#27ae60", "#e74c3c"])
    axes[0].set_ylabel("Count across all MIT-BIH records")
    axes[0].set_title("MIT-BIH: trigger peaks vs reference annotations")
    for x, v in enumerate(totals.values()):
        axes[0].text(x, v, f"{v:,}\n({v / total_triggers:.1%})", ha="center", va="bottom")

    rates = df["false_trigger_rate"].dropna().to_numpy()
    _violin(axes[1], [rates], [1], [""], ["#e67e22"])
    axes[1].set_ylabel("False trigger fraction per record")
    axes[1].set_title("MIT-BIH: per-record rate of triggers\nwith no matching reference beat")
    axes[1].set_xticks([])
    med = df["false_trigger_rate"].median()
    axes[1].text(
        0, med,
        f"median={med:.1%}\nrecords={len(df)}",
        ha="center", va="bottom", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_arrhythmia_rejection(beats: pd.DataFrame, out: Path) -> None:
    """For annotated arrhythmia beats: rejected vs wrongly triggered vs missed."""
    arr = _arrhythmia_outcomes(beats)
    triggered = arr["triggered_bool"]
    accepted = arr["accepted_bool"]
    candidate = arr["candidate_bool"]

    wrongly_triggered = int(triggered.sum())
    accepted_not_triggered = int((accepted & ~triggered).sum())
    rejected_after_candidate = int((candidate & ~accepted).sum())
    missed_no_candidate = int((~candidate).sum())
    correctly_not_triggered = int((~triggered).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: safety-oriented binary view
    binary = pd.Series({
        "Correctly not triggered": correctly_not_triggered,
        "Incorrectly triggered": wrongly_triggered,
    })
    axes[0].bar(binary.index, binary.values, color=[OUTCOME_COLORS[k] for k in binary.index])
    axes[0].set_ylabel("Annotated arrhythmia beats")
    axes[0].set_title(
        "MIT-BIH: arrhythmia-labeled reference beats\n"
        f"correct rejection rate = {correctly_not_triggered / len(arr):.1%}"
    )
    for i, v in enumerate(binary.values):
        axes[0].text(i, v, f"{v:,}\n({v / len(arr):.1%})", ha="center", va="bottom", fontsize=9)

    # Right: finer breakdown
    detail = pd.Series({
        "No detector match": missed_no_candidate,
        "Detected, supervisor rejected": rejected_after_candidate,
        "Accepted, not triggered": accepted_not_triggered,
        "Incorrectly triggered": wrongly_triggered,
    })
    axes[1].barh(detail.index[::-1], detail.values[::-1], color=[OUTCOME_COLORS[k] for k in detail.index[::-1]])
    axes[1].set_xlabel("Annotated arrhythmia beats")
    axes[1].set_title("MIT-BIH: what happened to arrhythmia-labeled beats?")
    for i, (name, v) in enumerate(detail[::-1].items()):
        axes[1].text(v, i, f"  {v:,} ({v / len(arr):.1%})", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_arrhythmia_rejection_by_record(beats: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Record-by-record safety view for arrhythmia-labeled beats."""
    arr = _arrhythmia_outcomes(beats)
    summary = (
        arr.groupby("record", as_index=False)
        .agg(
            n_arrhythmia=("record", "size"),
            n_correctly_not_triggered=("correctly_not_triggered", "sum"),
            n_incorrectly_triggered=("triggered_bool", "sum"),
        )
    )
    summary["rejection_rate"] = summary["n_correctly_not_triggered"] / summary["n_arrhythmia"]
    summary["trigger_rate"] = summary["n_incorrectly_triggered"] / summary["n_arrhythmia"]
    summary = summary.sort_values("rejection_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = np.where(summary["n_arrhythmia"] >= 20, "#27ae60", "#95a5a6")
    ax.bar(summary["record"], summary["rejection_rate"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Arrhythmia beats correctly not triggered")
    ax.set_xlabel("MIT-BIH record")
    ax.set_title(
        "MIT-BIH: arrhythmia rejection by record\n"
        "grey = fewer than 20 arrhythmia-labeled beats"
    )
    ax.tick_params(axis="x", rotation=90)
    for i, row in enumerate(summary.itertuples(index=False)):
        if row.n_arrhythmia >= 20:
            ax.text(i, row.rejection_rate + 0.015, f"{row.rejection_rate:.0%}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def plot_arrhythmia_rejection_record_violin(record_summary: pd.DataFrame, out: Path) -> None:
    """Distribution of per-record arrhythmia rejection rates."""
    df = record_summary[record_summary["n_arrhythmia"] >= 20].copy()
    values = df["rejection_rate"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _violin(ax, [values], [1], [""], ["#27ae60"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-record arrhythmia rejection rate")
    ax.set_title(
        "MIT-BIH: are results broad, or driven by a few bad records?\n"
        "records with at least 20 arrhythmia-labeled beats"
    )
    median = float(np.median(values)) if len(values) else float("nan")
    q1 = float(np.percentile(values, 25)) if len(values) else float("nan")
    q3 = float(np.percentile(values, 75)) if len(values) else float("nan")
    ax.text(1, median, f"median={median:.1%}\nIQR={q1:.1%}-{q3:.1%}\nn={len(values)}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_arrhythmia_rejection_by_symbol(beats: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Arrhythmia-type view using MIT-BIH beat annotation symbols."""
    arr = _arrhythmia_outcomes(beats)
    summary = (
        arr.groupby("symbol", as_index=False)
        .agg(
            n_arrhythmia=("symbol", "size"),
            n_correctly_not_triggered=("correctly_not_triggered", "sum"),
            n_incorrectly_triggered=("triggered_bool", "sum"),
        )
    )
    summary["rejection_rate"] = summary["n_correctly_not_triggered"] / summary["n_arrhythmia"]
    summary["trigger_rate"] = summary["n_incorrectly_triggered"] / summary["n_arrhythmia"]
    summary = summary.sort_values(["rejection_rate", "n_arrhythmia"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = np.where(summary["n_arrhythmia"] >= 50, "#27ae60", "#95a5a6")
    ax.bar(summary["symbol"], summary["rejection_rate"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Arrhythmia beats correctly not triggered")
    ax.set_xlabel("MIT-BIH beat annotation symbol")
    ax.set_title(
        "MIT-BIH: arrhythmia rejection by beat type\n"
        "grey = fewer than 50 beats"
    )
    for i, row in enumerate(summary.itertuples(index=False)):
        ax.text(i, row.rejection_rate + 0.015, f"{row.rejection_rate:.0%}\nn={row.n_arrhythmia}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def plot_arrhythmia_symbol_record_violin(beats: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Per-record rejection distributions inside each arrhythmia symbol."""
    arr = _arrhythmia_outcomes(beats)
    grouped = (
        arr.groupby(["symbol", "record"], as_index=False)
        .agg(
            n_arrhythmia=("symbol", "size"),
            n_correctly_not_triggered=("correctly_not_triggered", "sum"),
            n_incorrectly_triggered=("triggered_bool", "sum"),
        )
    )
    grouped["rejection_rate"] = grouped["n_correctly_not_triggered"] / grouped["n_arrhythmia"]
    grouped = grouped[grouped["n_arrhythmia"] >= 5].copy()

    symbol_counts = arr.groupby("symbol").size().sort_values(ascending=False)
    symbols = [symbol for symbol in symbol_counts.index if symbol_counts.loc[symbol] >= 50 and symbol in set(grouped["symbol"])]
    symbols = symbols[:8]
    groups = [grouped.loc[grouped["symbol"] == symbol, "rejection_rate"].to_numpy() for symbol in symbols]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if groups:
        _violin(ax, groups, list(range(1, len(symbols) + 1)), symbols, ["#8e44ad"] * len(symbols))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-record rejection rate")
    ax.set_xlabel("MIT-BIH beat annotation symbol")
    ax.set_title(
        "MIT-BIH: distribution of rejection rates by arrhythmia type\n"
        "symbol-record groups with at least 5 beats; symbols with at least 50 total beats"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return grouped


def write_readme(out_dir: Path) -> None:
    text = """# MIT-BIH Layer 1 analysis figures

These figures are generated from the Layer 1 benchmark CSVs and focus only on MIT-BIH Arrhythmia Database rows.

## Files

1. `01_real_accepted_delay_violin.png`
   Violin plot of **algorithm_after_label_ms** for accepted peaks that match a real reference beat.
   Split by healthy vs arrhythmia-labeled reference beats.

2. `02_false_triggers.png`
   Left: total true vs false triggers across MIT-BIH.
   Right: violin of per-record false-trigger rate.
   **False trigger** = supervisor trigger peak with no reference beat within 100 ms.

3. `03_arrhythmia_rejection.png`
   For reference beats labeled arrhythmia (V, F, E, ...):
   - **Correctly not triggered** = good (Layer 1 did not mark a running-mode trigger)
   - **Incorrectly triggered** = bad (abnormal beat passed to trigger list)

4. `04_arrhythmia_rejection_by_record.png`
   Record-by-record rejection rate for arrhythmia-labeled beats.

5. `05_arrhythmia_rejection_record_violin.png`
   Violin plot of per-record rejection rates. This shows whether the global result
   is typical or pulled down by a few bad records.

6. `06_arrhythmia_rejection_by_symbol.png`
   Rejection rate by MIT-BIH arrhythmia annotation symbol.

7. `07_arrhythmia_symbol_record_violin.png`
   Per-record rejection-rate distributions inside common arrhythmia symbols.

## CSV summaries

The `csv/` subfolder contains the data used for record-level and symbol-level plots.

## Delay columns

- `candidate_peak_delay_ms` = (detected peak sample - reference sample) / fs
- `detector_confirmation_delay_ms` = (confirmation sample - detected peak sample) / fs
- `algorithm_after_label_ms` = (confirmation sample - reference sample) / fs

Positive delay means the algorithm confirms **after** the annotated R-peak time.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("Results/layer1/current_benchmark_20260611/csv"),
        help="Directory containing per_beat_all.csv and per_record_all.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Results/layer1/mit_bih"),
        help="Directory where figures and summary CSVs are written",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    csv_out = out_dir / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)

    beats, records = _load(args.csv_dir)
    if beats.empty:
        raise SystemExit(f"No MIT-BIH rows found under {args.csv_dir}")

    plot_real_accepted_delay_violin(beats, out_dir / "01_real_accepted_delay_violin.png")
    plot_false_triggers(records, out_dir / "02_false_triggers.png")
    plot_arrhythmia_rejection(beats, out_dir / "03_arrhythmia_rejection.png")
    record_summary = plot_arrhythmia_rejection_by_record(beats, out_dir / "04_arrhythmia_rejection_by_record.png")
    plot_arrhythmia_rejection_record_violin(record_summary, out_dir / "05_arrhythmia_rejection_record_violin.png")
    symbol_summary = plot_arrhythmia_rejection_by_symbol(beats, out_dir / "06_arrhythmia_rejection_by_symbol.png")
    symbol_record_summary = plot_arrhythmia_symbol_record_violin(beats, out_dir / "07_arrhythmia_symbol_record_violin.png")
    record_summary.to_csv(csv_out / "arrhythmia_rejection_by_record.csv", index=False)
    symbol_summary.to_csv(csv_out / "arrhythmia_rejection_by_symbol.csv", index=False)
    symbol_record_summary.to_csv(csv_out / "arrhythmia_rejection_by_symbol_record.csv", index=False)
    write_readme(out_dir)

    print(f"Wrote MIT-BIH figures to {out_dir}")


if __name__ == "__main__":
    main()
