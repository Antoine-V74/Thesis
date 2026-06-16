"""
Plot the worst Layer-1 failure records in fixed-length segments.

For each of the top-N records (by oracle-permit / Layer1-inhibit count),
tiles the full recording into non-overlapping segments (default 10 s) and
saves one PNG per segment showing oracle vs fixed vs adaptive peaks.

Usage
-----
    cd "ECG Processing"

    .venv\\Scripts\\python Layer1\\tools\\plot_worst_records.py `
        --summary-csv Results/layer1_adaptive_diagnostics/summary_by_record.csv `
        --per-window-csv Results/layer2_v2_perrec/per_window.csv `
        --data-dir data `
        --out-dir Results/layer1_worst_10s `
        --top-n 10 `
        --segment-s 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_L1 = _HERE.parent
sys.path.insert(0, str(_L1))
from _bootstrap import setup_layer1_paths  # noqa: E402

setup_layer1_paths(include_archive=True)

from analysis_helpers import (  # noqa: E402
    apply_filters,
    load_record,
    load_validation_rows,
    match_peaks,
    rr_stats,
    run_layer1_detector,
)

adaptive_layer1_detect = None


def pick_worst_records(
    summary_csv: Path,
    top_n: int,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    df = df.sort_values("n_failure_windows", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def layer2_segment_stats(
    per_window: pd.DataFrame,
    dataset: str,
    record: str,
    start_s: float,
    segment_s: float,
) -> Dict:
    """Aggregate Layer-2 healthy-window outcomes inside one segment."""
    end_s = start_s + segment_s
    mask = (
        (per_window["dataset"] == dataset)
        & (per_window["record"].astype(str) == str(record))
        & (per_window["label"] == "healthy")
        & (per_window["window_start_s"] >= start_s)
        & (per_window["window_start_s"] < end_s)
    )
    sub = per_window.loc[mask]
    if sub.empty:
        return {
            "n_l2_windows": 0,
            "n_oracle_permit": 0,
            "n_l1_permit": 0,
            "n_l2_failures": 0,
            "failure_fraction": float("nan"),
        }

    piv = sub.pivot_table(
        index="window_start_s",
        columns="mode",
        values="permit",
        aggfunc="first",
    )
    if piv.empty or "oracle" not in piv.columns or "layer1" not in piv.columns:
        return {
            "n_l2_windows": 0,
            "n_oracle_permit": 0,
            "n_l1_permit": 0,
            "n_l2_failures": 0,
            "failure_fraction": float("nan"),
        }

    ora = piv["oracle"].astype(bool)
    l1 = piv["layer1"].astype(bool)
    n_fail = int((ora & ~l1).sum())
    return {
        "n_l2_windows": len(piv),
        "n_oracle_permit": int(ora.sum()),
        "n_l1_permit": int(l1.sum()),
        "n_l2_failures": n_fail,
        "failure_fraction": n_fail / len(piv),
    }


def plot_segment(
    filt: np.ndarray,
    fs: float,
    start_s: float,
    segment_s: float,
    oracle_peaks_s: np.ndarray,
    fixed_peaks_s: np.ndarray,
    adaptive_peaks_s: np.ndarray,
    metrics: Dict,
    l2_stats: Dict,
    save_path: Path,
    dataset: str,
    record: str,
    polarity: str = "",
    tol_ms: float = 80.0,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    end_s = start_s + segment_s
    i0 = int(start_s * fs)
    i1 = min(len(filt), int(end_s * fs))
    if i1 <= i0:
        return

    t = np.arange(i0, i1) / fs
    fig, ax = plt.subplots(figsize=(18, 4.5))

    fail_frac = l2_stats.get("failure_fraction", float("nan"))
    if l2_stats.get("n_l2_windows", 0) == 0:
        bg = "#f5f5f5"
    elif np.isfinite(fail_frac) and fail_frac > 0:
        bg = "#ffe6e6"
    else:
        bg = "#e8f5e9"
    ax.set_facecolor(bg)

    ax.plot(t, filt[i0:i1], color="0.25", lw=0.9, zorder=1, label="ECG filtered")
    ax.axvspan(start_s, end_s, color="royalblue", alpha=0.06)

    def _scatter(peaks_s, marker, color, label, size=55, zorder=5):
        vis = [p for p in peaks_s if start_s <= p < end_s]
        if vis:
            ax.scatter(
                vis,
                [filt[min(int(p * fs), len(filt) - 1)] for p in vis],
                marker=marker, color=color, s=size, label=label, zorder=zorder,
            )

    _scatter(oracle_peaks_s, "o", "green", "oracle", 55)
    _scatter(fixed_peaks_s, "^", "steelblue", "fixed L1", 55)
    if len(adaptive_peaks_s):
        _scatter(adaptive_peaks_s, "*", "magenta", "adaptive L1", 85)

    tol_samp = int(round(tol_ms * fs / 1000.0))
    fixed_set = set(int(round(p * fs)) for p in fixed_peaks_s)
    ad_set = set(int(round(p * fs)) for p in adaptive_peaks_s)
    missed_fixed, missed_ad, extra_fixed = [], [], []
    oracle_set = set(int(round(p * fs)) for p in oracle_peaks_s)

    for p in oracle_peaks_s:
        if not (start_s <= p < end_s):
            continue
        samp = int(round(p * fs))
        if not any(abs(samp - q) <= tol_samp for q in fixed_set):
            missed_fixed.append(p)
        if ad_set and not any(abs(samp - q) <= tol_samp for q in ad_set):
            missed_ad.append(p)

    for p in fixed_peaks_s:
        if not (start_s <= p < end_s):
            continue
        samp = int(round(p * fs))
        if not any(abs(samp - q) <= tol_samp for q in oracle_set):
            extra_fixed.append(p)

    _scatter(missed_fixed, "x", "red", "missed (fixed)", 95, 8)
    if missed_ad:
        _scatter(missed_ad, "P", "black", "missed (adaptive)", 65, 8)
    _scatter(extra_fixed, "D", "orange", "extra (fixed)", 50, 8)

    l2_line = (
        f"L2 healthy 5s windows: {l2_stats['n_l2_windows']}  "
        f"oracle-permit/L1-inhibit: {l2_stats['n_l2_failures']}"
    )
    if l2_stats["n_l2_windows"]:
        l2_line += f"  ({100 * l2_stats['failure_fraction']:.0f}% of segment windows)"

    ax.set_title(
        f"{dataset}/{record}  [{start_s:.0f}s - {end_s:.0f}s]  polarity={polarity}\n"
        f"fixed: missed={metrics['missed_peaks']} extra={metrics['extra_peaks']}  "
        f"sens={metrics['local_sensitivity']:.2f}  PPV={metrics['local_ppv']:.2f}  "
        f"driver={metrics['primary_driver']}  "
        f"long_rr={metrics.get('long_rr_fraction_layer1', float('nan')):.3f}\n"
        + l2_line,
        fontsize=8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(start_s, end_s)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right", ncol=4)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def plot_record_timeline(
    segment_rows: List[Dict],
    save_path: Path,
    dataset: str,
    record: str,
    segment_s: float,
) -> None:
    """One-page timeline: failure density + fixed sensitivity per segment."""
    if not segment_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    starts = [r["segment_start_s"] for r in segment_rows]
    fails = [r["n_l2_failures"] for r in segment_rows]
    sens = [r["fixed_sensitivity"] for r in segment_rows]
    missed = [r["fixed_missed_peaks"] for r in segment_rows]

    fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharex=True)

    axes[0].bar(starts, fails, width=segment_s * 0.9, align="edge", color="crimson", alpha=0.75)
    axes[0].set_ylabel("L2 failures\n(5s windows)")
    axes[0].set_title(f"{dataset}/{record} — segment timeline ({segment_s:.0f}s bins)")

    axes[1].bar(starts, missed, width=segment_s * 0.9, align="edge", color="darkorange", alpha=0.75)
    axes[1].set_ylabel("Missed oracle\nbeats (fixed)")

    valid_sens = [s if np.isfinite(s) else 0 for s in sens]
    axes[2].bar(starts, valid_sens, width=segment_s * 0.9, align="edge", color="steelblue", alpha=0.75)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Fixed sensitivity")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def process_record(
    dataset: str,
    record: str,
    n_failures: int,
    args: argparse.Namespace,
    per_window: pd.DataFrame,
) -> List[Dict]:
    stem = str(args.data_dir / dataset / record)
    try:
        raw, fs, ann = load_record(stem)
    except Exception as exc:
        print(f"  [WARN] skip {dataset}/{record}: {exc}")
        return []

    filt = apply_filters(raw, fs)
    duration_s = len(filt) / fs
    _, fixed_s, _ = run_layer1_detector(filt, fs)

    adaptive_s = np.array([])
    polarity = ""
    if adaptive_layer1_detect is not None:
        ad_res = adaptive_layer1_detect(filt, fs)
        adaptive_s = ad_res.accepted_s
        polarity = ad_res.polarity

    if ann is not None:
        ann_samples = np.asarray(ann.sample, dtype=int)
        ann_symbols = np.asarray(ann.symbol)
        beat_mask = np.isin(ann_symbols, list("NLRAaJSVFeEjf/!Qn?"))
        oracle_s = ann_samples[beat_mask] / fs
    else:
        oracle_s = np.array([])

    out_dir = args.out_dir / f"{dataset}_{record}"
    out_dir.mkdir(parents=True, exist_ok=True)

    segment_rows: List[Dict] = []
    seg_idx = 0
    start_s = 0.0

    while start_s < duration_s - 0.5:
        end_s = min(start_s + args.segment_s, duration_s)
        seg_len = end_s - start_s
        if seg_len < 1.0:
            break

        i0 = int(start_s * fs)
        i1 = int(end_s * fs)
        filt_seg = filt[i0:i1]

        o_win = oracle_s[(oracle_s >= start_s) & (oracle_s < end_s)] - start_s
        f_win = fixed_s[(fixed_s >= start_s) & (fixed_s < end_s)] - start_s
        a_win = adaptive_s[(adaptive_s >= start_s) & (adaptive_s < end_s)] - start_s if len(adaptive_s) else np.array([])

        m_fixed = match_peaks(o_win, f_win, fs, args.match_tolerance_ms)
        l_rr = rr_stats(f_win)
        if m_fixed["fn"] >= 1:
            primary_driver = "missed_beats"
        elif m_fixed["fp"] >= 1:
            primary_driver = "extra_detections"
        elif (np.isfinite(m_fixed["mean_abs_jitter_ms"])
              and m_fixed["mean_abs_jitter_ms"] > 20.0):
            primary_driver = "timing_jitter"
        else:
            primary_driver = "other"
        metrics = {
            "missed_peaks": m_fixed["fn"],
            "extra_peaks": m_fixed["fp"],
            "local_sensitivity": m_fixed["sensitivity"],
            "local_ppv": m_fixed["ppv"],
            "primary_driver": primary_driver,
            "long_rr_fraction_layer1": l_rr["long_rr_fraction"],
        }

        m_ad = match_peaks(o_win, a_win, fs, args.match_tolerance_ms) if len(a_win) else None

        l2_stats = layer2_segment_stats(
            per_window, dataset, record, start_s, args.segment_s,
        )

        fname = f"seg_{int(start_s):04d}_{int(end_s):04d}s.png"
        plot_segment(
            filt=filt,
            fs=fs,
            start_s=start_s,
            segment_s=seg_len,
            oracle_peaks_s=oracle_s,
            fixed_peaks_s=fixed_s,
            adaptive_peaks_s=adaptive_s,
            metrics=metrics,
            l2_stats=l2_stats,
            save_path=out_dir / fname,
            dataset=dataset,
            record=record,
            polarity=polarity,
            tol_ms=args.match_tolerance_ms,
        )

        row = {
            "dataset": dataset,
            "record": record,
            "rank_failure_windows": n_failures,
            "segment_index": seg_idx,
            "segment_start_s": round(start_s, 1),
            "segment_end_s": round(end_s, 1),
            "n_oracle_peaks": len(o_win),
            "n_fixed_peaks": len(f_win),
            "n_adaptive_peaks": len(a_win),
            "fixed_missed_peaks": metrics["missed_peaks"],
            "fixed_extra_peaks": metrics["extra_peaks"],
            "fixed_sensitivity": metrics["local_sensitivity"],
            "fixed_ppv": metrics["local_ppv"],
            "primary_driver": metrics["primary_driver"],
            "adaptive_missed_peaks": m_ad["fn"] if m_ad else float("nan"),
            "adaptive_sensitivity": m_ad["sensitivity"] if m_ad else float("nan"),
            **l2_stats,
            "plot_path": str(out_dir / fname),
        }
        segment_rows.append(row)
        seg_idx += 1
        start_s += args.segment_s

    plot_record_timeline(
        segment_rows,
        out_dir / "_timeline.png",
        dataset,
        record,
        args.segment_s,
    )
    print(f"  {dataset}/{record}: {len(segment_rows)} segments -> {out_dir}")
    return segment_rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("Results/layer1_adaptive_diagnostics/summary_by_record.csv"),
    )
    p.add_argument(
        "--per-window-csv",
        type=Path,
        default=Path("Results/layer2_v2_perrec/per_window.csv"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer1_worst_10s"))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--segment-s", type=float, default=10.0)
    p.add_argument("--match-tolerance-ms", type=float, default=80.0)
    p.add_argument("--feature-set", default="all")
    args = p.parse_args(argv)

    if adaptive_layer1_detect is None:
        print("[WARN] adaptive_layer1_detector not found; plots will omit adaptive peaks.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    worst = pick_worst_records(args.summary_csv, args.top_n, args.datasets)
    rank_path = args.out_dir / "worst_records_ranked.csv"
    worst.to_csv(rank_path, index=False)
    print(f"Top {len(worst)} worst records -> {rank_path}")

    per_window = load_validation_rows(
        args.per_window_csv,
        datasets=args.datasets,
        feature_set=args.feature_set,
    )

    all_rows: List[Dict] = []
    for _, rec in worst.iterrows():
        dataset = str(rec["dataset"])
        record = str(rec["record"])
        n_fail = int(rec["n_failure_windows"])
        print(f"Processing {dataset}/{record} ({n_fail} failure windows) ...")
        rows = process_record(dataset, record, n_fail, args, per_window)
        all_rows.extend(rows)

    summary_path = args.out_dir / "segment_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nDone. {len(all_rows)} segment plots saved under {args.out_dir}")
    print(f"  Per-segment metrics: {summary_path}")
    print(f"  Each record folder contains _timeline.png for navigation.")


if __name__ == "__main__":
    main()
