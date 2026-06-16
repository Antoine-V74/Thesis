"""
Generate representative diagnostic plots for the final report.

Plots:
  1. Wrong polarity – record 108 (segment 0–30 s)
  2. Supervisor recovery – record 116 (segment 460–490 s)
  3. Clean success – record 100 (segment 0–30 s)
  4. Arrhythmia inhibition – record 200 (~60–90 s, VT)

Each plot shows:
  - raw + filtered ECG (dual y-axes)
  - oracle peaks (gold)
  - Layer 1 accepted (blue)
  - supervisor rejected candidates with reason label (red)
  - optional horizontal RR band (dashed)

Saved to Results/final_mitbih_validation/plots/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = Path(__file__).resolve().parent
_L1 = _HERE.parent
sys.path.insert(0, str(_L1))
from _bootstrap import setup_layer1_paths  # noqa: E402

setup_layer1_paths(include_archive=True)

from analysis_helpers import (  # noqa: E402
    apply_filters,
    load_record,
    match_peaks,
    run_layer1_detector,
)
NORMAL = set("NLRAaJSVFeEjf/!Qn?")
TOL_MS = 80.0


def _run_layer1_with_decisions(filt, fs):
    cand_s, acc_s, sup = run_layer1_detector(filt, fs)
    decisions = {int(d.sample): (d.decision, d.rr_candidate_ms, d.rr_ref_ms)
                 for d in sup.state.decisions}
    pol = sup.state.mode  # supervisor mode for context
    return cand_s, acc_s, decisions, pol, 0.0


def _plot_segment(
    title: str,
    dataset_path: str,
    t_start: float,
    t_end: float,
    out_path: Path,
    annotate_rr_band: bool = False,
) -> None:
    try:
        import wfdb
        rec = wfdb.rdrecord(dataset_path)
        ann = wfdb.rdann(dataset_path, "atr")
    except Exception as exc:
        print(f"  [skip] {dataset_path}: {exc}")
        return

    raw = rec.p_signal[:, 0].astype(float)
    fs = float(rec.fs)
    filt = apply_filters(raw, fs)

    i_start = int(t_start * fs)
    i_end = int(t_end * fs)
    t_full = np.arange(len(filt)) / fs

    # Oracle peaks in segment
    oracle_t = np.array([
        s / fs for s, sym in zip(ann.sample, ann.symbol)
        if sym in NORMAL and t_start <= s / fs <= t_end
    ])

    # Layer 1 (fast causal + supervisor)
    cand_s, acc_s, decisions, pol, thr = _run_layer1_with_decisions(filt, fs)
    fixed_acc_t = acc_s[(acc_s >= t_start) & (acc_s <= t_end)]
    rejected_cands = [
        (c, decisions[int(round(c * fs))])
        for c in cand_s
        if t_start <= c <= t_end and int(round(c * fs)) not in
        set(int(round(p * fs)) for p in acc_s)
        and int(round(c * fs)) in decisions
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 4))
    t_seg = t_full[i_start:i_end]
    ecg_seg = filt[i_start:i_end]
    ax.plot(t_seg, ecg_seg, color="steelblue", lw=0.8, alpha=0.85, zorder=1)

    if len(oracle_t):
        ax.scatter(
            oracle_t, filt[(oracle_t * fs).astype(int).clip(0, len(filt) - 1)],
            marker="^", s=80, color="gold", edgecolors="darkorange", zorder=5,
            label="Oracle R-peaks",
        )
    if len(fixed_acc_t):
        ax.scatter(
            fixed_acc_t, filt[(fixed_acc_t * fs).astype(int).clip(0, len(filt) - 1)],
            marker="o", s=60, color="royalblue", edgecolors="navy", zorder=4,
            label="Layer 1 accepted",
        )

    # Rejected candidates
    reason_colors = {
        "reject_out_of_band_low": "firebrick",
        "reject_out_of_band_high": "tomato",
        "reject_short": "orangered",
        "reject_long": "darkorange",
        "recovery_reject_band": "red",
        "skip_refractory": "lightcoral",
        "skip_post_stim_protection": "lightsalmon",
    }
    for c, (dec, rr_cand, rr_ref) in rejected_cands:
        col = reason_colors.get(dec, "gray")
        ax.scatter(
            [c], [filt[min(int(round(c * fs)), len(filt) - 1)]],
            marker="x", s=70, color=col, zorder=3, linewidths=1.8,
        )
        ax.annotate(
            f"{dec}\n({rr_cand:.0f}/{rr_ref:.0f}ms)",
            xy=(c, filt[min(int(round(c * fs)), len(filt) - 1)]),
            xytext=(0, -35), textcoords="offset points",
            fontsize=6, color=col, ha="center",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6),
        )

    ax.set_xlim(t_start, t_end)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Filtered ECG (mV)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="gold", label="Oracle R-peaks"),
        mpatches.Patch(color="royalblue", label="Layer 1 accepted"),
        mpatches.Patch(color="firebrick", label="Rejected candidates (×)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("Results/final_mitbih_validation/plots"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mitdb = args.data_dir / "mit_bih_arrhythmia"

    plots = [
        (
            "Wrong polarity – MIT-BIH 108 (0–30 s)",
            str(mitdb / "108"), 0.0, 30.0,
            args.out_dir / "01_wrong_polarity_108.png",
        ),
        (
            "Supervisor recovery – MIT-BIH 116 (460–490 s)",
            str(mitdb / "116"), 460.0, 490.0,
            args.out_dir / "02_supervisor_recovery_116.png",
        ),
        (
            "Clean success – MIT-BIH 100 (0–30 s)",
            str(mitdb / "100"), 0.0, 30.0,
            args.out_dir / "03_clean_success_100.png",
        ),
        (
            "Arrhythmia inhibition – MIT-BIH 200 (60–90 s, VT region)",
            str(mitdb / "200"), 60.0, 90.0,
            args.out_dir / "04_arrhythmia_200.png",
        ),
    ]

    for title, path, t0, t1, out_p in plots:
        print(f"Plotting: {title}")
        _plot_segment(title, path, t0, t1, out_p)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
