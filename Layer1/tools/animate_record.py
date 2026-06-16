"""
animate_record.py — sliding ECG animation to watch Layer 1 live.

Example:
    python Layer1/tools/animate_record.py --record data/mit_bih_arrhythmia/100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.animation import FuncAnimation

_L1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_L1))
from _bootstrap import setup_layer1_paths

setup_layer1_paths(include_archive=False)

from main_pipeline import run_layer1
from reference_annotations import get_reference_beats


def run_pipeline(raw: np.ndarray, fs: float):
    result = run_layer1(raw, fs)
    return {
        "filt": result.filt,
        "candidate_samples": result.candidate_samples,
        "accepted_samples": result.accepted_samples,
        "supervisor": result.supervisor,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--window-s", type=float, default=5.0)
    args = parser.parse_args()

    rec = wfdb.rdrecord(args.record)
    raw = rec.p_signal[:, args.channel]
    fs = float(rec.fs)
    try:
        ann = wfdb.rdann(args.record, "atr")
        ref_samples, _ = get_reference_beats(ann)
    except Exception:
        ref_samples = np.array([], dtype=int)

    out = run_pipeline(raw, fs)
    filt = out["filt"]
    win = int(args.window_s * fs)
    n = len(filt)

    fig, ax = plt.subplots(figsize=(12, 4))
    line, = ax.plot([], [], "b-", lw=0.8)
    ax.set_ylim(np.percentile(filt, 1), np.percentile(filt, 99))
    ax.set_xlim(0, args.window_s)
    ax.set_title("Layer 1 — fast causal detector (press space to pause)")

    ref_scatter = ax.scatter([], [], c="gold", s=20, zorder=5, label="reference")
    cand_scatter = ax.scatter([], [], c="orange", s=30, zorder=6, label="candidates")
    acc_scatter = ax.scatter([], [], c="green", s=40, zorder=7, label="accepted")
    ax.legend(loc="upper right")

    paused = [False]

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(frame):
        if paused[0]:
            return line, ref_scatter, cand_scatter, acc_scatter
        end = min(n, frame * int(fs * 0.05 * args.speed) + win)
        start = max(0, end - win)
        t = np.arange(start, end) / fs - start / fs
        line.set_data(t, filt[start:end])

        def _mask(samples):
            m = (samples >= start) & (samples < end)
            return (samples[m] - start) / fs, filt[samples[m]]

        rx, ry = _mask(ref_samples)
        cx, cy = _mask(out["candidate_samples"])
        ax_, ay_ = _mask(out["accepted_samples"])
        ref_scatter.set_offsets(np.c_[rx, ry] if len(rx) else np.empty((0, 2)))
        cand_scatter.set_offsets(np.c_[cx, cy] if len(cx) else np.empty((0, 2)))
        acc_scatter.set_offsets(np.c_[ax_, ay_] if len(ax_) else np.empty((0, 2)))
        return line, ref_scatter, cand_scatter, acc_scatter

    ani = FuncAnimation(fig, update, frames=range(0, n, int(fs * 0.05)),
                        interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
