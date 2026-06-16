"""
stream_record.py — real-time-style Layer 1 replay.

Unlike animate_record.py, this tool does not precompute the full Layer 1 output.
It reveals a WFDB record chunk by chunk, filters only the samples available so
far, runs Layer 1 in causal mode, and updates the plot with what the system
would know at that moment.

Example:
    python Layer1/tools/stream_record.py --record data/mit_bih_arrhythmia/100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.animation import FuncAnimation

_L1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_L1))
from _bootstrap import setup_layer1_paths

setup_layer1_paths(include_archive=False)

from main_pipeline import run_layer1
from plot_helpers import DECISION_COLORS
from reference_annotations import get_reference_beats


def _empty_offsets() -> np.ndarray:
    return np.empty((0, 2))


def _decision_marker(name: str) -> str:
    if "reject" in name or "skip" in name:
        return "x"
    if "first" in name:
        return "s"
    if name in {"enter_recovery", "recovery_to_calibration", "calibration_to_running"}:
        return "D"
    return "o"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a WFDB ECG record as a real-time Layer 1 stream.",
    )
    parser.add_argument("--record", required=True, help="WFDB record stem/path")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--window-s", type=float, default=8.0)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="Optional limit for faster debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rec = wfdb.rdrecord(args.record)
    raw = rec.p_signal[:, args.channel].astype(float)
    fs = float(rec.fs)
    if args.max_duration_s is not None:
        raw = raw[: int(round(args.max_duration_s * fs))]

    try:
        ann = wfdb.rdann(args.record, "atr")
        ref_samples, _ = get_reference_beats(ann)
        ref_samples = ref_samples[ref_samples < len(raw)]
    except Exception:
        ref_samples = np.array([], dtype=int)

    n = len(raw)
    total_s = n / fs
    step = max(1, int(round(args.step_ms * fs / 1000.0)))
    window = max(step, int(round(args.window_s * fs)))
    interval_ms = max(1.0, args.step_ms / max(args.speed, 1e-6))

    state: Dict[str, object] = {
        "paused": False,
        "end": min(window, n),
        "last_result": None,
    }

    fig, (ax_ecg, ax_rr) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Layer 1 real-time replay — space: pause/resume")

    raw_line, = ax_ecg.plot([], [], color="0.65", lw=0.8, label="raw ECG")
    filt_line, = ax_ecg.plot([], [], color="navy", lw=1.0, label="causal filtered ECG")
    now_line = ax_ecg.axvline(0.0, color="black", linestyle="--", lw=1.0, label="current time")
    ref_scatter = ax_ecg.scatter([], [], c="gold", s=18, zorder=5, label="reference")
    cand_scatter = ax_ecg.scatter([], [], c="orange", s=24, zorder=6, label="candidate")

    decision_scatters = {}
    for decision, color in DECISION_COLORS.items():
        decision_scatters[decision] = ax_ecg.scatter(
            [],
            [],
            c=color,
            marker=_decision_marker(decision),
            s=42,
            zorder=7,
            label=decision,
        )

    rr_line, = ax_rr.plot([], [], "o-", color="black", ms=4, lw=1.0, label="RR candidate")
    rr_ref_line, = ax_rr.plot([], [], "-", color="dodgerblue", lw=2, label="RR ref")
    low_line, = ax_rr.plot([], [], "--", color="red", lw=1.0, label="lower band")
    high_line, = ax_rr.plot([], [], "--", color="red", lw=1.0, label="upper band")
    mode_text = ax_ecg.text(
        0.01,
        0.95,
        "",
        transform=ax_ecg.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )

    ax_ecg.set_ylabel("ECG")
    ax_rr.set_ylabel("RR (ms)")
    ax_rr.set_xlabel("Time (s)")
    ax_ecg.legend(loc="upper right", ncol=3, fontsize=8)
    ax_rr.legend(loc="upper right", ncol=4, fontsize=8)

    raw_ylim = np.percentile(raw[np.isfinite(raw)], [1, 99])
    raw_pad = 0.15 * max(raw_ylim[1] - raw_ylim[0], 1e-6)
    ax_ecg.set_ylim(raw_ylim[0] - raw_pad, raw_ylim[1] + raw_pad)

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not bool(state["paused"])

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        if not state["paused"]:
            state["end"] = min(n, int(state["end"]) + step)

        end = int(state["end"])
        start = max(0, end - window)
        now_s = end / fs
        t = np.arange(start, end) / fs

        result = run_layer1(
            raw[:end],
            fs,
            filter_mode="causal",
            match_tol_ms=50.0,
        )
        state["last_result"] = result

        filt = result.filt
        raw_line.set_data(t, raw[start:end])
        filt_line.set_data(t, filt[start:end])
        now_line.set_xdata([now_s, now_s])
        ax_ecg.set_xlim(max(0.0, now_s - args.window_s), max(args.window_s, now_s))

        local_raw = raw[start:end]
        local_filt = filt[start:end]
        if len(local_raw) and len(local_filt):
            lo = min(float(np.nanmin(local_raw)), float(np.nanmin(local_filt)))
            hi = max(float(np.nanmax(local_raw)), float(np.nanmax(local_filt)))
            pad = 0.15 * max(hi - lo, 1e-6)
            ax_ecg.set_ylim(lo - pad, hi + pad)

        def set_peak_offsets(scatter, samples: np.ndarray, y_source: np.ndarray) -> None:
            mask = (samples >= start) & (samples < end)
            selected = samples[mask]
            if len(selected):
                scatter.set_offsets(np.c_[selected / fs, y_source[selected]])
            else:
                scatter.set_offsets(_empty_offsets())

        set_peak_offsets(ref_scatter, ref_samples, raw)
        set_peak_offsets(cand_scatter, result.candidate_samples, filt)

        decisions_by_name = {name: [] for name in decision_scatters}
        rr_times = []
        rr_vals = []
        rr_refs = []
        low_vals = []
        high_vals = []

        for decision in result.supervisor.state.decisions:
            sample = int(decision.sample)
            if start <= sample < end and decision.decision in decisions_by_name:
                decisions_by_name[decision.decision].append(sample)

            tx = decision.t_ms / 1000.0
            if start / fs <= tx <= end / fs and decision.rr_candidate_ms > 0:
                rr_times.append(tx)
                rr_vals.append(decision.rr_candidate_ms)
                rr_refs.append(decision.rr_ref_ms)
                band_frac = (
                    decision.band_frac
                    if np.isfinite(decision.band_frac)
                    else result.supervisor.cfg.default_confidence_frac
                )
                low_vals.append(decision.rr_ref_ms * (1.0 - band_frac))
                high_vals.append(decision.rr_ref_ms * (1.0 + band_frac))

        for name, scatter in decision_scatters.items():
            samples = np.asarray(decisions_by_name[name], dtype=int)
            if len(samples):
                scatter.set_offsets(np.c_[samples / fs, filt[samples]])
            else:
                scatter.set_offsets(_empty_offsets())

        rr_line.set_data(rr_times, rr_vals)
        rr_ref_line.set_data(rr_times, rr_refs)
        low_line.set_data(rr_times, low_vals)
        high_line.set_data(rr_times, high_vals)
        ax_rr.set_xlim(ax_ecg.get_xlim())
        if rr_vals:
            all_rr = np.asarray(rr_vals + rr_refs + low_vals + high_vals, dtype=float)
            all_rr = all_rr[np.isfinite(all_rr)]
            if len(all_rr):
                lo = max(0.0, float(np.nanmin(all_rr)) - 100.0)
                hi = float(np.nanmax(all_rr)) + 100.0
                ax_rr.set_ylim(lo, hi)

        sup_state = result.supervisor.state
        latest_decision = sup_state.decisions[-1].decision if sup_state.decisions else "none"
        mode_text.set_text(
            f"t={now_s:6.2f}s / {total_s:6.2f}s\n"
            f"mode={sup_state.mode}\n"
            f"latest={latest_decision}\n"
            f"candidates={len(result.candidate_samples)}  "
            f"accepted={len(result.accepted_samples)}  "
            f"triggers={len(result.trigger_samples)}"
        )

        if end >= n:
            state["paused"] = True

        artists = [
            raw_line,
            filt_line,
            now_line,
            ref_scatter,
            cand_scatter,
            rr_line,
            rr_ref_line,
            low_line,
            high_line,
            mode_text,
        ]
        artists.extend(decision_scatters.values())
        return artists

    anim = FuncAnimation(
        fig,
        update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    fig._layer1_stream_animation = anim  # Keep animation alive for interactive backends.
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
