
"""
Diagnostic plotting helpers.
"""

import matplotlib.pyplot as plt
import numpy as np

from rhythm_supervisor import RRSupervisor


DECISION_COLORS = {
    "ignored_wait_arm": "lightgray",
    "first_beat": "black",
    "accept_calibration": "dodgerblue",
    "accept_running": "green",
    "skip_refractory": "orange",
    "skip_post_stim_protection": "goldenrod",
    "reject_short": "red",
    "reject_long": "purple",
    "reject_out_of_band_low": "crimson",
    "reject_out_of_band_high": "red",
    "enter_recovery": "brown",
    "recovery_first_anchor": "saddlebrown",
    "recovery_accept": "limegreen",
    "recovery_reject_short": "darkred",
    "recovery_reanchor_long": "purple",
    "recovery_reject_band": "darkorange",
    "recovery_to_calibration": "deepskyblue",
    "calibration_to_running": "navy",
}


def plot_record(
    record_name: str,
    raw: np.ndarray,
    filt: np.ndarray,
    fs: float,
    ref_samples: np.ndarray,
    candidate_samples: np.ndarray,
    supervisor: RRSupervisor,
    t0_s: float = 0.0,
    t1_s: float = 20.0,
) -> None:
    """
    Plot one time window of a record.

    The figure contains:
    1. Raw ECG with reference beats
    2. Filtered ECG with detector candidates and supervisor decisions
    3. RR candidate values, reference/EMA, and adaptive acceptance bands
    """
    n0 = int(round(t0_s * fs))
    n1 = min(int(round(t1_s * fs)), len(raw))
    t = np.arange(n0, n1) / fs
    decisions = supervisor.state.decisions

    plt.figure(figsize=(15, 10))

    # Panel 1: raw ECG with reference annotations.
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, raw[n0:n1], lw=1.0, color="black", label="raw ECG")
    ref_win = ref_samples[(ref_samples >= n0) & (ref_samples < n1)]
    if len(ref_win):
        ax1.scatter(ref_win / fs, raw[ref_win], s=18, color="cyan", label="reference beats")
    ax1.set_title(f"{record_name} | ECG and algorithm decisions")
    ax1.set_ylabel("Raw ECG")
    ax1.legend(loc="upper right")

    # Panel 2: filtered ECG and event-level decisions.
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, filt[n0:n1], lw=1.0, color="navy", label="filtered ECG")

    cand_win = candidate_samples[(candidate_samples >= n0) & (candidate_samples < n1)]
    if len(cand_win):
        ax2.scatter(
            cand_win / fs,
            filt[cand_win],
            s=16,
            color="gray",
            alpha=0.5,
            label="candidate peaks",
        )

    shown = set()
    for decision in decisions:
        sample = decision.sample
        if not (n0 <= sample < n1):
            continue

        color = DECISION_COLORS.get(decision.decision, "magenta")
        label = decision.decision if decision.decision not in shown else None
        shown.add(decision.decision)

        y = filt[sample]
        marker = "o"
        if "reject" in decision.decision or "skip" in decision.decision:
            marker = "x"
        elif "first" in decision.decision:
            marker = "s"
        elif decision.decision in {
            "enter_recovery",
            "recovery_to_calibration",
            "calibration_to_running",
        }:
            marker = "D"

        ax2.scatter(sample / fs, y, s=45, color=color, marker=marker, label=label)

        if decision.blanking_ms > 0:
            ax2.axvspan(
                decision.t_ms / 1000.0,
                (decision.t_ms + decision.blanking_ms) / 1000.0,
                color="gold",
                alpha=0.15,
            )

    ax2.set_ylabel("Filtered ECG")
    ax2.legend(loc="upper right", ncol=2)

    # Panel 3: RR candidates, RR reference/EMA, and acceptance band.
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    times = []
    rr_vals = []
    rr_ref_vals = []
    low_band = []
    high_band = []

    for decision in decisions:
        tx = decision.t_ms / 1000.0
        if not (t0_s <= tx <= t1_s):
            continue

        if decision.rr_candidate_ms > 0 and decision.decision in {
            "accept_calibration",
            "accept_running",
            "reject_out_of_band_low",
            "reject_out_of_band_high",
            "reject_short",
            "reject_long",
            "recovery_accept",
        }:
            times.append(tx)
            rr_vals.append(decision.rr_candidate_ms)
            rr_ref_vals.append(decision.rr_ref_ms)

            band_frac = (
                decision.band_frac
                if np.isfinite(decision.band_frac)
                else supervisor.cfg.default_confidence_frac
            )
            low_band.append(decision.rr_ref_ms * (1.0 - band_frac))
            high_band.append(decision.rr_ref_ms * (1.0 + band_frac))

    if len(times):
        ax3.plot(times, rr_vals, "o-", label="RR candidate")
        ax3.plot(times, rr_ref_vals, "-", lw=2, label="RR_ref / EMA")
        ax3.plot(times, low_band, "--", label="lower band")
        ax3.plot(times, high_band, "--", label="upper band")

    for decision in decisions:
        tx = decision.t_ms / 1000.0
        if not (t0_s <= tx <= t1_s):
            continue
        if decision.decision in {
            "recovery_reject_short",
            "recovery_reanchor_long",
            "recovery_reject_band",
        }:
            ax3.axvline(tx, color="gray", linestyle=":", alpha=0.5)

    for t_ms in supervisor.state.recovery_entry_times_ms:
        tx = t_ms / 1000.0
        if t0_s <= tx <= t1_s:
            ax3.axvline(tx, color="brown", linestyle=":", lw=2, label="enter recovery")

    for t_ms in supervisor.state.recalibration_times_ms:
        tx = t_ms / 1000.0
        if t0_s <= tx <= t1_s:
            ax3.axvline(tx, color="red", linestyle=":", lw=2, label="recalibration")

    handles, labels = ax3.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax3.legend(unique.values(), unique.keys(), loc="upper right")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RR (ms)")
    ax1.set_xlim(t0_s, t1_s)

    plt.tight_layout()
    plt.show()
