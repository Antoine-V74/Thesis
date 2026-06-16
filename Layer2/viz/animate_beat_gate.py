"""
Animated walkthrough of the Layer 2 beat-synchronous gate.

Shows how one trigger beat flows through:
  ECG window -> features -> hard rules -> Mahalanobis -> permit/inhibit

Output: Results/layer2/viz/layer2_gate_animation.gif (or .mp4 if ffmpeg available)

Run:
    .venv\\Scripts\\python Layer2\\viz\\animate_beat_gate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
sys.path.insert(0, str(_L2))
from _bootstrap import setup_layer2_paths  # noqa: E402

setup_layer2_paths()

from main_pipeline import calibrate_layer2, extract_layer2_features  # noqa: E402
from plot_style import DARK, GREEN, RED, apply_style  # noqa: E402


def _synthetic_ecg(fs: float, duration_s: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Simple sinus rhythm with one PVC-like beat."""
    t = np.arange(0, duration_s, 1 / fs)
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    rr = 0.85
    peaks = []
    beat_idx = 0
    while beat_idx * rr < duration_s - 0.2:
        center = beat_idx * rr
        if beat_idx == 7:
            # PVC: early, wider, inverted-ish
            center -= 0.15
            amp = -1.4
            width = 0.07
        else:
            amp = 1.0
            width = 0.04
        peaks.append(center)
        pulse = amp * np.exp(-0.5 * ((t - center) / width) ** 2)
        sig += pulse
        beat_idx += 1
    return sig.astype(float), np.array(peaks, dtype=float)


def _build_baseline(sig: np.ndarray, fs: float, peaks: np.ndarray, n: int = 6):
    feats = []
    for i in range(min(n, len(peaks))):
        t = peaks[i]
        i0 = max(0, int((t - 5.0) * fs))
        i1 = min(len(sig), int((t + 0.1) * fs))
        w = sig[i0:i1]
        f, _ = extract_layer2_features(w, fs, r_peaks_s=peaks[: i + 1], compute_entropy=False,
                                       compute_spectral_hrv=False, focus_peak_s=t)
        feats.append(f)
    return calibrate_layer2(feats)


def make_animation(out: Path, fps: int = 2) -> None:
    apply_style()
    fs = 360.0
    sig, peaks = _synthetic_ecg(fs)
    cal = _build_baseline(sig, fs, peaks)

    # Animate beats 5..9 (includes PVC at 7)
    beat_indices = list(range(5, min(10, len(peaks))))

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.2, 1], hspace=0.35, wspace=0.25)
    ax_ecg = fig.add_subplot(gs[0, :])
    ax_feat = fig.add_subplot(gs[1, 0])
    ax_gate = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    def _frame(k: int):
        for ax in (ax_ecg, ax_feat, ax_gate):
            ax.clear()
        ax_text.clear()
        ax_text.axis("off")

        bi = beat_indices[k]
        t_b = peaks[bi]
        i0 = max(0, int((t_b - 5.0) * fs))
        i1 = min(len(sig), int((t_b + 0.1) * fs))
        t_axis = np.arange(i0, i1) / fs
        window = sig[i0:i1]
        past_peaks = peaks[peaks <= t_b + 0.003]

        feats, _ = extract_layer2_features(
            window, fs, r_peaks_s=past_peaks, compute_entropy=False,
            compute_spectral_hrv=False, focus_peak_s=t_b,
        )
        decision = cal.decide(feats)
        permit = bool(decision.get("permit", False))
        reason = str(decision.get("reason", ""))
        mahal = float(decision.get("mahalanobis", 0))
        mahal_thr = float(decision.get("mahalanobis_threshold", 1))
        hard = str(decision.get("hard_rule_violated", "") or "")

        # ECG panel
        ax_ecg.plot(t_axis, window, color=DARK, lw=1.2)
        ax_ecg.axvline(t_b, color=RED if not permit else GREEN, lw=2, ls="--", label="Trigger R")
        ax_ecg.axvspan(t_b - 5.0, t_b + 0.1, alpha=0.08, color="steelblue", label="5s + 100ms window")
        ax_ecg.set_xlim(t_axis[0], t_axis[-1])
        ax_ecg.set_ylabel("ECG (a.u.)")
        ax_ecg.set_title(f"Beat {bi + 1} — Layer 2 beat-synchronous gate", fontweight="bold")
        ax_ecg.legend(loc="upper right", fontsize=8)

        # Key features
        keys = [
            "morph__template_corr", "rr__beat_coupling_ratio",
            "signal__hf_noise_ratio", "rr__short_rr_fraction",
        ]
        vals = [feats.get(k, float("nan")) for k in keys]
        colors = ["#2980b9" if np.isfinite(v) else "#ccc" for v in vals]
        ax_feat.barh(range(len(keys)), [v if np.isfinite(v) else 0 for v in vals], color=colors)
        ax_feat.set_yticks(range(len(keys)))
        ax_feat.set_yticklabels([k.replace("__", "\n") for k in keys], fontsize=8)
        ax_feat.set_title("Key features", fontweight="bold")

        # Gate meter
        ax_gate.barh([0], [min(mahal / max(mahal_thr, 1e-6), 2.0)], color=RED if not permit else GREEN, height=0.4)
        ax_gate.axvline(1.0, color="black", ls="--", lw=1.5, label="Mahalanobis threshold")
        ax_gate.set_xlim(0, 2)
        ax_gate.set_yticks([])
        ax_gate.set_xlabel("Mahalanobis / threshold")
        ax_gate.set_title("Statistical gate", fontweight="bold")
        ax_gate.legend(fontsize=8, loc="lower right")

        status = "PERMIT" if permit else "INHIBIT"
        status_color = GREEN if permit else RED
        steps = [
            ("1. Hard rules", "FAIL — " + hard if hard else "PASS"),
            ("2. Mahalanobis", f"{mahal:.2f} / {mahal_thr:.2f}"),
            ("3. Decision", status),
        ]
        y = 0.85
        for title, detail in steps:
            ax_text.text(0.02, y, title, fontsize=11, fontweight="bold", transform=ax_text.transAxes)
            ax_text.text(0.22, y, detail, fontsize=11, transform=ax_text.transAxes,
                         color=RED if "FAIL" in detail or status == "INHIBIT" and title.startswith("3") else DARK)
            y -= 0.28
        ax_text.add_patch(mpatches.FancyBboxPatch(
            (0.72, 0.15), 0.24, 0.7, boxstyle="round,pad=0.02",
            facecolor=status_color, alpha=0.15, transform=ax_text.transAxes,
        ))
        ax_text.text(0.84, 0.5, status, ha="center", va="center", fontsize=20, fontweight="bold",
                     color=status_color, transform=ax_text.transAxes)
        ax_text.text(0.84, 0.25, reason[:40], ha="center", va="center", fontsize=8,
                     transform=ax_text.transAxes)

        return []

    ani = animation.FuncAnimation(fig, _frame, frames=len(beat_indices), interval=1000 // fps)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(out.with_suffix(".mp4"), writer="ffmpeg", fps=fps, dpi=120)
        print(f"  Saved: {out.with_suffix('.mp4')}")
    except Exception:
        gif_path = out.with_suffix(".gif")
        ani.save(gif_path, writer="pillow", fps=fps, dpi=120)
        print(f"  Saved: {gif_path} (ffmpeg unavailable, used GIF)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("Results/layer2/viz/layer2_gate_animation"))
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args(argv)
    make_animation(args.out, fps=args.fps)


if __name__ == "__main__":
    main()
