"""
Artifact / lead-off stress test for the Layer 2 signal-quality (SQI) gate.

Purpose
-------
Deployment on animals and closed-loop rigs adds artifact classes that public
ECG benchmarks under-represent: stimulation pulse trains, electrode lead-off
(flatline), saturation/clipping, powerline pickup, EMG bursts, and baseline
wander. A false PERMIT during any of these is a safety failure. This script
verifies that the SQI ensemble (kSQI, pSQI, bSQI) plus the existing FFT SQI
gates INHIBIT under each injected artifact, while leaving clean windows mostly
permitted.

It is an OFFLINE robustness audit (numpy/scipy only, no torch). It reports, per
artifact type, the fraction of windows the SQI gate would inhibit (detection
rate = higher is safer) and the false-inhibit rate on clean windows.

Signal source
-------------
By default it synthesises clean single-lead ECG-like windows (documented as a
SMOKE surrogate, not physiological ground truth). If --data-dir points at a
WFDB dataset and wfdb is installed, it draws real clean windows instead
(preferred for any reportable number).

Usage
-----
    # Synthetic smoke run
    python Layer2/validation/run_artifact_stress_test.py \
        --out-dir Results/layer2_artifact_stress --n-windows 200

    # Real windows from MIT-BIH (preferred)
    python Layer2/validation/run_artifact_stress_test.py \
        --data-dir data --dataset mit_bih_arrhythmia \
        --out-dir Results/layer2_artifact_stress
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
for _p in (str(_ROOT), str(_L2), str(_L2 / "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from signal_features import (  # noqa: E402
    beat_detector_agreement_sqi,
    kurtosis_sqi,
    power_spectrum_sqi,
)
from full_features import spectral_sqi_features  # noqa: E402

try:
    from decision.config import SQI_ENSEMBLE_HARD_RULES, DEFAULT_HARD_RULES
except Exception:  # pragma: no cover - path fallback
    from pipeline.decision.config import SQI_ENSEMBLE_HARD_RULES, DEFAULT_HARD_RULES  # type: ignore


# ---------------------------------------------------------------------------
# Clean-window sources
# ---------------------------------------------------------------------------

def synth_clean_ecg(n: int, fs: float, hr_bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesise a clean single-lead ECG-like window (smoke surrogate)."""
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros(n)
    beat_t = 0.15
    while beat_t < t[-1]:
        # QRS: narrow biphasic spike; T: broad low bump.
        sig += 1.0 * np.exp(-0.5 * ((t - beat_t) / 0.010) ** 2)
        sig -= 0.20 * np.exp(-0.5 * ((t - beat_t + 0.020) / 0.012) ** 2)
        sig += 0.25 * np.exp(-0.5 * ((t - beat_t - 0.18) / 0.040) ** 2)
        beat_t += rr * (1.0 + 0.03 * rng.standard_normal())
    sig += 0.01 * rng.standard_normal(n)
    return sig


def load_clean_windows(
    data_dir: Path, dataset: str, fs_target: float, win_s: float, max_windows: int
) -> Optional[List[np.ndarray]]:
    """Load clean windows from a WFDB dataset if wfdb is available; else None."""
    try:
        import wfdb  # type: ignore
        from scipy.signal import resample_poly  # noqa: F401
    except Exception:
        return None
    ds_dir = data_dir / dataset
    recs = sorted({p.stem for p in ds_dir.glob("*.dat")})
    if not recs:
        return None
    win_n = int(round(win_s * fs_target))
    out: List[np.ndarray] = []
    for rec in recs:
        try:
            sig, fields = wfdb.rdsamp(str(ds_dir / rec))
        except Exception:
            continue
        x = np.asarray(sig[:, 0], dtype=float)
        fs = float(fields["fs"])
        if fs != fs_target:
            from scipy.signal import resample
            x = resample(x, int(len(x) * fs_target / fs))
        step = win_n
        for start in range(0, len(x) - win_n, step):
            out.append(x[start:start + win_n])
            if len(out) >= max_windows:
                return out
    return out or None


# ---------------------------------------------------------------------------
# Artifact injectors (each returns a corrupted copy)
# ---------------------------------------------------------------------------

def art_baseline_wander(x, fs, rng):
    t = np.arange(len(x)) / fs
    f = rng.uniform(0.15, 0.5)
    amp = 3.0 * (np.max(x) - np.min(x) + 1e-9)
    return x + amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))


def art_hf_emg(x, fs, rng):
    amp = 1.0 * np.std(x)
    return x + amp * rng.standard_normal(len(x))


def art_powerline(x, fs, rng):
    t = np.arange(len(x)) / fs
    f = rng.choice([50.0, 60.0])
    amp = 0.8 * (np.max(x) - np.min(x) + 1e-9)
    return x + amp * np.sin(2 * np.pi * f * t)


def art_saturation_clip(x, fs, rng):
    lim = np.percentile(np.abs(x), 40)
    return np.clip(x, -lim, lim)


def art_flatline_leadoff(x, fs, rng):
    return np.zeros_like(x) + 1e-3 * rng.standard_normal(len(x))


def art_stim_pulse(x, fs, rng):
    y = x.copy()
    period = int(fs * rng.uniform(0.2, 0.5))
    width = max(1, int(fs * 0.002))
    amp = 8.0 * (np.max(np.abs(x)) + 1e-9)
    for start in range(int(fs * 0.1), len(x), max(period, 1)):
        y[start:start + width] += amp
    return y


ARTIFACTS = {
    "baseline_wander": art_baseline_wander,
    "hf_emg": art_hf_emg,
    "powerline": art_powerline,
    "saturation_clip": art_saturation_clip,
    "flatline_leadoff": art_flatline_leadoff,
    "stim_pulse": art_stim_pulse,
}


# ---------------------------------------------------------------------------
# SQI gate on one window
# ---------------------------------------------------------------------------

def _simple_alt_detector(x: np.ndarray, fs: float) -> np.ndarray:
    """A crude alternative R-peak detector for bSQI (find_peaks on |x|)."""
    try:
        from scipy.signal import find_peaks
    except Exception:
        return np.array([], dtype=float)
    dist = max(1, int(0.25 * fs))
    thr = np.percentile(np.abs(x), 95)
    idx, _ = find_peaks(np.abs(x - np.median(x)), distance=dist, height=thr)
    return idx / fs


def _ref_detector(x: np.ndarray, fs: float) -> np.ndarray:
    try:
        from r_peak_detector import RPeakDetector, RPeakDetectorConfig
    except Exception:
        return np.array([], dtype=float)
    try:
        res = RPeakDetector(RPeakDetectorConfig(), fs).process_signal(x)
        return np.asarray(res.peak_samples, dtype=float) / fs
    except Exception:
        return np.array([], dtype=float)


def sqi_inhibit(x: np.ndarray, fs: float, include_bsqi: bool = True) -> Dict[str, object]:
    """
    Apply SQI ensemble + FFT SQI hard rules; return which (if any) fired.

    bSQI (two-detector agreement) is only meaningful with real detectors on real
    ECG. On the synthetic surrogate the reference and alternate detectors
    disagree even on clean windows, so bSQI is disabled unless include_bsqi is
    True (set only for the WFDB data path).
    """
    feats = {
        "signal__ksqi": kurtosis_sqi(x),
        "signal__psqi": power_spectrum_sqi(x, fs),
    }
    if include_bsqi:
        peaks_ref = _ref_detector(x, fs)
        peaks_alt = _simple_alt_detector(x, fs)
        feats["signal__bsqi"] = beat_detector_agreement_sqi(peaks_ref, peaks_alt)
    fft = spectral_sqi_features(x, fs)
    feats["signal__hf_noise_ratio"] = fft["hf_noise_ratio"]
    feats["signal__lf_wander_ratio"] = fft["lf_wander_ratio"]

    rules = {k: v for k, v in SQI_ENSEMBLE_HARD_RULES.items() if include_bsqi or k != "signal__bsqi"}
    for k in ("signal__hf_noise_ratio", "signal__lf_wander_ratio"):
        if k in DEFAULT_HARD_RULES:
            rules[k] = DEFAULT_HARD_RULES[k]

    fired = []
    for feat, (lo, hi) in rules.items():
        val = feats.get(feat, float("nan"))
        if not np.isfinite(val):
            continue
        if lo is not None and val < lo:
            fired.append(f"{feat}<{lo}")
        if hi is not None and val > hi:
            fired.append(f"{feat}>{hi}")
    return {"inhibit": bool(fired), "reasons": "|".join(fired), **feats}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--dataset", default="mit_bih_arrhythmia")
    p.add_argument("--fs", type=float, default=250.0)
    p.add_argument("--win-s", type=float, default=8.0)
    p.add_argument("--n-windows", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    clean: Optional[List[np.ndarray]] = None
    if args.data_dir is not None:
        clean = load_clean_windows(args.data_dir, args.dataset, args.fs, args.win_s, args.n_windows)
    source = "wfdb" if clean else "synthetic"
    if clean is None:
        n = int(round(args.win_s * args.fs))
        clean = [
            synth_clean_ecg(n, args.fs, hr_bpm=rng.uniform(60, 100), rng=rng)
            for _ in range(args.n_windows)
        ]

    include_bsqi = source == "wfdb"
    rows = []
    clean_inhibit = 0
    for x in clean:
        r = sqi_inhibit(x, args.fs, include_bsqi=include_bsqi)
        clean_inhibit += int(r["inhibit"])
        for art_name, fn in ARTIFACTS.items():
            xc = fn(x, args.fs, rng)
            rc = sqi_inhibit(xc, args.fs, include_bsqi=include_bsqi)
            rows.append({"artifact": art_name, "inhibit": int(rc["inhibit"]), "reasons": rc["reasons"]})

    per_window = pd.DataFrame(rows)
    summary = (
        per_window.groupby("artifact")["inhibit"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "detection_rate", "count": "n"})
        .reset_index()
        .sort_values("detection_rate")
    )
    clean_fi = clean_inhibit / max(1, len(clean))

    per_window.to_csv(args.out_dir / "artifact_per_window.csv", index=False)
    summary.to_csv(args.out_dir / "artifact_detection_summary.csv", index=False)
    pd.DataFrame([{"signal_source": source, "n_clean_windows": len(clean),
                   "clean_false_inhibit_rate": round(clean_fi, 4)}]).to_csv(
        args.out_dir / "artifact_clean_baseline.csv", index=False)

    print(f"signal source           : {source} ({len(clean)} clean windows)")
    print(f"bSQI in gate            : {include_bsqi}  (disabled on synthetic surrogate)")
    print(f"clean false-inhibit rate: {clean_fi:.3f}  (lower is better)")
    print("artifact detection rate (fraction inhibited; higher is safer):")
    for _, r in summary.iterrows():
        print(f"  {r['artifact']:<18} {r['detection_rate']:.3f}  (n={int(r['n'])})")
    print(f"Wrote: {args.out_dir}/artifact_detection_summary.csv (+ per_window, clean_baseline)")


if __name__ == "__main__":
    main()
