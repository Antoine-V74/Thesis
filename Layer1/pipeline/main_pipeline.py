"""
Layer 1 main pipeline.

Core signal path:

    ECG signal
        -> filter_ecg()
        -> detect_candidates()
        -> run_supervisor()
        -> run_layer1() result

This file is both:
    1. the importable API used by Layer 2/3 and analysis scripts;
    2. a small command-line entry point for running one record or a folder.
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter

from reference_annotations import greedy_match
from r_peak_detector import (
    FastCausalThresholdConfig,
    FastCausalThresholdResult,
    fast_causal_threshold_detect,
)
from rhythm_supervisor import RRSupervisor, SupervisorConfig


@dataclass
class Layer1Result:
    raw: np.ndarray
    filt: np.ndarray
    fs: float
    detector: FastCausalThresholdResult
    candidate_samples: np.ndarray
    accepted_samples: np.ndarray
    trigger_samples: np.ndarray
    supervisor: RRSupervisor
    metrics: Optional[Dict[str, object]] = None


def default_detector_config() -> FastCausalThresholdConfig:
    return FastCausalThresholdConfig(calibration_s=2.0, detector_refractory_ms=90.0)


def default_supervisor_config() -> SupervisorConfig:
    return SupervisorConfig(
        calibration_start_s=2.0,
        calibration_rr_count=10,
        ema_warmup_count=4,
        rr_min_ms=250.0,
        rr_max_ms=2500.0,
        rr_ema_alpha=0.20,
        default_confidence_frac=0.40,
        min_confidence_frac=0.10,
        max_confidence_frac=0.40,
        adaptive_band_history_len=10,
        adaptive_band_mad_scale=3.0,
        unstable_limit=5,
        blanking_fraction=0.50,
        min_blanking_ms=150.0,
        hard_refractory_ms=200.0,
        recovery_low_frac=0.50,
        recovery_high_frac=1.80,
        recovery_needed_count=2,
    )


def layer2_supervisor_config() -> SupervisorConfig:
    """Shorter calibration used by Layer 2 window validation."""
    return SupervisorConfig(
        rr_min_ms=200.0,
        rr_max_ms=2500.0,
        calibration_start_s=1.0,
        calibration_rr_count=5,
        ema_warmup_count=4,
        rr_ema_alpha=0.20,
        default_confidence_frac=0.40,
        min_confidence_frac=0.10,
        max_confidence_frac=0.40,
        adaptive_band_history_len=10,
        adaptive_band_mad_scale=3.0,
        unstable_limit=5,
        blanking_fraction=0.50,
        min_blanking_ms=150.0,
        hard_refractory_ms=200.0,
        recovery_low_frac=0.50,
        recovery_high_frac=1.80,
        recovery_needed_count=2,
    )


def filter_ecg(
    raw: np.ndarray,
    fs: float,
    mode: str = "zero_phase",
    low: float = 5.0,
    high: float = 20.0,
) -> np.ndarray:
    """Bandpass + notch. Use zero_phase offline, causal for deployment."""
    x = np.asarray(raw, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    nyq = 0.5 * fs
    lo = max(1e-4, low / nyq)
    hi = min(0.499, min(high, 0.45 * fs) / nyq)
    b, a = butter(4, [lo, hi], btype="band")

    if mode == "causal":
        filt = lfilter(b, a, x)
    else:
        filt = filtfilt(b, a, x)

    for f0 in (50.0, 60.0):
        if fs <= (110 if f0 == 50.0 else 125):
            continue
        if f0 >= 0.5 * fs:
            continue
        bn, an = iirnotch(f0 / nyq, 30.0)
        filt = lfilter(bn, an, filt) if mode == "causal" else filtfilt(bn, an, filt)
    return filt


def detect_candidates(
    filt: np.ndarray,
    fs: float,
    cfg: Optional[FastCausalThresholdConfig] = None,
) -> FastCausalThresholdResult:
    return fast_causal_threshold_detect(filt, fs, cfg or default_detector_config())


def run_supervisor(
    candidate_samples: np.ndarray,
    fs: float,
    cfg: Optional[SupervisorConfig] = None,
) -> RRSupervisor:
    sup = RRSupervisor(cfg or default_supervisor_config())
    for sample in np.asarray(candidate_samples, dtype=int):
        sup.process_candidate(int(sample), fs)
    return sup


def run_layer1(
    signal: np.ndarray,
    fs: float,
    *,
    already_filtered: bool = False,
    filter_mode: str = "zero_phase",
    detector_cfg: Optional[FastCausalThresholdConfig] = None,
    supervisor_cfg: Optional[SupervisorConfig] = None,
    use_supervisor: bool = True,
    ref_samples: Optional[np.ndarray] = None,
    match_tol_ms: float = 100.0,
) -> Layer1Result:
    """Run the full Layer 1 path on one ECG trace."""
    raw = np.asarray(signal, dtype=float)
    filt = raw if already_filtered else filter_ecg(raw, fs, mode=filter_mode)

    detector = detect_candidates(filt, fs, detector_cfg)
    candidates = np.asarray(detector.peak_samples, dtype=int)

    if use_supervisor:
        sup = run_supervisor(candidates, fs, supervisor_cfg)
        accepted = np.asarray(sup.state.accepted_samples, dtype=int)
        triggers = np.asarray(sup.state.trigger_samples, dtype=int)
    else:
        sup = RRSupervisor(supervisor_cfg or default_supervisor_config())
        accepted = candidates
        triggers = candidates

    metrics = None
    if ref_samples is not None and len(ref_samples) > 0:
        metrics = greedy_match(
            np.asarray(ref_samples, dtype=int),
            accepted,
            fs,
            tol_ms=match_tol_ms,
        )

    return Layer1Result(
        raw=raw,
        filt=filt,
        fs=float(fs),
        detector=detector,
        candidate_samples=candidates,
        accepted_samples=accepted,
        trigger_samples=triggers,
        supervisor=sup,
        metrics=metrics,
    )


def layer1_r_peaks(filt: np.ndarray, fs: float) -> np.ndarray:
    """Supervisor-accepted R-peak timestamps in seconds."""
    result = run_layer1(
        filt,
        fs,
        already_filtered=True,
        supervisor_cfg=layer2_supervisor_config(),
    )
    return result.accepted_samples.astype(float) / float(fs)


def layer1_detector_peaks(filt: np.ndarray, fs: float) -> np.ndarray:
    """Detector-only R-peak timestamps in seconds (no RR supervisor)."""
    detector = detect_candidates(filt, fs)
    return detector.peak_samples.astype(float) / float(fs)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for running the Layer 1 pipeline directly."""
    parser = argparse.ArgumentParser(
        description="Run Layer 1 detection + RR supervisor on WFDB records.",
    )
    parser.add_argument("--record", type=str, default="100", help="Record stem, e.g. 100")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder containing WFDB .hea/.dat/.atr files. Runs every .hea file.",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--artifact",
        action="store_true",
        help="Enable synthetic stimulation artifact injection",
    )
    parser.add_argument(
        "--artifact-source",
        type=str,
        default="baseline_triggers",
        choices=["baseline_triggers", "reference_beats"],
    )
    parser.add_argument("--artifact-delay-ms", type=float, default=40.0)
    parser.add_argument("--artifact-amp", type=float, default=1.0)
    parser.add_argument("--artifact-spike-ms", type=float, default=2.0)
    parser.add_argument("--artifact-decay-ms", type=float, default=18.0)
    parser.add_argument("--artifact-ring-freq", type=float, default=180.0)
    parser.add_argument("--artifact-window-ms", type=float, default=200.0)
    return parser


def run_record_command(args: argparse.Namespace) -> None:
    """
    Run Layer 1 from command-line arguments.

    Imports stay inside this function so importing main_pipeline.py remains cheap
    and does not create circular imports with evaluate_record.py.
    """
    from artifact_simulation import ArtifactConfig
    from evaluate_record import evaluate_record

    artifact_cfg = ArtifactConfig(
        enabled=args.artifact,
        source=args.artifact_source,
        stim_delay_ms=args.artifact_delay_ms,
        amp_mv=args.artifact_amp,
        spike_ms=args.artifact_spike_ms,
        decay_ms=args.artifact_decay_ms,
        ring_freq_hz=args.artifact_ring_freq,
        contamination_window_ms=args.artifact_window_ms,
    )

    if args.folder is not None:
        stems = sorted(path[:-4] for path in glob.glob(f"{args.folder}/*.hea"))
        if not stems:
            raise FileNotFoundError(f"No .hea files found in {args.folder}")
        for stem in stems:
            evaluate_record(
                record_stem=stem,
                channel=args.channel,
                show_plot=False,
                artifact_cfg=artifact_cfg,
            )
        return

    evaluate_record(
        record_stem=args.record,
        channel=args.channel,
        show_plot=not args.no_plot,
        artifact_cfg=artifact_cfg,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point: python Layer1/pipeline/main_pipeline.py --record 100."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_record_command(args)


if __name__ == "__main__":
    main()
