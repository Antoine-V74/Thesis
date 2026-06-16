"""
Record-level evaluation and diagnostic plotting for Layer 1.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from artifact_simulation import ArtifactConfig, inject_artifacts
from main_pipeline import run_layer1
from plot_helpers import plot_record
from reference_annotations import get_reference_beats
from rhythm_supervisor import RRSupervisor


def choose_even_windows(
    total_duration_s: float,
    win_s: float = 20.0,
    n_windows: int = 6,
) -> List[Tuple[float, float]]:
    if total_duration_s <= win_s:
        return [(0.0, total_duration_s)]
    centers = np.linspace(win_s / 2, total_duration_s - win_s / 2, n_windows)
    windows = []
    for center in centers:
        t0 = max(0.0, center - win_s / 2)
        t1 = min(total_duration_s, center + win_s / 2)
        windows.append((t0, t1))
    return windows


def choose_decision_windows(
    supervisor: RRSupervisor,
    total_duration_s: float,
    win_s: float = 20.0,
    max_windows: int = 6,
) -> List[Tuple[float, float]]:
    interesting = []
    for decision in supervisor.state.decisions:
        if decision.decision in {
            "reject_short", "reject_long",
            "reject_out_of_band_low", "reject_out_of_band_high",
            "enter_recovery", "recovery_reject_short",
            "recovery_reanchor_long", "recovery_reject_band",
            "recovery_to_calibration", "skip_refractory",
            "skip_post_stim_protection",
        }:
            interesting.append(decision.t_ms / 1000.0)

    chosen = []
    for center in interesting:
        if all(abs(center - x) > 0.7 * win_s for x in chosen):
            chosen.append(center)
        if len(chosen) >= max_windows:
            break

    return [
        (max(0.0, c - win_s / 2), min(total_duration_s, c + win_s / 2))
        for c in chosen
    ]


def _result_dict(result) -> Dict[str, object]:
    return {
        "raw": result.raw,
        "filt": result.filt,
        "candidate_samples": result.candidate_samples,
        "accepted_samples": result.accepted_samples,
        "trigger_samples": result.trigger_samples,
        "supervisor": result.supervisor,
        "metrics": result.metrics or {},
    }


def evaluate_record(
    record_stem: str,
    channel: int = 0,
    show_plot: bool = True,
    artifact_cfg: Optional[ArtifactConfig] = None,
    match_tol_ms: float = 50.0,
) -> None:
    """Evaluate one MIT-BIH record with the fast causal Layer 1 pipeline."""
    import wfdb

    record = wfdb.rdrecord(record_stem)
    ann = wfdb.rdann(record_stem, "atr")
    fs = float(record.fs)
    raw = record.p_signal[:, channel]
    ref_samples, _ = get_reference_beats(ann)

    if artifact_cfg is None:
        artifact_cfg = ArtifactConfig(enabled=False)

    clean_result = run_layer1(raw, fs, ref_samples=ref_samples, match_tol_ms=match_tol_ms)
    clean = _result_dict(clean_result)
    metrics = clean["metrics"]

    print("\n" + "=" * 72)
    print(f"Record: {record_stem}")
    print(f"fs = {fs} Hz | channel = {record.sig_name[channel]}")
    print("Detector: fast_causal_threshold")
    print("CLEAN RUN")
    print(
        f"Candidates = {len(clean['candidate_samples'])} | "
        f"accepted = {len(clean['accepted_samples'])} | "
        f"triggers = {len(clean['trigger_samples'])}"
    )
    if metrics:
        print(
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | "
            f"Se={metrics['sensitivity']:.3f} PPV={metrics['ppv']:.3f} "
            f"F1={metrics['f1']:.3f} | jitter={metrics['mean_abs_jitter_ms']:.2f} ms"
        )

    clean_state = clean["supervisor"].state
    print("-" * 72)
    print(f"recovery_entries = {clean_state.n_recovery_entries}")
    print(f"recalibrations   = {clean_state.n_recalibrations}")

    artifact = None
    if artifact_cfg.enabled:
        base_trigger_samples = (
            ref_samples if artifact_cfg.source == "reference_beats"
            else clean["trigger_samples"]
        )
        artifacted_raw, artifact_start_samples, _ = inject_artifacts(
            raw_ecg=raw,
            trigger_samples=base_trigger_samples,
            fs=fs,
            cfg=artifact_cfg,
        )
        artifact_result = run_layer1(
            artifacted_raw, fs, ref_samples=ref_samples, match_tol_ms=match_tol_ms,
        )
        artifact = _result_dict(artifact_result)
        print("-" * 72)
        print("ARTIFACT RUN")
        am = artifact["metrics"]
        if am:
            print(
                f"TP={am['tp']} FP={am['fp']} FN={am['fn']} | "
                f"Se={am['sensitivity']:.3f} PPV={am['ppv']:.3f}"
            )

    print("=" * 72)

    if show_plot:
        total_duration_s = len(raw) / fs
        plot_record(
            record_name=record_stem + " | CLEAN",
            raw=clean["raw"], filt=clean["filt"], fs=fs,
            ref_samples=ref_samples,
            candidate_samples=clean["candidate_samples"],
            supervisor=clean["supervisor"],
            t0_s=0.0, t1_s=min(20.0, total_duration_s),
        )
        for index, (t0, t1) in enumerate(
            choose_decision_windows(clean["supervisor"], total_duration_s), start=1,
        ):
            plot_record(
                record_name=record_stem + f" | difficult {index}",
                raw=clean["raw"], filt=clean["filt"], fs=fs,
                ref_samples=ref_samples,
                candidate_samples=clean["candidate_samples"],
                supervisor=clean["supervisor"],
                t0_s=t0, t1_s=t1,
            )
        if artifact is not None:
            plot_record(
                record_name=record_stem + " | ARTIFACT",
                raw=artifact["raw"], filt=artifact["filt"], fs=fs,
                ref_samples=ref_samples,
                candidate_samples=artifact["candidate_samples"],
                supervisor=artifact["supervisor"],
                t0_s=0.0, t1_s=min(20.0, total_duration_s),
            )
