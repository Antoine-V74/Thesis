
"""
Synthetic stimulation-artifact generation utilities.

These are used to stress-test the detector/supervisor pipeline by injecting
artifact waveforms after known trigger times.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ArtifactConfig:
    """
    Configuration for synthetic stimulation artifact injection.
    """
    enabled: bool = False
    source: str = "baseline_triggers"  # "baseline_triggers" or "reference_beats"
    stim_delay_ms: float = 40.0
    amp_mv: float = 1.0
    spike_ms: float = 2.0
    decay_ms: float = 18.0
    ring_freq_hz: float = 180.0
    ring_frac: float = 0.35
    dc_tail_frac: float = 0.15
    polarity: float = 1.0
    contamination_window_ms: float = 200.0


def synthesize_artifact_waveform(fs: float, cfg: ArtifactConfig) -> np.ndarray:
    """
    Build a damped artifact waveform with:
    - a narrow initial spike
    - a decaying ringing term
    - a small decaying DC tail

    The waveform is expressed in the same units as the ECG signal (mV in the
    current MIT-BIH workflow).
    """
    total_ms = max(20.0, cfg.spike_ms + 5.0 * cfg.decay_ms)
    n = max(int(round(total_ms * fs / 1000.0)), 3)
    t = np.arange(n) / fs

    spike_sigma_s = max(cfg.spike_ms / 1000.0 / 3.0, 1e-4)
    spike = cfg.amp_mv * np.exp(-0.5 * (t / spike_sigma_s) ** 2)

    ring = (
        cfg.amp_mv
        * cfg.ring_frac
        * np.exp(-t / max(cfg.decay_ms / 1000.0, 1e-4))
        * np.sin(2.0 * np.pi * cfg.ring_freq_hz * t)
    )

    tail = (
        cfg.amp_mv
        * cfg.dc_tail_frac
        * np.exp(-t / max(1.8 * cfg.decay_ms / 1000.0, 1e-4))
    )

    return cfg.polarity * (spike + ring + tail)


def inject_artifacts(
    raw_ecg: np.ndarray,
    trigger_samples: np.ndarray,
    fs: float,
    cfg: ArtifactConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add synthetic artifacts to a raw ECG trace after each trigger sample.

    Returns
    -------
    y:
        Artifact-contaminated signal.
    artifact_start_samples:
        Sample indices where each artifact begins.
    art:
        The artifact waveform that was injected.
    """
    y = raw_ecg.copy()
    trigger_samples = np.asarray(trigger_samples, dtype=int)
    art = synthesize_artifact_waveform(fs, cfg)
    delay_samples = int(round(cfg.stim_delay_ms * fs / 1000.0))

    artifact_start_samples: List[int] = []
    for s in trigger_samples:
        i0 = s + delay_samples
        if i0 < 0 or i0 >= len(y):
            continue

        i1 = min(len(y), i0 + len(art))
        y[i0:i1] += art[: i1 - i0]
        artifact_start_samples.append(i0)

    return y, np.asarray(artifact_start_samples, dtype=int), art


def count_events_near_artifacts(
    event_samples: np.ndarray,
    artifact_samples: np.ndarray,
    fs: float,
    window_ms: float = 200.0,
) -> int:
    """
    Count how many artifacts have at least one event falling shortly after
    them, within a configurable contamination window.

    This is useful to quantify whether artifacts are generating spurious
    detections or accepted beats.
    """
    event_samples = np.asarray(event_samples, dtype=int)
    artifact_samples = np.asarray(artifact_samples, dtype=int)

    if len(event_samples) == 0 or len(artifact_samples) == 0:
        return 0

    win = int(round(window_ms * fs / 1000.0))
    count = 0
    for a in artifact_samples:
        hit = np.any((event_samples >= a) & (event_samples <= a + win))
        count += int(hit)
    return count
