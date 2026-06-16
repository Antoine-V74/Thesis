"""
Layer 2 main pipeline entry point.

Workflow:
    ECG window + R-peaks (recommended at runtime)
        -> extract_layer2_features()
        -> calibrate_layer2()   # session start, on healthy windows
        -> decide_layer2()      # runtime permit/inhibit

R-peaks are optional in the API so signal-only features (wavelets, entropy,
SQI, amplitude) still work when peak detection fails. Without R-peaks, rr__
and morph__ features are omitted and rhythm/morphology gates cannot fire.
For normal stimulation safety, always pass accepted R-peak timestamps.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from decision import BaselineCalibrator
from full_features import full_features


FeatureDict = Dict[str, float]
DecisionDict = Dict[str, object]


def extract_layer2_features(
    window: np.ndarray,
    fs: float,
    r_peaks_s: Optional[np.ndarray] = None,
    species: str = "human",
    compute_spectral_hrv: bool = True,
    compute_entropy: bool = True,
    focus_peak_s: Optional[float] = None,
) -> Tuple[FeatureDict, Dict[str, str]]:
    """Compute the full Layer 2 feature vector for one ECG window."""
    return full_features(
        window=window,
        fs=fs,
        r_peaks_s=r_peaks_s,
        species=species,
        compute_spectral_hrv=compute_spectral_hrv,
        compute_entropy=compute_entropy,
        focus_peak_s=focus_peak_s,
    )


def calibrate_layer2(
    baseline_features: List[FeatureDict],
    feature_names: Optional[Sequence[str]] = None,
    threshold_quantile: float = 0.999,
    feature_set: str = "all",
) -> BaselineCalibrator:
    """
    Learn the healthy baseline from feature dictionaries.

    Run once at session start on stable healthy ECG windows.
    """
    calibrator = BaselineCalibrator()
    return calibrator.fit(
        baseline_features=baseline_features,
        feature_names=feature_names,
        threshold_quantile=threshold_quantile,
        feature_set=feature_set,
    )


def decide_layer2(
    window: np.ndarray,
    fs: float,
    calibrator: BaselineCalibrator,
    r_peaks_s: Optional[np.ndarray] = None,
    species: str = "human",
    compute_spectral_hrv: bool = True,
    compute_entropy: bool = True,
    focus_peak_s: Optional[float] = None,
) -> Tuple[DecisionDict, FeatureDict]:
    """
    Run the Layer 2 safety decision for one window.

    Returns the permit/inhibit decision and the feature vector used.
    """
    features, _groups = extract_layer2_features(
        window=window,
        fs=fs,
        r_peaks_s=r_peaks_s,
        species=species,
        compute_spectral_hrv=compute_spectral_hrv,
        compute_entropy=compute_entropy,
        focus_peak_s=focus_peak_s,
    )
    decision = calibrator.decide(features)
    return decision, features


__all__ = [
    "BaselineCalibrator",
    "DecisionDict",
    "FeatureDict",
    "calibrate_layer2",
    "decide_layer2",
    "extract_layer2_features",
]
