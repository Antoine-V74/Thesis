"""
Layer 2 main pipeline entry point.

Workflow:
    ECG window + R-peaks (recommended at runtime)
        -> extract_layer2_features()
        -> calibrate_layer2()   # session start, on healthy windows
        -> decide_layer2()      # runtime permit/inhibit

For prospective stimulation, use ProspectiveCadenceGate around decide_layer2():
observe 7 unstimulated beats, then apply that precomputed safety state to beat
8. The default cadence policy requires at least 6 safe observations and a safe
7th observation. Beat 8 only needs R-peak detection at trigger time.

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
from stimulation_cadence import ProspectiveCadenceGate


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
    compute_sqi_ensemble: bool = False,
    bsqi: Optional[float] = None,
    compute_onset_stability: bool = False,
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
        compute_sqi_ensemble=compute_sqi_ensemble,
        bsqi=bsqi,
        compute_onset_stability=compute_onset_stability,
    )


def calibrate_layer2(
    baseline_features: List[FeatureDict],
    feature_names: Optional[Sequence[str]] = None,
    threshold_quantile: float = 0.999,
    feature_set: str = "all",
    threshold_method: str = "conformal",
    conformal_alpha: float = 0.10,
    calibration_outlier_frac: float = 0.0,
    anomaly_model: str = "mahalanobis",
    knn_k: int = 5,
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
        threshold_method=threshold_method,
        conformal_alpha=conformal_alpha,
        calibration_outlier_frac=calibration_outlier_frac,
        anomaly_model=anomaly_model,
        knn_k=knn_k,
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
    compute_sqi_ensemble: bool = False,
    bsqi: Optional[float] = None,
    compute_onset_stability: bool = False,
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
        compute_sqi_ensemble=compute_sqi_ensemble,
        bsqi=bsqi,
        compute_onset_stability=compute_onset_stability,
    )
    decision = calibrator.decide(features)
    return decision, features


__all__ = [
    "BaselineCalibrator",
    "DecisionDict",
    "FeatureDict",
    "ProspectiveCadenceGate",
    "calibrate_layer2",
    "decide_layer2",
    "extract_layer2_features",
]
