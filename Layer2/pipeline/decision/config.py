"""
Layer 2 decision policy constants.

Contents:
    DEFAULT_HARD_RULES
        Fixed permit/inhibit limits checked before Mahalanobis.
        Example: too few beats, too much noise, low template correlation.

    DEFAULT_MAHAL_EXCLUDE
        Features removed from Mahalanobis because they are hard rules or too
        variable within healthy data.

    FROZEN_COUPLING_THRESHOLD / FROZEN_ZSCORE_QUANTILE
        Thresholds fixed by benchmark tuning; do not change casually.

    RRReliabilityConfig / check_rr_reliability()
        Settings for decide_hybrid(): is the current R-peak valid, and is the
        RR look-back history clean enough to trust rhythm features?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_MAHAL_EXCLUDE = [
    "rr__rr_count",
    "rr__short_rr_fraction",
    "rr__long_rr_fraction",
    "rr__beat_coupling_ratio",
    "morph__template_corr",
    "morph__neighbor_corr",
    "morph__qrs_width_ms",
    "morph__beat_amp",
    "morph__amp_vs_median",
    "morph__post_pre_area_ratio",
    "signal__hf_noise_ratio",
    "signal__lf_wander_ratio",
    "signal__raw_hf_noise_ratio",
    # Opt-in SQI ensemble and onset/stability discriminators are handled as
    # hard rules (see below), so they are excluded from Mahalanobis. These
    # entries are inert until the matching features are actually computed
    # (compute_sqi_ensemble / compute_onset_stability), because fit() drops
    # exclude names that are absent from the feature vector.
    "signal__ksqi",
    "signal__psqi",
    "signal__bsqi",
    "rr__onset_accel_frac",
    "rr__stability_ms",
    "rr__tachy_fraction",
]

NON_DROPPABLE_HARD_RULES = frozenset({
    "rr__beat_coupling_ratio",
})

FROZEN_COUPLING_THRESHOLD: float = 0.80

DEFAULT_HARD_RULES: Dict[str, List[Optional[float]]] = {
    "rr__rr_count":          [3,    None],
    "rr__short_rr_fraction": [None, 0.5],
    "rr__long_rr_fraction":  [None, 0.5],
    "rr__beat_coupling_ratio": [FROZEN_COUPLING_THRESHOLD, None],
    "morph__template_corr":  [0.55, None],
    "morph__neighbor_corr":  [0.50, None],
    "signal__hf_noise_ratio":  [None, 0.35],
    "signal__lf_wander_ratio": [None, 0.65],
    "signal__raw_hf_noise_ratio": [None, 0.15],
}

FEATURE_SET_CHOICES = ("all", "signal_only", "hybrid_rewarming")

FROZEN_ZSCORE_QUANTILE: float = 0.90


# ---------------------------------------------------------------------------
# Opt-in artifact / discriminator hard rules (NOT part of the frozen gate)
# ---------------------------------------------------------------------------
#
# These rules only fire when the matching opt-in features are computed
# (compute_sqi_ensemble / compute_onset_stability). They are provided as a
# separate dict so the frozen DEFAULT_HARD_RULES above stays byte-for-byte
# unchanged and existing benchmarks are unaffected. To use them, merge into the
# calibrator's hard rules via hard_rules_with_extensions() and calibrate with
# the ensemble features present.
#
# IMPORTANT: the numeric limits below are LITERATURE-INSPIRED PLACEHOLDERS
# (Clifford et al. 2012; Li et al. 2008; ICD Onset/Stability defaults). They
# MUST be recalibrated on the target setup (human dev-set first, then rat/pig)
# before any deployment claim. Direction is [lo, hi]: inhibit if value < lo or
# value > hi.

# SQI ensemble: inhibit on poor signal quality (mirrors an AED "noisy ECG,
# analysis suspended" state).
SQI_ENSEMBLE_HARD_RULES: Dict[str, List[Optional[float]]] = {
    "signal__ksqi": [4.0, None],   # low kurtosis -> flat / noisy, not ECG-like
    "signal__psqi": [0.40, None],  # too little QRS-band power -> broadband noise
    "signal__bsqi": [0.80, None],  # two detectors disagree -> untrustworthy beats
}

# Onset & stability: these are DISCRIMINATORS, not standalone vetoes, so they
# are intentionally NOT expressed as blanket hard rules. Rate-zone branching
# (below) decides when they matter. Kept here as documented reference limits
# for the analysis scripts and future decision wiring.
ONSET_STABILITY_REFERENCE_LIMITS: Dict[str, List[Optional[float]]] = {
    # In the VT rate zone, a sudden onset (large positive accel) plus a stable
    # (low ms) fast rhythm is the classic monomorphic-VT signature.
    "rr__onset_accel_frac": [None, 0.20],  # >0.20 fractional acceleration = abrupt
    "rr__stability_ms": [None, 50.0],      # <50 ms successive diff = organised/stable
}


def hard_rules_with_extensions(
    include_sqi_ensemble: bool = False,
) -> Dict[str, List[Optional[float]]]:
    """
    Return DEFAULT_HARD_RULES optionally extended with the opt-in SQI ensemble.

    Pass the result as ``hard_rules=`` to BaselineCalibrator.fit() when
    calibrating with the SQI ensemble features present. The frozen
    DEFAULT_HARD_RULES dict itself is never mutated.
    """
    rules: Dict[str, List[Optional[float]]] = {k: list(v) for k, v in DEFAULT_HARD_RULES.items()}
    if include_sqi_ensemble:
        for k, v in SQI_ENSEMBLE_HARD_RULES.items():
            rules[k] = list(v)
    return rules


@dataclass
class RRReliabilityConfig:
    """Thresholds for classifying RR history and current R-peak as reliable."""
    min_beats_window: int = 1
    min_beats_lookback: int = 15
    max_long_rr_frac: float = 0.25
    max_short_rr_frac: float = 0.25
    warm_beats: int = 5


RELIABILITY_DEFAULTS: Dict[str, RRReliabilityConfig] = {
    "human": RRReliabilityConfig(min_beats_window=1, min_beats_lookback=15,
                                 max_long_rr_frac=0.25, max_short_rr_frac=0.25,
                                 warm_beats=5),
    "rat":   RRReliabilityConfig(min_beats_window=1, min_beats_lookback=100,
                                 max_long_rr_frac=0.25, max_short_rr_frac=0.25,
                                 warm_beats=8),
    "pig":   RRReliabilityConfig(min_beats_window=1, min_beats_lookback=20,
                                 max_long_rr_frac=0.25, max_short_rr_frac=0.25,
                                 warm_beats=5),
}


def check_rr_reliability(
    features: Dict[str, float],
    n_beats_in_window: int,
    config: Optional[RRReliabilityConfig] = None,
) -> Tuple[bool, bool]:
    """Return (current_r_reliable, rr_history_reliable)."""
    if config is None:
        config = RRReliabilityConfig()

    current_r_reliable = n_beats_in_window >= config.min_beats_window

    rr_count = features.get("rr__rr_count", float("nan"))
    long_frac = features.get("rr__long_rr_fraction", float("nan"))
    short_frac = features.get("rr__short_rr_fraction", float("nan"))

    rr_history_reliable = bool(
        np.isfinite(rr_count) and rr_count >= config.min_beats_lookback
        and np.isfinite(long_frac) and long_frac < config.max_long_rr_frac
        and np.isfinite(short_frac) and short_frac < config.max_short_rr_frac
    )
    return current_r_reliable, rr_history_reliable
