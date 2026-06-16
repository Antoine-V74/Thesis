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
