"""
Layer 2 decision package.

Three modules:
    config.py      hard rules, frozen thresholds, RR reliability
    calibration.py baseline learning and persistence
    gate.py        runtime permit/inhibit decisions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .calibration import CalibrationMixin, features_to_vector
from .config import (
    DEFAULT_HARD_RULES,
    DEFAULT_MAHAL_EXCLUDE,
    FEATURE_SET_CHOICES,
    FROZEN_COUPLING_THRESHOLD,
    FROZEN_ZSCORE_QUANTILE,
    NON_DROPPABLE_HARD_RULES,
    RELIABILITY_DEFAULTS,
    RRReliabilityConfig,
    check_rr_reliability,
)
from .gate import GateMixin


@dataclass
class BaselineCalibrator(CalibrationMixin, GateMixin):
    """
    Per-animal baseline model and Layer 2 safety gate.

    Calibration (session start):
        fit healthy windows -> store baseline statistics and thresholds

    Runtime:
        score current features -> decide permit/inhibit
    """
    feature_names: List[str] = field(default_factory=list)
    mahal_feature_names: List[str] = field(default_factory=list)
    hard_rules: Dict[str, List] = field(default_factory=dict)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inv_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    diag_inv_cov: np.ndarray = field(default_factory=lambda: np.zeros(0))
    threshold_mahalanobis: float = 0.0
    threshold_max_zscore: float = 0.0
    threshold_signal_proxy: float = 0.0
    threshold_rr_proxy: float = 0.0
    threshold_quantile: float = 0.999
    use_diagonal: bool = False
    use_robust: bool = True
    use_shrinkage: bool = True
    val_frac: float = 0.3
    n_baseline_windows: int = 0
    n_calibration_windows: int = 0
    n_validation_windows: int = 0
    feature_set: str = "all"


__all__ = [
    "BaselineCalibrator",
    "DEFAULT_HARD_RULES",
    "DEFAULT_MAHAL_EXCLUDE",
    "FEATURE_SET_CHOICES",
    "FROZEN_COUPLING_THRESHOLD",
    "FROZEN_ZSCORE_QUANTILE",
    "NON_DROPPABLE_HARD_RULES",
    "RELIABILITY_DEFAULTS",
    "RRReliabilityConfig",
    "check_rr_reliability",
    "features_to_vector",
]
