"""
Layer 2 threshold-selection utilities.

Mirrors the conformal-prediction helpers used by Layer 3
(`Layer3/validation/layer3_validation_utils.py`) so both layers share the
same safety-conservative thresholding logic and vocabulary:

    quantile   legacy behaviour: threshold = quantile(healthy_val_scores, q)
    conformal  split-conformal upper-tail threshold with a stated healthy
               false-inhibit budget `alpha` (distribution-free, exchangeability
               only)

Safety note: `select_decision_threshold` never silently relaxes an infeasible
alpha. Callers MUST fail-safe inhibit (never permit) when status != "ok".
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

THRESHOLD_METHOD_CHOICES = ("quantile", "conformal")
ANOMALY_MODEL_CHOICES = ("mahalanobis", "knn")
DEFAULT_CONFORMAL_ALPHA = 0.10


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (default z = 95% CI)."""
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total)) / denom
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def conformal_threshold_from_scores(scores: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Upper-tail split-conformal threshold for healthy calibration scores.

    The finite-sample guarantee is only for the healthy false-inhibit rate
    under exchangeability. It is NOT a false-permit guarantee on abnormal
    beats/windows. If alpha is infeasible for the available sample size,
    callers must fail-safe inhibit rather than silently relaxing alpha.
    """
    s = np.asarray(scores, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    n = int(len(s))
    alpha = float(alpha)
    if n <= 0:
        return {
            "threshold": float("nan"), "status": "no_healthy_calibration_scores",
            "alpha": alpha, "n": n, "rank": 0, "alpha_min": float("nan"),
        }
    if not (0.0 < alpha < 1.0):
        return {
            "threshold": float("nan"), "status": "invalid_alpha",
            "alpha": alpha, "n": n, "rank": 0, "alpha_min": float(1.0 / (n + 1)),
        }
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    alpha_min = float(1.0 / (n + 1))
    if rank > n:
        return {
            "threshold": float("nan"), "status": "alpha_infeasible",
            "alpha": alpha, "n": n, "rank": rank, "alpha_min": alpha_min,
        }
    return {
        "threshold": float(s[rank - 1]), "status": "ok",
        "alpha": alpha, "n": n, "rank": rank, "alpha_min": alpha_min,
    }


def select_decision_threshold(
    val_scores: np.ndarray,
    method: str,
    threshold_quantile: float,
    conformal_alpha: float,
) -> Dict[str, Any]:
    """Choose a healthy-only decision threshold for a distance-based gate.

    Returns a dict with keys: threshold, method, status, target_false_inhibit,
    conformal_alpha, conformal_alpha_min, n_val.

    `target_false_inhibit` is `alpha` for conformal and `1 - quantile` for the
    quantile method, so a single coverage report can compare achieved vs
    targeted healthy false-inhibit rate regardless of method.

    Callers MUST fail-safe inhibit when status != "ok" (uncertainty -> inhibit).
    This function never relaxes alpha/quantile silently.
    """
    method = str(method).lower()
    s = np.asarray(val_scores, dtype=float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if method == "conformal":
        info = conformal_threshold_from_scores(s, float(conformal_alpha))
        return {
            "threshold": float(info["threshold"]),
            "method": "conformal",
            "status": str(info["status"]),
            "target_false_inhibit": float(conformal_alpha),
            "conformal_alpha": float(conformal_alpha),
            "conformal_alpha_min": float(info.get("alpha_min", float("nan"))),
            "n_val": n,
        }
    if method == "quantile":
        if n <= 0:
            return {
                "threshold": float("nan"), "method": "quantile",
                "status": "no_healthy_calibration_scores",
                "target_false_inhibit": float(1.0 - float(threshold_quantile)),
                "conformal_alpha": float("nan"), "conformal_alpha_min": float("nan"),
                "n_val": n,
            }
        q = float(np.clip(threshold_quantile, 0.0, 1.0))
        return {
            "threshold": float(np.quantile(s, q)),
            "method": "quantile",
            "status": "ok",
            "target_false_inhibit": float(1.0 - q),
            "conformal_alpha": float("nan"),
            "conformal_alpha_min": float("nan"),
            "n_val": n,
        }
    raise ValueError(f"Unsupported threshold method: {method!r} (use {THRESHOLD_METHOD_CHOICES})")
