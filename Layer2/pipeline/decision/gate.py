"""
Layer 2 runtime gate: score windows against baseline and return permit/inhibit.

Decision order in decide():
    1. hard rules (fixed absolute limits from config.py)
    2. Mahalanobis distance vs learned baseline
    3. max z-score vs learned threshold

decide_hybrid() adds RR reliability checks for beat-sync deployment:
    signal check -> current R-peak present -> RR history trusted or rewarming.

See pipeline/README.md for decide() vs decide_hybrid().
"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .calibration import features_to_vector

if TYPE_CHECKING:
    from . import BaselineCalibrator


class GateMixin:
    """Runtime safety decision logic."""

    def _decision_feature_indices(self: BaselineCalibrator) -> List[int]:
        """Indices used for max-zscore thresholding in decide()."""
        if self.feature_set in ("signal_only",):
            return [self.feature_names.index(f) for f in self.mahal_feature_names]
        return list(range(len(self.feature_names)))

    def _decision_feature_names(self: BaselineCalibrator) -> List[str]:
        """Feature names ranked for top-deviator logging."""
        if self.feature_set in ("signal_only",):
            return list(self.mahal_feature_names)
        return list(self.feature_names)

    def score(self: BaselineCalibrator, features: Dict[str, float]) -> Dict[str, float]:
        """Score one window against the fitted baseline."""
        x_all = features_to_vector(features, self.feature_names)
        z_all = (x_all - self.mean) / self.std

        mahal_idx = [self.feature_names.index(f) for f in self.mahal_feature_names]
        x_m = x_all[mahal_idx]
        mahal_val = self._mahalanobis_batch_m(x_m[None, :])[0]

        sig_idx = [i for i, n in enumerate(self.feature_names) if n.startswith("signal__")]
        rr_idx = [i for i, n in enumerate(self.feature_names) if n.startswith("rr__")]
        sig_proxy = float(np.sqrt(np.sum(z_all[sig_idx] ** 2))) if sig_idx else float("nan")
        rr_proxy = float(np.sqrt(np.sum(z_all[rr_idx] ** 2))) if rr_idx else float("nan")
        z_idx = self._decision_feature_indices()

        out: Dict[str, float] = {
            "mahalanobis": float(mahal_val),
            "signal_mahal_proxy": sig_proxy,
            "rr_mahal_proxy": rr_proxy,
            "max_abs_zscore": float(np.max(np.abs(z_all[z_idx]))),
        }
        for name, zi in zip(self.feature_names, z_all):
            out[f"zscore_{name}"] = float(zi)
        return out

    def check_hard_rules(
        self: BaselineCalibrator,
        features: Dict[str, float],
    ) -> Optional[str]:
        """Return the violated hard-rule name, or None if all pass."""
        for feat, (lo, hi) in self.hard_rules.items():
            val = features.get(feat, float("nan"))
            if not np.isfinite(val):
                continue
            if lo is not None and val < lo:
                return f"{feat}<{lo}"
            if hi is not None and val > hi:
                return f"{feat}>{hi}"
        return None

    def decide(self: BaselineCalibrator, features: Dict[str, float]) -> Dict[str, object]:
        """Standard permit/inhibit decision with full reasoning."""
        hard_violated = self.check_hard_rules(features)
        if hard_violated:
            s = self.score(features)
            z_items = [(n, s[f"zscore_{n}"]) for n in self._decision_feature_names()]
            z_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return {
                "permit": False,
                "inhibit": True,
                "reason": "hard_rule",
                "hard_rule_violated": hard_violated,
                "mahalanobis": s["mahalanobis"],
                "mahalanobis_threshold": self.threshold_mahalanobis,
                "signal_mahal_proxy": s["signal_mahal_proxy"],
                "rr_mahal_proxy": s["rr_mahal_proxy"],
                "max_abs_zscore": s["max_abs_zscore"],
                "zscore_threshold": self.threshold_max_zscore,
                "top_deviating_features": [
                    {"feature": n, "zscore": z} for n, z in z_items[:3]
                ],
            }

        s = self.score(features)
        inhibit_m = s["mahalanobis"] > self.threshold_mahalanobis
        inhibit_z = s["max_abs_zscore"] > self.threshold_max_zscore
        inhibit = bool(inhibit_m or inhibit_z)

        z_items = [(n, s[f"zscore_{n}"]) for n in self._decision_feature_names()]
        z_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_dev = [{"feature": n, "zscore": z} for n, z in z_items[:3]]

        return {
            "permit": not inhibit,
            "inhibit": inhibit,
            "reason": (
                "mahalanobis_exceeded" if inhibit_m
                else "max_zscore_exceeded" if inhibit_z
                else "within_baseline"
            ),
            "hard_rule_violated": None,
            "mahalanobis": s["mahalanobis"],
            "mahalanobis_threshold": self.threshold_mahalanobis,
            "signal_mahal_proxy": s["signal_mahal_proxy"],
            "rr_mahal_proxy": s["rr_mahal_proxy"],
            "max_abs_zscore": s["max_abs_zscore"],
            "zscore_threshold": self.threshold_max_zscore,
            "top_deviating_features": top_dev,
        }

    def decide_hybrid(
        self: BaselineCalibrator,
        features: Dict[str, float],
        current_r_reliable: bool,
        rr_history_reliable: bool,
        n_recent_clean_beats: int = 0,
        warm_beats: int = 5,
    ) -> Dict[str, object]:
        """Reliability-aware permit/inhibit decision (rewarming mode)."""
        s = self.score(features)
        z_items = [(n, s[f"zscore_{n}"]) for n in self._decision_feature_names()]
        z_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_dev = [{"feature": n, "zscore": z} for n, z in z_items[:3]]

        base: Dict[str, object] = {
            "mahalanobis": s["mahalanobis"],
            "mahalanobis_threshold": self.threshold_mahalanobis,
            "signal_mahal_proxy": s["signal_mahal_proxy"],
            "rr_mahal_proxy": s["rr_mahal_proxy"],
            "max_abs_zscore": s["max_abs_zscore"],
            "zscore_threshold": self.threshold_max_zscore,
            "hard_rule_violated": None,
            "current_r_reliable": current_r_reliable,
            "rr_history_reliable": rr_history_reliable,
            "n_recent_clean_beats": n_recent_clean_beats,
            "top_deviating_features": top_dev,
        }

        hard_violated = self.check_hard_rules(features)
        if hard_violated:
            return {
                "permit": False,
                "inhibit": True,
                "reason": "hard_rule",
                "hard_rule_violated": hard_violated,
                **{k: v for k, v in base.items() if k != "hard_rule_violated"},
            }

        signal_safe = s["signal_mahal_proxy"] <= self.threshold_signal_proxy
        if not signal_safe:
            return {"permit": False, "inhibit": True, "reason": "signal_not_safe", **base}

        if not current_r_reliable:
            return {
                "permit": False,
                "inhibit": True,
                "reason": "current_r_unreliable",
                **base,
            }

        if rr_history_reliable:
            rr_safe = s["mahalanobis"] <= self.threshold_mahalanobis
            if rr_safe:
                return {"permit": True, "inhibit": False, "reason": "all_safe", **base}
            return {"permit": False, "inhibit": True, "reason": "rr_abnormal", **base}

        if n_recent_clean_beats >= warm_beats:
            return {
                "permit": True,
                "inhibit": False,
                "reason": "rr_history_rewarming_permit",
                **base,
            }
        return {
            "permit": False,
            "inhibit": True,
            "reason": "rr_history_recovery_inhibit",
            **base,
        }
