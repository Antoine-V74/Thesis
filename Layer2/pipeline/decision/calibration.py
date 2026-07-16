"""
Layer 2 baseline calibration: fit healthy reference statistics and thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .config import (
    DEFAULT_HARD_RULES,
    DEFAULT_MAHAL_EXCLUDE,
    FEATURE_SET_CHOICES,
    FROZEN_ZSCORE_QUANTILE,
    NON_DROPPABLE_HARD_RULES,
)

if TYPE_CHECKING:
    from . import BaselineCalibrator


def features_to_vector(features: Dict[str, float], names: Sequence[str]) -> np.ndarray:
    """Project a feature dict onto a canonical ordered list of names."""
    return np.array([features[n] for n in names], dtype=float)


def regularized_inv(cov: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Ridge-regularized matrix inverse for numerical stability."""
    d = cov.shape[0]
    return np.linalg.inv(cov + ridge * np.eye(d))


def estimate_covariance(Z: np.ndarray, use_shrinkage: bool, ridge: float) -> np.ndarray:
    """Estimate covariance of z-scored data Z."""
    if use_shrinkage:
        try:
            from sklearn.covariance import LedoitWolf
            return LedoitWolf().fit(Z).covariance_
        except ImportError:
            pass
    return np.cov(Z, rowvar=False)


class CalibrationMixin:
    """Baseline learning: fit statistics, tune thresholds, save/load state."""

    def fit(
        self: BaselineCalibrator,
        baseline_features: List[Dict[str, float]],
        feature_names: Optional[Sequence[str]] = None,
        threshold_quantile: float = 0.999,
        zscore_quantile: float = FROZEN_ZSCORE_QUANTILE,
        ridge: float = 1e-4,
        force_diagonal: bool = False,
        use_robust: bool = True,
        use_shrinkage: bool = True,
        val_frac: float = 0.3,
        mahalanobis_exclude: Optional[List[str]] = None,
        hard_rules: Optional[Dict[str, List[Optional[float]]]] = None,
        use_default_hard_rules: bool = True,
        feature_set: str = "all",
        threshold_method: str = "conformal",
        conformal_alpha: float = 0.10,
        calibration_outlier_frac: float = 0.0,
        anomaly_model: str = "mahalanobis",
        knn_k: int = 5,
    ) -> BaselineCalibrator:
        """Fit baseline statistics on healthy windows."""
        if feature_set not in FEATURE_SET_CHOICES:
            raise ValueError(
                f"feature_set must be one of {FEATURE_SET_CHOICES}, got {feature_set!r}. "
                f"Note: 'hybrid_reliability' was renamed to 'hybrid_rewarming'."
            )
        if len(baseline_features) < 5:
            raise ValueError(
                f"Need at least 5 baseline windows, got {len(baseline_features)}."
            )
        if not 0.0 < val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0,1), got {val_frac}.")

        if feature_names is None:
            feature_names = list(baseline_features[0].keys())
        self.feature_names = list(feature_names)
        self.feature_set = str(feature_set)

        present = set(self.feature_names)
        if feature_set == "signal_only":
            self.hard_rules = {}
            self.mahal_feature_names = [
                f for f in self.feature_names if f.startswith("signal__")
            ]
        elif feature_set in ("all", "hybrid_rewarming"):
            if use_default_hard_rules:
                _exclude = (
                    mahalanobis_exclude
                    if mahalanobis_exclude is not None
                    else DEFAULT_MAHAL_EXCLUDE
                )
                _rules = hard_rules if hard_rules is not None else DEFAULT_HARD_RULES
            else:
                _exclude = mahalanobis_exclude or []
                _rules = hard_rules or {}

            self.hard_rules = {k: list(v) for k, v in _rules.items() if k in present}
            for feat in NON_DROPPABLE_HARD_RULES:
                if feat in _rules:
                    self.hard_rules[feat] = list(_rules[feat])
            excluded = [f for f in _exclude if f in present]
            self.mahal_feature_names = [f for f in self.feature_names if f not in excluded]

        if not self.mahal_feature_names:
            raise ValueError(
                f"No Mahalanobis features for feature_set={feature_set!r}."
            )

        X = np.stack([features_to_vector(f, self.feature_names) for f in baseline_features])
        self.n_baseline_windows = len(X)
        self.use_robust = bool(use_robust)
        self.use_shrinkage = bool(use_shrinkage)
        self.val_frac = float(val_frac)
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_method = str(threshold_method).lower()
        self.conformal_alpha = float(conformal_alpha)
        self.calibration_outlier_frac = float(calibration_outlier_frac)
        self.anomaly_model = str(anomaly_model).lower()
        self.knn_k = int(knn_k)
        self.n_outlier_removed = 0
        self._force_diagonal = bool(force_diagonal)
        self._ridge = float(ridge)

        n_cal = max(5, int(round(len(X) * (1.0 - val_frac))))
        X_cal = X[:n_cal]
        X_val = X[n_cal:] if len(X) > n_cal else X
        self.n_calibration_windows = len(X_cal)
        self.n_validation_windows = len(X_val)

        outlier_frac = float(np.clip(calibration_outlier_frac, 0.0, 0.5))
        if outlier_frac > 0.0 and len(X_cal) > 5:
            self._fit_core_statistics(X_cal)
            mahal_idx = [self.feature_names.index(f) for f in self.mahal_feature_names]
            prov_scores = self._mahalanobis_batch_m(X_cal[:, mahal_idx])
            keep_n = max(5, int(np.ceil((1.0 - outlier_frac) * len(X_cal))))
            keep_n = min(keep_n, len(X_cal))
            order = np.argsort(prov_scores, kind="mergesort")
            keep = np.zeros(len(X_cal), dtype=bool)
            keep[order[:keep_n]] = True
            self.n_outlier_removed = int((~keep).sum())
            X_cal = X_cal[keep]
            self.n_calibration_windows = len(X_cal)

        self._fit_core_statistics(X_cal)

        mahal_idx = [self.feature_names.index(f) for f in self.mahal_feature_names]
        X_m_val = X_val[:, mahal_idx]
        primary_val = self._primary_distance_batch(X_m_val)
        thr_info = self._select_primary_threshold(primary_val)
        self.threshold_mahalanobis = float(thr_info["threshold"])
        self._threshold_status = str(thr_info["status"])
        self._target_false_inhibit = float(thr_info["target_false_inhibit"])

        Z_val_all = (X_val - self.mean[None, :]) / self.std[None, :]
        z_idx = self._decision_feature_indices()
        max_z_val = np.max(np.abs(Z_val_all[:, z_idx]), axis=1)
        self.threshold_max_zscore = float(np.quantile(max_z_val, zscore_quantile))

        sig_idx = [i for i, n in enumerate(self.feature_names) if n.startswith("signal__")]
        rr_idx = [i for i, n in enumerate(self.feature_names) if n.startswith("rr__")]
        Z_sig_val = Z_val_all[:, sig_idx] if sig_idx else np.zeros((len(X_val), 1))
        Z_rr_val = Z_val_all[:, rr_idx] if rr_idx else np.zeros((len(X_val), 1))
        sig_proxy_val = np.sqrt(np.sum(Z_sig_val ** 2, axis=1))
        rr_proxy_val = np.sqrt(np.sum(Z_rr_val ** 2, axis=1))
        self.threshold_signal_proxy = float(np.quantile(sig_proxy_val, threshold_quantile))
        self.threshold_rr_proxy = float(np.quantile(rr_proxy_val, threshold_quantile))

        return self

    def _fit_core_statistics(self: BaselineCalibrator, X_cal: np.ndarray) -> None:
        """Fit location/scale/covariance and optional kNN bank from calibration rows."""
        if self.use_robust:
            center = np.median(X_cal, axis=0)
            mad = np.median(np.abs(X_cal - center[None, :]), axis=0)
            feat_range = np.percentile(X_cal, 95, axis=0) - np.percentile(X_cal, 5, axis=0)
            min_scale = np.maximum(0.005 * feat_range, 1e-4)
            scale = np.maximum(1.4826 * mad, min_scale)
        else:
            center = X_cal.mean(axis=0)
            scale = X_cal.std(axis=0) + 1e-8

        self.mean = center
        self.std = scale

        mahal_idx = [self.feature_names.index(f) for f in self.mahal_feature_names]
        X_m_cal = X_cal[:, mahal_idx]
        Z_m_cal = (X_m_cal - center[mahal_idx][None, :]) / scale[mahal_idx][None, :]
        d_m = len(mahal_idx)
        n_cal = len(X_cal)
        n_required = max(2 * d_m, 30)
        ridge = float(getattr(self, "_ridge", 1e-4))
        force_diagonal = bool(getattr(self, "_force_diagonal", False))

        if force_diagonal or n_cal < n_required or d_m < 2:
            self.use_diagonal = True
            self.diag_inv_cov = 1.0 / (scale[mahal_idx] ** 2)
            self.inv_cov = np.zeros((d_m, d_m))
        else:
            self.use_diagonal = False
            corr = estimate_covariance(Z_m_cal, self.use_shrinkage, ridge)
            inv_corr = regularized_inv(corr, ridge=ridge)
            inv_s = 1.0 / scale[mahal_idx]
            self.inv_cov = inv_s[:, None] * inv_corr * inv_s[None, :]
            self.diag_inv_cov = inv_s ** 2

        if self.anomaly_model == "knn":
            self.knn_calibration_vectors = Z_m_cal.copy()
        else:
            self.knn_calibration_vectors = np.zeros((0, d_m))

    def _select_primary_threshold(self: BaselineCalibrator, val_scores: np.ndarray) -> dict:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[3]
        _l2_val = _root / "Layer2" / "validation"
        if str(_l2_val) not in sys.path:
            sys.path.insert(0, str(_l2_val))
        from layer2_validation_utils import select_decision_threshold  # noqa: WPS433

        info = select_decision_threshold(
            val_scores,
            method=self.threshold_method,
            threshold_quantile=self.threshold_quantile,
            conformal_alpha=self.conformal_alpha,
        )
        if info["status"] != "ok" or not np.isfinite(info["threshold"]):
            info["threshold"] = float("inf")
        return info

    def _knn_distance_batch(self: BaselineCalibrator, X_m: np.ndarray) -> np.ndarray:
        if self.knn_calibration_vectors is None or self.knn_calibration_vectors.size == 0:
            return np.full(len(X_m), np.inf, dtype=float)
        ref = self.knn_calibration_vectors
        k_eff = int(max(1, min(int(self.knn_k), ref.shape[0])))
        diff = X_m[:, None, :] - ref[None, :, :]
        d = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        nearest = np.partition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
        return nearest.mean(axis=1)

    def _primary_distance_batch(self: BaselineCalibrator, X_m: np.ndarray) -> np.ndarray:
        if self.anomaly_model == "knn":
            center_m = self.mean[[self.feature_names.index(f) for f in self.mahal_feature_names]]
            scale_m = self.std[[self.feature_names.index(f) for f in self.mahal_feature_names]]
            z_m = (X_m - center_m[None, :]) / scale_m[None, :]
            return self._knn_distance_batch(z_m)
        return self._mahalanobis_batch_m(X_m)

    def calibrate_thresholds_for_abnormal_inhibit(
        self: BaselineCalibrator,
        abnormal_features: List[Dict[str, float]],
        target_inhibit_rate: float = 0.95,
        healthy_validation_features: Optional[List[Dict[str, float]]] = None,
        max_healthy_false_inhibit_rate: Optional[float] = None,
        tune_zscore: bool = True,
    ) -> Dict[str, float]:
        """Tune decision thresholds using labeled abnormal validation data."""
        if not 0.5 <= target_inhibit_rate <= 1.0:
            raise ValueError("target_inhibit_rate must be in [0.5, 1.0]")

        n_total = 0
        n_hard_caught = 0
        abn_m_remaining: List[float] = []
        abn_z_remaining: List[float] = []

        for f in abnormal_features:
            if not all(np.isfinite(f.get(k, float("nan"))) for k in self.mahal_feature_names):
                continue
            n_total += 1
            if self.check_hard_rules(f) is not None:
                n_hard_caught += 1
                continue
            s = self.score(f)
            abn_m_remaining.append(float(s["mahalanobis"]))
            abn_z_remaining.append(float(s["max_abs_zscore"]))

        if n_total < 5:
            return {"n_abnormal": n_total, "adjusted": False}

        hard_rate = n_hard_caught / n_total

        if hard_rate >= target_inhibit_rate:
            return {
                "n_abnormal": n_total,
                "n_hard_caught": n_hard_caught,
                "adjusted": False,
                "reason": "hard_rules_sufficient",
            }

        if len(abn_m_remaining) < 3:
            return {"n_abnormal": n_total, "adjusted": False}

        n_zscore_caught = sum(
            1 for z in abn_z_remaining if z > self.threshold_max_zscore
        )
        zscore_rate_on_remaining = n_zscore_caught / len(abn_m_remaining)
        already_caught_rate = hard_rate + zscore_rate_on_remaining * (1 - hard_rate)

        remaining_needed = target_inhibit_rate - already_caught_rate
        if remaining_needed <= 0:
            return {
                "n_abnormal": n_total,
                "n_hard_caught": n_hard_caught,
                "adjusted": False,
                "reason": "hard_rules_and_zscore_sufficient",
            }

        abn_m_after_z = [
            m for m, z in zip(abn_m_remaining, abn_z_remaining)
            if z <= self.threshold_max_zscore
        ]
        if not abn_m_after_z:
            return {
                "n_abnormal": n_total,
                "adjusted": False,
                "reason": "no_remaining_beats_for_mahal_tuning",
            }

        frac_of_remaining_to_catch = remaining_needed / (1 - already_caught_rate)
        frac_of_remaining_to_catch = min(frac_of_remaining_to_catch, 0.999)
        q = 1.0 - frac_of_remaining_to_catch
        thr_m = float(np.quantile(abn_m_after_z, q))

        h_mahal: List[float] = []
        h_zscore: List[float] = []
        n_h_total_valid = 0
        n_h_hard_blocked = 0
        if healthy_validation_features:
            for f in healthy_validation_features:
                if not all(np.isfinite(f.get(k, float("nan"))) for k in self.mahal_feature_names):
                    continue
                n_h_total_valid += 1
                if self.check_hard_rules(f) is not None:
                    n_h_hard_blocked += 1
                    continue
                s = self.score(f)
                hz = float(s["max_abs_zscore"])
                hm = float(s["mahalanobis"])
                if hz <= self.threshold_max_zscore:
                    h_mahal.append(hm)
                    h_zscore.append(hz)

        if h_mahal and max_healthy_false_inhibit_rate is not None:
            max_fi = float(max_healthy_false_inhibit_rate)
            budget = max(0, int(max_fi * n_h_total_valid) - n_h_hard_blocked)
            if budget >= len(h_mahal):
                thr_m_floor = -np.inf
            else:
                thr_m_floor = float(np.quantile(h_mahal, 1.0 - budget / len(h_mahal)))
            thr_m = max(thr_m, thr_m_floor)

        self.threshold_mahalanobis = thr_m

        zscore_thr_adjusted = False
        if tune_zscore:
            n_mahal_caught = sum(
                1 for m, z in zip(abn_m_remaining, abn_z_remaining)
                if z <= self.threshold_max_zscore and m > thr_m
            )
            n_blocked_total = n_hard_caught + n_zscore_caught + n_mahal_caught
            achieved_ai = n_blocked_total / n_total

            if achieved_ai < target_inhibit_rate:
                abn_z_still = [
                    z for m, z in zip(abn_m_remaining, abn_z_remaining)
                    if z <= self.threshold_max_zscore and m <= thr_m
                ]
                still_to_block = int(np.ceil(target_inhibit_rate * n_total)) - n_blocked_total
                if still_to_block > 0 and abn_z_still:
                    still_to_block = min(still_to_block, len(abn_z_still))
                    sorted_z = sorted(abn_z_still)
                    idx = len(sorted_z) - still_to_block
                    new_z = sorted_z[max(0, idx)]

                    if h_zscore and max_healthy_false_inhibit_rate is not None:
                        max_fi = float(max_healthy_false_inhibit_rate)
                        hard_fi = n_h_hard_blocked / max(1, n_h_total_valid)
                        remaining_fi_budget = max(0.0, max_fi - hard_fi)
                        h_mahal_blocked = sum(1 for m in h_mahal if m > thr_m)
                        mahal_fi = h_mahal_blocked / max(1, n_h_total_valid)
                        zscore_budget = max(0.0, remaining_fi_budget - mahal_fi)
                        if zscore_budget < 1.0:
                            n_z_budget = int(zscore_budget * n_h_total_valid)
                            n_z_budget = min(n_z_budget, len(h_zscore))
                            if n_z_budget <= 0:
                                new_z = self.threshold_max_zscore
                            else:
                                sorted_hz = sorted(h_zscore, reverse=True)
                                z_floor = sorted_hz[min(n_z_budget, len(sorted_hz) - 1)]
                                new_z = max(new_z, z_floor)

                    if new_z < self.threshold_max_zscore:
                        self.threshold_max_zscore = new_z
                        zscore_thr_adjusted = True

        return {
            "n_abnormal": n_total,
            "n_hard_caught": n_hard_caught,
            "n_zscore_caught_on_remaining": n_zscore_caught,
            "adjusted": True,
            "zscore_thr_adjusted": zscore_thr_adjusted,
            "threshold_mahalanobis": self.threshold_mahalanobis,
            "threshold_max_zscore": self.threshold_max_zscore,
            "threshold_signal_proxy": self.threshold_signal_proxy,
            "abnormal_mahal_q": q,
        }

    def _mahalanobis_batch_m(self: BaselineCalibrator, X_m: np.ndarray) -> np.ndarray:
        """Mahalanobis on already-sliced mahal_feature data."""
        center_m = self.mean[[self.feature_names.index(f) for f in self.mahal_feature_names]]
        diff = X_m - center_m[None, :]
        if self.use_diagonal:
            return np.sqrt(np.sum(diff ** 2 * self.diag_inv_cov[None, :], axis=1))
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff))

    def to_dict(self: BaselineCalibrator) -> dict:
        return {
            "feature_names": self.feature_names,
            "mahal_feature_names": self.mahal_feature_names,
            "hard_rules": self.hard_rules,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "inv_cov": self.inv_cov.tolist(),
            "diag_inv_cov": self.diag_inv_cov.tolist(),
            "threshold_mahalanobis": self.threshold_mahalanobis,
            "threshold_max_zscore": self.threshold_max_zscore,
            "threshold_signal_proxy": self.threshold_signal_proxy,
            "threshold_rr_proxy": self.threshold_rr_proxy,
            "threshold_quantile": self.threshold_quantile,
            "threshold_method": getattr(self, "threshold_method", "conformal"),
            "conformal_alpha": getattr(self, "conformal_alpha", 0.10),
            "calibration_outlier_frac": getattr(self, "calibration_outlier_frac", 0.0),
            "anomaly_model": getattr(self, "anomaly_model", "mahalanobis"),
            "knn_k": getattr(self, "knn_k", 5),
            "n_outlier_removed": getattr(self, "n_outlier_removed", 0),
            "knn_calibration_vectors": getattr(self, "knn_calibration_vectors", np.zeros((0, 0))).tolist(),
            "use_diagonal": self.use_diagonal,
            "use_robust": self.use_robust,
            "use_shrinkage": self.use_shrinkage,
            "val_frac": self.val_frac,
            "n_baseline_windows": self.n_baseline_windows,
            "n_calibration_windows": self.n_calibration_windows,
            "n_validation_windows": self.n_validation_windows,
            "feature_set": self.feature_set,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BaselineCalibrator:
        return cls(
            feature_names=list(d["feature_names"]),
            mahal_feature_names=list(d.get("mahal_feature_names", d["feature_names"])),
            hard_rules=dict(d.get("hard_rules", {})),
            mean=np.asarray(d["mean"]),
            std=np.asarray(d["std"]),
            inv_cov=np.asarray(d["inv_cov"]),
            diag_inv_cov=np.asarray(d["diag_inv_cov"]),
            threshold_mahalanobis=float(d["threshold_mahalanobis"]),
            threshold_max_zscore=float(d["threshold_max_zscore"]),
            threshold_signal_proxy=float(d.get("threshold_signal_proxy", 1e9)),
            threshold_rr_proxy=float(d.get("threshold_rr_proxy", 1e9)),
            threshold_quantile=float(d["threshold_quantile"]),
            threshold_method=str(d.get("threshold_method", "conformal")),
            conformal_alpha=float(d.get("conformal_alpha", 0.10)),
            calibration_outlier_frac=float(d.get("calibration_outlier_frac", 0.0)),
            anomaly_model=str(d.get("anomaly_model", "mahalanobis")),
            knn_k=int(d.get("knn_k", 5)),
            n_outlier_removed=int(d.get("n_outlier_removed", 0)),
            knn_calibration_vectors=np.asarray(d.get("knn_calibration_vectors", []), dtype=float),
            use_diagonal=bool(d["use_diagonal"]),
            use_robust=bool(d.get("use_robust", True)),
            use_shrinkage=bool(d.get("use_shrinkage", True)),
            val_frac=float(d.get("val_frac", 0.3)),
            n_baseline_windows=int(d["n_baseline_windows"]),
            n_calibration_windows=int(d.get("n_calibration_windows", 0)),
            n_validation_windows=int(d.get("n_validation_windows", 0)),
            feature_set=str(d.get("feature_set", "all")),
        )

    def save(self: BaselineCalibrator, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BaselineCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self: BaselineCalibrator) -> str:
        lines = [
            "BaselineCalibrator",
            f"  feature_set     : {self.feature_set}",
            f"  features        : {len(self.feature_names)} total",
            f"  mahal features  : {len(self.mahal_feature_names)} "
            f"({len(self.feature_names) - len(self.mahal_feature_names)} excluded)",
            f"  hard rules      : {list(self.hard_rules.keys())}",
            f"  use_robust      : {self.use_robust}",
            f"  use_shrinkage   : {self.use_shrinkage}",
            f"  use_diagonal    : {self.use_diagonal}",
            f"  n_cal / n_val   : {self.n_calibration_windows} / {self.n_validation_windows}",
            f"  thr_mahalanobis : {self.threshold_mahalanobis:.3f}",
            f"  thr_signal_proxy: {self.threshold_signal_proxy:.3f}",
            f"  thr_rr_proxy    : {self.threshold_rr_proxy:.3f}",
        ]
        return "\n".join(lines)
