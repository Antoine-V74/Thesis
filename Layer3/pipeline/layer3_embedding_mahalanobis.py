#!/usr/bin/env python3
"""
Embedding-space healthy-baseline anomaly models for the Layer 3 ECG veto.

Safety framing:
- This class does NOT command stimulation.
- It only returns a distance score and a permit/inhibit decision for an upstream gate.
- Thresholds must be fitted from held-out healthy calibration embeddings, not from abnormal labels,
  unless explicitly marked as offline supervised analysis.

Main deployable options:
- Gaussian Mahalanobis distance: simple, transparent baseline around the healthy cloud.
- kNN distance: nonparametric baseline that can better tolerate multimodal healthy ECG.

Deep SVDD lives in layer3_anomaly.py as an optional ablation, not the primary
runtime framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
import numpy as np


@dataclass
class EmbeddingPreprocessConfig:
    """Optional embedding preprocessing before the healthy-baseline model.

    Fit ONLY on healthy calibration embeddings. This avoids leakage from
    abnormal labels and keeps the runtime story deployment-compatible.
    """
    l2_normalize: bool = False
    pca_dim: Optional[int] = 32
    pca_whiten: bool = False
    eps: float = 1e-12


class EmbeddingPreprocessor:
    """L2-normalization + PCA projection fitted on healthy calibration embeddings."""

    def __init__(self, config: Optional[EmbeddingPreprocessConfig] = None):
        self.config = config or EmbeddingPreprocessConfig()
        self.pca_ = None
        self.input_dim_: Optional[int] = None
        self.output_dim_: Optional[int] = None

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        return EmbeddingMahalanobisBaseline._as_2d(x)

    def _l2(self, x: np.ndarray) -> np.ndarray:
        if not self.config.l2_normalize:
            return x
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, float(self.config.eps))
        return x / norm

    def fit(self, embeddings: np.ndarray) -> "EmbeddingPreprocessor":
        x = self._as_2d(embeddings)
        self.input_dim_ = int(x.shape[1])
        x = self._l2(x)
        pca_dim = self.config.pca_dim
        if pca_dim is not None and int(pca_dim) > 0:
            n_components = int(min(int(pca_dim), x.shape[0] - 1, x.shape[1]))
            if n_components >= 1:
                try:
                    from sklearn.decomposition import PCA
                except Exception as exc:
                    raise RuntimeError("PCA preprocessing requires scikit-learn.") from exc
                self.pca_ = PCA(n_components=n_components, whiten=bool(self.config.pca_whiten), svd_solver="auto")
                self.pca_.fit(x)
                self.output_dim_ = int(n_components)
                return self
        self.pca_ = None
        self.output_dim_ = int(x.shape[1])
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = self._as_2d(embeddings)
        x = self._l2(x)
        if self.pca_ is not None:
            x = self.pca_.transform(x)
        return np.asarray(x, dtype=np.float64)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.fit(embeddings).transform(embeddings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l2_normalize": bool(self.config.l2_normalize),
            "pca_dim": None if self.config.pca_dim is None else int(self.config.pca_dim),
            "pca_whiten": bool(self.config.pca_whiten),
            "input_dim": self.input_dim_,
            "output_dim": self.output_dim_,
            "pca_enabled": self.pca_ is not None,
        }


@dataclass
class EmbeddingMahalanobisConfig:
    shrinkage: float = 0.10
    eps: float = 1e-6
    threshold_quantile: float = 0.99
    use_sqrt_distance: bool = True
    covariance_estimator: str = "ledoit_wolf"  # "ledoit_wolf" or "diagonal_shrinkage"


class EmbeddingMahalanobisBaseline:
    """Fit a Gaussian baseline in encoder embedding space and score deviations."""

    def __init__(self, config: Optional[EmbeddingMahalanobisConfig] = None):
        self.config = config or EmbeddingMahalanobisConfig()
        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.n_fit_: int = 0
        self.n_val_: int = 0
        self.n_outlier_removed_: int = 0
        self.robust_keep_mask_: Optional[np.ndarray] = None
        self.robust_fit_scores_: Optional[np.ndarray] = None

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D embedding array, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Embedding array contains NaN or inf values")
        return x

    def fit(self, embeddings: np.ndarray) -> "EmbeddingMahalanobisBaseline":
        x = self._as_2d(embeddings)
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 embeddings to fit Mahalanobis baseline")

        self.n_fit_ = int(x.shape[0])
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_

        estimator = str(self.config.covariance_estimator).lower()
        if estimator == "ledoit_wolf" and x.shape[0] <= 2:
            estimator = "diagonal_shrinkage"
        if estimator == "ledoit_wolf" and x.shape[0] > 2:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(x)
                cov = np.asarray(lw.covariance_, dtype=np.float64)
            except Exception:
                # Fall back to the historical diagonal shrinkage path if sklearn
                # is unavailable or Ledoit-Wolf fails on a degenerate calibration.
                estimator = "diagonal_shrinkage"

        if estimator != "ledoit_wolf":
            if x.shape[0] <= 2:
                # Degenerate fallback for tiny calibration segments.
                var = np.var(centered, axis=0) + self.config.eps
                cov = np.diag(var)
            else:
                cov = np.cov(centered, rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]], dtype=np.float64)

            # Historical fallback: shrink empirical covariance toward its diagonal
            # for stability when dim is high or calibration data are limited.
            diag = np.diag(np.diag(cov))
            shrink = float(np.clip(self.config.shrinkage, 0.0, 1.0))
            cov = (1.0 - shrink) * cov + shrink * diag

        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = cov + self.config.eps * np.eye(cov.shape[0], dtype=np.float64)

        self.cov_ = cov
        self.precision_ = np.linalg.pinv(cov)
        return self

    def fit_robust(
        self,
        embeddings: np.ndarray,
        outlier_frac: float = 0.0,
        min_keep: int = 2,
    ) -> "EmbeddingMahalanobisBaseline":
        """Fit, prune the most distant calibration embeddings, then refit.

        This protects the personalized healthy baseline from occasional PVC/noise
        contamination in the calibration segment. The threshold is still set from
        held-out healthy validation scores, not abnormal labels.
        """
        x = self._as_2d(embeddings)
        frac = float(np.clip(outlier_frac, 0.0, 0.5))
        if frac <= 0.0 or x.shape[0] <= max(2, int(min_keep)):
            self.n_outlier_removed_ = 0
            self.robust_keep_mask_ = np.ones(x.shape[0], dtype=bool)
            self.robust_fit_scores_ = np.full(x.shape[0], np.nan, dtype=float)
            return self.fit(x)

        provisional = EmbeddingMahalanobisBaseline(self.config).fit(x)
        scores = provisional.score(x)
        keep_n = max(int(min_keep), int(np.ceil((1.0 - frac) * x.shape[0])))
        keep_n = min(max(2, keep_n), x.shape[0])
        order = np.argsort(scores, kind="mergesort")
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[order[:keep_n]] = True
        self.n_outlier_removed_ = int((~mask).sum())
        self.robust_keep_mask_ = mask
        self.robust_fit_scores_ = scores.astype(float)
        return self.fit(x[mask])

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("Baseline must be fitted before scoring")
        x = self._as_2d(embeddings)
        diff = x - self.mean_
        d2 = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2) if self.config.use_sqrt_distance else d2

    def set_threshold_from_scores(self, healthy_scores: np.ndarray) -> float:
        scores = np.asarray(healthy_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            raise ValueError("No finite healthy validation scores available for thresholding")
        q = float(np.clip(self.config.threshold_quantile, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, q))
        self.n_val_ = int(scores.size)
        return self.threshold_

    def set_threshold(self, healthy_validation_embeddings: np.ndarray) -> float:
        return self.set_threshold_from_scores(self.score(healthy_validation_embeddings))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold must be set before prediction")
        return self.score(embeddings) <= float(self.threshold_)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shrinkage": self.config.shrinkage,
            "eps": self.config.eps,
            "threshold_quantile": self.config.threshold_quantile,
            "use_sqrt_distance": self.config.use_sqrt_distance,
            "covariance_estimator": self.config.covariance_estimator,
            "threshold": self.threshold_,
            "n_fit": self.n_fit_,
            "n_val": self.n_val_,
            "n_outlier_removed": self.n_outlier_removed_,
            "embedding_dim": None if self.mean_ is None else int(self.mean_.shape[0]),
        }

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.cov_ is None or self.precision_ is None:
            raise RuntimeError("Cannot save an unfitted baseline")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean_,
            cov=self.cov_,
            precision=self.precision_,
            threshold=np.array([np.nan if self.threshold_ is None else self.threshold_]),
            meta=json.dumps(self.to_dict()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingMahalanobisBaseline":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        cfg = EmbeddingMahalanobisConfig(
            shrinkage=float(meta.get("shrinkage", 0.10)),
            eps=float(meta.get("eps", 1e-6)),
            threshold_quantile=float(meta.get("threshold_quantile", 0.99)),
            use_sqrt_distance=bool(meta.get("use_sqrt_distance", True)),
            covariance_estimator=str(meta.get("covariance_estimator", "ledoit_wolf")),
        )
        obj = cls(cfg)
        obj.mean_ = data["mean"]
        obj.cov_ = data["cov"]
        obj.precision_ = data["precision"]
        thr = float(data["threshold"][0])
        obj.threshold_ = None if np.isnan(thr) else thr
        obj.n_fit_ = int(meta.get("n_fit", 0))
        obj.n_val_ = int(meta.get("n_val", 0))
        obj.n_outlier_removed_ = int(meta.get("n_outlier_removed", 0))
        return obj


@dataclass
class EmbeddingKNNConfig:
    k: int = 5
    threshold_quantile: float = 0.99
    score_aggregation: str = "mean"  # "mean" or "kth"


class EmbeddingKNNBaseline:
    """Fit a nonparametric healthy embedding baseline and score kNN distance."""

    def __init__(self, config: Optional[EmbeddingKNNConfig] = None):
        self.config = config or EmbeddingKNNConfig()
        self.embeddings_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.n_fit_: int = 0
        self.n_val_: int = 0
        self.n_outlier_removed_: int = 0
        self.robust_keep_mask_: Optional[np.ndarray] = None
        self.robust_fit_scores_: Optional[np.ndarray] = None

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        return EmbeddingMahalanobisBaseline._as_2d(x)

    def fit(self, embeddings: np.ndarray) -> "EmbeddingKNNBaseline":
        x = self._as_2d(embeddings)
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 embeddings to fit kNN baseline")
        self.embeddings_ = x.copy()
        self.n_fit_ = int(x.shape[0])
        return self

    def _aggregate_distances(self, distances: np.ndarray, k_eff: int) -> np.ndarray:
        nearest = np.partition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
        if str(self.config.score_aggregation).lower() == "kth":
            return nearest.max(axis=1)
        return nearest.mean(axis=1)

    def score(self, embeddings: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        if self.embeddings_ is None:
            raise RuntimeError("Baseline must be fitted before scoring")
        x = self._as_2d(embeddings)
        ref = self.embeddings_
        k_eff = int(max(1, min(int(self.config.k), ref.shape[0])))
        scores = []
        for start in range(0, x.shape[0], int(chunk_size)):
            chunk = x[start:start + int(chunk_size)]
            diff = chunk[:, None, :] - ref[None, :, :]
            d = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
            scores.append(self._aggregate_distances(d, k_eff))
        return np.concatenate(scores).astype(float)

    def score_fit_leave_one_out(self, chunk_size: int = 2048) -> np.ndarray:
        if self.embeddings_ is None:
            raise RuntimeError("Baseline must be fitted before scoring")
        x = self.embeddings_
        if x.shape[0] < 3:
            return np.zeros(x.shape[0], dtype=float)
        k_eff = int(max(1, min(int(self.config.k), x.shape[0] - 1)))
        scores = []
        for start in range(0, x.shape[0], int(chunk_size)):
            chunk = x[start:start + int(chunk_size)]
            diff = chunk[:, None, :] - x[None, :, :]
            d = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
            rows = np.arange(d.shape[0])
            cols = np.arange(start, start + d.shape[0])
            d[rows, cols] = np.inf
            scores.append(self._aggregate_distances(d, k_eff))
        return np.concatenate(scores).astype(float)

    def fit_robust(
        self,
        embeddings: np.ndarray,
        outlier_frac: float = 0.0,
        min_keep: int = 2,
    ) -> "EmbeddingKNNBaseline":
        x = self._as_2d(embeddings)
        frac = float(np.clip(outlier_frac, 0.0, 0.5))
        if frac <= 0.0 or x.shape[0] <= max(2, int(min_keep)):
            self.n_outlier_removed_ = 0
            self.robust_keep_mask_ = np.ones(x.shape[0], dtype=bool)
            self.robust_fit_scores_ = np.full(x.shape[0], np.nan, dtype=float)
            return self.fit(x)

        provisional = EmbeddingKNNBaseline(self.config).fit(x)
        scores = provisional.score_fit_leave_one_out()
        keep_n = max(int(min_keep), int(np.ceil((1.0 - frac) * x.shape[0])))
        keep_n = min(max(2, keep_n), x.shape[0])
        order = np.argsort(scores, kind="mergesort")
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[order[:keep_n]] = True
        self.n_outlier_removed_ = int((~mask).sum())
        self.robust_keep_mask_ = mask
        self.robust_fit_scores_ = scores.astype(float)
        return self.fit(x[mask])

    def set_threshold_from_scores(self, healthy_scores: np.ndarray) -> float:
        scores = np.asarray(healthy_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            raise ValueError("No finite healthy validation scores available for thresholding")
        q = float(np.clip(self.config.threshold_quantile, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, q))
        self.n_val_ = int(scores.size)
        return self.threshold_

    def set_threshold(self, healthy_validation_embeddings: np.ndarray) -> float:
        return self.set_threshold_from_scores(self.score(healthy_validation_embeddings))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold must be set before prediction")
        return self.score(embeddings) <= float(self.threshold_)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": int(self.config.k),
            "threshold_quantile": self.config.threshold_quantile,
            "score_aggregation": self.config.score_aggregation,
            "threshold": self.threshold_,
            "n_fit": self.n_fit_,
            "n_val": self.n_val_,
            "n_outlier_removed": self.n_outlier_removed_,
            "embedding_dim": None if self.embeddings_ is None else int(self.embeddings_.shape[1]),
        }

    def save(self, path: str | Path) -> None:
        if self.embeddings_ is None:
            raise RuntimeError("Cannot save an unfitted baseline")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings_,
            threshold=np.array([np.nan if self.threshold_ is None else self.threshold_]),
            meta=json.dumps(self.to_dict()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingKNNBaseline":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        cfg = EmbeddingKNNConfig(
            k=int(meta.get("k", 5)),
            threshold_quantile=float(meta.get("threshold_quantile", 0.99)),
            score_aggregation=str(meta.get("score_aggregation", "mean")),
        )
        obj = cls(cfg)
        obj.embeddings_ = data["embeddings"]
        obj.n_fit_ = int(meta.get("n_fit", obj.embeddings_.shape[0]))
        obj.n_val_ = int(meta.get("n_val", 0))
        obj.n_outlier_removed_ = int(meta.get("n_outlier_removed", 0))
        thr = float(data["threshold"][0])
        obj.threshold_ = None if np.isnan(thr) else thr
        return obj
