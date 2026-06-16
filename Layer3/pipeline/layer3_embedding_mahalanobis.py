#!/usr/bin/env python3
"""
Embedding-space Gaussian Mahalanobis baseline for Layer 3 ECG anomaly veto.

Safety framing:
- This class does NOT command stimulation.
- It only returns a distance score and a permit/inhibit decision for an upstream gate.
- Thresholds must be fitted from held-out healthy calibration embeddings, not from abnormal labels,
  unless explicitly marked as offline supervised analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
import numpy as np


@dataclass
class EmbeddingMahalanobisConfig:
    shrinkage: float = 0.10
    eps: float = 1e-6
    threshold_quantile: float = 0.99
    use_sqrt_distance: bool = True


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

        if x.shape[0] <= 2:
            # Degenerate fallback for tiny calibration segments.
            var = np.var(centered, axis=0) + self.config.eps
            cov = np.diag(var)
        else:
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]], dtype=np.float64)

        # Shrink empirical covariance toward its diagonal for stability when dim is high
        # or calibration data are limited.
        diag = np.diag(np.diag(cov))
        shrink = float(np.clip(self.config.shrinkage, 0.0, 1.0))
        cov = (1.0 - shrink) * cov + shrink * diag
        cov = cov + self.config.eps * np.eye(cov.shape[0], dtype=np.float64)

        self.cov_ = cov
        self.precision_ = np.linalg.pinv(cov)
        return self

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
            "threshold": self.threshold_,
            "n_fit": self.n_fit_,
            "n_val": self.n_val_,
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
        )
        obj = cls(cfg)
        obj.mean_ = data["mean"]
        obj.cov_ = data["cov"]
        obj.precision_ = data["precision"]
        thr = float(data["threshold"][0])
        obj.threshold_ = None if np.isnan(thr) else thr
        obj.n_fit_ = int(meta.get("n_fit", 0))
        obj.n_val_ = int(meta.get("n_val", 0))
        return obj
