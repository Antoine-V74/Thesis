"""
Gaussian Process surrogate model.

Wraps sklearn's GaussianProcessRegressor with:
  - automatic input normalisation to [0, 1]^d
  - output standardisation (handled by sklearn's normalize_y)
  - a clean predict() interface returning (mean, std) in original output scale
  - logging of fitted kernel hyperparameters and log marginal likelihood

How GP hyperparameter optimisation works
-----------------------------------------
A GP is defined by a kernel k(x, x'; θ).  The hyperparameters θ are estimated
by maximising the log marginal likelihood (LML):

    log p(y | X, θ) = -½ yᵀ K_y⁻¹ y  -  ½ log|K_y|  -  n/2 log(2π)

where  K_y = K(X, X; θ) + σ²_n I.

The first term rewards data fit; the second term penalises model complexity
(Occam factor) — the LML automatically balances over- and under-fitting.

Optimisation is done with L-BFGS-B in log-hyperparameter space, starting from
`n_restarts_optimizer` random initialisations (sampled uniformly in the bounds
you defined in KernelConfig).  The run that achieves the highest LML is kept.

Practical tips
--------------
- More restarts → more robust but slower.  5–10 is usually sufficient.
- If a hyperparameter collapses to its bound, widen the bounds in KernelConfig.
- Always normalise inputs (this class does it automatically).
- For a near-deterministic simulator, set noise_level_bounds=(1e-10, 1e-6).
- For a stochastic simulator, allow noise_level_bounds=(1e-4, 1.0).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


class GPSurrogate:
    """Gaussian Process surrogate wrapping sklearn's GaussianProcessRegressor.

    Inputs are normalised to [0, 1]^d before fitting / predicting.
    Output standardisation (zero mean, unit variance) is handled by sklearn
    when normalize_y=True.

    Parameters
    ----------
    kernel               : sklearn Kernel object (build one with kernels.get_kernel)
    n_restarts_optimizer : number of L-BFGS-B restarts for LML maximisation
    normalize_y          : standardise y before fitting (recommended)
    alpha                : nugget added to diagonal of K for numerical stability
                           (also acts as a minimum observation noise)
    random_state         : seed for reproducible hyperparameter optimisation
    """

    def __init__(
        self,
        kernel: Kernel,
        n_restarts_optimizer: int = 5,
        normalize_y: bool = True,
        alpha: float = 1e-6,
        random_state: Optional[int] = 42,
    ) -> None:
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.random_state = random_state

        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            alpha=alpha,
            random_state=random_state,
        )

        self._bounds: Optional[np.ndarray] = None  # (d, 2)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: Optional[np.ndarray] = None,
    ) -> "GPSurrogate":
        """Fit the GP on observations (X, y).

        Parameters
        ----------
        X      : (n, d) parameter matrix
        y      : (n,)  score values
        bounds : (d, 2) parameter bounds [[lo, hi], ...] for normalisation.
                 If None, uses column-wise min/max of X (may be unreliable with few points).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()

        if bounds is not None:
            self._bounds = np.asarray(bounds, dtype=float)
        else:
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            same = (hi == lo)
            hi[same] = lo[same] + 1.0   # avoid zero-range dimensions
            self._bounds = np.column_stack([lo, hi])

        X_norm = self._normalise(X)
        self._gpr.fit(X_norm, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (mean, std) at candidate points X.

        Parameters
        ----------
        X : (n, d) or (d,) parameter vectors in original (un-normalised) space

        Returns
        -------
        mu    : (n,) posterior mean
        sigma : (n,) posterior standard deviation
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_norm = self._normalise(X)
        mu, sigma = self._gpr.predict(X_norm, return_std=True)
        return mu, np.maximum(sigma, 0.0)

    def sample_posterior(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw posterior samples at X. Returns (n_points, n_samples)."""
        self._check_fitted()
        X_norm = self._normalise(np.atleast_2d(X))
        seed = int(rng.integers(1 << 30)) if rng is not None else self.random_state
        return self._gpr.sample_y(X_norm, n_samples=n_samples, random_state=seed)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def log_marginal_likelihood(self) -> float:
        """LML of the fitted model — compare across kernels: higher is better."""
        self._check_fitted()
        return float(self._gpr.log_marginal_likelihood_value_)

    @property
    def fitted_kernel(self) -> Kernel:
        """Kernel with optimised hyperparameters (available after fit)."""
        self._check_fitted()
        return self._gpr.kernel_

    def hyperparameter_summary(self) -> Dict[str, float]:
        """Return optimised hyperparameters as a flat dict for logging."""
        self._check_fitted()
        k = self._gpr.kernel_
        summary: Dict[str, float] = {"log_marginal_likelihood": self.log_marginal_likelihood}
        for hp in k.hyperparameters:
            val = k.get_params()[hp.name]
            if np.ndim(val) == 0:
                summary[hp.name] = float(val)
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        span = hi - lo
        span[span == 0.0] = 1.0
        return (X - lo) / span

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GPSurrogate: call fit() before using this method.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"GPSurrogate(kernel={self.kernel.__class__.__name__}, "
            f"restarts={self.n_restarts_optimizer}, {status})"
        )
