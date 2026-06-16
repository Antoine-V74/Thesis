"""
Acquisition functions for Bayesian optimisation.

All functions take a fitted GPSurrogate and return a utility score for each
candidate point (higher = more worth evaluating).  The optimiser maximises
this score to pick the next simulation to run.

Acquisition functions available
--------------------------------
EI   ExpectedImprovement     Classic; balances exploration and exploitation via ξ.
PI   ProbabilityOfImprovement Aggressive; exploits quickly but may get stuck early.
UCB  UpperConfidenceBound     Explicit β trades off exploration vs exploitation.
TS   ThompsonSampling         Stochastic; naturally diverse, hard to fool.

Exploration / exploitation scheduling strategies
-------------------------------------------------
CoolingUCB      UCB with a decaying β schedule.  [recommended]
                β(t) = β₀ · (1 - t/T)^γ + β_min
                Large β at the start → wide exploration.
                Shrinks to β_min at the end → exploitation of best region.

WarmupThenEI    UCB for the first `warmup` iterations, then EI.
                Useful when you want broad initial coverage then fine-grained search.

AdaptiveEI      EI with ξ that starts high (exploration) and decays to near 0
                (pure improvement), following the same annealing idea.

Practical guidance
------------------
- With a budget of N iterations:
    * CoolingUCB(budget=N, beta_0=2.0, beta_min=0.1) — good all-around default.
    * WarmupThenEI(warmup=N//3) — good if you trust EI's exploitation.
    * Pure UCB(beta=2.0) — fine for exploration-heavy early runs.
    * EI(xi=0.01) — good once you have ≥10 points and want to converge.
- Thompson Sampling is a good secondary strategy: run it in parallel with UCB
  to ensure diverse proposals.
- For a near-deterministic simulator (no noise), EI converges faster than UCB.
- For a noisy simulator, UCB or TS tends to be more robust.

Acquisition maximisation
------------------------
Each class exposes a `maximize(gp, bounds, best_y, ...)` method that uses:
  1. Large random grid (n_random=1024) to find promising seeds.
  2. L-BFGS-B gradient descent from the top-k seeds (multi-start).
This avoids local optima in the acquisition landscape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        """Return acquisition value for each row of X.

        Parameters
        ----------
        X      : (n, d) candidate parameter vectors (original / un-normalised space)
        gp     : fitted GPSurrogate
        best_y : current best observed objective value (we are maximising)

        Returns
        -------
        acq : (n,) values; higher means more worth evaluating
        """

    def maximize(
        self,
        gp,
        bounds: np.ndarray,
        best_y: float,
        n_restarts: int = 20,
        n_random: int = 1024,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Return the parameter vector that maximises this acquisition function.

        Strategy: random search over `n_random` points, keep top-`n_restarts`
        as seeds, then refine each with L-BFGS-B.

        Parameters
        ----------
        gp         : fitted GPSurrogate
        bounds     : (d, 2) array [[lo, hi], ...]
        best_y     : current best observation
        n_restarts : number of L-BFGS-B restarts
        n_random   : size of the initial random grid
        rng        : numpy Generator (for reproducibility)

        Returns
        -------
        x_best : (d,) parameter vector
        """
        if rng is None:
            rng = np.random.default_rng()

        d = bounds.shape[0]
        lo, hi = bounds[:, 0], bounds[:, 1]
        scipy_bounds = list(zip(lo.tolist(), hi.tolist()))

        # Random grid
        X_rand = rng.uniform(lo, hi, size=(n_random, d))
        acq_rand = self(X_rand, gp, best_y)

        # Top seeds for gradient restarts
        k = min(n_restarts, n_random)
        top_idx = np.argsort(acq_rand)[-k:][::-1]
        seeds = X_rand[top_idx]

        best_x = seeds[0].copy()
        best_val = acq_rand[top_idx[0]]

        for x0 in seeds:
            def neg_acq(x: np.ndarray) -> float:
                return -float(self(x[np.newaxis, :], gp, best_y)[0])

            res = minimize(
                neg_acq,
                x0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            candidate_val = -res.fun
            if candidate_val > best_val:
                best_val = candidate_val
                best_x = np.clip(res.x, lo, hi)

        return best_x

    def step(self) -> None:
        """Advance the internal schedule by one iteration (override if stateful)."""


# ---------------------------------------------------------------------------
# Concrete acquisition functions
# ---------------------------------------------------------------------------

class ExpectedImprovement(AcquisitionFunction):
    """EI = E[max(f(x) - (f* + ξ), 0)] under the GP posterior.

    Closed form:
        EI(x) = (μ - f* - ξ) · Φ(Z) + σ · φ(Z),   Z = (μ - f* - ξ) / σ

    Parameters
    ----------
    xi : exploration parameter ξ ≥ 0.
         ξ = 0   → pure greedy improvement (exploitation-heavy).
         ξ = 0.1 → encourages exploration of uncertain regions.
         Typical values: [0.0, 0.1].
    """

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        mu, sigma = gp.predict(np.atleast_2d(X))
        sigma = np.maximum(sigma, 1e-9)
        improvement = mu - best_y - self.xi
        Z = improvement / sigma
        return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    def __repr__(self) -> str:
        return f"ExpectedImprovement(xi={self.xi})"


class ProbabilityOfImprovement(AcquisitionFunction):
    """PI = P(f(x) > f* + ξ) = Φ((μ - f* - ξ) / σ).

    Converges faster than EI in noise-free settings but is more prone to
    over-exploitation (gets stuck in a local optimum).

    Parameters
    ----------
    xi : same role as in EI; increase to force more exploration.
    """

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        mu, sigma = gp.predict(np.atleast_2d(X))
        sigma = np.maximum(sigma, 1e-9)
        Z = (mu - best_y - self.xi) / sigma
        return norm.cdf(Z)

    def __repr__(self) -> str:
        return f"ProbabilityOfImprovement(xi={self.xi})"


class UpperConfidenceBound(AcquisitionFunction):
    """UCB(x) = μ(x) + β · σ(x).

    Explicitly trades off mean (exploitation) and uncertainty (exploration).

    Parameters
    ----------
    beta : exploration weight β ≥ 0.
           β = 0   → pure exploitation (pick point with highest predicted mean).
           β = 2   → classic GP-UCB; good default for early exploration.
           β = 0.1 → late-stage exploitation.
           Theoretical (Srinivas 2010): β_t = 2 log(d t² π² / 6δ) but empirically
           a fixed β ∈ [0.5, 3] works well.
    """

    def __init__(self, beta: float = 2.0) -> None:
        self.beta = beta

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        mu, sigma = gp.predict(np.atleast_2d(X))
        return mu + self.beta * sigma

    def __repr__(self) -> str:
        return f"UpperConfidenceBound(beta={self.beta})"


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling: draw one GP posterior sample and pick its maximum.

    Properties:
    - Stochastic — naturally encourages diversity between proposals.
    - Asymptotically optimal regret bounds.
    - Avoids pathological over-exploitation of EI / PI in noisy settings.

    Parameters
    ----------
    rng : numpy Generator for reproducibility.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng or np.random.default_rng()

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        X = np.atleast_2d(X)
        seed = int(self._rng.integers(1 << 30))
        samples = gp.sample_posterior(X, n_samples=1, rng=self._rng)
        return samples[:, 0]

    def __repr__(self) -> str:
        return "ThompsonSampling()"


# ---------------------------------------------------------------------------
# Scheduling strategies
# ---------------------------------------------------------------------------

class CoolingUCB(AcquisitionFunction):
    """UCB with a monotonically decaying β schedule.  [recommended default]

        β(t) = β₀ · (1 - t / T)^γ + β_min

    At iteration 0 (t=0):  β = β₀ + β_min   → strong exploration.
    At iteration T (t=T):  β = β_min          → focused exploitation.

    Call `.step()` after each BO iteration to advance the schedule.

    Parameters
    ----------
    budget  : total number of BO iterations T (used to normalise t/T)
    beta_0  : initial extra exploration weight (default 2.0)
    beta_min: minimum β after full annealing (default 0.1; ensures some exploration)
    gamma   : annealing exponent.
              1.0 = linear decay (recommended),
              2.0 = quadratic (slower start, faster end),
              0.5 = fast initial decay.
    """

    def __init__(
        self,
        budget: int,
        beta_0: float = 2.0,
        beta_min: float = 0.1,
        gamma: float = 1.0,
    ) -> None:
        self.budget = budget
        self.beta_0 = beta_0
        self.beta_min = beta_min
        self.gamma = gamma
        self._t: int = 0

    @property
    def current_beta(self) -> float:
        frac = min(self._t / max(self.budget, 1), 1.0)
        return self.beta_0 * (1.0 - frac) ** self.gamma + self.beta_min

    def step(self) -> None:
        self._t += 1

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        mu, sigma = gp.predict(np.atleast_2d(X))
        return mu + self.current_beta * sigma

    def __repr__(self) -> str:
        return (
            f"CoolingUCB(budget={self.budget}, beta0={self.beta_0}, "
            f"beta_min={self.beta_min}, gamma={self.gamma}, t={self._t})"
        )


class WarmupThenEI(AcquisitionFunction):
    """Use UCB for the first `warmup` iterations, then switch to EI.

    Rationale: UCB provides broad coverage of the parameter space early on;
    EI then focuses the remaining budget on the most promising region.

    Parameters
    ----------
    warmup : number of iterations to run UCB before switching (recommend budget//3)
    beta   : UCB exploration weight during warmup phase
    xi     : EI exploration parameter after warmup
    """

    def __init__(
        self,
        warmup: int,
        beta: float = 2.0,
        xi: float = 0.01,
    ) -> None:
        self.warmup = warmup
        self._ucb = UpperConfidenceBound(beta=beta)
        self._ei = ExpectedImprovement(xi=xi)
        self._t: int = 0

    @property
    def phase(self) -> str:
        return "UCB" if self._t < self.warmup else "EI"

    def step(self) -> None:
        self._t += 1

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        if self._t < self.warmup:
            return self._ucb(X, gp, best_y)
        return self._ei(X, gp, best_y)

    def __repr__(self) -> str:
        return f"WarmupThenEI(warmup={self.warmup}, phase={self.phase}, t={self._t})"


class AdaptiveEI(AcquisitionFunction):
    """EI with ξ decaying from ξ_max to ξ_min over the optimisation budget.

        ξ(t) = ξ_max · (1 - t/T)^γ + ξ_min

    Provides the same exploration/exploitation transition as CoolingUCB but
    via the EI framework.

    Parameters
    ----------
    budget : total BO iterations
    xi_max : initial ξ (encourages exploration; default 0.1)
    xi_min : final ξ (near-greedy; default 0.001)
    gamma  : annealing speed (default 1.0)
    """

    def __init__(
        self,
        budget: int,
        xi_max: float = 0.1,
        xi_min: float = 0.001,
        gamma: float = 1.0,
    ) -> None:
        self.budget = budget
        self.xi_max = xi_max
        self.xi_min = xi_min
        self.gamma = gamma
        self._t: int = 0

    @property
    def current_xi(self) -> float:
        frac = min(self._t / max(self.budget, 1), 1.0)
        return self.xi_max * (1.0 - frac) ** self.gamma + self.xi_min

    def step(self) -> None:
        self._t += 1

    def __call__(self, X: np.ndarray, gp, best_y: float) -> np.ndarray:
        mu, sigma = gp.predict(np.atleast_2d(X))
        sigma = np.maximum(sigma, 1e-9)
        xi = self.current_xi
        improvement = mu - best_y - xi
        Z = improvement / sigma
        return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    def __repr__(self) -> str:
        return (
            f"AdaptiveEI(budget={self.budget}, xi_max={self.xi_max}, "
            f"xi_min={self.xi_min}, xi_now={self.current_xi:.4f}, t={self._t})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AVAILABLE_ACQUISITIONS = {
    "ei": ExpectedImprovement,
    "pi": ProbabilityOfImprovement,
    "ucb": UpperConfidenceBound,
    "ts": ThompsonSampling,
    "cooling_ucb": CoolingUCB,
    "warmup_then_ei": WarmupThenEI,
    "adaptive_ei": AdaptiveEI,
}


def get_acquisition(name: str, **kwargs) -> AcquisitionFunction:
    """Factory: instantiate an acquisition function by name.

    Examples
    --------
    >>> get_acquisition("cooling_ucb", budget=30)
    >>> get_acquisition("ei", xi=0.05)
    >>> get_acquisition("ucb", beta=1.5)
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in AVAILABLE_ACQUISITIONS:
        raise ValueError(
            f"Unknown acquisition: {name!r}. "
            f"Choose from: {list(AVAILABLE_ACQUISITIONS)}"
        )
    return AVAILABLE_ACQUISITIONS[key](**kwargs)
