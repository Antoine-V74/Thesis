"""
BayesianOptimizer: main optimisation loop.

Typical usage (5 lines of integration code)
--------------------------------------------
    from optimizer import BayesianOptimizer
    from kernels import get_kernel
    from acquisition import CoolingUCB
    from objective import BaselineProfile, LiteratureObjective
    from api import FunctionAdapter

    baseline = BaselineProfile.from_outputs(my_lv_sim(contraction=0.0, contraction_velocity=0.0))
    bridge = FunctionAdapter(my_lv_sim, param_names=["contraction", "contraction_velocity"])
    opt = BayesianOptimizer(
        bridge=bridge,
        objective=LiteratureObjective(baseline=baseline),
        param_bounds={"contraction": (0.1, 1.0), "contraction_velocity": (0.05, 2.0)},
        kernel="matern52",
        acquisition=CoolingUCB(budget=30),
    )
    opt.run(n_init=5, n_iter=25)
    print(opt.best_params)   # {'contraction': ..., 'contraction_velocity': ...}
    print(opt.best_outputs)  # {'LVEDP': ..., 'aortic_flow': ..., ...}

Manual / asynchronous loop
----------------------------
    params = opt.suggest()                          # ask for next params
    outputs = my_lv_sim(**params)                   # run simulation externally
    opt.observe(params, outputs)                    # record result
    # repeat...

Warm-starting from a previous run
-----------------------------------
    opt.load_history("results/run1.json")
    opt.run(n_init=0, n_iter=15)   # skip random init, continue BO
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from acquisition import AcquisitionFunction, CoolingUCB
from gp_surrogate import GPSurrogate
from kernels import get_kernel
from objective import Objective
from api import SimulatorAdapter as SimulatorBridge


ParamBounds = Dict[str, Tuple[float, float]]


class BayesianOptimizer:
    """Bayesian optimisation over MNA / cardiac simulation parameters.

    Parameters
    ----------
    bridge        : SimulatorBridge — connection to the simulation
    objective     : Objective       — maps sim outputs to a scalar to maximise
    param_bounds  : {name: (lo, hi)} for each parameter
    kernel        : sklearn Kernel or name string (required; e.g. "matern52")
    acquisition   : AcquisitionFunction (default CoolingUCB)
    n_restarts_gp : GP hyperparameter optimisation restarts per fit (default 5)
    seed          : global random seed for reproducibility
    verbose       : print iteration logs by default
    """

    def __init__(
        self,
        bridge: SimulatorBridge,
        objective: Objective,
        param_bounds: ParamBounds,
        kernel,
        acquisition: Optional[AcquisitionFunction] = None,
        n_restarts_gp: int = 5,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.bridge = bridge
        self.objective = objective
        self.param_names: List[str] = list(param_bounds.keys())
        self.param_bounds: ParamBounds = param_bounds
        self.bounds_array: np.ndarray = np.array(
            [param_bounds[k] for k in self.param_names], dtype=float
        )
        self.verbose = verbose

        if kernel is None:
            raise ValueError(
                "BayesianOptimizer requires an explicit kernel. "
                "Pass a kernel name string such as 'matern52', 'matern32', "
                "'rbf', 'rq', or 'periodic'."
            )
        elif isinstance(kernel, str):
            kernel = get_kernel(kernel)
        self.kernel = kernel

        # Acquisition
        self.acquisition: AcquisitionFunction = acquisition or CoolingUCB(budget=20)

        # GP surrogate
        self.gp = GPSurrogate(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_gp,
        )

        self.rng = np.random.default_rng(seed)

        # History buffers
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._raw_outputs: List[Dict[str, float]] = []
        self._timestamps: List[float] = []
        self._labels: List[str] = []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def run(
        self,
        n_init: int = 5,
        n_iter: int = 20,
        verbose: Optional[bool] = None,
    ) -> "BayesianOptimizer":
        """Execute the full optimisation loop.

        Parameters
        ----------
        n_init  : number of initial Latin Hypercube samples (random exploration)
        n_iter  : number of model-guided BO iterations
        verbose : override the instance's verbose flag for this run

        Returns self for method chaining.
        """
        loud = self.verbose if verbose is None else verbose

        if loud:
            print("=" * 60)
            print("  BayesianOptimizer")
            print("=" * 60)
            print(f"  Parameters : {self.param_names}")
            print(f"  Bounds     : {dict(self.param_bounds)}")
            print(f"  Kernel     : {self.kernel.__class__.__name__}")
            print(f"  Acquisition: {self.acquisition}")
            print(f"  Budget     : {n_init} init + {n_iter} BO = {n_init + n_iter} total")
            print("=" * 60)

        # --- initial random phase ---
        if n_init > 0:
            X_init = self._latin_hypercube(n_init)
            for i, x in enumerate(X_init):
                label = f"init {i + 1:2d}/{n_init}"
                self._evaluate(x, label=label, loud=loud)

        # Fit GP after init phase so hyperparameters are available even if n_iter=0
        self._fit_gp()

        # --- BO phase ---
        for i in range(n_iter):
            self._fit_gp()
            x_next = self.acquisition.maximize(
                self.gp,
                bounds=self.bounds_array,
                best_y=self.best_value,
                rng=self.rng,
            )
            label = f"BO   {i + 1:2d}/{n_iter}"
            self._evaluate(x_next, label=label, loud=loud)
            self.acquisition.step()

        if loud:
            print("-" * 60)
            print(f"  Best value : {self.best_value:.4f}")
            print(f"  Best params: {self.best_params}")
            print("=" * 60)

        return self

    def suggest(self) -> Dict[str, float]:
        """Suggest the next set of parameters to evaluate.

        Does not run the simulation.  Useful for manual or asynchronous loops.

        Returns
        -------
        params dict ready to pass to your simulation.
        """
        if len(self._y) < 2:
            x = self._latin_hypercube(1)[0]
        else:
            self._fit_gp()
            x = self.acquisition.maximize(
                self.gp,
                bounds=self.bounds_array,
                best_y=self.best_value,
                rng=self.rng,
            )
        return self._vec_to_dict(x)

    def observe(
        self,
        params: Dict[str, float],
        outputs: Dict[str, float],
    ) -> None:
        """Record an externally evaluated (params, outputs) pair.

        Call this after `suggest()` when you run the simulation yourself.
        """
        x = np.array([params[k] for k in self.param_names], dtype=float)
        y = self.objective(outputs)
        self._X.append(x)
        self._y.append(y)
        self._raw_outputs.append(outputs)
        self._timestamps.append(time.time())
        self._labels.append("external")
        self.acquisition.step()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_value(self) -> float:
        """Best objective value observed so far."""
        if not self._y:
            return float("-inf")
        finite = [v for v in self._y if np.isfinite(v)]
        return float(max(finite)) if finite else float("-inf")

    @property
    def best_params(self) -> Dict[str, float]:
        """Parameter dict that achieved the best objective value."""
        if not self._y:
            return {}
        idx = int(np.argmax(self._y))
        return self._vec_to_dict(self._X[idx])

    @property
    def best_outputs(self) -> Dict[str, float]:
        """Full simulation output dict at the best observed point."""
        if not self._y:
            return {}
        idx = int(np.argmax(self._y))
        return dict(self._raw_outputs[idx])

    @property
    def n_evaluations(self) -> int:
        return len(self._y)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full evaluation history as a list of dicts (for analysis / plotting)."""
        return [
            {
                "iteration": i,
                "label": self._labels[i] if i < len(self._labels) else "",
                "params": self._vec_to_dict(x),
                "objective": float(y),
                "best_so_far": float(max(self._y[: i + 1])),
                "outputs": dict(out),
                "timestamp": ts,
            }
            for i, (x, y, out, ts) in enumerate(
                zip(self._X, self._y, self._raw_outputs, self._timestamps)
            )
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise optimisation history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "param_names": self.param_names,
            "param_bounds": {k: list(v) for k, v in self.param_bounds.items()},
            "kernel": str(self.kernel),
            "acquisition": repr(self.acquisition),
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_evaluations": self.n_evaluations,
            "history": self.history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if self.verbose:
            print(f"[BayesOpt] Saved {self.n_evaluations} evaluations to {path}")

    def load_history(self, path: str | Path) -> None:
        """Warm-start by loading a previous run's history.

        Call before `run(n_init=0, n_iter=...)` to continue from a saved state.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("history", []):
            x = np.array([entry["params"][k] for k in self.param_names], dtype=float)
            self._X.append(x)
            self._y.append(float(entry["objective"]))
            self._raw_outputs.append(entry.get("outputs", {}))
            self._timestamps.append(entry.get("timestamp", 0.0))
            self._labels.append(entry.get("label", "loaded"))

        if self.verbose:
            print(f"[BayesOpt] Loaded {len(data.get('history', []))} points from {path}")

    # ------------------------------------------------------------------
    # GP diagnostic
    # ------------------------------------------------------------------

    def gp_hyperparameters(self) -> Dict[str, float]:
        """Return the fitted GP kernel hyperparameters (after fit)."""
        if not self.gp._fitted:
            return {}
        return self.gp.hyperparameter_summary()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, label: str = "", loud: bool = True) -> float:
        params = self._vec_to_dict(x)
        outputs, elapsed = self.bridge.run_timed(params)

        if outputs is None:
            y = float("-inf")
            msg = "FAILED"
        else:
            y = self.objective(outputs)
            msg = f"y={y:+.4f}"

        self._X.append(x)
        self._y.append(y)
        self._raw_outputs.append(outputs or {})
        self._timestamps.append(time.time())
        self._labels.append(label)

        if loud:
            param_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
            best_str = f"  [best={self.best_value:+.4f}]" if self._y else ""
            print(f"  [{label}]  {msg}  |  {param_str}  ({elapsed:.2f}s){best_str}")

        return y

    def _fit_gp(self) -> None:
        X = np.array(self._X)
        y = np.array(self._y)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return
        self.gp.fit(X[valid], y[valid], bounds=self.bounds_array)

    def _latin_hypercube(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling over the parameter bounds."""
        d = len(self.param_names)
        try:
            from scipy.stats.qmc import LatinHypercube, scale
            sampler = LatinHypercube(d=d, seed=int(self.rng.integers(1 << 30)))
            unit_samples = sampler.random(n=n)
            return scale(unit_samples, self.bounds_array[:, 0], self.bounds_array[:, 1])
        except ImportError:
            # Fallback: jittered uniform grid
            unit = (np.arange(n)[:, None] + self.rng.random((n, d))) / n
            for col in range(d):
                self.rng.shuffle(unit[:, col])
            lo = self.bounds_array[:, 0]
            hi = self.bounds_array[:, 1]
            return lo + unit * (hi - lo)

    def _vec_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_names, x)}
