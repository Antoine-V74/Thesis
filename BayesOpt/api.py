"""
BayesOpt public API — connect a simulation and run MNA parameter optimisation.

Most users only need:

    from api import run_mna_bayesopt

    def simulate_episode(contraction, contraction_velocity, n_beats):
        # run one full LV + MNA simulation episode
        return {"LVEDP": ..., "aortic_flow": ..., ...}

    results = run_mna_bayesopt(
        simulate_episode=simulate_episode,
        n_beats=10000,
        kernel_name="matern52",
    )

If the model is launched as a Python script, use ``make_script_episode_function``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from acquisition import CoolingUCB
from kernels import get_kernel
from objective import BaselineProfile, LiteratureObjective, Objective


SimParams = Dict[str, float]
SimOutputs = Dict[str, float]
EpisodeFunction = Callable[[float, float, int], SimOutputs]


# ---------------------------------------------------------------------------
# Simulation adapters (used internally by the optimiser)
# ---------------------------------------------------------------------------

class SimulatorAdapter(ABC):
    """Interface between BayesianOptimizer and a simulation."""

    @abstractmethod
    def run(self, params: SimParams) -> SimOutputs:
        """Run the simulation and return haemodynamic outputs."""

    def run_safe(self, params: SimParams) -> Optional[SimOutputs]:
        try:
            return self.run(params)
        except Exception as exc:  # noqa: BLE001
            print(f"[BayesOpt] Simulation failed for params={params}: {exc}")
            return None

    def run_timed(self, params: SimParams) -> tuple[Optional[SimOutputs], float]:
        t0 = time.perf_counter()
        outputs = self.run_safe(params)
        return outputs, time.perf_counter() - t0


class FunctionAdapter(SimulatorAdapter):
    """Wrap a plain Python callable as a simulation adapter."""

    def __init__(
        self,
        fn: Callable,
        param_names: List[str],
        accept_dict: bool = False,
        output_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.fn = fn
        self.param_names = param_names
        self.accept_dict = accept_dict
        self.output_map = output_map or {}

    def run(self, params: SimParams) -> SimOutputs:
        if self.accept_dict:
            result = self.fn(params)
        else:
            kwargs = {k: params[k] for k in self.param_names if k in params}
            result = self.fn(**kwargs)

        if not isinstance(result, dict):
            raise TypeError(
                f"Simulation function must return a dict, got {type(result).__name__}"
            )

        if self.output_map:
            result = {self.output_map.get(k, k): v for k, v in result.items()}

        return result


class MockSimulation(SimulatorAdapter):
    """Synthetic cardiac response for algorithm testing (no real model required)."""

    def __init__(
        self,
        param_names: Optional[List[str]] = None,
        noise_std: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        self.param_names = param_names or ["contraction", "contraction_velocity"]
        self.noise_std = noise_std
        import numpy as np
        self._rng = np.random.default_rng(seed)

    def run(self, params: SimParams) -> SimOutputs:
        import numpy as np

        c = float(params.get(self.param_names[0], 0.5))
        cv = float(params.get(self.param_names[1], 0.5)) if len(self.param_names) > 1 else 0.5
        r = float(np.exp(-((c - 0.65) ** 2 / 0.05 + (cv - 0.55) ** 2 / 0.03)))

        def n() -> float:
            return float(self._rng.normal(0.0, self.noise_std))

        lvedv = 110.0 + 30.0 * (1.0 - c) + n()
        lvesv = 40.0 - 20.0 * r + n()
        lvesp = 80.0 + 40.0 * r + n()
        lvedp = 6.0 + 8.0 * (1.0 - r) + n()
        rvedv = 95.0 + 20.0 * (1.0 - c) + n()
        rvesv = 35.0 - 15.0 * r + n()
        rvesp = 20.0 + 8.0 * r + n()
        rvedp = 4.0 + 5.0 * (1.0 - r) + n()

        return {
            "LVEDV": lvedv,
            "LVESV": lvesv,
            "LVESP": lvesp,
            "LVEDP": lvedp,
            "RVEDV": rvedv,
            "RVESV": rvesv,
            "RVESP": rvesp,
            "RVEDP": rvedp,
            "aortic_flow": 4.0 + 2.5 * r + n(),
            "pulmonary_flow": 4.0 + 2.3 * r + n(),
        }


# Backward-compatible aliases used by run_demo / advanced scripts
SimulatorBridge = SimulatorAdapter
FunctionBridge = FunctionAdapter
MockBridge = MockSimulation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mna_bayesopt(
    simulate_episode: EpisodeFunction,
    *,
    n_beats: int = 1000,
    contraction_bounds: Tuple[float, float] = (0.1, 1.0),
    contraction_velocity_bounds: Tuple[float, float] = (0.05, 2.0),
    baseline_params: Tuple[float, float] = (0.0, 0.0),
    baseline_outputs: Optional[SimOutputs] = None,
    objective: Optional[Objective] = None,
    n_init: int = 5,
    n_iter: int = 25,
    kernel_name: str,
    seed: int = 42,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run episode-level BO for MNA contraction parameters.

    Each BO trial runs one full simulation episode at fixed
    ``(contraction, contraction_velocity)`` for ``n_beats`` beats.

    ``kernel_name`` is required on purpose so the GP kernel choice is explicit
    in every optimisation run. Common choices are "matern52", "matern32",
    "rbf", "rq", and "periodic".
    """
    from optimizer import BayesianOptimizer

    def wrapped_episode(contraction: float, contraction_velocity: float) -> SimOutputs:
        return simulate_episode(contraction, contraction_velocity, n_beats)

    if baseline_outputs is None:
        baseline_outputs = wrapped_episode(*baseline_params)

    if objective is None:
        baseline = BaselineProfile.from_outputs(baseline_outputs)
        objective = LiteratureObjective(baseline=baseline)

    adapter = FunctionAdapter(
        wrapped_episode,
        param_names=["contraction", "contraction_velocity"],
    )

    opt = BayesianOptimizer(
        bridge=adapter,
        objective=objective,
        param_bounds={
            "contraction": contraction_bounds,
            "contraction_velocity": contraction_velocity_bounds,
        },
        kernel=kernel_name,
        acquisition=CoolingUCB(budget=n_iter),
        seed=seed,
        verbose=verbose,
    )
    opt.run(n_init=n_init, n_iter=n_iter)

    if save_path is not None:
        opt.save(save_path)

    return {
        "best_params": opt.best_params,
        "best_outputs": opt.best_outputs,
        "best_value": opt.best_value,
        "baseline_outputs": baseline_outputs,
        "history": opt.history,
        "n_evaluations": opt.n_evaluations,
        "optimizer": opt,
    }


def make_script_episode_function(
    script_path: str | Path,
    *,
    python_executable: str | Path | None = None,
    extra_args: Optional[list[str]] = None,
    output_map: Optional[Dict[str, str]] = None,
) -> EpisodeFunction:
    """Build an episode function from a Python script.

    The script is called as:

        python script.py --contraction C --contraction-velocity CV --n-beats N

    It must print one JSON object to stdout with the haemodynamic outputs.
    """
    script_path = Path(script_path)
    executable = str(python_executable or sys.executable)
    extra_args = extra_args or []
    output_map = output_map or {}

    def run_script_episode(
        contraction: float,
        contraction_velocity: float,
        n_beats: int,
    ) -> SimOutputs:
        cmd = [
            executable,
            str(script_path),
            "--contraction",
            str(contraction),
            "--contraction-velocity",
            str(contraction_velocity),
            "--n-beats",
            str(n_beats),
            *extra_args,
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = json.loads(completed.stdout)
        if not isinstance(outputs, dict):
            raise TypeError("Simulation script must print a JSON object.")
        if output_map:
            outputs = {output_map.get(k, k): v for k, v in outputs.items()}
        return {k: float(v) for k, v in outputs.items()}

    return run_script_episode
