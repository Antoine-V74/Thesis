"""
Small public API for Bayesian optimization of MNA/stimulation parameters.

Most users provide one function:

    def simulate_episode(n_beats, **params):
        return {"LVEDP": ..., "LVEDV": ..., "LVESV": ..., ...}

and then call:

    results = run_bayesopt(simulate_episode, param_bounds={...}, n_beats=10000)
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
from score_function import Baseline, HemodynamicScore, SafetyLimits, ScoreFunction


SimParams = Dict[str, float]
SimOutputs = Dict[str, float]
EpisodeFunction = Callable[[float, float, int], SimOutputs]
GenericEpisodeFunction = Callable[..., SimOutputs]


class SimulatorAdapter(ABC):
    """Small interface between the optimizer and a simulation."""

    @abstractmethod
    def run(self, params: SimParams) -> SimOutputs:
        """Run the simulation and return output variables."""

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
    """Wrap a normal Python function as a simulation."""

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
            result = self.fn(**{k: params[k] for k in self.param_names})

        if not isinstance(result, dict):
            raise TypeError(
                f"Simulation function must return a dict, got {type(result).__name__}"
            )

        if self.output_map:
            result = {self.output_map.get(k, k): v for k, v in result.items()}

        return {k: float(v) for k, v in result.items()}


class MockSimulation(SimulatorAdapter):
    """Small synthetic LV + MNA response used for smoke tests and demos."""

    def __init__(
        self,
        param_names: Optional[List[str]] = None,
        noise_std: float = 0.02,
        seed: Optional[int] = 42,
    ) -> None:
        import numpy as np

        self.param_names = param_names or ["contraction", "contraction_velocity"]
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def run(self, params: SimParams) -> SimOutputs:
        import numpy as np

        c = float(params.get(self.param_names[0], 0.5))
        cv = float(params.get(self.param_names[1], 0.5))
        response = float(np.exp(-((c - 0.65) ** 2 / 0.05 + (cv - 0.55) ** 2 / 0.03)))

        def noise() -> float:
            return float(self._rng.normal(0.0, self.noise_std))

        return {
            "LVEDV": 110.0 + 30.0 * (1.0 - c) + noise(),
            "LVESV": 40.0 - 20.0 * response + noise(),
            "LVESP": 80.0 + 40.0 * response + noise(),
            "LVEDP": 6.0 + 8.0 * (1.0 - response) + noise(),
            "RVEDP": 4.0 + 5.0 * (1.0 - response) + noise(),
            "aortic_flow": 4.0 + 2.5 * response + noise(),
            "aortic_pressure": 80.0 + 35.0 * response + noise(),
            "pulmonary_flow": 4.0 + 2.3 * response + noise(),
        }


def run_bayesopt(
    simulate_episode: GenericEpisodeFunction,
    *,
    param_bounds: Dict[str, Tuple[float, float]],
    n_beats: int = 1000,
    baseline_params: Optional[SimParams] = None,
    baseline_outputs: Optional[SimOutputs] = None,
    score_function: Optional[ScoreFunction] = None,
    limits: Optional[SafetyLimits] = None,
    weights: Optional[Dict[str, float]] = None,
    accepts_dict: bool = False,
    n_init: int = 5,
    n_iter: int = 25,
    kernel_name: str = "matern52",
    seed: int = 42,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run BO on any number of named parameters."""
    from optimizer import BayesianOptimizer

    if not param_bounds:
        raise ValueError("param_bounds must contain at least one parameter.")

    param_names = list(param_bounds)
    _validate_param_bounds(param_bounds)
    if baseline_params is not None:
        _validate_param_values(baseline_params, param_names, label="baseline_params")

    def wrapped_episode(**params: float) -> SimOutputs:
        if accepts_dict:
            return simulate_episode(params, n_beats)
        return simulate_episode(n_beats=n_beats, **params)

    if baseline_outputs is None and score_function is None:
        if baseline_params is None:
            raise ValueError(
                "baseline_params is required when BayesOpt builds the default "
                "HemodynamicScore. Pass the unassisted/nominal parameter "
                "values explicitly."
            )
        baseline_outputs = wrapped_episode(**baseline_params)

    if score_function is None:
        if baseline_outputs is None:
            raise ValueError(
                "baseline_outputs is required when score_function is not provided."
            )
        baseline = Baseline.from_outputs(baseline_outputs)
        score_function = HemodynamicScore(
            baseline=baseline,
            limits=limits,
            **(weights or {}),
        )

    adapter = FunctionAdapter(
        wrapped_episode,
        param_names=param_names,
    )

    opt = BayesianOptimizer(
        bridge=adapter,
        score_function=score_function,
        param_bounds=param_bounds,
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


def _validate_param_bounds(param_bounds: Dict[str, Tuple[float, float]]) -> None:
    for name, bounds in param_bounds.items():
        if len(bounds) != 2:
            raise ValueError(f"Bounds for {name!r} must be a (low, high) pair.")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise ValueError(f"Bounds for {name!r} must satisfy low < high.")


def _validate_param_values(
    values: SimParams,
    param_names: List[str],
    *,
    label: str,
) -> None:
    missing = [name for name in param_names if name not in values]
    if missing:
        raise ValueError(f"{label} is missing values for: {missing}")


def run_mna_bayesopt(
    simulate_episode: EpisodeFunction,
    *,
    n_beats: int = 1000,
    contraction_bounds: Tuple[float, float] = (0.1, 1.0),
    contraction_velocity_bounds: Tuple[float, float] = (0.05, 2.0),
    baseline_params: Tuple[float, float] = (0.0, 0.0),
    baseline_outputs: Optional[SimOutputs] = None,
    score_function: Optional[ScoreFunction] = None,
    limits: Optional[SafetyLimits] = None,
    weights: Optional[Dict[str, float]] = None,
    n_init: int = 5,
    n_iter: int = 25,
    kernel_name: str = "matern52",
    seed: int = 42,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Backward-compatible wrapper for contraction and contraction_velocity."""

    def wrapped_mna_episode(
        *,
        contraction: float,
        contraction_velocity: float,
        n_beats: int,
    ) -> SimOutputs:
        return simulate_episode(contraction, contraction_velocity, n_beats)

    return run_bayesopt(
        wrapped_mna_episode,
        param_bounds={
            "contraction": contraction_bounds,
            "contraction_velocity": contraction_velocity_bounds,
        },
        n_beats=n_beats,
        baseline_params={
            "contraction": baseline_params[0],
            "contraction_velocity": baseline_params[1],
        },
        baseline_outputs=baseline_outputs,
        score_function=score_function,
        limits=limits,
        weights=weights,
        n_init=n_init,
        n_iter=n_iter,
        kernel_name=kernel_name,
        seed=seed,
        save_path=save_path,
        verbose=verbose,
    )


def make_script_episode_function(
    script_path: str | Path,
    *,
    param_names: Optional[List[str]] = None,
    python_executable: str | Path | None = None,
    extra_args: Optional[list[str]] = None,
    output_map: Optional[Dict[str, str]] = None,
) -> GenericEpisodeFunction:
    """Wrap a simulation script that prints one JSON output dictionary."""
    script_path = Path(script_path)
    executable = str(python_executable or sys.executable)
    param_names = param_names or ["contraction", "contraction_velocity"]
    extra_args = extra_args or []
    output_map = output_map or {}

    def run_script_episode(*args, **kwargs) -> SimOutputs:
        if len(args) == 3 and not kwargs:
            params = {
                param_names[0]: float(args[0]),
                param_names[1]: float(args[1]),
            }
            n_beats = int(args[2])
        else:
            n_beats = int(kwargs.pop("n_beats"))
            params = {name: float(kwargs[name]) for name in param_names}

        cmd = [
            executable,
            str(script_path),
            "--n-beats",
            str(n_beats),
            *extra_args,
        ]
        for name, value in params.items():
            cmd.extend([f"--{name.replace('_', '-')}", str(value)])
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = _parse_json_stdout(completed.stdout)
        if not isinstance(outputs, dict):
            raise TypeError("Simulation script must print a JSON object.")
        if output_map:
            outputs = {output_map.get(k, k): v for k, v in outputs.items()}
        return {k: float(v) for k, v in outputs.items()}

    return run_script_episode


def _parse_json_stdout(stdout: str) -> object:
    """Accept pure JSON stdout or a final JSON line after logs."""
    text = stdout.strip()
    if not text:
        raise ValueError("Simulation script printed no JSON output.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for line in reversed(text.splitlines()):
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            continue

    raise ValueError("Simulation script must print a JSON object.")


# Backward-compatible aliases for older scripts.
SimulatorBridge = SimulatorAdapter
FunctionBridge = FunctionAdapter
MockBridge = MockSimulation
