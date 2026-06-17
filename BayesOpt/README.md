# BayesOpt

## Motivation

The main purpose of Bayesian Optimization is to provide a mathematical framework
to find good MNA or stimulation parameters with as few simulation runs as
possible.

In short, we create a probabilistic model of the relationship:

```text
MNA / stimulation parameters -> haemodynamic performance score
```

This lets us choose the next set of parameters by balancing:

- exploration: try uncertain regions;
- exploitation: refine regions that already look good.

## Quick Start

The parameters used for now are:

- `contraction`
- `contraction_velocity`

Later, we can add things like:

- `stim_amplitude`
- `pulse_width`
- `stim_delay`
- `frequency`

The main function that runs the whole process is `run_bayesopt`.

```python
from api import run_bayesopt


def simulate_episode(n_beats, **params):
    # Replace this with the Abaqus + LV/MNA simulation call.
    contraction = params["contraction"]
    contraction_velocity = params["contraction_velocity"]

    return {
        "LVEDP": ...,
        "LVEDV": ...,
        "LVESV": ...,
        "LVESP": ...,
        "aortic_pressure": ...,
        "aortic_flow": ...,
        "pulmonary_flow": ...,
    }


results = run_bayesopt(
    simulate_episode,
    param_bounds={
        "contraction": (0.1, 1.0),
        "contraction_velocity": (0.05, 2.0),
    },
    baseline_params={
        "contraction": 0.0,
        "contraction_velocity": 0.0,
    },
    n_beats=10000,
    n_init=5,
    n_iter=25,
    save_path="BayesOpt/results/my_run.json",
)

print(results["best_params"])
print(results["best_outputs"])
```

`simulate_episode` can be any function from the simulation side. The only rule
is that it must return a dictionary with the output metrics.

## Important Detail

For now, each complete ECG/simulation episode gives one dictionary of summary
metrics. For example, `simulate_episode` can take the median or another summary
value of each output over the full run.

That dictionary is then passed into the `HemodynamicScore`, which returns one
scalar number. This scalar score represents how good that stimulation setting
was for that particular run and that particular parameter set.

## Baseline Parameters Matter

`baseline_params` corresponds to the case where the MNA or stimulation is not
active.

Example:

```python
baseline_params={
    "contraction": 0.0,
    "contraction_velocity": 0.0,
}
```

If the baseline simulation has already been run elsewhere, you can pass
`baseline_outputs` directly instead.

## Score Function

I tried to design a score function that makes sense as a first pass, but this
should definitely be improved with medical/literature input.

The current `HemodynamicScore` does this:

```text
reward stroke volume
+ reward aortic flow, if available
+ reward aortic pressure, if available
+ small reward for a pressure-work proxy, if available
- penalize high LVEDP
- penalize high RVEDP, if available
```

Hard safety limits are handled separately by `SafetyLimits`.

## Main Files

```text
BayesOpt/
|-- api.py                  simple entry point
|-- score_function.py       Baseline, SafetyLimits, HemodynamicScore
|-- optimizer.py            Bayesian optimization loop
|-- gp_surrogate.py         Gaussian Process wrapper
|-- acquisition.py          acquisition functions
|-- kernels.py              GP kernel choices
`-- run_demo.py             minimal mock example
```

## Useful Names

- `run_bayesopt`: generic API that takes any number of parameters as input.
- `Baseline`: output of the unassisted simulation.
- `SafetyLimits`: hard reject rules, for example excessive LVEDP.
- `HemodynamicScore`: weighted score used by the optimizer.
- `BayesianOptimizer`: lower-level optimizer class.

## Demo

```powershell
.\.venv\Scripts\python.exe BayesOpt\run_demo.py
```
