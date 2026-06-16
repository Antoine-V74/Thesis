"""
Demo: optimise MNA parameters using the mock cardiac simulation.

Run from this folder:
    python run_demo.py

Or from the project root:
    python BayesOpt/run_demo.py

What this script demonstrates
------------------------------
1. Connecting to a simulation via the api module.
2. Comparing four kernels by their log marginal likelihood.
3. Running the full BO loop with the winning kernel.
4. The suggest() / observe() manual loop (asynchronous pattern).
5. Saving results to JSON and warm-starting a second run.
6. Producing convergence and GP surface plots.

Replace MockSimulation with run_mna_bayesopt(your_sim, ...) to use your real model.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make BayesOpt importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquisition import CoolingUCB, WarmupThenEI, get_acquisition
from kernels import get_kernel, AVAILABLE_KERNELS
from objective import BaselineProfile, LiteratureObjective
from optimizer import BayesianOptimizer
from api import MockSimulation


# ---------------------------------------------------------------------------
# 0.  Define your simulation connection
# ---------------------------------------------------------------------------
# Option A — Use the built-in mock (runs without any real model):
bridge = MockSimulation(
    param_names=["contraction", "contraction_velocity"],
    noise_std=0.01,   # small noise to test robustness; set 0.0 for deterministic
    seed=42,
)

# Option B — Wrap your real Python simulation function:
# def my_lv_sim(contraction, contraction_velocity):
#     # ... call your model here ...
#     return {
#         "LVEDP": ..., "LVEDV": ...,
#         "LVESP": ..., "LVESV": ...,
#         "RVEDP": ..., "RVEDV": ...,
#         "RVESP": ..., "RVESV": ...,
#         "aortic_flow": ..., "pulmonary_flow": ...,
#     }
# adapter = FunctionAdapter(my_lv_sim, param_names=["contraction", "contraction_velocity"])


# ---------------------------------------------------------------------------
# 1.  Define the parameter search space
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "contraction":          (0.1, 1.0),   # MNA contraction level [0..1]
    "contraction_velocity": (0.05, 2.0),  # MNA contraction velocity [arb. units]
}


# ---------------------------------------------------------------------------
# 2.  Define the literature-based suitability score
# ---------------------------------------------------------------------------
baseline_outputs = bridge.run({"contraction": 0.0, "contraction_velocity": 0.0})
baseline = BaselineProfile.from_outputs(baseline_outputs)
objective = LiteratureObjective(baseline=baseline)


# ---------------------------------------------------------------------------
# 3.  Kernel comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Step 1: Kernel comparison on warm-up data")
print("=" * 60)

N_INIT_COMPARISON = 8
kernel_lmls: dict[str, float] = {}

for kernel_name in AVAILABLE_KERNELS:
    _opt = BayesianOptimizer(
        bridge=bridge,
        objective=objective,
        param_bounds=PARAM_BOUNDS,
        kernel=get_kernel(kernel_name),
        acquisition=get_acquisition("ucb", beta=2.0),
        seed=42,
        verbose=False,
    )
    _opt.run(n_init=N_INIT_COMPARISON, n_iter=0, verbose=False)
    lml = _opt.gp.log_marginal_likelihood if _opt.gp._fitted else float("nan")
    kernel_lmls[kernel_name] = lml
    print(f"  kernel={kernel_name:<12}  LML={lml:+.3f}")

best_kernel_name = max(kernel_lmls, key=lambda k: kernel_lmls[k])
print(f"\n  Best kernel by LML: {best_kernel_name!r}")


# ---------------------------------------------------------------------------
# 4.  Full optimisation run with the best kernel
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Step 2: Full BO run")
print("=" * 60)

N_INIT = 5
N_ITER = 25
BUDGET = N_ITER   # used to schedule CoolingUCB

opt = BayesianOptimizer(
    bridge=bridge,
    objective=objective,
    param_bounds=PARAM_BOUNDS,
    kernel=get_kernel(best_kernel_name),
    acquisition=CoolingUCB(budget=BUDGET, beta_0=2.0, beta_min=0.1, gamma=1.0),
    n_restarts_gp=5,
    seed=0,
    verbose=True,
)

opt.run(n_init=N_INIT, n_iter=N_ITER)

print("\nFitted GP hyperparameters:")
for name, val in opt.gp_hyperparameters().items():
    print(f"  {name}: {val:.4f}")


# ---------------------------------------------------------------------------
# 5.  Demonstrate manual suggest / observe loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Step 3: Manual suggest -> evaluate -> observe loop")
print("=" * 60)

for step in range(3):
    params = opt.suggest()
    print(f"  Suggested params: {params}")
    # In real use: outputs = my_lv_sim(**params)
    outputs = bridge.run(params)
    opt.observe(params, outputs)
    print(f"  Observed aortic_flow={outputs.get('aortic_flow', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# 6.  Save history and demonstrate warm-start
# ---------------------------------------------------------------------------
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

save_path = results_dir / "demo_run.json"
opt.save(save_path)

print("\n" + "=" * 60)
print("  Step 4: Warm-start continuation from saved history")
print("=" * 60)

opt2 = BayesianOptimizer(
    bridge=bridge,
    objective=objective,
    param_bounds=PARAM_BOUNDS,
    kernel=get_kernel(best_kernel_name),
    acquisition=CoolingUCB(budget=10, beta_0=1.0, beta_min=0.05),
    seed=99,
    verbose=True,
)
opt2.load_history(save_path)
opt2.run(n_init=0, n_iter=5)


# ---------------------------------------------------------------------------
# 7.  Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; change to "TkAgg" for display
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent / "tools"))
    from plot_convergence import (
        plot_convergence,
        plot_gp_surface,
        plot_haemodynamics,
        plot_kernel_comparison,
    )

    fig1 = plot_convergence(opt, title="Demo - Convergence")
    fig1.savefig(results_dir / "convergence.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved convergence plot -> {results_dir / 'convergence.png'}")

    fig2 = plot_gp_surface(
        opt,
        param_x="contraction",
        param_y="contraction_velocity",
        resolution=40,
        title="Demo - GP Posterior Surface",
    )
    fig2.savefig(results_dir / "gp_surface.png", dpi=150, bbox_inches="tight")
    print(f"  Saved GP surface plot -> {results_dir / 'gp_surface.png'}")

    fig3 = plot_haemodynamics(
        opt,
        keys=["aortic_flow", "LVEDP", "LVESP", "LVESV", "pulmonary_flow"],
        title="Demo - Haemodynamic Outputs",
    )
    fig3.savefig(results_dir / "haemodynamics.png", dpi=150, bbox_inches="tight")
    print(f"  Saved haemodynamics plot -> {results_dir / 'haemodynamics.png'}")

    fig4 = plot_kernel_comparison(kernel_lmls)
    fig4.savefig(results_dir / "kernel_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved kernel comparison -> {results_dir / 'kernel_comparison.png'}")

    plt.close("all")

except ImportError:
    print("\n  [info] matplotlib not found; skipping plots.")

print("\n  Done.")
