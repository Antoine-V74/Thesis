"""
Minimal BayesOpt demo using the built-in mock simulation.

Run from the repository root:
    python BayesOpt/run_demo.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # noqa: BLE001
    pass

from api import MockSimulation, run_bayesopt


def main() -> None:
    bridge = MockSimulation(noise_std=0.01, seed=42)

    def simulate_episode(
        n_beats: int,
        **params: float,
    ) -> dict[str, float]:
        return bridge.run(params)

    results = run_bayesopt(
        simulate_episode=simulate_episode,
        param_bounds={
            "contraction": (0.1, 1.0),
            "contraction_velocity": (0.05, 2.0),
        },
        baseline_params={
            "contraction": 0.0,
            "contraction_velocity": 0.0,
        },
        n_beats=1000,
        n_init=5,
        n_iter=20,
        weights={
            "w_sv": 2.0,
            "w_flow": 1.5,
            "w_aortic_pressure": 0.5,
        },
        save_path=Path(__file__).parent / "results" / "demo_run.json",
        verbose=True,
    )

    print("\nBest parameters:")
    print(results["best_params"])
    print("\nBest outputs:")
    print(results["best_outputs"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(Path(__file__).parent / "tools"))
        from plot_convergence import plot_convergence

        fig = plot_convergence(results["optimizer"], title="BayesOpt demo")
        out_path = Path(__file__).parent / "results" / "convergence.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved convergence plot -> {out_path}")
    except ImportError:
        print("\nMatplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
