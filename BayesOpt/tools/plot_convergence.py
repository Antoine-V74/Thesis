"""
Visualisation tools for Bayesian optimisation results.

Functions
---------
plot_convergence      Objective value and best-so-far over iterations.
plot_gp_surface       2-D GP posterior mean + uncertainty (2-parameter problems).
plot_haemodynamics    Haemodynamic outputs over iterations.
plot_kernel_comparison Compare LML across multiple kernel/run combinations.

All functions return a matplotlib Figure.  Call plt.show() or fig.savefig()
after calling them.

Usage
-----
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from tools.plot_convergence import plot_convergence, plot_gp_surface

    fig = plot_convergence(opt)
    fig.savefig("convergence.png", dpi=150)

    fig = plot_gp_surface(opt, param_x="contraction",
                          param_y="contraction_velocity", resolution=60)
    fig.savefig("gp_surface.png", dpi=150)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    opt,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Bayesian Optimisation Convergence",
) -> "plt.Figure":
    """Plot objective value per iteration and running best.

    Parameters
    ----------
    opt : BayesianOptimizer instance after calling .run()

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    history = opt.history
    iterations = [e["iteration"] for e in history]
    objectives = [e["objective"] for e in history]
    best_so_far = [e["best_so_far"] for e in history]
    labels = [e["label"] for e in history]

    # Colour-code init vs BO iterations
    colours = [
        "#a8c8f0" if "init" in lbl else "#f4a460"
        for lbl in labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Left: per-iteration value
    ax = axes[0]
    ax.bar(iterations, objectives, color=colours, edgecolor="white", linewidth=0.5)
    ax.plot(iterations, best_so_far, color="#d62728", linewidth=2,
            label="Best so far", zorder=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("Objective per iteration")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Colour legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#a8c8f0", label="Random init"),
        Patch(facecolor="#f4a460", label="BO iteration"),
    ]
    ax.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="Best so far")
    ], fontsize=8)

    # Right: best-so-far convergence
    ax2 = axes[1]
    ax2.plot(iterations, best_so_far, color="#d62728", linewidth=2.5, marker="o",
             markersize=4)
    ax2.fill_between(iterations, min(best_so_far) * 0.98, best_so_far,
                     alpha=0.15, color="#d62728")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best objective so far")
    ax2.set_title("Convergence curve")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GP posterior surface (2-parameter problems only)
# ---------------------------------------------------------------------------

def plot_gp_surface(
    opt,
    param_x: str = "contraction",
    param_y: str = "contraction_velocity",
    resolution: int = 50,
    figsize: Tuple[float, float] = (14, 4),
    title: str = "GP Posterior",
) -> "plt.Figure":
    """Plot the GP mean, standard deviation, and acquisition surface.

    Only works for problems with exactly 2 parameters.  For higher-dimensional
    problems, fix all other parameters at their best-so-far value.

    Parameters
    ----------
    opt        : fitted BayesianOptimizer
    param_x    : name of the x-axis parameter
    param_y    : name of the y-axis parameter
    resolution : grid resolution per axis
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    if not opt.gp._fitted:
        raise RuntimeError("GP not fitted yet. Run opt.run() first.")

    bounds = opt.param_bounds
    px = param_x
    py = param_y
    bx = bounds[px]
    by = bounds[py]

    xv = np.linspace(bx[0], bx[1], resolution)
    yv = np.linspace(by[0], by[1], resolution)
    XX, YY = np.meshgrid(xv, yv)

    # Build full parameter array (fix other dims at best-so-far)
    best = opt.best_params
    n_pts = resolution * resolution
    X_grid = np.zeros((n_pts, len(opt.param_names)))
    for i, name in enumerate(opt.param_names):
        if name == px:
            X_grid[:, i] = XX.ravel()
        elif name == py:
            X_grid[:, i] = YY.ravel()
        else:
            X_grid[:, i] = best.get(name, 0.5)

    mu, sigma = opt.gp.predict(X_grid)
    MU = mu.reshape(resolution, resolution)
    SG = sigma.reshape(resolution, resolution)

    # Acquisition values
    acq_vals = opt.acquisition(X_grid, opt.gp, opt.best_value)
    ACQ = acq_vals.reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    obs_x = [e["params"][px] for e in opt.history]
    obs_y_ax = [e["params"][py] for e in opt.history]
    obs_obj = [e["objective"] for e in opt.history]

    def _scatter(ax):
        sc = ax.scatter(obs_x, obs_y_ax, c=obs_obj, cmap="RdYlGn",
                        edgecolors="k", linewidths=0.5, s=40, zorder=10)
        if opt.best_params:
            ax.scatter([opt.best_params[px]], [opt.best_params[py]],
                       marker="*", s=200, color="gold", edgecolors="k",
                       linewidths=0.8, zorder=11, label="Best")
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        return sc

    # GP mean
    ax = axes[0]
    im = ax.pcolormesh(XX, YY, MU, cmap="RdYlGn", shading="auto")
    ax.contour(XX, YY, MU, levels=8, colors="k", linewidths=0.4, alpha=0.4)
    fig.colorbar(im, ax=ax, label="GP mean")
    _scatter(ax)
    ax.set_title("Posterior mean μ(x)")

    # GP uncertainty
    ax = axes[1]
    im2 = ax.pcolormesh(XX, YY, SG, cmap="Blues", shading="auto")
    fig.colorbar(im2, ax=ax, label="GP std")
    _scatter(ax)
    ax.set_title("Posterior std σ(x)")

    # Acquisition function
    ax = axes[2]
    im3 = ax.pcolormesh(XX, YY, ACQ, cmap="plasma", shading="auto")
    fig.colorbar(im3, ax=ax, label="Acquisition")
    _scatter(ax)
    ax.set_title(f"Acquisition: {opt.acquisition.__class__.__name__}")

    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Haemodynamic output evolution
# ---------------------------------------------------------------------------

def plot_haemodynamics(
    opt,
    keys: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Haemodynamic Outputs Over Iterations",
) -> "plt.Figure":
    """Plot how each haemodynamic output evolves across iterations.

    Parameters
    ----------
    opt  : BayesianOptimizer instance
    keys : list of output keys to plot.  If None, all found keys are plotted.
    """
    import matplotlib.pyplot as plt

    history = opt.history
    iterations = [e["iteration"] for e in history]

    all_keys = list({k for e in history for k in e["outputs"]})
    if keys is None:
        keys = sorted(all_keys)
    else:
        keys = [k for k in keys if k in all_keys]

    if not keys:
        raise ValueError("No haemodynamic output keys found in history.")

    n_keys = len(keys)
    ncols = min(n_keys, 3)
    nrows = (n_keys + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Best iteration marker
    best_iter = int(np.argmax([e["objective"] for e in history]))

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        vals = [e["outputs"].get(key, float("nan")) for e in history]
        colours = ["#a8c8f0" if "init" in e["label"] else "#f4a460" for e in history]
        ax.bar(iterations, vals, color=colours, edgecolor="white", linewidth=0.4)
        ax.axvline(best_iter, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"Best (iter {best_iter})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    # Hide empty subplots
    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Kernel comparison (LML bar chart)
# ---------------------------------------------------------------------------

def plot_kernel_comparison(
    results: Dict[str, float],
    figsize: Tuple[float, float] = (7, 4),
    title: str = "Kernel Comparison — Log Marginal Likelihood",
) -> "plt.Figure":
    """Bar chart comparing log marginal likelihood across kernels.

    Parameters
    ----------
    results : {kernel_name: log_marginal_likelihood}
              Build with: {name: opt.gp.log_marginal_likelihood for name, opt in runs}

    Example
    -------
        from tools.plot_convergence import plot_kernel_comparison
        lmls = {"matern52": -12.3, "rbf": -15.1, "matern32": -13.8, "rq": -14.0}
        fig = plot_kernel_comparison(lmls)
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    lmls = [results[k] for k in names]
    best_idx = int(np.argmax(lmls))
    colours = ["#2ca02c" if i == best_idx else "#1f77b4" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, lmls, color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Log marginal likelihood (higher = better fit)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
