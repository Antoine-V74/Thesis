"""Plotting helpers for BayesOpt results."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_convergence(
    opt,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Bayesian Optimization Convergence",
) -> "plt.Figure":
    """Plot score per iteration and best score so far."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    history = opt.history
    iterations = [e["iteration"] for e in history]
    scores = [_score(e) for e in history]
    best_so_far = [e["best_so_far"] for e in history]
    labels = [e["label"] for e in history]
    colors = ["#a8c8f0" if "init" in label else "#f4a460" for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.bar(iterations, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.plot(iterations, best_so_far, color="#d62728", linewidth=2, zorder=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title("Score per iteration")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor="#a8c8f0", label="Random init"),
            Patch(facecolor="#f4a460", label="BO iteration"),
            plt.Line2D([0], [0], color="#d62728", linewidth=2, label="Best so far"),
        ],
        fontsize=8,
    )

    ax2 = axes[1]
    ax2.plot(iterations, best_so_far, color="#d62728", linewidth=2.5, marker="o")
    ax2.fill_between(iterations, min(best_so_far) * 0.98, best_so_far,
                     alpha=0.15, color="#d62728")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best score so far")
    ax2.set_title("Convergence curve")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_gp_surface(
    opt,
    param_x: str = "contraction",
    param_y: str = "contraction_velocity",
    resolution: int = 50,
    figsize: Tuple[float, float] = (14, 4),
    title: str = "GP Posterior",
) -> "plt.Figure":
    """Plot GP mean, uncertainty, and acquisition for two selected parameters."""
    import matplotlib.pyplot as plt

    if not opt.gp._fitted:
        raise RuntimeError("GP not fitted yet. Run opt.run() first.")

    bounds = opt.param_bounds
    bx = bounds[param_x]
    by = bounds[param_y]

    xv = np.linspace(bx[0], bx[1], resolution)
    yv = np.linspace(by[0], by[1], resolution)
    xx, yy = np.meshgrid(xv, yv)

    n_pts = resolution * resolution
    grid = np.zeros((n_pts, len(opt.param_names)))
    best = opt.best_params

    for i, name in enumerate(opt.param_names):
        if name == param_x:
            grid[:, i] = xx.ravel()
        elif name == param_y:
            grid[:, i] = yy.ravel()
        else:
            grid[:, i] = best.get(name, 0.5)

    mu, sigma = opt.gp.predict(grid)
    mean_grid = mu.reshape(resolution, resolution)
    std_grid = sigma.reshape(resolution, resolution)
    acq_grid = opt.acquisition(grid, opt.gp, opt.best_value).reshape(
        resolution,
        resolution,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    obs_x = [e["params"][param_x] for e in opt.history]
    obs_y = [e["params"][param_y] for e in opt.history]
    obs_scores = [_score(e) for e in opt.history]

    def scatter_observations(ax):
        sc = ax.scatter(
            obs_x,
            obs_y,
            c=obs_scores,
            cmap="RdYlGn",
            edgecolors="k",
            linewidths=0.5,
            s=40,
            zorder=10,
        )
        if opt.best_params:
            ax.scatter(
                [opt.best_params[param_x]],
                [opt.best_params[param_y]],
                marker="*",
                s=200,
                color="gold",
                edgecolors="k",
                linewidths=0.8,
                zorder=11,
                label="Best",
            )
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        return sc

    im = axes[0].pcolormesh(xx, yy, mean_grid, cmap="RdYlGn", shading="auto")
    axes[0].contour(xx, yy, mean_grid, levels=8, colors="k", linewidths=0.4, alpha=0.4)
    fig.colorbar(im, ax=axes[0], label="GP mean")
    scatter_observations(axes[0])
    axes[0].set_title("Posterior mean")

    im2 = axes[1].pcolormesh(xx, yy, std_grid, cmap="Blues", shading="auto")
    fig.colorbar(im2, ax=axes[1], label="GP std")
    scatter_observations(axes[1])
    axes[1].set_title("Posterior uncertainty")

    im3 = axes[2].pcolormesh(xx, yy, acq_grid, cmap="plasma", shading="auto")
    fig.colorbar(im3, ax=axes[2], label="Acquisition")
    scatter_observations(axes[2])
    axes[2].set_title(f"Acquisition: {opt.acquisition.__class__.__name__}")

    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


def plot_haemodynamics(
    opt,
    keys: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Hemodynamic Outputs Over Iterations",
) -> "plt.Figure":
    """Plot selected simulation outputs over iterations."""
    import matplotlib.pyplot as plt

    history = opt.history
    iterations = [e["iteration"] for e in history]
    all_keys = list({k for e in history for k in e["outputs"]})

    if keys is None:
        keys = sorted(all_keys)
    else:
        keys = [k for k in keys if k in all_keys]

    if not keys:
        raise ValueError("No hemodynamic output keys found in history.")

    n_keys = len(keys)
    ncols = min(n_keys, 3)
    nrows = (n_keys + ncols - 1) // ncols
    figsize = figsize or (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    best_iter = int(np.argmax([_score(e) for e in history]))

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        vals = [e["outputs"].get(key, float("nan")) for e in history]
        colors = ["#a8c8f0" if "init" in e["label"] else "#f4a460" for e in history]
        ax.bar(iterations, vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(best_iter, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"Best (iter {best_iter})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_kernel_comparison(
    results: Dict[str, float],
    figsize: Tuple[float, float] = (7, 4),
    title: str = "Kernel Comparison - Log Marginal Likelihood",
) -> "plt.Figure":
    """Bar chart comparing log marginal likelihood across kernels."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    lmls = [results[k] for k in names]
    best_idx = int(np.argmax(lmls))
    colors = ["#2ca02c" if i == best_idx else "#1f77b4" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, lmls, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Log marginal likelihood (higher = better fit)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _score(entry: Dict[str, float]) -> float:
    """Read score from a history entry."""
    return float(entry["score"])
