"""
Kernel zoo for the Gaussian Process surrogate.

Available kernels
-----------------
rbf         RBF / Squared Exponential  — smooth, infinitely differentiable.
            Best when the response surface is very smooth.
matern32    Matérn ν=3/2               — once differentiable.
            More realistic for physical/mechanical systems than RBF.
matern52    Matérn ν=5/2               — twice differentiable.  [recommended default]
            Slightly smoother than Matérn 3/2; good balance for cardiac dynamics.
rq          Rational Quadratic          — mixture of RBF kernels at different scales.
            Useful when the response has features at multiple length scales.
periodic    Periodic (Exp-Sine-Squared) — for cyclic / heart-rate-periodic functions.
            Combine with RBF: periodic * RBF to model "periodic with decay".

How to pick a kernel
--------------------
Start with matern52. If the surrogate under-fits, try matern32 (rougher surface)
or rbf (smoother). The marginal likelihood after fit tells you which kernel
describes the data best — see GPSurrogate.log_marginal_likelihood.

Hyperparameter tuning (what each parameter controls)
------------------------------------------------------
All kernels are multiplied by a ConstantKernel (signal variance, σ²_f) and
a WhiteKernel (observation noise, σ²_n):

    k_total = σ²_f · k_core(x, x'; θ) + σ²_n

signal_variance (σ²_f):
    Overall amplitude of function variation.
    If too small → GP predicts near-zero everywhere.
    If too large → GP overfits / oscillates.
    Tuned automatically by LML; typical range [0.01, 100].

length_scale (l):
    How quickly the function changes along each input dimension.
    Small l → very wiggly function (more expressive but may overfit).
    Large l → very smooth function (may under-fit).
    Tuned automatically; set bounds to a physically reasonable range.
    If your parameters span [0, 1] after normalisation, l ∈ [0.1, 3] is typical.

noise_level (σ²_n):
    Observation noise.  Set this to the expected noise variance of your simulator.
    A deterministic simulator → set noise_level_bounds=(1e-10, 1e-5) (near-zero).
    A noisy/stochastic simulator → allow noise_level_bounds=(1e-4, 1.0).

Matérn ν (nu): fixed; choose the class, not a tunable parameter.

RQ alpha: shape parameter mixing short and long-range components.

Periodic periodicity: the expected period of the cyclic pattern.

Composite kernels (via sklearn operators)
-----------------------------------------
    from kernels import get_kernel
    from sklearn.gaussian_process.kernels import RBF
    k1 = get_kernel("matern52")
    k2 = get_kernel("periodic")
    k_sum  = k1 + k2   # captures trend + oscillation
    k_prod = k1 * k2   # captures locally-periodic functions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    Kernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


@dataclass
class KernelConfig:
    """Typed configuration for a named kernel + noise model.

    All hyperparameter bounds are in the *original* (not log) domain.
    sklearn internally converts to log-scale for optimisation.
    """
    name: str
    # amplitude
    signal_variance: float = 1.0
    signal_variance_bounds: Tuple[float, float] = (1e-3, 1e3)
    # length scale (shared for all dimensions; anisotropic not yet exposed)
    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-3, 1e3)
    # observation noise
    noise_level: float = 1e-4
    noise_level_bounds: Tuple[float, float] = (1e-8, 1e-1)
    # RationalQuadratic only
    alpha_rq: float = 1.0
    alpha_rq_bounds: Tuple[float, float] = (1e-3, 1e3)
    # Periodic only
    periodicity: float = 1.0
    periodicity_bounds: Tuple[float, float] = (1e-2, 1e2)


def build_kernel(cfg: KernelConfig) -> Kernel:
    """Assemble a full sklearn Kernel from a KernelConfig.

    Structure: ConstantKernel * core_kernel + WhiteKernel
    """
    amplitude = ConstantKernel(
        constant_value=cfg.signal_variance,
        constant_value_bounds=cfg.signal_variance_bounds,
    )
    noise = WhiteKernel(
        noise_level=cfg.noise_level,
        noise_level_bounds=cfg.noise_level_bounds,
    )

    name = cfg.name.lower().replace("-", "").replace("_", "").replace(" ", "")

    if name in ("rbf", "squaredexponential", "se", "gaussianrbf"):
        core = RBF(
            length_scale=cfg.length_scale,
            length_scale_bounds=cfg.length_scale_bounds,
        )
    elif name in ("matern32", "matern15", "mat32"):
        core = Matern(
            length_scale=cfg.length_scale,
            length_scale_bounds=cfg.length_scale_bounds,
            nu=1.5,
        )
    elif name in ("matern52", "matern25", "mat52", "matern"):
        core = Matern(
            length_scale=cfg.length_scale,
            length_scale_bounds=cfg.length_scale_bounds,
            nu=2.5,
        )
    elif name in ("rq", "rationalquadratic", "rational"):
        core = RationalQuadratic(
            length_scale=cfg.length_scale,
            length_scale_bounds=cfg.length_scale_bounds,
            alpha=cfg.alpha_rq,
            alpha_bounds=cfg.alpha_rq_bounds,
        )
    elif name in ("periodic", "expsinesquared", "exp_sine", "expsine"):
        core = ExpSineSquared(
            length_scale=cfg.length_scale,
            length_scale_bounds=cfg.length_scale_bounds,
            periodicity=cfg.periodicity,
            periodicity_bounds=cfg.periodicity_bounds,
        )
    else:
        raise ValueError(
            f"Unknown kernel name: {cfg.name!r}. "
            f"Choose from: {AVAILABLE_KERNELS}"
        )

    return amplitude * core + noise


def get_kernel(
    name: str = "matern52",
    *,
    signal_variance: float = 1.0,
    signal_variance_bounds: Tuple[float, float] = (1e-3, 1e3),
    length_scale: float = 1.0,
    length_scale_bounds: Tuple[float, float] = (1e-3, 1e3),
    noise_level: float = 1e-4,
    noise_level_bounds: Tuple[float, float] = (1e-8, 1e-1),
    **kwargs,
) -> Kernel:
    """Convenience factory — get a fully configured kernel by name.

    Parameters
    ----------
    name                  : kernel name (see AVAILABLE_KERNELS)
    signal_variance       : initial σ²_f (tuned by LML)
    signal_variance_bounds: search range for σ²_f
    length_scale          : initial length scale (tuned by LML)
    length_scale_bounds   : search range for length scale
    noise_level           : initial σ²_n (tuned by LML)
    noise_level_bounds    : search range for σ²_n; tighten for near-deterministic sims
    **kwargs              : forwarded to KernelConfig (e.g. periodicity, alpha_rq)
    """
    cfg = KernelConfig(
        name=name,
        signal_variance=signal_variance,
        signal_variance_bounds=signal_variance_bounds,
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        noise_level=noise_level,
        noise_level_bounds=noise_level_bounds,
        **kwargs,
    )
    return build_kernel(cfg)


AVAILABLE_KERNELS: Tuple[str, ...] = (
    "rbf",
    "matern32",
    "matern52",   # recommended default
    "rq",
    "periodic",
)
