"""
Simple haemodynamic score function for LV + MNA Bayesian optimization.

The optimizer needs one scalar score: higher is better. This file converts one
simulation output dictionary into that score while enforcing basic safety limits.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SimOutputs = Dict[str, float]


@dataclass
class Baseline:
    """Reference outputs from the unassisted or nominal simulation."""

    stroke_volume: float
    lvedp: float
    aortic_flow: Optional[float] = None
    aortic_pressure: Optional[float] = None
    lvesp: Optional[float] = None
    pulmonary_flow: Optional[float] = None

    @classmethod
    def from_outputs(cls, outputs: SimOutputs) -> "Baseline":
        return cls(
            stroke_volume=stroke_volume(outputs) or 0.0,
            lvedp=float(outputs["LVEDP"]),
            aortic_flow=finite(outputs.get("aortic_flow")),
            aortic_pressure=finite(outputs.get("aortic_pressure")),
            lvesp=finite(outputs.get("LVESP")),
            pulmonary_flow=finite(outputs.get("pulmonary_flow")),
        )

    @property
    def sv_lv(self) -> float:
        """Backward-compatible name used by older scripts."""
        return self.stroke_volume


class ScoreFunction(ABC):
    """Maps simulation outputs to a scalar score."""

    @abstractmethod
    def __call__(self, outputs: SimOutputs) -> float:
        """Return a score to maximize."""


@dataclass
class SafetyLimits:
    """Hard limits. Failed trials receive ``-inf`` and are not used by the GP."""

    baseline: Baseline
    max_lvedp_rise: Optional[float] = 2.0
    max_lvedp: Optional[float] = None
    min_ef: Optional[float] = None
    max_flow_mismatch: Optional[float] = 0.25
    max_aortic_pressure: Optional[float] = None
    require_sv_gain: bool = False
    require_flow_gain: bool = False

    def check(self, outputs: SimOutputs) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        sv = stroke_volume(outputs)
        ef = ejection_fraction(outputs)
        lvedp = finite(outputs.get("LVEDP"))
        flow = finite(outputs.get("aortic_flow"))
        pulmonary_flow = finite(outputs.get("pulmonary_flow"))
        aortic_pressure = finite(outputs.get("aortic_pressure"))

        if lvedp is None:
            violations.append("missing_lvedp")
        else:
            if self.max_lvedp_rise is not None:
                if lvedp > self.baseline.lvedp + self.max_lvedp_rise:
                    violations.append("lvedp_too_high_vs_baseline")
            if self.max_lvedp is not None and lvedp >= self.max_lvedp:
                violations.append("lvedp_too_high")

        if self.min_ef is not None and (ef is None or ef < self.min_ef):
            violations.append("ef_too_low")

        if self.max_aortic_pressure is not None:
            if aortic_pressure is None or aortic_pressure > self.max_aortic_pressure:
                violations.append("aortic_pressure_too_high")

        if self.require_sv_gain and (
            sv is None or sv <= self.baseline.stroke_volume
        ):
            violations.append("stroke_volume_not_improved")

        if self.require_flow_gain and self.baseline.aortic_flow is not None:
            if flow is None or flow <= self.baseline.aortic_flow:
                violations.append("aortic_flow_not_improved")

        if (
            self.max_flow_mismatch is not None
            and flow is not None
            and pulmonary_flow is not None
        ):
            ref = max(abs(self.baseline.aortic_flow or flow), 1e-9)
            mismatch = abs(flow - pulmonary_flow) / ref
            if mismatch > self.max_flow_mismatch:
                violations.append("flow_mismatch")

        return len(violations) == 0, violations


class HemodynamicScore(ScoreFunction):
    """Weighted compromise between performance and safety."""

    def __init__(
        self,
        baseline: Baseline,
        limits: Optional[SafetyLimits] = None,
        *,
        w_sv: float = 2.0,
        w_flow: float = 1.5,
        w_aortic_pressure: float = 0.5,
        w_work: float = 0.25,
        lvedp_target: float = 12.0,
        lvedp_penalty: float = 0.8,
        rvedp_target: float = 8.0,
        rvedp_penalty: float = 0.4,
        **legacy_kwargs: float,
    ) -> None:
        # Keep older notebooks usable while the public names become simpler.
        limits = legacy_kwargs.pop("safety_filter", limits)  # type: ignore[assignment]
        w_flow = float(legacy_kwargs.pop("w_aortic_flow", w_flow))
        w_sv = float(legacy_kwargs.pop("w_stroke", w_sv))
        w_work = float(legacy_kwargs.pop("w_stroke_work", w_work))
        lvedp_target = float(legacy_kwargs.pop("lvedp_soft_limit", lvedp_target))
        rvedp_target = float(legacy_kwargs.pop("rvedp_soft_limit", rvedp_target))

        self.baseline = baseline
        self.limits = limits or SafetyLimits(baseline=baseline)
        self.w_sv = w_sv
        self.w_flow = w_flow
        self.w_aortic_pressure = w_aortic_pressure
        self.w_work = w_work
        self.lvedp_target = lvedp_target
        self.lvedp_penalty = lvedp_penalty
        self.rvedp_target = rvedp_target
        self.rvedp_penalty = rvedp_penalty
        self._baseline_work = pressure_work_proxy(
            baseline.lvesp,
            baseline.lvedp,
            baseline.stroke_volume,
        )

    def safety_check(self, outputs: SimOutputs) -> Tuple[bool, List[str]]:
        return self.limits.check(outputs)

    def __call__(self, outputs: SimOutputs) -> float:
        safe, _ = self.safety_check(outputs)
        if not safe:
            return float("-inf")

        sv = stroke_volume(outputs)
        lvedp = finite(outputs.get("LVEDP"))
        if sv is None or lvedp is None:
            return float("-inf")

        score = self.w_sv * ratio(sv, self.baseline.stroke_volume)

        flow = finite(outputs.get("aortic_flow"))
        if flow is not None and self.baseline.aortic_flow is not None:
            score += self.w_flow * ratio(flow, self.baseline.aortic_flow)

        aortic_pressure = finite(outputs.get("aortic_pressure"))
        if aortic_pressure is not None and self.baseline.aortic_pressure is not None:
            score += self.w_aortic_pressure * ratio(
                aortic_pressure,
                self.baseline.aortic_pressure,
            )

        work = pressure_work_proxy(finite(outputs.get("LVESP")), lvedp, sv)
        if work is not None and self._baseline_work is not None:
            score += self.w_work * ratio(work, self._baseline_work)

        score -= self.lvedp_penalty * max(0.0, lvedp - self.lvedp_target) ** 2

        rvedp = finite(outputs.get("RVEDP"))
        if rvedp is not None:
            score -= self.rvedp_penalty * max(0.0, rvedp - self.rvedp_target) ** 2

        return score

    def __repr__(self) -> str:
        return (
            "HemodynamicScore("
            f"baseline_sv={self.baseline.stroke_volume:.3f}, "
            f"baseline_lvedp={self.baseline.lvedp:.2f})"
        )


def finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def ratio(value: float, reference: float) -> float:
    return value / max(abs(reference), 1e-9)


def stroke_volume(outputs: SimOutputs) -> Optional[float]:
    lvedv = finite(outputs.get("LVEDV"))
    lvesv = finite(outputs.get("LVESV"))
    if lvedv is None or lvesv is None:
        return None
    return lvedv - lvesv


def ejection_fraction(outputs: SimOutputs) -> Optional[float]:
    lvedv = finite(outputs.get("LVEDV"))
    sv = stroke_volume(outputs)
    if lvedv is None or sv is None or lvedv <= 0:
        return None
    return sv / lvedv


def pressure_work_proxy(
    systolic_pressure: Optional[float],
    filling_pressure: Optional[float],
    sv: Optional[float],
) -> Optional[float]:
    if systolic_pressure is None or filling_pressure is None or sv is None:
        return None
    work = (systolic_pressure - filling_pressure) * sv
    return work if work > 0 else None


class LiteratureSafetyFilter(SafetyLimits):
    """Backward-compatible wrapper for the previous class name."""

    def __init__(
        self,
        baseline: Baseline,
        require_aortic_flow_improvement: bool = False,
        require_sv_improvement: bool = False,
        lvedp_margin_mmHg: Optional[float] = 2.0,
        lvedp_absolute_max: Optional[float] = None,
        min_ef: Optional[float] = None,
        max_flow_imbalance_ratio: Optional[float] = 0.25,
    ) -> None:
        super().__init__(
            baseline=baseline,
            max_lvedp_rise=lvedp_margin_mmHg,
            max_lvedp=lvedp_absolute_max,
            min_ef=min_ef,
            max_flow_mismatch=max_flow_imbalance_ratio,
            require_sv_gain=require_sv_improvement,
            require_flow_gain=require_aortic_flow_improvement,
        )

