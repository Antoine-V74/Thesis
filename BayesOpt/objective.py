"""
Literature-based suitability scoring for MNA / cardiac simulation optimisation.

The simulation returns haemodynamic outputs from one episode. ``LiteratureObjective``
converts those outputs into one scalar suitability score (higher = better).

Expected simulation output keys
--------------------------------
    LVEDP, LVEDV, LVESP, LVESV
    RVEDP (optional soft penalty)
    aortic_flow, pulmonary_flow

Derived inside this module:
    SV_LV  = LVEDV - LVESV
    EF_LV  = SV_LV / LVEDV
    stroke_work_proxy = (LVESP - LVEDP) * SV_LV
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SimOutputs = Dict[str, float]

# Per-component literature / rationale notes used in thesis documentation.
COMPONENT_SOURCES: Dict[str, str] = {
    "aortic_flow_above_baseline": (
        "Moreira et al., J Thorac Cardiovasc Surg 1992; Caputo et al., ASAIO 2000; "
        "Frey et al., J Thorac Cardiovasc Surg 1993"
    ),
    "sv_lv_above_baseline": (
        "Moreira et al., 1992; Goldenberg et al., Eur J Cardiothorac Surg 1996; "
        "Chiu et al., Ann Biomed Eng 1997"
    ),
    "lvedp_vs_baseline": (
        "Caputo et al., ASAIO 2000; Goldenberg et al., 1996; Frey et al., 1993"
    ),
    "lvedp_absolute_max": (
        "Nagueh et al., ASE diastolic function update 2025"
    ),
    "min_ef": (
        "Common systolic dysfunction threshold used as conservative engineering limit; "
        "should be recalibrated for advanced HF / rat models"
    ),
    "flow_balance": (
        "Engineering circulation-consistency check inspired by balanced-flow assumptions "
        "in cardiovascular modelling"
    ),
    "aortic_flow_term": (
        "Moreira et al., 1992; Timm et al., Bioengineering 2014; Azarnoush et al., 2021"
    ),
    "stroke_volume_term": (
        "Moreira et al., 1992; Goldenberg et al., 1996"
    ),
    "stroke_work_term": (
        "Kass, CV Physiology (PV-loop stroke work); Moreira et al., 1992"
    ),
    "lvedp_soft_penalty": (
        "Caputo et al., 2000; Grosan et al., Ann Biomed Eng 2021"
    ),
    "rvedp_soft_penalty": (
        "Engineering extension for biventricular filling-pressure control"
    ),
}


@dataclass
class BaselineProfile:
    """Haemodynamic outputs from one unassisted baseline simulation episode."""

    aortic_flow: float
    sv_lv: float
    lvedp: float
    lvesp: Optional[float] = None
    pulmonary_flow: Optional[float] = None

    @classmethod
    def from_outputs(cls, outputs: SimOutputs) -> "BaselineProfile":
        lvedv = float(outputs["LVEDV"])
        lvesv = float(outputs["LVESV"])
        return cls(
            aortic_flow=float(outputs["aortic_flow"]),
            sv_lv=lvedv - lvesv,
            lvedp=float(outputs["LVEDP"]),
            lvesp=float(outputs.get("LVESP", float("nan"))),
            pulmonary_flow=float(outputs.get("pulmonary_flow", float("nan"))),
        )


class Objective(ABC):
    """Abstract base: maps simulation output dict to a scalar suitability score."""

    @abstractmethod
    def __call__(self, outputs: SimOutputs) -> float:
        """Compute scalar score (higher = better)."""


@dataclass
class LiteratureSafetyFilter:
    """Hard safety constraints applied before suitability scoring.

    If any constraint fails, the trial score is set to ``-inf`` and excluded
    from GP training.
    """

    baseline: BaselineProfile
    lvedp_margin_mmHg: float = 2.0
    lvedp_absolute_max: float = 16.0
    min_ef: float = 0.35
    max_flow_imbalance_ratio: float = 0.25

    CONSTRAINT_SOURCES: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.CONSTRAINT_SOURCES = {
            "aortic_flow_not_above_baseline": COMPONENT_SOURCES["aortic_flow_above_baseline"],
            "sv_lv_not_above_baseline": COMPONENT_SOURCES["sv_lv_above_baseline"],
            "lvedp_above_baseline_margin": COMPONENT_SOURCES["lvedp_vs_baseline"],
            "lvedp_absolute_limit": COMPONENT_SOURCES["lvedp_absolute_max"],
            "ef_below_minimum": COMPONENT_SOURCES["min_ef"],
            "flow_imbalance": COMPONENT_SOURCES["flow_balance"],
        }

    def check(self, outputs: SimOutputs) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        aortic_flow = _finite(outputs.get("aortic_flow"))
        sv = _lv_stroke_volume(outputs)
        lvedp = _finite(outputs.get("LVEDP"))
        ef = _lv_ejection_fraction(outputs)
        pulmonary_flow = _finite(outputs.get("pulmonary_flow"))

        if aortic_flow is None or aortic_flow <= self.baseline.aortic_flow:
            violations.append("aortic_flow_not_above_baseline")

        if sv is None or sv <= self.baseline.sv_lv:
            violations.append("sv_lv_not_above_baseline")

        if lvedp is None:
            violations.append("lvedp_missing")
        else:
            if lvedp > self.baseline.lvedp + self.lvedp_margin_mmHg:
                violations.append("lvedp_above_baseline_margin")
            if lvedp >= self.lvedp_absolute_max:
                violations.append("lvedp_absolute_limit")

        if ef is None or ef < self.min_ef:
            violations.append("ef_below_minimum")

        if aortic_flow is not None and pulmonary_flow is not None:
            q0_a = self.baseline.aortic_flow
            q0_p = self.baseline.pulmonary_flow
            if q0_p is not None and math.isfinite(q0_p):
                ref = max(abs(q0_a), abs(q0_p), 1e-9)
            else:
                ref = max(abs(q0_a), 1e-9)
            imbalance = abs(aortic_flow - pulmonary_flow) / ref
            if imbalance > self.max_flow_imbalance_ratio:
                violations.append("flow_imbalance")

        return len(violations) == 0, violations


class LiteratureObjective(Objective):
    """Literature-informed constrained suitability score.

    Pipeline
    --------
    1. Hard safety constraints -> reject with ``-inf`` if unsafe
    2. Soft score:
       + weighted normalised aortic_flow, SV_LV, stroke-work proxy
       - squared penalties for elevated LVEDP / RVEDP

    Default weights are initial engineering priorities (not fitted from data):
    aortic_flow > stroke volume > stroke work > pressure penalties.
    """

    TERM_SOURCES: Dict[str, str] = {
        "aortic_flow": COMPONENT_SOURCES["aortic_flow_term"],
        "stroke_volume": COMPONENT_SOURCES["stroke_volume_term"],
        "stroke_work": COMPONENT_SOURCES["stroke_work_term"],
        "lvedp_penalty": COMPONENT_SOURCES["lvedp_soft_penalty"],
        "rvedp_penalty": COMPONENT_SOURCES["rvedp_soft_penalty"],
    }

    def __init__(
        self,
        baseline: BaselineProfile,
        safety_filter: Optional[LiteratureSafetyFilter] = None,
        w_aortic_flow: float = 2.0,
        w_stroke: float = 1.0,
        w_stroke_work: float = 0.5,
        lvedp_soft_limit: float = 12.0,
        lvedp_penalty: float = 0.8,
        rvedp_soft_limit: float = 8.0,
        rvedp_penalty: float = 0.4,
    ) -> None:
        self.baseline = baseline
        self.safety_filter = safety_filter or LiteratureSafetyFilter(baseline=baseline)
        self.w_aortic_flow = w_aortic_flow
        self.w_stroke = w_stroke
        self.w_stroke_work = w_stroke_work
        self.lvedp_soft_limit = lvedp_soft_limit
        self.lvedp_penalty = lvedp_penalty
        self.rvedp_soft_limit = rvedp_soft_limit
        self.rvedp_penalty = rvedp_penalty

        sw0 = None
        if baseline.lvesp is not None and math.isfinite(baseline.lvesp):
            sw0 = (baseline.lvesp - baseline.lvedp) * baseline.sv_lv
        self._baseline_stroke_work = sw0 if sw0 and sw0 > 0 else None

    def safety_check(self, outputs: SimOutputs) -> Tuple[bool, List[str]]:
        return self.safety_filter.check(outputs)

    def __call__(self, outputs: SimOutputs) -> float:
        passed, _ = self.safety_filter.check(outputs)
        if not passed:
            return float("-inf")

        aortic_flow = _finite(outputs.get("aortic_flow"))
        sv = _lv_stroke_volume(outputs)
        lvedp = _finite(outputs.get("LVEDP"))
        lvesp = _finite(outputs.get("LVESP"))
        rvedp = _finite(outputs.get("RVEDP"))

        if aortic_flow is None or sv is None or lvedp is None:
            return float("-inf")

        b = self.baseline
        aortic_flow_ratio = aortic_flow / max(b.aortic_flow, 1e-9)
        stroke_ratio = sv / max(b.sv_lv, 1e-9)

        stroke_work_term = 0.0
        if self._baseline_stroke_work is not None and lvesp is not None:
            stroke_work = (lvesp - lvedp) * sv
            stroke_work_term = self.w_stroke_work * (stroke_work / self._baseline_stroke_work)

        pen_lvedp = self.lvedp_penalty * max(0.0, lvedp - self.lvedp_soft_limit) ** 2
        pen_rvedp = 0.0
        if rvedp is not None:
            pen_rvedp = self.rvedp_penalty * max(0.0, rvedp - self.rvedp_soft_limit) ** 2

        return (
            self.w_aortic_flow * aortic_flow_ratio
            + self.w_stroke * stroke_ratio
            + stroke_work_term
            - pen_lvedp
            - pen_rvedp
        )

    def __repr__(self) -> str:
        return (
            f"LiteratureObjective(baseline_aortic_flow={self.baseline.aortic_flow:.3f}, "
            f"baseline_lvedp={self.baseline.lvedp:.2f})"
        )


def _finite(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    v = float(val)
    return v if math.isfinite(v) else None


def _lv_stroke_volume(outputs: SimOutputs) -> Optional[float]:
    lvedv = _finite(outputs.get("LVEDV"))
    lvesv = _finite(outputs.get("LVESV"))
    if lvedv is None or lvesv is None:
        return None
    return lvedv - lvesv


def _lv_ejection_fraction(outputs: SimOutputs) -> Optional[float]:
    lvedv = _finite(outputs.get("LVEDV"))
    sv = _lv_stroke_volume(outputs)
    if lvedv is None or sv is None or lvedv <= 0:
        return None
    return sv / lvedv
