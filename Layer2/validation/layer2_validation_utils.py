"""Shared thresholding, coverage reporting, and guard helpers for Layer 2 validation."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_L3_PIPELINE = _ROOT / "Layer3" / "pipeline"
if str(_L3_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_L3_PIPELINE))

from label_grouping import (  # noqa: E402
    AF_TREATED_AS_DEFAULT,
    build_rhythm_spans,
    group_for_beat,
    safety_expectation,
)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def conformal_threshold_from_scores(scores: np.ndarray, alpha: float) -> Dict[str, Any]:
    s = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = int(len(s))
    alpha = float(alpha)
    if n <= 0:
        return {"threshold": float("nan"), "status": "no_healthy_calibration_scores", "alpha": alpha, "n": n, "rank": 0, "alpha_min": float("nan")}
    if not (0.0 < alpha < 1.0):
        return {"threshold": float("nan"), "status": "invalid_alpha", "alpha": alpha, "n": n, "rank": 0, "alpha_min": float(1.0 / (n + 1))}
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    alpha_min = float(1.0 / (n + 1))
    if rank > n:
        return {"threshold": float("nan"), "status": "alpha_infeasible", "alpha": alpha, "n": n, "rank": rank, "alpha_min": alpha_min}
    return {"threshold": float(s[rank - 1]), "status": "ok", "alpha": alpha, "n": n, "rank": rank, "alpha_min": alpha_min}


def select_decision_threshold(
    val_scores: np.ndarray,
    method: str,
    threshold_quantile: float,
    conformal_alpha: float,
) -> Dict[str, Any]:
    method = str(method).lower()
    s = np.asarray(val_scores, dtype=float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if method == "conformal":
        info = conformal_threshold_from_scores(s, float(conformal_alpha))
        return {
            "threshold": float(info["threshold"]),
            "method": "conformal",
            "status": str(info["status"]),
            "target_false_inhibit": float(conformal_alpha),
            "conformal_alpha": float(conformal_alpha),
            "conformal_alpha_min": float(info.get("alpha_min", float("nan"))),
            "n_val": n,
        }
    if method == "healthy_quantile":
        if n <= 0:
            return {
                "threshold": float("nan"),
                "method": "healthy_quantile",
                "status": "no_healthy_calibration_scores",
                "target_false_inhibit": float(1.0 - float(threshold_quantile)),
                "conformal_alpha": float("nan"),
                "conformal_alpha_min": float("nan"),
                "n_val": n,
            }
        q = float(np.clip(threshold_quantile, 0.0, 1.0))
        return {
            "threshold": float(np.quantile(s, q)),
            "method": "healthy_quantile",
            "status": "ok",
            "target_false_inhibit": float(1.0 - q),
            "conformal_alpha": float("nan"),
            "conformal_alpha_min": float("nan"),
            "n_val": n,
        }
    raise ValueError(f"Unsupported threshold method: {method!r}")


def write_threshold_coverage(
    scored: pd.DataFrame,
    out_dir: Path,
    *,
    threshold_method: str,
    target_false_inhibit: float,
    healthy_col: str = "is_healthy",
    record_col: str = "record",
) -> pd.DataFrame:
    df = scored.copy()
    if "decision" not in df.columns:
        df["decision"] = np.where(df["permit"].astype(bool), "permit", "inhibit")
    mask = (
        (df.get("split", "test") == "test")
        & df["decision"].isin(["permit", "inhibit"])
        & df[healthy_col].astype(bool)
    )
    eval_df = df[mask]

    def _row(name: str, gg: pd.DataFrame) -> Dict[str, Any]:
        n = int(len(gg))
        fi = int((gg["decision"].astype(str).str.lower() == "inhibit").sum())
        rate = float(fi / n) if n else float("nan")
        lo, hi = wilson_ci(fi, n)
        return {
            record_col: name,
            "threshold_method": str(threshold_method),
            "target_false_inhibit": float(target_false_inhibit),
            "n_healthy_test": n,
            "false_inhibit_healthy_n": fi,
            "achieved_false_inhibit_healthy": rate,
            "achieved_false_inhibit_ci_low": lo,
            "achieved_false_inhibit_ci_high": hi,
            "within_target": bool(rate <= target_false_inhibit) if n else False,
        }

    rows = [_row(str(rk), gg) for rk, gg in eval_df.groupby(record_col)]
    coverage = pd.DataFrame(rows)
    overall = _row("ALL", eval_df)
    coverage = pd.concat([coverage, pd.DataFrame([overall])], ignore_index=True)
    coverage.to_csv(out_dir / "threshold_coverage.csv", index=False)
    return coverage


def compute_guard_end_time(cal_end_t: float, guard_s: float) -> float:
    if guard_s is None or not np.isfinite(float(guard_s)) or float(guard_s) <= 0.0:
        return float(cal_end_t)
    return float(cal_end_t) + float(guard_s)


def beat_split_label(beat_time_s: float, cal_end_t: float, guard_end_t: float) -> str:
    if beat_time_s <= cal_end_t:
        return "calibration"
    if beat_time_s <= guard_end_t:
        return "guard"
    return "test"


def annotate_rhythm_spans(ann_samples, ann_symbols, ann_aux, sig_len: int):
    aux = ann_aux if ann_aux is not None else [""] * len(ann_symbols)
    return build_rhythm_spans(ann_samples, ann_symbols, aux, sig_len)


def safety_group_for_beat(sample: int, symbol: str, spans, dataset: str) -> str:
    return group_for_beat(int(sample), str(symbol), spans, dataset=dataset)


def add_policy_expectation(df: pd.DataFrame, af_treated_as: str = AF_TREATED_AS_DEFAULT) -> pd.DataFrame:
    out = df.copy()
    if "safety_group" not in out.columns:
        out["safety_expectation_policy"] = np.where(
            out.get("is_healthy", False).astype(bool), "permit_expected", "inhibit_expected"
        )
        return out
    out["safety_expectation_policy"] = [
        safety_expectation(str(g), af_treated_as=af_treated_as)
        for g in out["safety_group"].astype(str).tolist()
    ]
    return out


def policy_metrics_by_group(df: pd.DataFrame, af_treated_as: str = AF_TREATED_AS_DEFAULT) -> pd.DataFrame:
    """Policy-aware metrics keyed by safety_group (test split only)."""
    d = add_policy_expectation(df.copy(), af_treated_as=af_treated_as)
    d = d[d.get("split", "test") == "test"].copy()
    if "decision" not in d.columns:
        d["decision"] = np.where(d["permit"].astype(bool), "permit", "inhibit")
    rows = []
    for (fset, mode, group), grp in d.groupby(["feature_set", "mode", "safety_group"], dropna=False):
        exp = safety_expectation(str(group), af_treated_as=af_treated_as)
        n = len(grp)
        permits = int((grp["decision"] == "permit").sum())
        inhibits = int((grp["decision"] == "inhibit").sum())
        row = {
            "feature_set": fset,
            "mode": mode,
            "safety_group": group,
            "safety_expectation": exp,
            "n_beats": n,
            "n_permit": permits,
            "n_inhibit": inhibits,
        }
        if exp == "permit_expected" and n:
            row["false_inhibit_rate"] = float((grp["decision"] == "inhibit").mean())
        elif exp == "inhibit_expected" and n:
            row["false_permit_rate"] = float((grp["decision"] == "permit").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def infer_risk_family(reason: str, hard_rule: str, signal_proxy: float, rr_proxy: float,
                      signal_thr: float, rr_thr: float) -> str:
    if reason in ("within_baseline", "all_safe", "rr_history_rewarming_permit"):
        return "none"
    if reason == "hard_rule" or hard_rule:
        feat = str(hard_rule or "")
        if feat.startswith("signal__"):
            return "noise"
        if feat.startswith("morph__"):
            return "morphology"
        if feat.startswith("rr__"):
            return "rhythm"
        return "hard_rule"
    if reason == "signal_not_safe":
        return "noise"
    if reason in ("rr_abnormal",) or (np.isfinite(rr_proxy) and np.isfinite(rr_thr) and rr_proxy > rr_thr):
        return "rhythm"
    if reason in ("max_zscore_exceeded", "mahalanobis_exceeded", "knn_exceeded"):
        if np.isfinite(signal_proxy) and np.isfinite(signal_thr) and signal_proxy > signal_thr:
            return "noise"
        if np.isfinite(rr_proxy) and np.isfinite(rr_thr) and rr_proxy > rr_thr:
            return "rhythm"
        return "baseline_deviation"
    return "other"
