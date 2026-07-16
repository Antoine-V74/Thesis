#!/usr/bin/env python3
"""Policy-aware safety metrics for Layer 3 danger-grouped labels.

Unlike the legacy healthy-vs-abnormal metrics, these metrics honor the safety
policy in `label_grouping.py`:

- NORMAL and AF_CONTEXT-as-permit are permit-expected groups.
- DANGEROUS, NOISE, and AF_CONTEXT-as-inhibit are inhibit-expected groups.
- BENIGN_ABNORMAL is reported but not penalized.
- UNLABELED / AF_CONTEXT-as-exclude are ignored.

This prevents isolated ectopy or explicitly ignored rows from being counted as
false permits in the final safety tables.
"""
from __future__ import annotations

from typing import Any, Dict
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PIPELINE_DIR = LAYER3_ROOT / "pipeline"
for path in (PIPELINE_DIR, LAYER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from label_grouping import (
    AF_TREATED_AS_DEFAULT,
    DONT_CARE,
    IGNORE,
    INHIBIT_EXPECTED,
    PERMIT_EXPECTED,
    safety_expectation,
)


def add_policy_columns(
    df: pd.DataFrame,
    *,
    safety_group_col: str = "safety_group",
    af_treated_as: str = AF_TREATED_AS_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()
    if safety_group_col not in out.columns:
        out["safety_expectation_policy"] = np.where(
            out.get("is_healthy", out.get("is_healthy_window", False)).astype(bool),
            PERMIT_EXPECTED,
            INHIBIT_EXPECTED,
        )
        return out
    out["safety_expectation_policy"] = [
        safety_expectation(str(g), af_treated_as=af_treated_as)
        for g in out[safety_group_col].astype(str).tolist()
    ]
    return out


def policy_decision_metrics(
    df: pd.DataFrame,
    *,
    decision_col: str = "decision",
    safety_group_col: str = "safety_group",
    af_treated_as: str = AF_TREATED_AS_DEFAULT,
) -> Dict[str, Any]:
    d = add_policy_columns(df, safety_group_col=safety_group_col, af_treated_as=af_treated_as)
    if d.empty:
        return {
            "n": 0,
            "n_evaluable": 0,
            "n_permit_expected": 0,
            "n_inhibit_expected": 0,
            "n_dont_care": 0,
            "n_ignored": 0,
            "permit_rate_evaluable": float("nan"),
            "false_permit_rate": float("nan"),
            "false_inhibit_rate": float("nan"),
            "inhibit_rate_on_dont_care": float("nan"),
        }

    permit = d[decision_col].astype(str).str.lower().eq("permit")
    exp = d["safety_expectation_policy"].astype(str)
    permit_expected = exp.eq(PERMIT_EXPECTED)
    inhibit_expected = exp.eq(INHIBIT_EXPECTED)
    dont_care = exp.eq(DONT_CARE)
    ignored = exp.eq(IGNORE)
    evaluable = permit_expected | inhibit_expected

    out: Dict[str, Any] = {
        "n": int(len(d)),
        "n_evaluable": int(evaluable.sum()),
        "n_permit_expected": int(permit_expected.sum()),
        "n_inhibit_expected": int(inhibit_expected.sum()),
        "n_dont_care": int(dont_care.sum()),
        "n_ignored": int(ignored.sum()),
        "permit_rate_evaluable": float(permit[evaluable].mean()) if evaluable.any() else float("nan"),
        "false_permit_rate": float(permit[inhibit_expected].mean()) if inhibit_expected.any() else float("nan"),
        "false_inhibit_rate": float((~permit[permit_expected]).mean()) if permit_expected.any() else float("nan"),
        "inhibit_rate_on_dont_care": float((~permit[dont_care]).mean()) if dont_care.any() else float("nan"),
    }
    # Compatibility aliases for safety tables.
    out["false_permit"] = out["false_permit_rate"]
    out["false_inhibit"] = out["false_inhibit_rate"]
    return out


def add_normal_vs_danger_auroc(
    metrics: Dict[str, Any],
    df: pd.DataFrame,
    *,
    score_col: str = "anomaly_score",
    safety_group_col: str = "safety_group",
) -> Dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score
        if safety_group_col not in df.columns or score_col not in df.columns:
            metrics["auroc_normal_vs_dangerous"] = np.nan
            return metrics
        sub = df[df[safety_group_col].astype(str).isin(["NORMAL", "DANGEROUS"])].copy()
        scores = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
        y = sub[safety_group_col].astype(str).eq("DANGEROUS").astype(int).to_numpy()
        mask = np.isfinite(scores)
        if len(np.unique(y[mask])) == 2:
            metrics["auroc_normal_vs_dangerous"] = float(roc_auc_score(y[mask], scores[mask]))
        else:
            metrics["auroc_normal_vs_dangerous"] = np.nan
    except Exception:
        metrics["auroc_normal_vs_dangerous"] = np.nan
    return metrics


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "safety_group": ["NORMAL", "DANGEROUS", "BENIGN_ABNORMAL", "UNLABELED"],
            "decision": ["permit", "permit", "permit", "inhibit"],
        }
    )
    m = policy_decision_metrics(demo)
    assert m["n_evaluable"] == 2
    assert m["n_dont_care"] == 1
    assert m["n_ignored"] == 1
    assert abs(m["false_permit_rate"] - 1.0) < 1e-12
    print("layer3_group_metrics smoke test: OK")
