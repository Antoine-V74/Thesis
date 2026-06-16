"""
Shared helpers for Layer 1 analysis and plotting tools.

Used by scripts in Layer1/tools/ — not part of the real-time pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError as exc:
    raise ImportError("wfdb required for diagnostics") from exc

from main_pipeline import filter_ecg, layer2_supervisor_config, run_layer1
from reference_annotations import greedy_match
from rhythm_supervisor import RRSupervisor


def apply_filters(raw: np.ndarray, fs: float) -> np.ndarray:
    return filter_ecg(raw, fs, mode="zero_phase")


def run_layer1_detector(
    filt: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, RRSupervisor]:
    """Return candidate peaks (s), accepted peaks (s), and supervisor."""
    result = run_layer1(
        filt, fs,
        already_filtered=True,
        supervisor_cfg=layer2_supervisor_config(),
    )
    return (
        result.candidate_samples.astype(float) / fs,
        result.accepted_samples.astype(float) / fs,
        result.supervisor,
    )


def load_validation_rows(
    csv_path: Path,
    datasets: Optional[List[str]] = None,
    feature_set: str = "all",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "feature_set" in df.columns:
        df = df[df["feature_set"] == feature_set]
    df = df[df["label"] == "healthy"]
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    return df.reset_index(drop=True)


def load_record(stem: str) -> Tuple[np.ndarray, float, object]:
    rec = wfdb.rdrecord(stem)
    raw = rec.p_signal[:, 0].astype(float)
    fs = float(rec.fs)
    try:
        ann = wfdb.rdann(stem, "atr")
    except Exception:
        ann = None
    return raw, fs, ann


def match_peaks(
    oracle_s: np.ndarray,
    test_s: np.ndarray,
    fs: float,
    tol_ms: float = 80.0,
) -> Dict:
    if len(oracle_s) == 0 or len(test_s) == 0:
        n_o, n_t = len(oracle_s), len(test_s)
        return {
            "tp": 0, "fp": n_t, "fn": n_o,
            "sensitivity": 0.0 if n_o else float("nan"),
            "ppv": 0.0 if n_t else float("nan"),
            "f1": 0.0,
            "mean_abs_jitter_ms": float("nan"),
            "max_abs_jitter_ms": float("nan"),
            "matches": [],
        }
    oracle_idx = np.round(oracle_s * fs).astype(int)
    test_idx = np.round(test_s * fs).astype(int)
    res = greedy_match(oracle_idx, test_idx, fs=fs, tol_ms=tol_ms)
    jitters = [abs(dt) * 1000.0 / fs for _, _, dt in res["matches"]]
    return {
        "tp": int(res["tp"]),
        "fp": int(res["fp"]),
        "fn": int(res["fn"]),
        "sensitivity": float(res["sensitivity"]),
        "ppv": float(res["ppv"]),
        "f1": float(res["f1"]),
        "mean_abs_jitter_ms": float(np.mean(jitters)) if jitters else float("nan"),
        "max_abs_jitter_ms": float(np.max(jitters)) if jitters else float("nan"),
        "matches": res["matches"],
    }


def rr_stats(
    peaks_s: np.ndarray,
    rr_short_ms: float = 400.0,
    rr_long_ms: float = 1200.0,
) -> Dict:
    if len(peaks_s) < 2:
        return {
            "n_beats": len(peaks_s),
            "rr_min_ms": float("nan"), "rr_max_ms": float("nan"),
            "rr_mean_ms": float("nan"), "rr_rmssd": float("nan"),
            "rr_diff_abs_mean": float("nan"),
            "n_short_rr": 0, "n_long_rr": 0,
            "short_rr_fraction": float("nan"),
            "long_rr_fraction": float("nan"),
        }
    rr_ms = np.diff(np.sort(peaks_s)) * 1000.0
    n = len(rr_ms)
    n_short = int(np.sum(rr_ms < rr_short_ms))
    n_long = int(np.sum(rr_ms > rr_long_ms))
    rr_diffs = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(rr_diffs ** 2))) if len(rr_diffs) else float("nan")
    diff_mean = float(np.mean(np.abs(rr_diffs))) if len(rr_diffs) else float("nan")
    return {
        "n_beats": len(peaks_s),
        "rr_min_ms": float(rr_ms.min()),
        "rr_max_ms": float(rr_ms.max()),
        "rr_mean_ms": float(rr_ms.mean()),
        "rr_rmssd": rmssd,
        "rr_diff_abs_mean": diff_mean,
        "n_short_rr": n_short,
        "n_long_rr": n_long,
        "short_rr_fraction": n_short / n,
        "long_rr_fraction": n_long / n,
    }
