
"""
Helpers for MIT-BIH annotations and event matching.
"""

from typing import Dict, List, Tuple

import numpy as np


def get_reference_beats(ann) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract beat annotations from a WFDB annotation structure.

    The symbol set follows the original single-file implementation and keeps
    the same accepted classes for evaluation.
    """
    beat_symbols = {
        "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/",
        "f", "Q", "n", "?",
    }
    samples: List[int] = []
    symbols: List[str] = []

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in beat_symbols:
            samples.append(sample)
            symbols.append(symbol)

    return np.asarray(samples, dtype=int), np.asarray(symbols)


def greedy_match(
    ref_samples: np.ndarray,
    test_samples: np.ndarray,
    fs: float,
    tol_ms: float = 100.0,
) -> Dict[str, object]:
    """
    We match the detected samples to reference beats within a tolerance.

    Returns a dictionary with TP/FP/FN and standard detection metrics.
    """
    tol = int(round(tol_ms * fs / 1000.0))
    ref_samples = np.asarray(ref_samples, dtype=int)
    test_samples = np.asarray(test_samples, dtype=int)

    i = 0
    j = 0
    matches = []

    while i < len(ref_samples) and j < len(test_samples):
        dt = test_samples[j] - ref_samples[i]

        if abs(dt) <= tol:
            matches.append((ref_samples[i], test_samples[j], dt))
            i += 1
            j += 1
        elif test_samples[j] < ref_samples[i] - tol:
            j += 1
        else:
            i += 1

    matched_ref = {m[0] for m in matches}
    matched_test = {m[1] for m in matches}

    tp = len(matches)
    fp = sum(1 for x in test_samples if x not in matched_test)
    fn = sum(1 for x in ref_samples if x not in matched_ref)
    dts = np.array([m[2] for m in matches], dtype=float) if matches else np.array([])

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "mean_abs_jitter_ms": (1000.0 * np.mean(np.abs(dts)) / fs) if len(dts) else np.nan,
        "matches": matches,
    }
