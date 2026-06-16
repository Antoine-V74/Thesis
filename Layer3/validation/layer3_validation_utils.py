#!/usr/bin/env python3
"""Shared utilities for Layer 3 ECG embedding validation scripts.

Safety note: nothing in this module can command stimulation. All decision
helpers here only return permit / inhibit / calibration_no_stim labels for
an upstream gate that combines Layer 1, Layer 2 and (optionally) Layer 3.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Layer 3 validation requires PyTorch. Install torch first.") from exc


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGER_NAME = "layer3"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch RNGs. Inference encoders are deterministic in
    eval() with no_grad regardless, but we still seed for reproducibility of any
    sampling that happens during validation (DataLoader shuffles, dropout if any).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch may not support warn_only; fall back to best-effort.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seeding so DataLoader workers don't all see the same RNG state."""
    base = int(torch.initial_seed()) % (2 ** 32 - 1)
    seed = (base + worker_id) % (2 ** 32 - 1)
    np.random.seed(seed)
    random.seed(seed)


def make_torch_generator(seed: int) -> "torch.Generator":
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def resolve_torch_device(device: Optional[str] = "auto", require_cuda: bool = False) -> str:
    """Resolve user-facing device strings to a torch device name.

    `auto` uses CUDA when the installed PyTorch build can see it, otherwise CPU.
    """
    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        msg = (
            f"Requested device `{device}`, but this PyTorch install cannot see CUDA. "
            "Install a CUDA-enabled torch build or use --device cpu."
        )
        if require_cuda:
            raise RuntimeError(msg)
        get_logger().warning("%s Falling back to CPU.", msg)
        return "cpu"
    return requested


# Conservative default for a stimulation safety gate: only normal sinus beat labels are
# considered baseline-safe unless the user explicitly broadens the set.
DEFAULT_NORMAL_SYMBOLS = {"N"}
KNOWN_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?", "!"
}
NON_BEAT_SYMBOLS = {"+", "~", "|", "\"", "=", "x", "(", ")", "[", "]"}


def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        parts: List[str] = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
        return parts
    out: List[str] = []
    for item in value:
        out.extend(parse_csv_list(item))
    return out


def import_wfdb():
    try:
        import wfdb  # type: ignore
        return wfdb
    except Exception as exc:
        raise RuntimeError(
            "This script needs the WFDB package for PhysioNet records. Install with: pip install wfdb"
        ) from exc


def resolve_record_path(data_dir: str | Path, dataset: str, record: str, record_path: Optional[str] = None) -> Path:
    if record_path:
        p = Path(record_path)
        if p.exists() or p.with_suffix(".hea").exists():
            return p
        p2 = Path(data_dir) / record_path
        if p2.exists() or p2.with_suffix(".hea").exists():
            return p2
    return Path(data_dir) / dataset / record


def robust_normalize_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    x = x / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def resample_if_needed(x: np.ndarray, fs: float, target_fs: Optional[float]) -> np.ndarray:
    if target_fs is None or abs(float(fs) - float(target_fs)) < 1e-6:
        return x
    try:
        from scipy.signal import resample_poly
    except Exception as exc:
        raise RuntimeError("Resampling requested but scipy is not installed. Install scipy or omit --target-fs.") from exc
    from math import gcd
    fs_i = int(round(float(fs)))
    target_i = int(round(float(target_fs)))
    g = gcd(fs_i, target_i)
    y = resample_poly(x, target_i // g, fs_i // g)
    return y.astype(np.float32)


def read_wfdb_window(
    record_path: str | Path,
    start_sample: int,
    end_sample: int,
    lead_index: int = 0,
    target_fs: Optional[float] = 250.0,
) -> np.ndarray:
    wfdb = import_wfdb()
    record_path = str(record_path)
    start_sample = max(0, int(start_sample))
    end_sample = max(start_sample + 1, int(end_sample))
    rec = wfdb.rdrecord(record_path, sampfrom=start_sample, sampto=end_sample, channels=[int(lead_index)])
    if rec.p_signal is not None:
        x = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
    elif rec.d_signal is not None:
        x = np.asarray(rec.d_signal[:, 0], dtype=np.float32)
    else:
        raise RuntimeError(f"No signal found in {record_path}")
    x = resample_if_needed(x, float(rec.fs), target_fs)
    return robust_normalize_window(x)


class RandomConvEncoder(nn.Module):
    """Small fallback encoder used only when the project ECGEncoder1D file is unavailable.

    This is intentionally simple and randomly initialized. It lets the validation pipeline run
    end-to-end as a smoke test, but it is NOT expected to be a meaningful learned Layer 3 model.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        torch.manual_seed(7)
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(96, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


def _try_instantiate_encoder(cls: Any) -> nn.Module:
    constructors = [
        {},
        {"in_channels": 1},
        {"input_channels": 1},
        {"n_channels": 1},
        {"embedding_dim": 128},
        {"in_channels": 1, "embedding_dim": 128},
    ]
    last_exc: Optional[Exception] = None
    for kwargs in constructors:
        try:
            return cls(**kwargs)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not instantiate ECGEncoder1D with common signatures: {last_exc}")


def build_encoder(checkpoint: Optional[str], device: str = "auto", allow_random_fallback: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    device = resolve_torch_device(device)
    info: Dict[str, Any] = {
        "source": None,
        "checkpoint": checkpoint,
        "checkpoint_loaded": False,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "warnings": [],
    }

    model: Optional[nn.Module] = None
    for mod_name in ("Layer3.layer3_encoder", "layer3_encoder"):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "ECGEncoder1D")
            model = _try_instantiate_encoder(cls)
            info["source"] = mod_name + ".ECGEncoder1D"
            break
        except Exception as exc:
            info["warnings"].append(f"Could not import/instantiate {mod_name}.ECGEncoder1D: {exc}")

    if model is None:
        if not allow_random_fallback:
            raise RuntimeError("No ECGEncoder1D available and random fallback disabled")
        model = RandomConvEncoder()
        info["source"] = "RandomConvEncoder fallback"
        info["warnings"].append("Using randomly initialized fallback encoder. This is only a pipeline smoke test.")

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            # PyTorch 2.6+ defaults weights_only=True, which can raise for checkpoints
            # that contain a config dict. We try the safe path first and fall back to
            # weights_only=False only if necessary. Checkpoints are trusted artifacts
            # produced by our own pretraining; the fallback is acceptable here.
            ckpt = None
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            except TypeError:
                # Older torch without weights_only kwarg
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
            except Exception as exc:
                info["warnings"].append(
                    f"weights_only torch.load failed for {checkpoint}: {exc}. "
                    f"Retrying with weights_only=False (trusted checkpoint)."
                )
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = ckpt
            if isinstance(ckpt, dict):
                for key in ("encoder_state_dict", "model_state_dict", "state_dict", "encoder", "model"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state = ckpt[key]
                        break
            if isinstance(state, dict):
                cleaned = {}
                for k, v in state.items():
                    nk = str(k)
                    # Handle common checkpoint prefixes, including nested forms such as
                    # module.encoder.stem.weight or model.encoder.stem.weight.
                    changed = True
                    while changed:
                        changed = False
                        for prefix in ("module.", "model.", "encoder."):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix):]
                                changed = True
                    cleaned[nk] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                info["checkpoint_loaded"] = True
                info["missing_keys"] = list(missing)
                info["unexpected_keys"] = list(unexpected)
                if len(missing) > 0 or len(unexpected) > 0:
                    get_logger().warning(
                        "Checkpoint loaded with %d missing and %d unexpected keys; "
                        "verify the encoder architecture matches the pretraining config.",
                        len(missing), len(unexpected),
                    )
            else:
                info["warnings"].append(f"Checkpoint {checkpoint} did not contain a state dict; ignored.")
        else:
            info["warnings"].append(f"Checkpoint {checkpoint} not found; using current/random encoder weights.")

    model = model.to(device)
    model.eval()
    return model, info


def tensor_to_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "z", "features", "h", "projection"):
            if key in output:
                return tensor_to_embedding(output[key])
        # Fall back to first tensor-like value.
        for value in output.values():
            if torch.is_tensor(value):
                return tensor_to_embedding(value)
    if isinstance(output, (tuple, list)):
        # Prefer a 2D tensor if available.
        for value in output:
            if torch.is_tensor(value) and value.ndim == 2:
                return value
        for value in output:
            if torch.is_tensor(value):
                return tensor_to_embedding(value)
    if not torch.is_tensor(output):
        raise RuntimeError(f"Encoder output type {type(output)} cannot be converted to embeddings")
    z = output
    if z.ndim == 3:
        z = z.mean(dim=-1)
    if z.ndim > 2:
        z = z.flatten(start_dim=1)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    return z


@torch.no_grad()
def encode_windows(
    model: nn.Module,
    windows: Sequence[np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        max_len = max(len(x) for x in batch)
        arr = np.zeros((len(batch), 1, max_len), dtype=np.float32)
        for j, x in enumerate(batch):
            arr[j, 0, : len(x)] = x
        xb = torch.from_numpy(arr).to(device)
        z = tensor_to_embedding(model(xb)).detach().cpu().numpy().astype(np.float32)
        embeddings.append(z)
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def decision_metrics(df: pd.DataFrame, decision_col: str = "decision", healthy_col: str = "is_healthy") -> Dict[str, Any]:
    """Safety-gate metrics, with both verbose and short-name aliases.

    Stimulation-safety semantics:
        permit on healthy            -> good (therapy availability)
        inhibit on healthy           -> false_inhibit (reduces therapy availability)
        inhibit on abnormal          -> good (safety catch)
        permit on abnormal           -> false_permit (the dangerous case)

    A row with decision == "permit" counts as a permit. Everything else
    (including "inhibit", "calibration_no_stim", "unknown") counts as a
    non-permit. For per-window/beat metrics, callers should restrict to
    `split == "test"` BEFORE calling this to avoid mixing calibration rows in.
    """
    if df.empty:
        empty = {
            "n": 0, "healthy_n": 0, "abnormal_n": 0,
            "permit_rate": float("nan"), "inhibit_rate": float("nan"),
            "healthy_permit_rate": float("nan"),
            "healthy_false_inhibit_rate": float("nan"),
            "abnormal_inhibit_rate": float("nan"),
            "abnormal_false_permit_rate": float("nan"),
        }
        # Short aliases for direct comparison with Layer 2 tables.
        empty["healthy_permit"] = empty["healthy_permit_rate"]
        empty["abnormal_inhibit"] = empty["abnormal_inhibit_rate"]
        empty["false_permit"] = empty["abnormal_false_permit_rate"]
        empty["false_inhibit"] = empty["healthy_false_inhibit_rate"]
        return empty

    d = df
    permit = d[decision_col].astype(str).str.lower().eq("permit")
    healthy = d[healthy_col].astype(bool)
    abnormal = ~healthy
    n = int(len(d))
    out: Dict[str, Any] = {
        "n": n,
        "healthy_n": int(healthy.sum()),
        "abnormal_n": int(abnormal.sum()),
        "permit_rate": float(permit.mean()) if n else float("nan"),
        "inhibit_rate": float((~permit).mean()) if n else float("nan"),
        "healthy_permit_rate": float(permit[healthy].mean()) if healthy.any() else float("nan"),
        "healthy_false_inhibit_rate": float((~permit[healthy]).mean()) if healthy.any() else float("nan"),
        "abnormal_inhibit_rate": float((~permit[abnormal]).mean()) if abnormal.any() else float("nan"),
        "abnormal_false_permit_rate": float(permit[abnormal].mean()) if abnormal.any() else float("nan"),
    }
    # Short aliases (Layer 2 reports these names too).
    out["healthy_permit"] = out["healthy_permit_rate"]
    out["abnormal_inhibit"] = out["abnormal_inhibit_rate"]
    out["false_permit"] = out["abnormal_false_permit_rate"]
    out["false_inhibit"] = out["healthy_false_inhibit_rate"]
    return out


def add_auroc_auprc(metrics: Dict[str, Any], y_abnormal: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = np.asarray(y_abnormal).astype(int)
        s = np.asarray(scores, dtype=float)
        mask = np.isfinite(s)
        y = y[mask]
        s = s[mask]
        if len(np.unique(y)) == 2:
            metrics["auroc_abnormal_score"] = float(roc_auc_score(y, s))
            metrics["auprc_abnormal_score"] = float(average_precision_score(y, s))
        else:
            metrics["auroc_abnormal_score"] = np.nan
            metrics["auprc_abnormal_score"] = np.nan
    except Exception:
        metrics["auroc_abnormal_score"] = np.nan
        metrics["auprc_abnormal_score"] = np.nan
    return metrics


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def compute_guard_samples(window_s: float, guard_s: float | None, fs: float) -> int:
    """Number of samples to skip between the last calibration window's center
    and the first test window's center to guarantee no temporal overlap
    between calibration and test windows. Default in callers is guard_s == window_s
    so a centered window of length W around a test center cannot overlap the
    centered val window around the last val center.
    """
    if guard_s is None:
        guard_s = window_s
    if guard_s is None or not np.isfinite(float(guard_s)) or float(guard_s) <= 0.0:
        return 0
    return int(round(float(guard_s) * float(fs)))


SAFETY_DISCLAIMER = (
    "Safety framing:\n"
    "- This is a stimulation safety gate, not a clinical arrhythmia classifier.\n"
    "- Layer 3 is a veto layer only; it cannot command stimulation by itself.\n"
    "- Final therapy rule: permit only if Layer 1 trigger is reliable AND Layer 2 permits AND Layer 3 permits.\n"
    "- Uncertainty or runtime failure must inhibit.\n"
    "- Calibration windows never trigger stimulation; they are reported as `calibration_no_stim`.\n"
    "- Human MIT-BIH evaluation is proxy validation. Animal deployment requires per-session\n"
    "  animal calibration and prospective animal-ECG validation.\n"
    "- Runtime does not consume oracle annotations. Offline threshold tuning that uses\n"
    "  abnormal-beat labels is NOT deployable unsupervised calibration.\n"
    "- Oracle beat-synchronous validation is an offline upper bound, not runtime deployable.\n"
    "- Centered beat windows are non-causal unless `--causal-window` is used.\n"
)
