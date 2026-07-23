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
DEFAULT_DANGEROUS_SYMBOLS = {"V", "F", "E", "/", "f", "!"}
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


def load_record_allowlist(path: str | Path) -> set[Tuple[str, str]]:
    """
    Load (dataset, record) pairs from a CSV with columns dataset, record.

    Used to restrict beat validation to gold transition records
    (e.g. Results/layer3/transition_analysis/pilot_primary_mitbih_gold.csv).
    """
    import pandas as pd

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Record allowlist not found: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "dataset" not in cols or "record" not in cols:
        raise ValueError(
            f"Record allowlist must have columns 'dataset' and 'record' (got {list(df.columns)})"
        )
    pairs: set[Tuple[str, str]] = set()
    for _, row in df.iterrows():
        ds = str(row[cols["dataset"]]).strip()
        rec = str(row[cols["record"]]).strip().replace("\\", "/")
        if ds and rec and ds.lower() != "nan" and rec.lower() != "nan":
            pairs.add((ds, rec))
    return pairs


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


def _try_instantiate_encoder(
    cls: Any,
    encoder_config_cls: Any = None,
    pooling_mode: str = "global_avg",
    local_fraction: float = 0.125,
) -> nn.Module:
    if encoder_config_cls is not None:
        try:
            cfg = encoder_config_cls(
                pooling_mode=str(pooling_mode),
                local_fraction=float(local_fraction),
            )
            return cls(cfg)
        except Exception:
            # Fall through to legacy constructor probing for external encoders.
            pass
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


def build_encoder(
    checkpoint: Optional[str],
    device: str = "auto",
    allow_random_fallback: bool = True,
    pooling_mode: str = "auto",
    local_fraction: float = 0.125,
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = resolve_torch_device(device)
    requested_pooling_mode = str(pooling_mode).lower()
    if requested_pooling_mode not in {"auto", "global_avg", "avg_max", "causal_local_global"}:
        raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}")
    loaded_checkpoint: Any = None
    checkpoint_config: Dict[str, Any] = {}
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            loaded_checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except TypeError:
            loaded_checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        except Exception:
            # Trusted project checkpoint; mirror the established fallback below.
            loaded_checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(loaded_checkpoint, dict) and isinstance(loaded_checkpoint.get("config"), dict):
            checkpoint_config = dict(loaded_checkpoint["config"])

    checkpoint_pooling_mode = str(checkpoint_config.get("encoder_pooling_mode", "global_avg")).lower()
    checkpoint_local_fraction = float(checkpoint_config.get("encoder_local_fraction", local_fraction))
    if requested_pooling_mode == "auto":
        resolved_pooling_mode = checkpoint_pooling_mode
        resolved_local_fraction = checkpoint_local_fraction
    else:
        resolved_pooling_mode = requested_pooling_mode
        resolved_local_fraction = float(local_fraction)
        if checkpoint_config and resolved_pooling_mode != checkpoint_pooling_mode:
            raise RuntimeError(
                f"Encoder pooling mismatch: checkpoint uses {checkpoint_pooling_mode!r}, "
                f"but evaluation requested {resolved_pooling_mode!r}."
            )
    info: Dict[str, Any] = {
        "source": None,
        "checkpoint": checkpoint,
        "checkpoint_loaded": False,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "warnings": [],
        "pooling_mode": resolved_pooling_mode,
        "local_fraction": resolved_local_fraction,
        "checkpoint_pooling_mode": checkpoint_pooling_mode if checkpoint_config else None,
        "checkpoint_window_s": checkpoint_config.get("window_s") if checkpoint_config else None,
        "checkpoint_window_n_samples": checkpoint_config.get("window_n_samples") if checkpoint_config else None,
        "checkpoint_window_target_fs": checkpoint_config.get("window_target_fs") if checkpoint_config else None,
    }

    model: Optional[nn.Module] = None
    for mod_name in ("Layer3.layer3_encoder", "layer3_encoder"):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "ECGEncoder1D")
            config_cls = getattr(mod, "EncoderConfig", None)
            model = _try_instantiate_encoder(
                cls,
                encoder_config_cls=config_cls,
                pooling_mode=resolved_pooling_mode,
                local_fraction=resolved_local_fraction,
            )
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
            ckpt = loaded_checkpoint
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
                info["missing_keys"] = list(missing)
                info["unexpected_keys"] = list(unexpected)
                # Only call it "loaded" if the core encoder weights were all found.
                # Missing keys mean the encoder is partly random (architecture or
                # objective mismatch), which would silently produce garbage Layer 3
                # scores — fail closed unless random fallback is explicitly allowed.
                info["checkpoint_loaded"] = len(missing) == 0
                if len(missing) > 0:
                    msg = (
                        f"Checkpoint {checkpoint} is missing {len(missing)} encoder key(s); "
                        "the encoder architecture/objective likely does not match, so weights are partly random."
                    )
                    if not allow_random_fallback:
                        raise RuntimeError(msg + " Refusing to continue (--no-random-fallback).")
                    get_logger().warning(msg)
                elif len(unexpected) > 0:
                    get_logger().warning(
                        "Checkpoint loaded cleanly with %d unexpected key(s) ignored.", len(unexpected)
                    )
            else:
                msg = f"Checkpoint {checkpoint} did not contain a recognizable state dict."
                if not allow_random_fallback:
                    raise RuntimeError(msg + " Refusing to continue (--no-random-fallback).")
                info["warnings"].append(msg + " Ignored; using random weights.")
        else:
            msg = f"Checkpoint {checkpoint} not found; the encoder would use random weights."
            if not allow_random_fallback:
                raise RuntimeError(msg + " Refusing to continue (--no-random-fallback). Check the --checkpoint path.")
            info["warnings"].append(msg)

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


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial rate."""
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
        / denom
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def conformal_threshold_from_scores(scores: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Upper-tail split-conformal threshold for healthy calibration scores.

    The finite-sample guarantee is only for healthy false alarms / false inhibits
    under exchangeability. It is not a false-permit guarantee on dangerous beats.
    If alpha is infeasible for the calibration sample size, callers should
    fail-safe inhibit instead of silently relaxing alpha.
    """
    s = np.asarray(scores, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    n = int(len(s))
    alpha = float(alpha)
    if n <= 0:
        return {
            "threshold": float("nan"),
            "status": "no_healthy_calibration_scores",
            "alpha": alpha,
            "n": n,
            "rank": 0,
            "alpha_min": float("nan"),
        }
    if not (0.0 < alpha < 1.0):
        return {
            "threshold": float("nan"),
            "status": "invalid_alpha",
            "alpha": alpha,
            "n": n,
            "rank": 0,
            "alpha_min": float(1.0 / (n + 1)),
        }
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    alpha_min = float(1.0 / (n + 1))
    if rank > n:
        return {
            "threshold": float("nan"),
            "status": "alpha_infeasible",
            "alpha": alpha,
            "n": n,
            "rank": rank,
            "alpha_min": alpha_min,
        }
    return {
        "threshold": float(s[rank - 1]),
        "status": "ok",
        "alpha": alpha,
        "n": n,
        "rank": rank,
        "alpha_min": alpha_min,
    }


def select_decision_threshold(
    val_scores: np.ndarray,
    method: str,
    threshold_quantile: float,
    conformal_alpha: float,
) -> Dict[str, Any]:
    """Choose a healthy-only decision threshold for the main permit/inhibit gate.

    Two methods are supported:

    - ``conformal``: split-conformal upper-tail threshold with a stated healthy
      false-inhibit budget ``alpha`` (distribution-free, exchangeability only).
    - ``healthy_quantile``: the historical ``quantile(healthy_val_scores, q)``.

    The returned ``target_false_inhibit`` is ``alpha`` for conformal and
    ``1 - quantile`` for the quantile method, so a coverage report can compare
    the achieved healthy false-inhibit rate against a single stated target.

    Callers MUST fail-safe inhibit when ``status != "ok"`` (uncertainty ->
    inhibit). This helper never relaxes ``alpha`` silently.
    """
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
    raise ValueError(f"Unsupported threshold method: {method!r} (use conformal or healthy_quantile)")


def write_threshold_coverage(
    scored: pd.DataFrame,
    out_dir: Path,
    *,
    threshold_method: str,
    target_false_inhibit: float,
    healthy_col: str,
    record_col: str = "record_key",
) -> pd.DataFrame:
    """Write ``threshold_coverage.csv``: achieved vs targeted healthy false-inhibit.

    Only scored test rows with a genuine ``permit``/``inhibit`` decision on
    healthy beats/windows are counted. ``within_target`` is a per-group
    diagnostic; conformal coverage holds in expectation over calibration draws,
    not necessarily for every record, so small-``n`` records may exceed target.
    """
    df = scored.copy()
    if healthy_col not in df.columns:
        raise KeyError(f"healthy column {healthy_col!r} missing from scored frame")
    mask = (
        (df["split"] == "test")
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


def phase1_label_group(symbol: Any, normal_symbols: Iterable[str], dangerous_symbols: Iterable[str]) -> str:
    """Map a beat symbol to Phase 1 reporting groups without changing policy."""
    sym = str(symbol)
    normal = {str(s) for s in normal_symbols}
    dangerous = {str(s) for s in dangerous_symbols}
    if sym in normal:
        return "NORMAL"
    if sym in dangerous:
        return "DANGEROUS"
    if sym in NON_BEAT_SYMBOLS:
        return "NOISE"
    if sym in KNOWN_BEAT_SYMBOLS:
        return "BENIGN_ABNORMAL"
    return "UNLABELED"


def safe_auroc(y_positive: np.ndarray, scores: np.ndarray) -> float:
    """Return AUROC for a positive class scored by larger anomaly scores."""
    try:
        from sklearn.metrics import roc_auc_score
        y = np.asarray(y_positive).astype(int)
        s = np.asarray(scores, dtype=float)
        mask = np.isfinite(s)
        y = y[mask]
        s = s[mask]
        if len(np.unique(y)) != 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


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
