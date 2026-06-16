#!/usr/bin/env python3
"""
Build a CSV index of ECG windows for Layer 3 embedding anomaly validation and
contrastive pretraining.

Outputs both validation-friendly WFDB/native columns:
    dataset, record, record_path, start_sample, end_sample, center_sample, labels...

and pretraining-friendly columns expected by existing layer3_pretrain.py:
    record_id, signal_path, start_idx, n_samples

If --signal-dir is omitted, a sibling folder next to --out-csv is used and one
lead of each WFDB record is cached as a .npy file. start_idx/n_samples are in
this cached-signal coordinate system. When --target-fs > 0, the cached signal is
resampled to that frequency, so pretraining receives consistent 5 s windows.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow running as `python Layer3/build_window_index.py` from repo root.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from layer3_validation_utils import (  # noqa: E402
    DEFAULT_NORMAL_SYMBOLS,
    KNOWN_BEAT_SYMBOLS,
    NON_BEAT_SYMBOLS,
    get_logger,
    import_wfdb,
    parse_csv_list,
    resample_if_needed,
    robust_normalize_window,
)

LOG = get_logger("layer3.build_index")


def find_wfdb_records(dataset_dir: Path) -> List[str]:
    """Return WFDB record names relative to dataset_dir, without .hea suffix."""
    records: List[str] = []
    for hea in sorted(dataset_dir.rglob("*.hea")):
        rel = hea.relative_to(dataset_dir).with_suffix("")
        records.append(str(rel).replace("\\", "/"))
    return records


def safe_record_file_stem(dataset: str, record: str, lead_index: int, target_fs: Optional[float]) -> str:
    fs_tag = "native" if target_fs is None else f"fs{int(round(float(target_fs)))}"
    raw = f"{dataset}__{record}__lead{lead_index}__{fs_tag}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def read_annotations(record_path: Path, ann_ext: str) -> Tuple[np.ndarray, List[str], List[str]]:
    wfdb = import_wfdb()
    try:
        ann = wfdb.rdann(str(record_path), ann_ext)
    except Exception:
        return np.array([], dtype=np.int64), [], []

    samples = np.asarray(ann.sample, dtype=np.int64)
    symbols = [str(s) for s in ann.symbol]
    if getattr(ann, "aux_note", None) is not None:
        aux = [str(a) for a in ann.aux_note]
    else:
        aux = [""] * len(symbols)
    return samples, symbols, aux


def window_label_summary(
    ann_samples: np.ndarray,
    ann_symbols: Sequence[str],
    ann_aux: Sequence[str],
    start: int,
    end: int,
    normal_symbols: set[str],
    min_beats: int,
) -> Dict[str, object]:
    lo = int(np.searchsorted(ann_samples, start, side="left"))
    hi = int(np.searchsorted(ann_samples, end, side="right"))
    symbols_all = list(ann_symbols[lo:hi])
    aux_all = list(ann_aux[lo:hi])

    beat_symbols = [s for s in symbols_all if s in KNOWN_BEAT_SYMBOLS and s not in NON_BEAT_SYMBOLS]
    ignored = [s for s in symbols_all if s not in KNOWN_BEAT_SYMBOLS or s in NON_BEAT_SYMBOLS]
    normal = [s for s in beat_symbols if s in normal_symbols]
    abnormal = [s for s in beat_symbols if s not in normal_symbols]

    is_healthy = len(beat_symbols) >= min_beats and len(abnormal) == 0 and len(normal) > 0
    if len(abnormal) > 0:
        dominant = max(set(abnormal), key=abnormal.count)
    elif len(normal) > 0:
        dominant = max(set(normal), key=normal.count)
    else:
        dominant = "unlabeled"

    rhythm_aux = "|".join([a for a in aux_all if a])
    return {
        "n_annotations": len(symbols_all),
        "n_beats": len(beat_symbols),
        "n_normal_beats": len(normal),
        "n_abnormal_beats": len(abnormal),
        "n_ignored_annotations": len(ignored),
        "beat_symbols": "".join(beat_symbols),
        "dominant_label": dominant,
        "rhythm_aux": rhythm_aux,
        "is_healthy_window": bool(is_healthy),
        "has_labels": bool(len(beat_symbols) > 0),
    }


def cache_record_signal(
    record_path: Path,
    dataset: str,
    record: str,
    signal_dir: Path,
    lead_index: int,
    target_fs: Optional[float],
    overwrite: bool = False,
) -> Tuple[Path, float, int]:
    """Cache one WFDB lead as normalized .npy and return path, cached fs, cached length."""
    wfdb = import_wfdb()
    signal_dir.mkdir(parents=True, exist_ok=True)
    out_path = signal_dir / f"{safe_record_file_stem(dataset, record, lead_index, target_fs)}.npy"
    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        cached_fs = target_fs
        if cached_fs is None:
            cached_fs = float(wfdb.rdheader(str(record_path)).fs)
        return out_path, float(cached_fs), int(arr.shape[0])

    rec = wfdb.rdrecord(str(record_path), channels=[int(lead_index)])
    if rec.p_signal is not None:
        x = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
    elif rec.d_signal is not None:
        x = np.asarray(rec.d_signal[:, 0], dtype=np.float32)
    else:
        raise RuntimeError(f"No signal found in {record_path}")

    x = resample_if_needed(x, float(rec.fs), target_fs)
    x = robust_normalize_window(x)
    np.save(out_path, x.astype(np.float32))
    cached_fs = float(rec.fs if target_fs is None else target_fs)
    return out_path, cached_fs, int(len(x))


def build_index_for_record(
    data_dir: Path,
    dataset: str,
    record: str,
    window_s: float,
    stride_s: float,
    ann_ext: str,
    normal_symbols: set[str],
    min_beats: int,
    signal_dir: Optional[Path],
    lead_index: int,
    target_fs: Optional[float],
    overwrite_signal_cache: bool,
) -> List[Dict[str, object]]:
    wfdb = import_wfdb()
    record_path = data_dir / dataset / record
    header = wfdb.rdheader(str(record_path))
    native_fs = float(header.fs)
    native_sig_len = int(header.sig_len)
    sig_names = list(getattr(header, "sig_name", []) or [])

    signal_path = ""
    cached_fs = native_fs
    cached_len = native_sig_len
    if signal_dir is not None:
        cached_path, cached_fs, cached_len = cache_record_signal(
            record_path=record_path,
            dataset=dataset,
            record=record,
            signal_dir=signal_dir,
            lead_index=lead_index,
            target_fs=target_fs,
            overwrite=overwrite_signal_cache,
        )
        signal_path = str(cached_path)

    ann_samples, ann_symbols, ann_aux = read_annotations(record_path, ann_ext)
    if ann_samples.size:
        order = np.argsort(ann_samples)
        ann_samples = ann_samples[order]
        ann_symbols = [ann_symbols[i] for i in order]
        ann_aux = [ann_aux[i] for i in order]

    native_window_n = int(round(window_s * native_fs))
    native_stride_n = int(round(stride_s * native_fs))
    cache_window_n = int(round(window_s * cached_fs))
    cache_stride_n = int(round(stride_s * cached_fs))
    if native_window_n <= 1 or native_stride_n <= 0 or cache_window_n <= 1 or cache_stride_n <= 0:
        raise ValueError("window_s and stride_s must produce positive sample counts")

    rows: List[Dict[str, object]] = []
    n_windows = max(0, (native_sig_len - native_window_n) // native_stride_n + 1)
    for wi in range(n_windows):
        native_start = wi * native_stride_n
        native_end = native_start + native_window_n
        cache_start = wi * cache_stride_n
        cache_end = cache_start + cache_window_n
        if cache_end > cached_len:
            break

        label_info = window_label_summary(
            ann_samples,
            ann_symbols,
            ann_aux,
            native_start,
            native_end,
            normal_symbols=normal_symbols,
            min_beats=min_beats,
        )
        record_id = f"{dataset}/{record}"
        rows.append(
            {
                # Shared identifiers
                "record_id": record_id,
                "dataset": dataset,
                "record": record,
                "record_path": str((Path(dataset) / record)).replace("\\", "/"),
                # Native WFDB coordinate system used by validation scripts
                "fs": native_fs,
                "signal_len": native_sig_len,
                "lead_names": "|".join(sig_names),
                "ann_ext": ann_ext,
                "start_sample": int(native_start),
                "end_sample": int(native_end),
                "center_sample": int(native_start + native_window_n // 2),
                "start_s": float(native_start / native_fs),
                "end_s": float(native_end / native_fs),
                "center_s": float((native_start + native_window_n // 2) / native_fs),
                # Cached-signal coordinate system expected by layer3_pretrain.py
                "signal_path": signal_path,
                "cached_fs": float(cached_fs),
                "cached_signal_len": int(cached_len),
                "start_idx": int(cache_start),
                "n_samples": int(cache_window_n),
                # Window parameters and labels
                "window_s": float(window_s),
                "stride_s": float(stride_s),
                "lead_index": int(lead_index),
                **label_info,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build Layer 3 ECG window index CSV from WFDB datasets.")
    p.add_argument("--data-dir", required=True, help="Root data directory, e.g. data")
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset folder names, comma-separated or space-separated")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--stride-s", type=float, default=1.0)
    p.add_argument("--ann-ext", default="atr", help="WFDB annotation extension, e.g. atr for MIT-BIH")
    p.add_argument("--normal-symbols", default="N", help="Comma-separated beat symbols treated as healthy baseline")
    p.add_argument("--min-beats", type=int, default=1, help="Minimum labeled beats required to call a window healthy")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--lead-index", type=int, default=0)
    p.add_argument("--target-fs", type=float, default=250.0, help="Cached .npy signal fs for pretraining. Set <=0 to keep native fs.")
    p.add_argument("--signal-dir", default=None, help="Where to cache .npy full-record signals. Default: <out-csv parent>/signals_npy")
    p.add_argument("--no-signal-cache", action="store_true", help="Only build validation index; signal_path/start_idx remain unusable for pretraining.")
    p.add_argument("--overwrite-signal-cache", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    datasets = parse_csv_list(args.datasets)
    normal_symbols = set(parse_csv_list(args.normal_symbols)) or set(DEFAULT_NORMAL_SYMBOLS)
    out_csv = Path(args.out_csv)
    target_fs = None if args.target_fs is not None and args.target_fs <= 0 else float(args.target_fs)
    signal_dir: Optional[Path]
    if args.no_signal_cache:
        signal_dir = None
    elif args.signal_dir is None:
        signal_dir = out_csv.parent / "signals_npy"
    else:
        signal_dir = Path(args.signal_dir)

    all_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds_dir = data_dir / dataset
        if not ds_dir.exists():
            LOG.warning("Dataset directory not found: %s", ds_dir)
            continue
        records = find_wfdb_records(ds_dir)
        if not records:
            LOG.warning("No .hea files under %s", ds_dir)
            continue
        if args.max_records is not None:
            records = records[: int(args.max_records)]
        LOG.info("%s: %d WFDB records to process", dataset, len(records))
        for record in records:
            try:
                rows = build_index_for_record(
                    data_dir=data_dir,
                    dataset=dataset,
                    record=record,
                    window_s=args.window_s,
                    stride_s=args.stride_s,
                    ann_ext=args.ann_ext,
                    normal_symbols=normal_symbols,
                    min_beats=args.min_beats,
                    signal_dir=signal_dir,
                    lead_index=args.lead_index,
                    target_fs=target_fs,
                    overwrite_signal_cache=args.overwrite_signal_cache,
                )
                all_rows.extend(rows)
                LOG.info("%s/%s: %d windows", dataset, record, len(rows))
            except Exception as exc:
                LOG.warning("Failed on %s/%s: %s", dataset, record, exc)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    if df.empty:
        LOG.warning("Index is empty. Check --data-dir / --datasets / --ann-ext. Writing empty CSV anyway.")
    else:
        df.insert(0, "window_id", np.arange(len(df), dtype=int))
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    LOG.info("Wrote %d windows to %s", len(df), out_csv)
    if signal_dir is not None:
        LOG.info("Cached .npy signals in %s", signal_dir)


if __name__ == "__main__":
    main()
