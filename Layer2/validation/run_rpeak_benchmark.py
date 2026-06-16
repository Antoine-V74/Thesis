"""
Benchmark Layer 2 r_peak_detector.py against annotated R-peaks.

Compares detected peaks to expert annotations and writes detection metrics only.

Usage:
    .venv\\Scripts\\python.exe Layer2\\validation\\run_rpeak_benchmark.py `
        --data-dir data --datasets mitdb `
        --out-dir Results\\layer2\\rpeak_benchmark_mitdb
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    sys.exit("Missing dependency: pip install wfdb")

_HERE = Path(__file__).resolve().parent
_L2 = _HERE.parent
_ROOT = _L2.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_L2))
sys.path.insert(0, str(_DATA))

from _bootstrap import setup_layer2_paths  # noqa: E402
from dataset_registry import dataset_dir, resolve_dataset  # noqa: E402
from r_peak_detector import detect_r_peaks  # noqa: E402

setup_layer2_paths()

# NSTDB SNR encoded in record stem (dB). See data/noise_stress_test/*.hea headers.
NSTDB_SNR: Dict[str, Optional[int]] = {
    "118e_6": -6, "119e_6": -6,
    "118e00": 0, "119e00": 0,
    "118e06": 6, "119e06": 6,
    "118e12": 12, "119e12": 12,
    "118e18": 18, "119e18": 18,
    "118e24": 24, "119e24": 24,
}


def nstdb_record_snr(record_stem: str) -> Optional[int]:
    return NSTDB_SNR.get(record_stem)


def should_include_record(dataset_folder: str, record_stem: str, nstdb_min_snr: int) -> bool:
    """Skip NSTDB records below the SNR threshold; keep other datasets unchanged."""
    if dataset_folder != "noise_stress_test":
        return True
    snr = nstdb_record_snr(record_stem)
    if snr is None:
        return False
    return snr >= nstdb_min_snr


def beat_label(symbol: str, dataset: str) -> str:
    info = resolve_dataset(dataset)
    if info.group == "vt_vfib":
        return "abnormal_v"
    if info.folder == "noise_stress_test":
        return "abnormal_noise"
    if symbol in info.normal_beats:
        return "healthy"
    if symbol in info.abnormal_beats:
        return "abnormal_v"
    return "mixed"


def greedy_match(
    ref_s: np.ndarray,
    det_s: np.ndarray,
    tol_s: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    pairs: List[Tuple[float, int, int]] = []
    for i, r in enumerate(ref_s):
        cand = np.where((det_s >= r - tol_s) & (det_s <= r + tol_s))[0]
        for j in cand:
            pairs.append((abs(float(det_s[j] - r)), i, int(j)))
    pairs.sort(key=lambda x: x[0])

    used_ref, used_det = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for dt, i, j in pairs:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        matches.append((i, j, float(det_s[j] - ref_s[i])))

    missed = [i for i in range(len(ref_s)) if i not in used_ref]
    extras = [j for j in range(len(det_s)) if j not in used_det]
    return matches, missed, extras


def detection_metrics(
    n_ref: int,
    n_det: int,
    n_matched: int,
    timing_ms: np.ndarray,
    duration_s: float,
) -> Dict[str, float]:
    sensitivity = n_matched / n_ref if n_ref else float("nan")
    ppv = n_matched / n_det if n_det else float("nan")
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if n_ref and n_det and (sensitivity + ppv) > 0
        else float("nan")
    )
    missed = n_ref - n_matched
    extra = n_det - n_matched
    hours = max(duration_s / 3600.0, 1e-9)
    return {
        "n_annotated_beats": n_ref,
        "n_detected_peaks": n_det,
        "n_matched": n_matched,
        "n_missed": missed,
        "n_extra": extra,
        "sensitivity": round(sensitivity, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "miss_rate": round(missed / n_ref, 4) if n_ref else float("nan"),
        "extra_rate": round(extra / n_det, 4) if n_det else float("nan"),
        "extra_peaks_per_hour": round(extra / hours, 2),
        "median_abs_timing_error_ms": round(float(np.median(np.abs(timing_ms))), 2) if len(timing_ms) else float("nan"),
        "mean_abs_timing_error_ms": round(float(np.mean(np.abs(timing_ms))), 2) if len(timing_ms) else float("nan"),
        "p95_abs_timing_error_ms": round(float(np.percentile(np.abs(timing_ms), 95)), 2) if len(timing_ms) else float("nan"),
    }


def benchmark_record(
    stem: Path,
    dataset: str,
    match_tol_s: float,
) -> Tuple[List[Dict], List[Dict], Dict, List[Dict]]:
    info = resolve_dataset(dataset)
    rec = wfdb.rdrecord(str(stem))
    ann = wfdb.rdann(str(stem), info.ann_ext)
    ch = min(info.channel, rec.p_signal.shape[1] - 1)
    raw = rec.p_signal[:, ch].astype(float)
    fs = float(rec.fs)
    duration_s = len(raw) / fs

    det = detect_r_peaks(raw, fs)
    det_s = det.peak_samples.astype(float) / fs

    ref_rows: List[Dict] = []
    ref_s_list: List[float] = []
    for sample, sym in zip(ann.sample, ann.symbol):
        lbl = beat_label(sym, dataset)
        if lbl == "mixed":
            continue
        t = sample / fs
        ref_s_list.append(t)
        ref_rows.append({
            "dataset": dataset,
            "record": stem.name,
            "beat_time_s": round(t, 3),
            "beat_symbol": sym,
            "label": lbl,
        })

    ref_s = np.asarray(ref_s_list, dtype=float)
    matches, missed_idx, extra_idx = greedy_match(ref_s, det_s, match_tol_s)

    match_by_ref = {i: (j, err) for i, j, err in matches}
    beat_rows: List[Dict] = []
    for i, row in enumerate(ref_rows):
        matched = i in match_by_ref
        j, err_s = match_by_ref.get(i, (-1, float("nan")))
        beat_rows.append({
            **row,
            "detected": matched,
            "detected_time_s": round(float(det_s[j]), 3) if matched else float("nan"),
            "timing_error_ms": round(float(err_s * 1000.0), 3) if matched else float("nan"),
        })

    extra_rows = [
        {
            "dataset": dataset,
            "record": stem.name,
            "detected_time_s": round(float(det_s[j]), 3),
        }
        for j in extra_idx
    ]

    timing_ms = np.array([err * 1000.0 for _, _, err in matches], dtype=float)

    label_summaries: List[Dict] = []
    for lbl in ("healthy", "abnormal_v", "abnormal_noise"):
        idx = [i for i, r in enumerate(ref_rows) if r["label"] == lbl]
        if not idx:
            continue
        n_ref = len(idx)
        n_matched = sum(1 for i in idx if i in match_by_ref)
        lbl_timing = np.array([
            match_by_ref[i][1] * 1000.0 for i in idx if i in match_by_ref
        ], dtype=float)
        m = detection_metrics(n_ref, len(det_s), n_matched, lbl_timing, duration_s)
        label_summaries.append({
            "dataset": dataset,
            "record": stem.name,
            "label": lbl,
            **m,
        })

    overall = detection_metrics(len(ref_s), len(det_s), len(matches), timing_ms, duration_s)
    record_summary = {
        "dataset": dataset,
        "record": stem.name,
        "duration_s": round(duration_s, 1),
        "fs_hz": fs,
        "polarity": det.polarity,
        "match_tol_ms": round(match_tol_s * 1000.0, 1),
        **overall,
    }
    snr = nstdb_record_snr(stem.name)
    if snr is not None:
        record_summary["snr_db"] = snr
    if det.confirmation_delays_ms.size:
        record_summary["mean_confirmation_delay_ms"] = round(
            float(np.mean(det.confirmation_delays_ms)), 2)

    return beat_rows, extra_rows, record_summary, label_summaries


def run_benchmark(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "rpeak_benchmark.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    tol_s = args.match_tol_ms / 1000.0
    beat_rows: List[Dict] = []
    extra_rows: List[Dict] = []
    record_summaries: List[Dict] = []
    label_summaries: List[Dict] = []

    datasets = [resolve_dataset(d).folder for d in args.datasets]
    for dataset in datasets:
        ds_dir = dataset_dir(args.data_dir, dataset)
        hea_files = sorted(ds_dir.glob("*.hea"))
        if args.record_limit:
            hea_files = hea_files[:args.record_limit]
        included = [
            h for h in hea_files
            if should_include_record(dataset, h.stem, args.nstdb_min_snr)
        ]
        skipped = len(hea_files) - len(included)
        logging.info(
            "[%s] %d records (%d skipped: SNR < %d dB or non-benchmark)",
            dataset, len(included), skipped, args.nstdb_min_snr,
        )

        for hea in included:
            t0 = time.time()
            stem = hea.with_suffix("")
            try:
                beats, extras, rec_sum, lbl_rows = benchmark_record(stem, dataset, tol_s)
                beat_rows.extend(beats)
                extra_rows.extend(extras)
                record_summaries.append(rec_sum)
                label_summaries.extend(lbl_rows)
                logging.info("  %s %.1fs", hea.stem, time.time() - t0)
            except Exception as exc:
                logging.exception("  %s failed: %s", hea.stem, exc)

    per_beat = pd.DataFrame(beat_rows)
    extras_df = pd.DataFrame(extra_rows)
    per_record = pd.DataFrame(record_summaries)
    per_record_label = pd.DataFrame(label_summaries)

    per_beat.to_csv(args.out_dir / "per_beat_detection.csv", index=False)
    extras_df.to_csv(args.out_dir / "extra_peaks.csv", index=False)
    per_record.to_csv(args.out_dir / "per_record_overall.csv", index=False)
    per_record_label.to_csv(args.out_dir / "per_record_by_label.csv", index=False)

    if not per_record_label.empty:
        overall = []
        for (dataset, label), grp in per_record_label.groupby(["dataset", "label"]):
            n_ref = int(grp["n_annotated_beats"].sum())
            n_det = int(grp["n_detected_peaks"].sum())
            n_matched = int(grp["n_matched"].sum())
            dur = float(per_record[per_record["dataset"] == dataset]["duration_s"].sum())
            sens = n_matched / n_ref if n_ref else float("nan")
            ppv = n_matched / n_det if n_det else float("nan")
            f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv > 0 else float("nan")
            overall.append({
                "dataset": dataset,
                "label": label,
                "n_annotated_beats": n_ref,
                "n_detected_peaks": n_det,
                "n_matched": n_matched,
                "n_missed": n_ref - n_matched,
                "n_extra": n_det - n_matched,
                "sensitivity": round(sens, 4),
                "ppv": round(ppv, 4),
                "f1": round(f1, 4),
                "miss_rate": round((n_ref - n_matched) / n_ref, 4) if n_ref else float("nan"),
                "extra_peaks_per_hour": round((n_det - n_matched) / max(dur / 3600.0, 1e-9), 2),
            })
        pd.DataFrame(overall).to_csv(args.out_dir / "dataset_summary_by_label.csv", index=False)

    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "out_dir": str(args.out_dir), "data_dir": str(args.data_dir)}, f, indent=2)

    logging.info("Done -> %s", args.out_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/rpeak_benchmark"))
    p.add_argument("--datasets", nargs="+", default=["mitdb"])
    p.add_argument("--record-limit", type=int, default=None)
    p.add_argument("--match-tol-ms", type=float, default=100.0)
    p.add_argument(
        "--nstdb-min-snr", type=int, default=12,
        help="For noise_stress_test (NSTDB), only include records at or above this SNR (dB).",
    )
    args = p.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
