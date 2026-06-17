"""
Compare multiple R-peak detection algorithms on PhysioNet-style datasets.

Writes pooled and per-record metrics plus confirmation/timing summaries to
Results/rpeak_comparison/.

Usage:
    .venv\\Scripts\\python.exe RPeakDetection\\benchmark\\run_comparison.py `
        --data-dir data --datasets mitdb `
        --out-dir Results\\rpeak_comparison\\mitdb

    .venv\\Scripts\\python.exe RPeakDetection\\benchmark\\run_comparison.py `
        --data-dir data --datasets mitdb nstdb --record-limit 3
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

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DATA))

from dataset_registry import dataset_dir, is_beat_symbol, resolve_dataset  # noqa: E402
from RPeakDetection.algorithms import DEFAULT_ALGORITHMS, get_detector, list_algorithms  # noqa: E402
from RPeakDetection.benchmark.metrics import greedy_match, summarize_detection  # noqa: E402

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
    if dataset_folder != "noise_stress_test":
        return True
    snr = nstdb_record_snr(record_stem)
    if snr is None:
        return False
    return snr >= nstdb_min_snr


def dataset_rpeak_skip_reason(dataset: str, include_nondefault: bool) -> Optional[str]:
    info = resolve_dataset(dataset)
    if info.rpeak_ann_ext is None:
        return f"{info.folder}: {info.rpeak_notes}"
    if not info.rpeak_benchmark_default and not include_nondefault:
        return (
            f"{info.folder}: non-default R-peak reference "
            f"({info.rpeak_reference}). {info.rpeak_notes}"
        )
    return None


def load_reference_beats(
    stem: Path,
    dataset: str,
    fs: float,
    max_time_s: Optional[float] = None,
) -> Tuple[List[float], int]:
    info = resolve_dataset(dataset)
    if info.rpeak_ann_ext is None:
        raise ValueError(f"{info.folder} has no beat-level R-peak annotation extension")
    ann = wfdb.rdann(str(stem), info.rpeak_ann_ext)
    ref_s: List[float] = []
    for sample, sym in zip(ann.sample, ann.symbol):
        if not is_beat_symbol(sym):
            continue
        t = sample / fs
        if max_time_s is not None and t > max_time_s:
            continue
        ref_s.append(t)
    return ref_s, len(ann.sample)


def benchmark_one(
    algorithm: str,
    stem: Path,
    dataset: str,
    raw: np.ndarray,
    fs: float,
    duration_s: float,
    match_tol_s: float,
) -> Dict:
    detector = get_detector(algorithm)
    t0 = time.perf_counter()
    result = detector.detect(raw, fs)
    processing_ms = (time.perf_counter() - t0) * 1000.0

    max_time_s = duration_s if duration_s < float("inf") else None
    ref_s_list, _ = load_reference_beats(stem, dataset, fs, max_time_s=max_time_s)
    ref_s = np.asarray(ref_s_list, dtype=float)
    det_s = result.peak_times_s(fs)

    matches, _, _ = greedy_match(ref_s, det_s, match_tol_s)
    timing_errors = np.array([err for _, _, err in matches], dtype=float)
    conf_s = result.confirmation_times_s(fs)
    confirmation_timing_errors = np.array([], dtype=float)
    if len(conf_s) == len(det_s):
        confirmation_timing_errors = np.array(
            [float(conf_s[j] - ref_s[i]) for i, j, _ in matches],
            dtype=float,
        )

    summary = summarize_detection(
        n_ref=len(ref_s),
        n_det=len(det_s),
        n_matched=len(matches),
        timing_errors_s=timing_errors,
        confirmation_timing_errors_s=confirmation_timing_errors,
        duration_s=duration_s,
        processing_ms=processing_ms,
        result=result,
        fs=fs,
    )
    info = resolve_dataset(dataset)
    return {
        "algorithm": algorithm,
        "dataset": dataset,
        "record": stem.name,
        "duration_s": round(duration_s, 1),
        "fs_hz": fs,
        "match_tol_ms": round(match_tol_s * 1000.0, 1),
        "annotation_ext": info.rpeak_ann_ext,
        "annotation_reference": info.rpeak_reference,
        "notes": result.notes,
        "snr_db": nstdb_record_snr(stem.name),
        **summary,
    }


def aggregate_algorithm(rows: pd.DataFrame) -> Dict:
    n_ref = int(rows["n_annotated_beats"].sum())
    n_det = int(rows["n_detected_peaks"].sum())
    n_matched = int(rows["n_matched"].sum())
    if "total_duration_s" in rows.columns:
        dur = float(rows["total_duration_s"].sum())
    else:
        dur = float(rows["duration_s"].sum())
    sens = n_matched / n_ref if n_ref else float("nan")
    ppv = n_matched / n_det if n_det else float("nan")
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv > 0 else float("nan")

    def _wmean(col: str) -> float:
        if rows.empty or col not in rows.columns:
            return float("nan")
        vals = rows[col].astype(float)
        if vals.isna().all():
            return float("nan")
        weights = rows["n_matched"].astype(float)
        mask = vals.notna() & weights > 0
        if not mask.any():
            return float(np.nanmean(vals))
        return float(np.average(vals[mask], weights=weights[mask]))

    if "processing_ms" in rows.columns:
        mean_processing_ms_per_record = float(rows["processing_ms"].mean())
        total_processing_ms = float(rows["processing_ms"].sum())
    else:
        mean_processing_ms_per_record = float(rows["mean_processing_ms_per_record"].mean())
        total_processing_ms = float(
            (rows["mean_processing_ms_per_record"].astype(float) * rows["n_records"].astype(float)).sum()
        )

    return {
        "n_records": int(rows["n_records"].sum()) if "n_records" in rows.columns else len(rows),
        "total_duration_s": round(dur, 1),
        "n_annotated_beats": n_ref,
        "n_detected_peaks": n_det,
        "n_matched": n_matched,
        "n_missed": n_ref - n_matched,
        "n_extra": n_det - n_matched,
        "sensitivity": round(sens, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "extra_peaks_per_hour": round((n_det - n_matched) / max(dur / 3600.0, 1e-9), 2),
        "mean_confirmation_delay_ms": round(_wmean("mean_confirmation_delay_ms"), 2),
        "median_confirmation_delay_ms": round(_wmean("median_confirmation_delay_ms"), 2),
        "p95_confirmation_delay_ms": round(_wmean("p95_confirmation_delay_ms"), 2),
        "mean_detection_lag_ms": round(_wmean("mean_detection_lag_ms"), 2),
        "median_detection_lag_ms": round(_wmean("median_detection_lag_ms"), 2),
        "p95_detection_lag_ms": round(_wmean("p95_detection_lag_ms"), 2),
        "mean_confirmed_event_lag_ms": round(_wmean("mean_confirmed_event_lag_ms"), 2),
        "median_confirmed_event_lag_ms": round(_wmean("median_confirmed_event_lag_ms"), 2),
        "p95_confirmed_event_lag_ms": round(_wmean("p95_confirmed_event_lag_ms"), 2),
        "mean_abs_confirmed_event_error_ms": round(_wmean("mean_abs_confirmed_event_error_ms"), 2),
        "mean_abs_timing_error_ms": round(_wmean("mean_abs_timing_error_ms"), 2),
        "median_abs_timing_error_ms": round(_wmean("median_abs_timing_error_ms"), 2),
        "mean_processing_ms_per_record": round(mean_processing_ms_per_record, 2),
        "mean_processing_ms_per_second_signal": round(
            total_processing_ms / max(dur, 1e-9), 4
        ),
        "is_causal": bool(rows["is_causal"].iloc[0]) if len(rows) else False,
        "uses_prefilter": bool(rows["uses_prefilter"].iloc[0]) if len(rows) else False,
        "has_explicit_confirmation": bool(rows["has_explicit_confirmation"].iloc[0]) if len(rows) else False,
    }


def build_algorithm_summaries(per_algo_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    algo_rows: List[Dict] = []
    timing_rows: List[Dict] = []
    if per_algo_dataset.empty:
        return pd.DataFrame(), pd.DataFrame()

    for algorithm, grp in per_algo_dataset.groupby("algorithm"):
        agg = aggregate_algorithm(grp)
        algo_rows.append({"algorithm": algorithm, **agg})
        timing_rows.append({
            "algorithm": algorithm,
            "is_causal": agg["is_causal"],
            "has_explicit_confirmation": agg["has_explicit_confirmation"],
            "mean_confirmation_delay_ms": agg["mean_confirmation_delay_ms"],
            "median_confirmation_delay_ms": agg["median_confirmation_delay_ms"],
            "p95_confirmation_delay_ms": agg["p95_confirmation_delay_ms"],
            "mean_detection_lag_ms": agg["mean_detection_lag_ms"],
            "median_detection_lag_ms": agg["median_detection_lag_ms"],
            "p95_detection_lag_ms": agg["p95_detection_lag_ms"],
            "mean_confirmed_event_lag_ms": agg["mean_confirmed_event_lag_ms"],
            "median_confirmed_event_lag_ms": agg["median_confirmed_event_lag_ms"],
            "p95_confirmed_event_lag_ms": agg["p95_confirmed_event_lag_ms"],
            "mean_abs_confirmed_event_error_ms": agg["mean_abs_confirmed_event_error_ms"],
            "mean_abs_timing_error_ms": agg["mean_abs_timing_error_ms"],
            "median_abs_timing_error_ms": agg["median_abs_timing_error_ms"],
            "notes": (
                "detection_lag_ms: matched reported peak minus annotation. "
                "confirmed_event_lag_ms: live confirmation event minus annotation for causal detectors. "
                "Stim command time = confirmed event time + policy offset when using confirmed causal output."
            ),
        })
    return pd.DataFrame(algo_rows), pd.DataFrame(timing_rows)


def build_macro_summary(per_algo_dataset: pd.DataFrame) -> pd.DataFrame:
    """Mean of per-dataset metrics, so large datasets do not dominate the table."""
    if per_algo_dataset.empty:
        return pd.DataFrame()

    valid = per_algo_dataset[per_algo_dataset["n_annotated_beats"].astype(float) > 0]
    rows: List[Dict] = []
    for algorithm, grp in valid.groupby("algorithm"):
        row = {"algorithm": algorithm, "n_datasets": int(grp["dataset"].nunique())}
        for col in (
            "sensitivity",
            "ppv",
            "f1",
            "extra_peaks_per_hour",
            "mean_detection_lag_ms",
            "mean_confirmed_event_lag_ms",
            "mean_abs_timing_error_ms",
            "mean_abs_confirmed_event_error_ms",
        ):
            if col in grp.columns:
                vals = grp[col].astype(float)
                row[f"macro_{col}"] = round(float(vals.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summaries(
    per_record: pd.DataFrame,
    out_dir: Path,
    write_per_record: bool = False,
    merge_per_dataset: Optional[Path] = None,
    run_datasets: Optional[List[str]] = None,
) -> None:
    """Write summary CSVs; optionally merge with an existing per_algorithm_per_dataset file."""
    if write_per_record and not per_record.empty:
        per_record.to_csv(out_dir / "per_record.csv", index=False)
    elif not write_per_record:
        per_record_path = out_dir / "per_record.csv"
        if per_record_path.exists():
            per_record_path.unlink()

    algo_dataset_rows: List[Dict] = []
    if not per_record.empty:
        for (algorithm, dataset), grp in per_record.groupby(["algorithm", "dataset"]):
            agg = aggregate_algorithm(grp)
            algo_dataset_rows.append({"algorithm": algorithm, "dataset": dataset, **agg})

    new_per_algo_dataset = pd.DataFrame(algo_dataset_rows)
    if merge_per_dataset is not None and merge_per_dataset.exists():
        prior = pd.read_csv(merge_per_dataset)
        if run_datasets:
            prior = prior[~prior["dataset"].isin(run_datasets)]
        if not new_per_algo_dataset.empty:
            per_algo_dataset = pd.concat([prior, new_per_algo_dataset], ignore_index=True)
        else:
            per_algo_dataset = prior
    else:
        per_algo_dataset = new_per_algo_dataset

    per_algo_dataset.to_csv(out_dir / "per_algorithm_per_dataset.csv", index=False)
    algo_summary, timing_summary = build_algorithm_summaries(per_algo_dataset)
    algo_summary.to_csv(out_dir / "algorithm_summary.csv", index=False)
    timing_summary.to_csv(out_dir / "timing_summary.csv", index=False)
    build_macro_summary(per_algo_dataset).to_csv(out_dir / "macro_algorithm_summary.csv", index=False)

    if not per_record.empty and "snr_db" in per_record.columns:
        snr_rows: List[Dict] = []
        snr_df = per_record[per_record["snr_db"].notna()]
        for (algorithm, dataset, snr_db), grp in snr_df.groupby(["algorithm", "dataset", "snr_db"]):
            agg = aggregate_algorithm(grp)
            snr_rows.append({
                "algorithm": algorithm,
                "dataset": dataset,
                "snr_db": int(float(snr_db)),
                **agg,
            })
        snr_path = out_dir / "per_algorithm_per_snr.csv"
        if snr_rows:
            pd.DataFrame(snr_rows).to_csv(snr_path, index=False)
        elif snr_path.exists():
            snr_path.unlink()


def run_comparison(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "comparison.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    algorithms = args.algorithms or DEFAULT_ALGORITHMS
    tol_s = args.match_tol_ms / 1000.0
    per_record_rows: List[Dict] = []

    datasets: List[str] = []
    skipped_datasets: List[Dict[str, str]] = []
    for d in args.datasets:
        info = resolve_dataset(d)
        reason = dataset_rpeak_skip_reason(info.folder, args.include_nondefault_annotations)
        if reason is not None:
            logging.warning("Skipping %s", reason)
            skipped_datasets.append({"dataset": info.folder, "reason": reason})
            continue
        datasets.append(info.folder)

    for dataset in datasets:
        ds_dir = dataset_dir(args.data_dir, dataset)
        hea_files = sorted(ds_dir.glob("*.hea"))
        if args.record_limit:
            hea_files = hea_files[:args.record_limit]

        included = [
            h for h in hea_files
            if should_include_record(dataset, h.stem, args.nstdb_min_snr)
        ]
        logging.info("[%s] %d records", dataset, len(included))

        for hea in included:
            stem = hea.with_suffix("")
            try:
                rec = wfdb.rdrecord(str(stem))
            except Exception as exc:
                logging.exception("  %s load failed: %s", hea.stem, exc)
                continue

            info = resolve_dataset(dataset)
            ch = min(info.channel, rec.p_signal.shape[1] - 1)
            raw = rec.p_signal[:, ch].astype(float)
            fs = float(rec.fs)
            if args.max_duration_s is not None:
                max_n = int(args.max_duration_s * fs)
                raw = raw[:max_n]
            duration_s = len(raw) / fs

            for alg in algorithms:
                t0 = time.time()
                try:
                    row = benchmark_one(alg, stem, dataset, raw, fs, duration_s, tol_s)
                    per_record_rows.append(row)
                    logging.info(
                        "  %s %s ppv=%.3f sens=%.3f conf=%.1fms lag=%.1fms %.1fs",
                        alg, hea.stem, row["ppv"], row["sensitivity"],
                        row.get("mean_confirmation_delay_ms", float("nan")),
                        row.get("mean_detection_lag_ms", float("nan")),
                        time.time() - t0,
                    )
                except Exception as exc:
                    logging.exception("  %s %s failed: %s", alg, hea.stem, exc)

        write_summaries(
            pd.DataFrame(per_record_rows),
            args.out_dir,
            write_per_record=args.write_per_record,
            merge_per_dataset=args.merge_per_dataset,
            run_datasets=datasets,
        )
        logging.info("[%s] summaries updated", dataset)

    write_summaries(
        pd.DataFrame(per_record_rows),
        args.out_dir,
        write_per_record=args.write_per_record,
        merge_per_dataset=args.merge_per_dataset,
        run_datasets=datasets,
    )

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "available_algorithms": list_algorithms(),
        "included_datasets": datasets,
        "skipped_datasets": skipped_datasets,
    }
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logging.info("Done -> %s", args.out_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("Results/rpeak_comparison"))
    p.add_argument("--datasets", nargs="+", default=["mitdb"])
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=f"Subset of: {', '.join(DEFAULT_ALGORITHMS)}",
    )
    p.add_argument("--record-limit", type=int, default=None)
    p.add_argument("--match-tol-ms", type=float, default=100.0)
    p.add_argument("--nstdb-min-snr", type=int, default=12)
    p.add_argument(
        "--max-duration-s",
        type=float,
        default=1800.0,
        help="Truncate each record to this many seconds (default 1800, like Layer 1 benchmark).",
    )
    p.add_argument(
        "--write-per-record",
        action="store_true",
        help="Write per_record.csv (large). Default: summary CSVs only.",
    )
    p.add_argument(
        "--include-nondefault-annotations",
        action="store_true",
        help=(
            "Also score datasets whose R-peak references are secondary or explicitly "
            "non-definitive. Rhythm-only datasets are still skipped."
        ),
    )
    p.add_argument(
        "--merge-per-dataset",
        type=Path,
        default=None,
        help="Merge new dataset rows into an existing per_algorithm_per_dataset.csv.",
    )
    args = p.parse_args(argv)
    run_comparison(args)


if __name__ == "__main__":
    main()
