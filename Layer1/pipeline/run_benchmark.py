"""
run_benchmark.py
----------------
Evaluates the fast causal Layer 1 pipeline across PhysioNet datasets.

Reports detector-only and supervisor-accepted performance vs oracle beats.

USAGE
-----
    python Layer1/pipeline/run_benchmark.py --data-dir ./data --out-dir ./benchmark_results
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install wfdb scipy numpy pandas")

_PIPELINE = Path(__file__).resolve().parent
_DATA = _PIPELINE.parent.parent / "data"
sys.path.insert(0, str(_PIPELINE))
sys.path.insert(0, str(_DATA))

from dataset_registry import GROUP_ORDER, dataset_dir, list_datasets
from main_pipeline import run_layer1
from reference_annotations import get_reference_beats, greedy_match


def _metrics(ref_samples: np.ndarray, test_samples: np.ndarray, fs: float, tol_ms: float) -> Dict:
    if len(ref_samples) == 0:
        return dict(tp=0, fp=0, fn=0, sensitivity=float("nan"), ppv=float("nan"),
                      f1=float("nan"), mean_abs_jitter_ms=float("nan"))
    return greedy_match(ref_samples, test_samples, fs, tol_ms=tol_ms)


def run_one_record(
    record_stem: str,
    channel: int,
    tol_ms: float = 100.0,
) -> Optional[Dict]:
    rec = wfdb.rdrecord(record_stem)
    fs = float(rec.fs)
    if rec.p_signal is None or rec.p_signal.shape[0] == 0:
        return None
    ch = min(channel, rec.p_signal.shape[1] - 1)
    raw = rec.p_signal[:, ch].astype(float)
    duration_s = len(raw) / fs

    try:
        ann = wfdb.rdann(record_stem, "atr")
        ref_samples, _ = get_reference_beats(ann)
    except Exception:
        ref_samples = np.asarray([], dtype=int)

    result = run_layer1(raw, fs, ref_samples=ref_samples, match_tol_ms=tol_ms)
    sup = result.supervisor.state

    det_m = _metrics(ref_samples, result.candidate_samples, fs, tol_ms)
    acc_m = result.metrics or _metrics(ref_samples, result.accepted_samples, fs, tol_ms)

    def _row(prefix: str, m: Dict, n_test: int) -> Dict:
        fp_per_hour = m["fp"] / (duration_s / 3600.0) if duration_s > 0 else float("nan")
        return {
            "n_ref_beats": int(len(ref_samples)),
            "n_candidates": int(len(result.candidate_samples)),
            "n_accepted": int(len(result.accepted_samples)),
            f"{prefix}_tp": m["tp"], f"{prefix}_fp": m["fp"], f"{prefix}_fn": m["fn"],
            f"{prefix}_sensitivity": round(float(m["sensitivity"]), 4) if np.isfinite(m["sensitivity"]) else float("nan"),
            f"{prefix}_ppv": round(float(m["ppv"]), 4) if np.isfinite(m["ppv"]) else float("nan"),
            f"{prefix}_f1": round(float(m["f1"]), 4) if np.isfinite(m["f1"]) else float("nan"),
            f"{prefix}_fp_per_hour": round(float(fp_per_hour), 2) if np.isfinite(fp_per_hour) else float("nan"),
            f"{prefix}_jitter_ms": round(float(m["mean_abs_jitter_ms"]), 2) if np.isfinite(m["mean_abs_jitter_ms"]) else float("nan"),
        }

    recovery_per_min = sup.n_recovery_entries / (duration_s / 60.0) if duration_s > 0 else float("nan")

    return {
        "duration_s": round(duration_s, 1),
        "fs_hz": fs,
        "polarity": result.detector.polarity,
        **{k: v for k, v in _row("det", det_m, len(result.candidate_samples)).items()},
        **{k: v for k, v in _row("acc", acc_m, len(result.accepted_samples)).items()},
        "n_recovery_entries": sup.n_recovery_entries,
        "n_recalibrations": sup.n_recalibrations,
        "recovery_per_min": round(float(recovery_per_min), 4) if np.isfinite(recovery_per_min) else float("nan"),
        "n_reject_short": sup.n_reject_short,
        "n_reject_long": sup.n_reject_long,
        "n_reject_out_of_band": sup.n_reject_out_of_band,
    }


def run_benchmark(
    data_dir: Path,
    out_dir: Path,
    channel: int = 0,
    tol_ms: float = 100.0,
    record_limit: Optional[int] = None,
    datasets: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("benchmark")
    log.info("Fast causal Layer 1 benchmark start")

    rows: List[Dict] = []
    specs = list_datasets(datasets)
    for spec in specs:
        ds_dir = dataset_dir(data_dir, spec.folder)
        if not ds_dir.is_dir():
            log.warning(f"[{spec.folder}] not found — skipping")
            continue
        hea_files = sorted(ds_dir.glob("*.hea"))
        if record_limit:
            hea_files = hea_files[:record_limit]
        log.info(f"[{spec.folder}] ({spec.title}) {len(hea_files)} records")

        for hea in hea_files:
            stem = str(hea.with_suffix(""))
            t0 = time.perf_counter()
            try:
                res = run_one_record(stem, channel, tol_ms)
                if res is None:
                    continue
                rows.append({
                    "dataset": spec.folder,
                    "physionet_id": spec.physionet_id,
                    "group": spec.group,
                    "record": hea.stem,
                    **res,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                })
                log.info(
                    f"  {hea.stem:12s} det Se={res['det_sensitivity']:.3f} "
                    f"acc Se={res['acc_sensitivity']:.3f} FP/h={res['acc_fp_per_hour']:.0f}"
                )
            except Exception as exc:
                log.error(f"  {hea.stem}: FAILED — {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_record.csv", index=False)

    summary_rows = []
    for score in ("det", "acc"):
        df_valid = df[df[f"{score}_sensitivity"].notna() & (df["n_ref_beats"] > 0)]
        for group in GROUP_ORDER:
            g = df_valid[df_valid["group"] == group]
            if g.empty:
                continue
            summary_rows.append({
                "group": group, "score": score, "n_records": len(g),
                "sensitivity_mean": round(float(g[f"{score}_sensitivity"].mean()), 4),
                "ppv_mean": round(float(g[f"{score}_ppv"].mean()), 4),
                "f1_mean": round(float(g[f"{score}_f1"].mean()), 4),
                "fp_per_hour_mean": round(float(g[f"{score}_fp_per_hour"].mean()), 2),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "per_group_summary.csv", index=False)
    log.info("Benchmark complete.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--tol-ms", type=float, default=100.0)
    p.add_argument("--record-limit", type=int, default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args(argv)
    run_benchmark(args.data_dir, args.out_dir, args.channel, args.tol_ms,
                  args.record_limit, args.datasets)


if __name__ == "__main__":
    main()
