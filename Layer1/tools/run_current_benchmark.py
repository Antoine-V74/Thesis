"""
Run a detailed Layer 1 benchmark and write fresh structured results.

Outputs are written under:

    Results/layer1/current_benchmark_<timestamp>/
        csv/
            per_record_all.csv
            per_dataset_summary.csv
            per_label_summary_all.csv
            per_beat_all.csv
            per_window_10s_all.csv
            worst_windows_10s.csv
            good_windows_10s.csv
            <dataset>_per_record.csv
            <dataset>_per_label_summary.csv
            <dataset>_per_window_10s.csv
        README.md

The goal is to answer:
  - Which datasets/records dominate the statistics?
  - How many normal vs arrhythmia beats are detected, accepted, and triggered?
  - How long after the annotated R-peak does Layer 1 confirm a candidate?
  - Which 10 s windows are worst/good examples for inspection?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\n  pip install wfdb")

_HERE = Path(__file__).resolve().parent
_L1 = _HERE.parent
_ROOT = _L1.parent
_DATA = _ROOT / "data"
sys.path.insert(0, str(_L1))
sys.path.insert(0, str(_DATA))

from _bootstrap import setup_layer1_paths  # noqa: E402

setup_layer1_paths(include_archive=False)

from dataset_registry import DatasetInfo, dataset_dir, list_datasets  # noqa: E402
from main_pipeline import run_layer1  # noqa: E402
from reference_annotations import get_reference_beats  # noqa: E402


BEAT_SYMBOLS = frozenset({"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q", "n", "?", "!"})


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _classify_symbol(symbol: str, spec: DatasetInfo) -> str:
    if symbol in spec.normal_beats:
        return "healthy"
    if symbol in spec.abnormal_beats:
        return "arrhythmia"
    if symbol in BEAT_SYMBOLS:
        return "other_beat"
    return "nonbeat"


def _load_annotations(stem: str, ann_ext: str) -> Tuple[np.ndarray, np.ndarray]:
    ann = wfdb.rdann(stem, ann_ext)
    samples, symbols = get_reference_beats(ann)
    return samples, symbols


def _match_refs_to_tests(
    ref_samples: np.ndarray,
    test_samples: np.ndarray,
    tol_samples: int,
) -> Tuple[List[Optional[int]], np.ndarray]:
    """Greedy monotonic one-to-one matching from reference beats to test beats."""
    matches: List[Optional[int]] = [None] * len(ref_samples)
    used = np.zeros(len(test_samples), dtype=bool)
    j = 0
    for i, ref in enumerate(ref_samples):
        while j < len(test_samples) and test_samples[j] < ref - tol_samples:
            j += 1
        best = None
        best_dist = tol_samples + 1
        k = j
        while k < len(test_samples) and test_samples[k] <= ref + tol_samples:
            if not used[k]:
                dist = abs(int(test_samples[k]) - int(ref))
                if dist < best_dist:
                    best = k
                    best_dist = dist
            k += 1
        if best is not None:
            matches[i] = int(best)
            used[best] = True
    return matches, used


def _decision_by_sample(result) -> Dict[int, object]:
    return {int(d.sample): d for d in result.supervisor.state.decisions}


def analyze_record(
    spec: DatasetInfo,
    stem_path: Path,
    *,
    channel_override: Optional[int],
    match_tol_ms: float,
    filter_mode: str,
    max_duration_s: Optional[float],
) -> Tuple[Dict, List[Dict], List[Dict]]:
    rec = wfdb.rdrecord(str(stem_path))
    if rec.p_signal is None or rec.p_signal.shape[0] == 0:
        raise RuntimeError("empty signal")

    fs = float(rec.fs)
    channel = spec.channel if channel_override is None else channel_override
    channel = min(channel, rec.p_signal.shape[1] - 1)
    raw = rec.p_signal[:, channel].astype(float)
    if max_duration_s is not None:
        raw = raw[: int(round(max_duration_s * fs))]
    duration_s = len(raw) / fs

    ref_samples, ref_symbols = _load_annotations(str(stem_path), spec.ann_ext)
    valid_mask = (ref_samples >= 0) & (ref_samples < len(raw))
    ref_samples = ref_samples[valid_mask]
    ref_symbols = ref_symbols[valid_mask]

    result = run_layer1(raw, fs, filter_mode=filter_mode)
    decisions = _decision_by_sample(result)
    candidate_samples = np.asarray(result.candidate_samples, dtype=int)
    accepted_samples = np.asarray(result.accepted_samples, dtype=int)
    trigger_samples = np.asarray(result.trigger_samples, dtype=int)
    confirmation_samples = np.asarray(result.detector.confirmation_samples, dtype=int)

    tol_samples = int(round(match_tol_ms * fs / 1000.0))
    candidate_matches, used_candidates = _match_refs_to_tests(
        ref_samples, candidate_samples, tol_samples,
    )
    accepted_matches, used_accepted = _match_refs_to_tests(
        ref_samples, accepted_samples, tol_samples,
    )
    trigger_matches, used_triggers = _match_refs_to_tests(
        ref_samples, trigger_samples, tol_samples,
    )

    per_beat_rows: List[Dict] = []
    label_counts: Dict[str, Dict[str, float]] = {}

    def label_bucket(label: str) -> Dict[str, float]:
        if label not in label_counts:
            label_counts[label] = {
                "n_ref": 0,
                "n_candidate": 0,
                "n_accepted": 0,
                "n_triggered": 0,
                "n_missed_candidate": 0,
                "n_missed_accepted": 0,
                "n_missed_trigger": 0,
                "candidate_delay_ms_sum": 0.0,
                "candidate_delay_ms_abs_sum": 0.0,
                "confirmation_delay_ms_sum": 0.0,
                "algorithm_after_label_ms_sum": 0.0,
                "candidate_delay_ms_values": [],
                "algorithm_after_label_ms_values": [],
            }
        return label_counts[label]

    for beat_idx, (ref_sample, symbol) in enumerate(zip(ref_samples, ref_symbols)):
        label = _classify_symbol(str(symbol), spec)
        bucket = label_bucket(label)
        bucket["n_ref"] += 1

        cand_idx = candidate_matches[beat_idx]
        acc_idx = accepted_matches[beat_idx]
        trig_idx = trigger_matches[beat_idx]

        candidate_sample = int(candidate_samples[cand_idx]) if cand_idx is not None else None
        accepted_sample = int(accepted_samples[acc_idx]) if acc_idx is not None else None
        trigger_sample = int(trigger_samples[trig_idx]) if trig_idx is not None else None

        if cand_idx is not None:
            bucket["n_candidate"] += 1
        else:
            bucket["n_missed_candidate"] += 1
        if acc_idx is not None:
            bucket["n_accepted"] += 1
        else:
            bucket["n_missed_accepted"] += 1
        if trig_idx is not None:
            bucket["n_triggered"] += 1
        else:
            bucket["n_missed_trigger"] += 1

        peak_delay_ms = float("nan")
        algorithm_after_label_ms = float("nan")
        detector_confirmation_delay_ms = float("nan")
        confirmation_sample = None
        if cand_idx is not None and candidate_sample is not None:
            peak_delay_ms = 1000.0 * (candidate_sample - int(ref_sample)) / fs
            bucket["candidate_delay_ms_sum"] += peak_delay_ms
            bucket["candidate_delay_ms_abs_sum"] += abs(peak_delay_ms)
            bucket["candidate_delay_ms_values"].append(peak_delay_ms)
            if cand_idx < len(confirmation_samples):
                confirmation_sample = int(confirmation_samples[cand_idx])
                detector_confirmation_delay_ms = 1000.0 * (confirmation_sample - candidate_sample) / fs
                algorithm_after_label_ms = 1000.0 * (confirmation_sample - int(ref_sample)) / fs
                bucket["confirmation_delay_ms_sum"] += detector_confirmation_delay_ms
                bucket["algorithm_after_label_ms_sum"] += algorithm_after_label_ms
                bucket["algorithm_after_label_ms_values"].append(algorithm_after_label_ms)

        decision = decisions.get(accepted_sample or candidate_sample)
        per_beat_rows.append({
            "dataset": spec.folder,
            "group": spec.group,
            "record": stem_path.name,
            "fs_hz": fs,
            "channel": channel,
            "ref_sample": int(ref_sample),
            "ref_time_s": int(ref_sample) / fs,
            "symbol": str(symbol),
            "label": label,
            "candidate_matched": cand_idx is not None,
            "accepted_matched": acc_idx is not None,
            "triggered_matched": trig_idx is not None,
            "candidate_sample": candidate_sample,
            "accepted_sample": accepted_sample,
            "trigger_sample": trigger_sample,
            "confirmation_sample": confirmation_sample,
            "candidate_peak_delay_ms": peak_delay_ms,
            "detector_confirmation_delay_ms": detector_confirmation_delay_ms,
            "algorithm_after_label_ms": algorithm_after_label_ms,
            "supervisor_decision": decision.decision if decision is not None else "",
            "supervisor_mode": decision.mode if decision is not None else "",
        })

    extra_candidates = candidate_samples[~used_candidates]
    extra_accepted = accepted_samples[~used_accepted]
    extra_triggers = trigger_samples[~used_triggers]

    label_rows: List[Dict] = []
    for label, values in sorted(label_counts.items()):
        n_ref = int(values["n_ref"])
        cand_values = np.asarray(values["candidate_delay_ms_values"], dtype=float)
        alg_values = np.asarray(values["algorithm_after_label_ms_values"], dtype=float)
        label_rows.append({
            "dataset": spec.folder,
            "group": spec.group,
            "record": stem_path.name,
            "label": label,
            "n_ref": n_ref,
            "n_candidate": int(values["n_candidate"]),
            "n_accepted": int(values["n_accepted"]),
            "n_triggered": int(values["n_triggered"]),
            "candidate_sensitivity": _safe_div(values["n_candidate"], n_ref),
            "accepted_sensitivity": _safe_div(values["n_accepted"], n_ref),
            "trigger_rate_vs_ref": _safe_div(values["n_triggered"], n_ref),
            "candidate_peak_delay_ms_mean": float(np.nanmean(cand_values)) if len(cand_values) else float("nan"),
            "candidate_peak_abs_delay_ms_mean": float(np.nanmean(np.abs(cand_values))) if len(cand_values) else float("nan"),
            "candidate_peak_delay_ms_p95_abs": float(np.nanpercentile(np.abs(cand_values), 95)) if len(cand_values) else float("nan"),
            "algorithm_after_label_ms_mean": float(np.nanmean(alg_values)) if len(alg_values) else float("nan"),
            "algorithm_after_label_ms_p95": float(np.nanpercentile(alg_values, 95)) if len(alg_values) else float("nan"),
        })

    sup_state = result.supervisor.state
    summary = {
        "dataset": spec.folder,
        "physionet_id": spec.physionet_id,
        "group": spec.group,
        "record": stem_path.name,
        "duration_s": duration_s,
        "fs_hz": fs,
        "channel": channel,
        "polarity": result.detector.polarity,
        "filter_mode": filter_mode,
        "n_ref_beats": int(len(ref_samples)),
        "n_ref_healthy": int(sum(_classify_symbol(str(s), spec) == "healthy" for s in ref_symbols)),
        "n_ref_arrhythmia": int(sum(_classify_symbol(str(s), spec) == "arrhythmia" for s in ref_symbols)),
        "n_candidates": int(len(candidate_samples)),
        "n_accepted": int(len(accepted_samples)),
        "n_triggers": int(len(trigger_samples)),
        "extra_candidates": int(len(extra_candidates)),
        "extra_accepted": int(len(extra_accepted)),
        "extra_triggers": int(len(extra_triggers)),
        "candidate_ppv_vs_all_ref": _safe_div(len(candidate_samples) - len(extra_candidates), len(candidate_samples)),
        "accepted_ppv_vs_all_ref": _safe_div(len(accepted_samples) - len(extra_accepted), len(accepted_samples)),
        "trigger_ppv_vs_all_ref": _safe_div(len(trigger_samples) - len(extra_triggers), len(trigger_samples)),
        "n_recovery_entries": int(sup_state.n_recovery_entries),
        "n_recalibrations": int(sup_state.n_recalibrations),
        "n_reject_short": int(sup_state.n_reject_short),
        "n_reject_long": int(sup_state.n_reject_long),
        "n_reject_out_of_band": int(sup_state.n_reject_out_of_band),
        "mean_detector_confirmation_delay_ms": float(np.nanmean(result.detector.confirmation_delays_ms)) if len(result.detector.confirmation_delays_ms) else float("nan"),
        "p95_detector_confirmation_delay_ms": float(np.nanpercentile(result.detector.confirmation_delays_ms, 95)) if len(result.detector.confirmation_delays_ms) else float("nan"),
    }

    for row in label_rows:
        prefix = row["label"]
        summary[f"{prefix}_candidate_sensitivity"] = row["candidate_sensitivity"]
        summary[f"{prefix}_accepted_sensitivity"] = row["accepted_sensitivity"]
        summary[f"{prefix}_trigger_rate_vs_ref"] = row["trigger_rate_vs_ref"]
        summary[f"{prefix}_algorithm_after_label_ms_mean"] = row["algorithm_after_label_ms_mean"]
    for prefix in ("healthy", "arrhythmia", "other_beat"):
        summary.setdefault(f"{prefix}_candidate_sensitivity", float("nan"))
        summary.setdefault(f"{prefix}_accepted_sensitivity", float("nan"))
        summary.setdefault(f"{prefix}_trigger_rate_vs_ref", float("nan"))
        summary.setdefault(f"{prefix}_algorithm_after_label_ms_mean", float("nan"))

    return summary, per_beat_rows, label_rows


def build_windows(
    record_summary: Dict,
    beat_rows: List[Dict],
    *,
    window_s: float,
) -> List[Dict]:
    duration_s = float(record_summary["duration_s"])
    rows: List[Dict] = []
    starts = np.arange(0.0, max(duration_s, 0.0), window_s)
    for idx, start in enumerate(starts):
        end = min(start + window_s, duration_s)
        sub = [r for r in beat_rows if start <= float(r["ref_time_s"]) < end]
        if not sub and end - start <= 0:
            continue

        def _count(label: Optional[str], key: str) -> int:
            return sum(
                1 for r in sub
                if (label is None or r["label"] == label) and bool(r[key])
            )

        def _n(label: Optional[str]) -> int:
            return sum(1 for r in sub if label is None or r["label"] == label)

        alg_delays = np.asarray([
            float(r["algorithm_after_label_ms"]) for r in sub
            if np.isfinite(float(r["algorithm_after_label_ms"]))
        ], dtype=float)
        healthy_n = _n("healthy")
        arr_n = _n("arrhythmia")
        all_n = _n(None)
        all_acc = _count(None, "accepted_matched")
        all_trig = _count(None, "triggered_matched")
        rows.append({
            "dataset": record_summary["dataset"],
            "group": record_summary["group"],
            "record": record_summary["record"],
            "window_index": idx,
            "window_start_s": round(float(start), 3),
            "window_end_s": round(float(end), 3),
            "n_ref": all_n,
            "n_ref_healthy": healthy_n,
            "n_ref_arrhythmia": arr_n,
            "n_accepted": all_acc,
            "n_triggered": all_trig,
            "n_healthy_accepted": _count("healthy", "accepted_matched"),
            "n_healthy_triggered": _count("healthy", "triggered_matched"),
            "n_arrhythmia_accepted": _count("arrhythmia", "accepted_matched"),
            "n_arrhythmia_triggered": _count("arrhythmia", "triggered_matched"),
            "accepted_sensitivity": _safe_div(all_acc, all_n),
            "trigger_rate_vs_ref": _safe_div(all_trig, all_n),
            "healthy_trigger_rate": _safe_div(_count("healthy", "triggered_matched"), healthy_n),
            "arrhythmia_trigger_rate": _safe_div(_count("arrhythmia", "triggered_matched"), arr_n),
            "n_missed_accepted": all_n - all_acc,
            "n_missed_trigger": all_n - all_trig,
            "algorithm_after_label_ms_mean": float(np.nanmean(alg_delays)) if len(alg_delays) else float("nan"),
            "algorithm_after_label_ms_p95": float(np.nanpercentile(alg_delays, 95)) if len(alg_delays) else float("nan"),
            "score_badness": (all_n - all_acc) + 2 * _count("arrhythmia", "triggered_matched"),
        })
    return rows


def aggregate_label_rows(label_df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()
    group_cols = list(keys) + ["label"]
    agg = label_df.groupby(group_cols, dropna=False).agg(
        n_ref=("n_ref", "sum"),
        n_candidate=("n_candidate", "sum"),
        n_accepted=("n_accepted", "sum"),
        n_triggered=("n_triggered", "sum"),
        candidate_peak_abs_delay_ms_mean=("candidate_peak_abs_delay_ms_mean", "mean"),
        algorithm_after_label_ms_mean=("algorithm_after_label_ms_mean", "mean"),
        algorithm_after_label_ms_p95=("algorithm_after_label_ms_p95", "mean"),
    ).reset_index()
    agg["candidate_sensitivity"] = agg["n_candidate"] / agg["n_ref"].replace(0, np.nan)
    agg["accepted_sensitivity"] = agg["n_accepted"] / agg["n_ref"].replace(0, np.nan)
    agg["trigger_rate_vs_ref"] = agg["n_triggered"] / agg["n_ref"].replace(0, np.nan)
    return agg


def write_readme(out_dir: Path, args: argparse.Namespace, datasets: List[str]) -> None:
    text = f"""# Layer 1 Current Benchmark

Generated: {datetime.now().isoformat(timespec='seconds')}

## Command

```powershell
.\\.venv\\Scripts\\python.exe Layer1\\tools\\run_current_benchmark.py --data-dir {args.data_dir} --out-root {args.out_root}
```

## Configuration

- datasets: {', '.join(datasets)}
- filter mode: `{args.filter_mode}`
- match tolerance: `{args.match_tol_ms}` ms
- window length: `{args.window_s}` s

## Main CSVs

- `csv/per_record_all.csv`: one row per record, with healthy/arrhythmia sensitivities, trigger rates, false/extra counts, recovery counts, and detector delay.
- `csv/<dataset>_per_record.csv`: same information split per dataset.
- `csv/per_beat_all.csv`: one row per labeled beat, including symbol, healthy/arrhythmia label, candidate/accepted/trigger match flags, and delay.
- `csv/per_label_summary_all.csv`: per record and label (`healthy`, `arrhythmia`, `other_beat`) summary.
- `csv/per_dataset_summary.csv`: dataset-level aggregation to see whether one dataset dominates.
- `csv/per_window_10s_all.csv`: 10-second windows for local failure/good-run inspection.
- `csv/worst_windows_10s.csv`: highest-badness windows.
- `csv/good_windows_10s.csv`: clean high-beat-count windows.

## Delay Columns

- `candidate_peak_delay_ms`: detected peak sample minus labeled R-peak sample.
- `detector_confirmation_delay_ms`: confirmation sample minus detected peak sample.
- `algorithm_after_label_ms`: confirmation sample minus labeled R-peak sample. This is closest to "when did the algorithm know after the label?"

## Label Interpretation

Labels use `data/dataset_registry.py`:

- `healthy`: dataset-specific normal beat symbols.
- `arrhythmia`: dataset-specific abnormal beat symbols.
- `other_beat`: beat annotation exists, but is not in the normal/abnormal sets for that dataset.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("Results/layer1"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--match-tol-ms", type=float, default=100.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--filter-mode", choices=["causal", "zero_phase"], default="causal")
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--top-windows", type=int, default=200)
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"current_benchmark_{timestamp}"
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    specs = list_datasets(args.datasets)
    datasets_run = [s.folder for s in specs]
    all_records: List[Dict] = []
    all_beats: List[Dict] = []
    all_labels: List[Dict] = []
    all_windows: List[Dict] = []
    errors: List[Dict] = []

    t0_all = time.perf_counter()
    print(f"Writing fresh Layer 1 results to {out_dir}", flush=True)
    for spec in specs:
        ds_dir = dataset_dir(args.data_dir, spec.folder)
        hea_files = sorted(ds_dir.glob("*.hea"))
        if args.record_limit is not None:
            hea_files = hea_files[:args.record_limit]
        if not hea_files:
            print(f"[WARN] no records for {spec.folder} at {ds_dir}")
            continue

        print(f"\n[{spec.folder}] {len(hea_files)} records", flush=True)
        dataset_records: List[Dict] = []
        dataset_labels: List[Dict] = []
        dataset_windows: List[Dict] = []

        for idx, hea in enumerate(hea_files, start=1):
            stem = hea.with_suffix("")
            t0 = time.perf_counter()
            try:
                summary, beat_rows, label_rows = analyze_record(
                    spec,
                    stem,
                    channel_override=args.channel,
                    match_tol_ms=args.match_tol_ms,
                    filter_mode=args.filter_mode,
                    max_duration_s=args.max_duration_s,
                )
                window_rows = build_windows(summary, beat_rows, window_s=args.window_s)
                summary["elapsed_s"] = round(time.perf_counter() - t0, 3)

                all_records.append(summary)
                all_beats.extend(beat_rows)
                all_labels.extend(label_rows)
                all_windows.extend(window_rows)
                dataset_records.append(summary)
                dataset_labels.extend(label_rows)
                dataset_windows.extend(window_rows)

                print(
                    f"  {idx:>3}/{len(hea_files)} {hea.stem}: "
                    f"healthy_acc={summary.get('healthy_accepted_sensitivity', float('nan')):.3f} "
                    f"arr_trig={summary.get('arrhythmia_trigger_rate_vs_ref', float('nan')):.3f} "
                    f"delay={summary['mean_detector_confirmation_delay_ms']:.1f}ms",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [ERR] {hea.stem}: {exc}", flush=True)
                errors.append({"dataset": spec.folder, "record": hea.stem, "error": str(exc)})

        if dataset_records:
            pd.DataFrame(dataset_records).to_csv(csv_dir / f"{spec.folder}_per_record.csv", index=False)
        if dataset_labels:
            pd.DataFrame(dataset_labels).to_csv(csv_dir / f"{spec.folder}_per_label_summary.csv", index=False)
        if dataset_windows:
            pd.DataFrame(dataset_windows).to_csv(csv_dir / f"{spec.folder}_per_window_10s.csv", index=False)

    record_df = pd.DataFrame(all_records)
    beat_df = pd.DataFrame(all_beats)
    label_df = pd.DataFrame(all_labels)
    window_df = pd.DataFrame(all_windows)

    record_df.to_csv(csv_dir / "per_record_all.csv", index=False)
    beat_df.to_csv(csv_dir / "per_beat_all.csv", index=False)
    label_df.to_csv(csv_dir / "per_label_summary_all.csv", index=False)
    window_df.to_csv(csv_dir / "per_window_10s_all.csv", index=False)

    aggregate_label_rows(label_df, ["dataset", "group"]).to_csv(
        csv_dir / "per_dataset_label_summary.csv", index=False,
    )
    if not record_df.empty:
        dataset_summary = record_df.groupby(["dataset", "physionet_id", "group"], dropna=False).agg(
            n_records=("record", "count"),
            total_duration_s=("duration_s", "sum"),
            n_ref_beats=("n_ref_beats", "sum"),
            n_ref_healthy=("n_ref_healthy", "sum"),
            n_ref_arrhythmia=("n_ref_arrhythmia", "sum"),
            n_candidates=("n_candidates", "sum"),
            n_accepted=("n_accepted", "sum"),
            n_triggers=("n_triggers", "sum"),
            extra_accepted=("extra_accepted", "sum"),
            n_recovery_entries=("n_recovery_entries", "sum"),
            mean_detector_confirmation_delay_ms=("mean_detector_confirmation_delay_ms", "mean"),
            p95_detector_confirmation_delay_ms=("p95_detector_confirmation_delay_ms", "mean"),
            healthy_accepted_sensitivity=("healthy_accepted_sensitivity", "mean"),
            healthy_trigger_rate_vs_ref=("healthy_trigger_rate_vs_ref", "mean"),
            arrhythmia_accepted_sensitivity=("arrhythmia_accepted_sensitivity", "mean"),
            arrhythmia_trigger_rate_vs_ref=("arrhythmia_trigger_rate_vs_ref", "mean"),
        ).reset_index()
        dataset_summary.to_csv(csv_dir / "per_dataset_summary.csv", index=False)

    if not window_df.empty:
        worst = window_df.sort_values(
            ["score_badness", "n_missed_accepted", "n_arrhythmia_triggered", "n_ref"],
            ascending=[False, False, False, False],
        ).head(args.top_windows)
        good = window_df[
            (window_df["n_ref"] >= 5)
            & (window_df["n_missed_accepted"] == 0)
            & (window_df["n_arrhythmia_triggered"] == 0)
        ].sort_values(["n_ref", "algorithm_after_label_ms_mean"], ascending=[False, True]).head(args.top_windows)
        worst.to_csv(csv_dir / "worst_windows_10s.csv", index=False)
        good.to_csv(csv_dir / "good_windows_10s.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(csv_dir / "errors.csv", index=False)

    config = vars(args).copy()
    config["datasets"] = datasets_run
    config["elapsed_s"] = round(time.perf_counter() - t0_all, 3)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    write_readme(out_dir, args, datasets_run)

    pointer = args.out_root / "latest_current_benchmark.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(out_dir), encoding="utf-8")

    print(f"\nDone. Wrote {out_dir}", flush=True)
    print(f"Main CSV folder: {csv_dir}", flush=True)


if __name__ == "__main__":
    main()
