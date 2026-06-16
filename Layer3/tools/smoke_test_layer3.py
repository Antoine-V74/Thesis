#!/usr/bin/env python3
"""
End-to-end smoke test for the Layer 3 validation pipeline.

Generates synthetic WFDB records into a temporary directory and runs the
full pipeline:
    1. build_window_index.py
    2. layer3_pretrain.py    (2 epochs, tiny batch)
    3. layer3_validate.py    (window-level)
    4. layer3_validate_beat_sync.py
    5. compare_layer2_layer3.py (against a synthetic Layer 2 per-beat CSV)

Asserts that the expected output files exist. Does NOT validate metric
quality; with a random encoder and synthetic data, metrics are meaningless.
This only checks that the wiring is correct.

Run from anywhere:
    python smoke_test_layer3.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def ensure_wfdb_available() -> None:
    try:
        import wfdb  # noqa: F401
    except ModuleNotFoundError:
        print("wfdb is required; install with pip install wfdb", file=sys.stderr)
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_ecg(n_samples: int, fs: int, hr_bpm: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Make a single synthetic ECG with QRS-like pulses and annotations."""
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.standard_normal(n_samples).astype(np.float32)
    period = int(round(fs * 60.0 / hr_bpm))
    qrs = np.array([0.0, 0.2, 1.0, -0.6, 0.1], dtype=np.float32)
    beat_samples: list[int] = []
    beat_symbols: list[str] = []
    i = period // 2
    beat_idx = 0
    while i + len(qrs) < n_samples:
        x[i : i + len(qrs)] += qrs
        # 10% of beats labeled abnormal (V) for the second half of the record only
        if i > n_samples // 2 and beat_idx % 7 == 0:
            beat_symbols.append("V")
            # Make abnormal beats noticeably different: wider, taller
            x[i : i + len(qrs)] += np.array([0.4, 0.6, 0.6, 0.4, 0.4], dtype=np.float32)
        else:
            beat_symbols.append("N")
        beat_samples.append(int(i))
        # Vary period slightly
        period_i = period + int(rng.integers(-fs // 20, fs // 20 + 1))
        i += max(period // 2, period_i)
        beat_idx += 1
    return x, np.array(beat_samples, dtype=np.int64), beat_symbols


def write_synthetic_wfdb_record(out_dir: Path, record_name: str, signal: np.ndarray, fs: int,
                                ann_samples: np.ndarray, ann_symbols: list[str]) -> None:
    """Write one synthetic WFDB record and .atr annotation file."""
    import wfdb
    out_dir.mkdir(parents=True, exist_ok=True)
    # Re-scale and clip to a realistic mV-ish range for digital storage
    sig_2d = signal.astype(np.float64).reshape(-1, 1)
    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"],
        sig_name=["MLII"],
        p_signal=sig_2d,
        write_dir=str(out_dir),
        fmt=["16"],
    )
    wfdb.wrann(
        record_name=record_name,
        extension="atr",
        sample=ann_samples,
        symbol=ann_symbols,
        write_dir=str(out_dir),
    )


def build_synthetic_dataset(root: Path, n_records: int = 3, duration_s: float = 60.0, fs: int = 250) -> None:
    """Create a tiny synthetic 'mitdb' dataset."""
    mitdb = root / "mitdb"
    mitdb.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * fs)
    for i in range(n_records):
        hr = 65.0 + 5.0 * i
        sig, samples, symbols = make_synthetic_ecg(n_samples=n, fs=fs, hr_bpm=hr, seed=100 + i)
        write_synthetic_wfdb_record(
            out_dir=mitdb,
            record_name=f"synth{i:03d}",
            signal=sig,
            fs=fs,
            ann_samples=samples,
            ann_symbols=symbols,
        )


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def assert_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise AssertionError(f"Missing expected {what}: {path}")
    print(f"  ok: {what} -> {path}")


# ---------------------------------------------------------------------------
# Synthetic Layer 2 CSV for the compare step
# ---------------------------------------------------------------------------

def make_fake_layer2_per_beat(layer3_per_beat_csv: Path, out_csv: Path) -> None:
    """Build a synthetic Layer 2 per-beat CSV that shares merge keys with Layer 3."""
    l3 = pd.read_csv(layer3_per_beat_csv)
    keep = [c for c in ["dataset", "record", "beat_sample", "beat_symbol", "is_healthy_beat"] if c in l3.columns]
    l2 = l3[keep].copy()
    # Simple Layer 2 policy: permit if healthy beat, inhibit otherwise.
    if "is_healthy_beat" in l2.columns:
        l2["decision"] = np.where(l2["is_healthy_beat"].astype(bool), "permit", "inhibit")
    else:
        l2["decision"] = "inhibit"
    l2["layer2_reason"] = "smoke_test_fake_layer2_policy"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    l2.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_wfdb_available()
    tmp = Path(tempfile.mkdtemp(prefix="layer3_smoke_"))
    print(f"[INFO] smoke test workspace: {tmp}")
    try:
        data_dir = tmp / "data"
        results_dir = tmp / "Results" / "layer3_validation"
        pretrain_dir = tmp / "Results" / "layer3_pretrain"
        layer2_dir = tmp / "Results" / "fake_layer2" / "beat_sync"
        windows_csv = results_dir / "synth_windows.csv"

        # 1. Build synthetic WFDB dataset
        build_synthetic_dataset(data_dir, n_records=3, duration_s=60.0, fs=250)
        print(f"[INFO] synthetic dataset built at {data_dir / 'mitdb'}")

        # 2. build_window_index.py
        run([
            PYTHON, str(THIS_DIR / "build_window_index.py"),
            "--data-dir", str(data_dir),
            "--datasets", "mitdb",
            "--out-csv", str(windows_csv),
            "--window-s", "5",
            "--stride-s", "2",
            "--target-fs", "250",
        ])
        assert_exists(windows_csv, "windows index CSV")
        df = pd.read_csv(windows_csv)
        assert len(df) > 0, "windows CSV is empty"
        for col in ["record_id", "signal_path", "start_idx", "n_samples"]:
            assert col in df.columns, f"missing pretrain column: {col}"
        print(f"  ok: {len(df)} windows, {df['record_id'].nunique()} records")

        # 3. layer3_pretrain.py (very short)
        run([
            PYTHON, str(THIS_DIR / "layer3_pretrain.py"),
            "--window-index", str(windows_csv),
            "--epochs", "2",
            "--batch-size", "8",
            "--checkpoint-dir", str(pretrain_dir),
            "--num-workers", "0",
            "--seed", "0",
        ])
        last_ckpt = pretrain_dir / "encoder_last.pt"
        assert_exists(last_ckpt, "pretrain last checkpoint")
        assert_exists(pretrain_dir / "pretrain_history.csv", "pretrain history CSV")

        # 4. layer3_validate.py (window-level)
        window_out = results_dir / "window_level"
        run([
            PYTHON, str(THIS_DIR / "layer3_validate.py"),
            "--data-dir", str(data_dir),
            "--datasets", "mitdb",
            "--window-index", str(windows_csv),
            "--checkpoint", str(last_ckpt),
            "--out-dir", str(window_out),
            "--per-record-calibration",
            "--seed", "0",
            "--guard-s", "5",
            "--min-fit-windows", "4",
            "--min-val-windows", "2",
        ])
        for fname in ["per_window.csv", "metrics_overall.csv", "metrics_by_record.csv",
                       "thresholds.csv", "FINAL_LAYER3_SUMMARY.md"]:
            assert_exists(window_out / fname, f"window-level output {fname}")

        # 5. layer3_validate_beat_sync.py
        bs_out = results_dir / "beat_sync"
        run([
            PYTHON, str(THIS_DIR / "layer3_validate_beat_sync.py"),
            "--data-dir", str(data_dir),
            "--datasets", "mitdb",
            "--checkpoint", str(last_ckpt),
            "--out-dir", str(bs_out),
            "--per-record-calibration",
            "--seed", "0",
            "--guard-s", "5",
            "--min-fit-beats", "10",
            "--min-val-beats", "5",
        ])
        for fname in ["per_beat.csv", "metrics_overall.csv", "metrics_by_record.csv",
                       "thresholds.csv", "FINAL_LAYER3_SUMMARY.md"]:
            assert_exists(bs_out / fname, f"beat-sync output {fname}")

        # 6. compare_layer2_layer3.py (with a synthetic Layer 2 CSV)
        fake_l2_csv = layer2_dir / "per_beat.csv"
        make_fake_layer2_per_beat(bs_out / "per_beat.csv", fake_l2_csv)
        compare_out = results_dir / "comparison"
        run([
            PYTHON, str(THIS_DIR / "compare_layer2_layer3.py"),
            "--layer2-dir", str(layer2_dir),
            "--layer3-dir", str(bs_out),
            "--out-dir", str(compare_out),
        ])
        for fname in ["combined_per_beat.csv", "comparison_layer2_layer3.csv",
                       "final_comparison_table.csv", "FINAL_COMPARISON_SUMMARY.md"]:
            assert_exists(compare_out / fname, f"comparison output {fname}")

        print("\n[SMOKE TEST PASSED]\n")
    finally:
        # Keep workspace on failure for inspection; clean it up on success.
        if os.environ.get("LAYER3_SMOKE_KEEP"):
            print(f"[INFO] keeping workspace at {tmp} (LAYER3_SMOKE_KEEP set)")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
