# Layer 1 Pipeline Files

This folder contains the production Layer 1 path:

```
ECG signal
  -> filters.py
  -> r_peak_detector.py
  -> rhythm_supervisor.py
  -> main_pipeline.py result
```

## Run These

| File | Purpose |
|------|---------|
| `main_pipeline.py` | Main API and single-record CLI: `run_layer1()`, `layer1_r_peaks()`, `main()` |
| `ALGORITHM_SUMMARY.md` | Plain-language explanation of Layer 1 filtering, detection, RR supervision, and outputs |
| `run_benchmark.py` | Run Layer 1 across datasets and write CSV summaries |
| `run_record.py` | Backward-compatible wrapper around `main_pipeline.py` |

## Core Building Blocks

| File | Purpose |
|------|---------|
| `r_peak_detector.py` | Fast causal adaptive threshold R-peak detector |
| `rhythm_supervisor.py` | RR timing supervisor: blanking, rejection, recovery |
| `filters.py` | Bandpass/notch filtering helpers |
| `reference_annotations.py` | WFDB beat labels and oracle matching |
| `artifact_simulation.py` | Synthetic stimulation artifact injection |

## Support Helpers

| File | Purpose |
|------|---------|
| `evaluate_record.py` | Record-level evaluation and plotting glue for `run_record.py` |
| `analysis_helpers.py` | Shared utilities used by `Layer1/tools/` |
| `plot_helpers.py` | Diagnostic ECG plotting helpers |
| `../tools/stream_record.py` | Real-time-style ECG replay: causal filtering + live Layer 1 decisions |

Start with:

```powershell
.\.venv\Scripts\python.exe Layer1\pipeline\main_pipeline.py --record 100
.\.venv\Scripts\python.exe Layer1\tools\stream_record.py --record data\mit_bih_arrhythmia\100
.\.venv\Scripts\python.exe Layer1\pipeline\run_benchmark.py --data-dir data --out-dir Results\layer1_benchmark
```
