# Layer 1 - R-Peak Detection and Rhythm Supervision

Layer 1 is the fast deterministic safety layer:

```text
ECG signal -> filter -> R-peak detector -> rhythm supervisor -> accepted beats
```

The classifier/anomaly layers can veto stimulation later, but Layer 1 is the
first real-time timing gate.

## Folder Map

```text
Layer1/
  pipeline/   Main code to run and import
  tools/      Optional analysis and plotting helpers
  archive/    Older experiments kept for reference only
```

## Main Pipeline

Use `pipeline/` for normal work.

| File | Purpose |
|------|---------|
| `pipeline/main_pipeline.py` | Main API: `run_layer1()`, `layer1_r_peaks()` |
| `pipeline/r_peak_detector.py` | Fast causal adaptive R-peak detector |
| `pipeline/rhythm_supervisor.py` | RR timing safety gate: blanking, rejection, recovery |
| `pipeline/filters.py` | ECG bandpass/notch filtering helpers |
| `pipeline/reference_annotations.py` | WFDB annotation loading and oracle matching |
| `pipeline/run_benchmark.py` | Benchmark Layer 1 across datasets |
| `pipeline/run_record.py` | Run one WFDB record |

More detail: `pipeline/README.md`.

## Commands

```powershell
# Full Layer 1 benchmark
.\.venv\Scripts\python.exe Layer1\pipeline\run_benchmark.py `
    --data-dir data `
    --out-dir Results\layer1_benchmark

# One record, with plot
.\.venv\Scripts\python.exe Layer1\pipeline\run_record.py `
    --record data\mit_bih_arrhythmia\100

# Quick summary after a benchmark
.\.venv\Scripts\python.exe Layer1\tools\summarize_benchmark.py `
    Results\layer1_benchmark\per_record.csv
```

## Tools

Optional helper scripts live in `tools/`.

| Tool | Use when |
|------|----------|
| `summarize_benchmark.py` | You want a quick table from `per_record.csv` |
| `analyze_records.py` | Specific records fail and you want a failure breakdown |
| `plot_thesis_figures.py` | You need report-ready example figures |
| `plot_worst_records.py` | You want segment plots for bad records |
| `animate_record.py` | You want to watch detector/supervisor behavior live |

More detail: `tools/README.md`.

## Archive

`archive/` contains older or non-default approaches:

- old fixed-threshold hybrid detector
- adaptive-threshold experiment
- Pan-Tompkins comparison
- old detector-only benchmark

Do not start there unless you are comparing historical approaches.

## Import From Layer 2 / Layer 3

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Layer1")))
from _bootstrap import setup_layer1_paths

setup_layer1_paths()

from main_pipeline import layer1_r_peaks, run_layer1
```

## Smoke-Test Status

Recent checks:

- Production pipeline imports passed.
- `run_layer1()` synthetic ECG smoke test passed.
- Layer 2 import of Layer 1 passed.
- A full import pass over Layer 1 was interrupted, but all live pipeline/tools
  files passed before the interruption after removing the obsolete
  `analyze_layer2_gaps.py` tool.
