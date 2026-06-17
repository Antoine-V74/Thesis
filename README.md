# Master Thesis - ECG Processing

This repository contains the ECG-processing code used for a master thesis on
ECG-triggered cardiac stimulation and MyoNeural Actuator (MNA) control research.

Only this `ECG Processing/` folder is intended to be pushed to GitHub. Raw ECG
databases, virtual environments, and generated result folders are deliberately
kept out of Git.

## Repository Map

```text
ECG Processing/
  Layer1/          Deterministic R-peak timing and rhythm supervision
  Layer2/          Handcrafted ECG feature safety gate
  Layer3/          Learned anomaly-detection research layer
  RPeakDetection/  Standalone detector comparison benchmark
  BayesOpt/        Bayesian optimization for MNA/LV simulation parameters
  data/            Tracked registry plus local untracked datasets
  reports/         Small curated summaries and figures for GitHub
  Results/         Local generated outputs, ignored by Git
```

## Status

| Area | Status | Use |
|---|---|---|
| `Layer1/` | active | Fast deterministic timing layer and rhythm supervision. |
| `Layer2/` | active | Main interpretable ECG safety gate under validation. |
| `RPeakDetection/` | active | Detector comparison before choosing timing components. |
| `BayesOpt/` | active research | MNA/LV simulation parameter optimization. |
| `Layer3/` | exploratory | Learned anomaly layer; not the primary validated safety path yet. |
| `Layer2/archive/` | archive | Historical scripts kept for traceability, not the current workflow. |

## Installation

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, use `python ...` for the commands below.

## Quick Reproducibility Check

Download one small benchmark dataset:

```powershell
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mit_bih_arrhythmia')"
```

Run a two-record R-peak detector smoke test:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets mitdb `
    --record-limit 2 `
    --out-dir Results\rpeak_comparison\smoke `
    --write-per-record
```

Expected local outputs:

```text
Results/rpeak_comparison/smoke/
  algorithm_summary.csv
  macro_algorithm_summary.csv
  per_algorithm_per_dataset.csv
  timing_summary.csv
  per_record.csv
  run_config.json
```

## Layer 1 - Deterministic Timing Layer

Layer 1 is the fast deterministic safety layer:

```text
ECG signal -> filter -> R-peak detector -> RR/rhythm supervisor
```

Main files:

```text
Layer1/
  pipeline/   Main detector, rhythm supervisor, filters, benchmark scripts
  tools/      Optional analysis, plots, summaries, and animations
```

Example:

```powershell
python Layer1\pipeline\run_benchmark.py --data-dir data --out-dir Results\layer1_benchmark
```

## Layer 2 - Feature Safety Gate

Layer 2 extracts beat-synchronous ECG features, calibrates a healthy baseline,
and returns permit/inhibit decisions. It is interpretable and is intended as
the main handcrafted safety upgrade after Layer 1.

Main files:

```text
Layer2/
  pipeline/     Feature extraction and baseline decision gate
  validation/   Beat-synchronous, cross-dataset, and Pareto validation scripts
  viz/          Thesis plots and gate animation helpers
```

Example:

```powershell
python Layer2\validation\run_beat_validation.py --data-dir data --datasets mit_bih_arrhythmia --out-dir Results\layer2_beat_validation
```

## Layer 3 - Learned Anomaly Research Layer

Layer 3 explores learned ECG embeddings and anomaly detection. Treat it as an
optional research extension until Layers 1 and 2 are fully validated.

Example smoke test:

```powershell
python Layer3\tools\smoke_test_layer3.py
```

## RPeakDetection - Detector Comparison

`RPeakDetection/` compares causal and batch R-peak/QRS detectors on the same
WFDB records and annotations.

Compared methods currently include:

- local causal adaptive threshold detector from `Layer2/r_peak_detector.py`
- causal AMPT-style simplified Pan-Tompkins baseline
- Hamilton, Christov, Pan-Tompkins, Engzee, and Two Average from `py-ecg-detectors`

Example validated cross-dataset command:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets mitdb nsrdb svdb ltafdb nstdb incartdb `
    --out-dir Results\rpeak_comparison\validated_datasets
```

See [RPeakDetection/README.md](RPeakDetection/README.md) for timing metrics,
annotation rules, skipped datasets, and result files.

## BayesOpt - MNA Parameter Optimization

`BayesOpt/` contains Bayesian optimization code for tuning MNA/LV simulation
parameters such as `contraction` and `contraction_velocity`.

Example:

```powershell
python BayesOpt\run_demo.py
```

## Data

The repository does not include ECG waveforms. Put downloaded PhysioNet WFDB
datasets under the local `data/` folder. The tracked registry and download
instructions are in [data/README.md](data/README.md) and
[data/dataset_registry.py](data/dataset_registry.py).

Scripts accept descriptive folder names and old PhysioNet IDs:

```powershell
python Layer1\pipeline\run_benchmark.py --data-dir data --datasets mit_bih_arrhythmia
python Layer1\pipeline\run_benchmark.py --data-dir data --datasets mitdb
```

## Results

`Results/` is ignored by Git and is recreated by running benchmarks,
validations, and plotting scripts. This keeps the repository small and avoids
committing multi-GB generated files.

Use `reports/` for small curated outputs that should be visible on GitHub:

- final summary tables
- selected thesis figures
- short notes explaining how a result was generated

Large full-result archives should be stored outside Git, for example on Zenodo,
OSF, institutional storage, or GitHub Releases, then linked from the README or
from files in `reports/`.
