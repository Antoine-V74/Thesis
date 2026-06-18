# Master Thesis - ECG Processing

This repository contains code for two parts of a master thesis on ECG-triggered
cardiac stimulation and MyoNeural Actuator (MNA) control:

- ECG processing for rhythm safety supervision.
- Bayesian optimization for MNA/LV simulation parameters.

The central software question is:

```text
Is the current ECG/rhythm state safe enough to permit stimulation,
or should stimulation be inhibited?
```

The ECG pipeline is conservative by design: it can permit continuation or veto
stimulation, but it does not independently command stimulation.

## Repository Structure

```text
Layer1/          Deterministic R-peak detection and rhythm supervision
Layer2/          Interpretable ECG feature safety gate
Layer3/          Learned anomaly-detection research layer
RPeakDetection/  Standalone R-peak detector benchmark
BayesOpt/        Bayesian optimization for MNA/LV simulation parameters
data/            Dataset registry and local dataset instructions
requirements.txt Python dependencies
```

## Main Components

| Component | Purpose | Status |
|---|---|---|
| `Layer1/` | Low-latency R-peak timing and RR/rhythm supervision | active |
| `Layer2/` | Handcrafted feature gate calibrated against a safe ECG baseline | active |
| `RPeakDetection/` | Comparison of local and classical R-peak detectors | active |
| `BayesOpt/` | Bayesian optimization of MNA/LV simulation parameters | active research |
| `Layer3/` | Learned embedding/anomaly-detection experiments | exploratory |

## Safety Logic

The ECG layers are evaluated independently. Layer 1 provides deterministic
R-peak/rhythm supervision, Layer 2 evaluates handcrafted ECG features against a
safe baseline, and Layer 3 is an exploratory learned anomaly-detection path.

All safety decisions are conservative: uncertainty, poor signal quality,
unstable rhythm, failed calibration, or unavailable optional analysis should not
produce a false permit.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Example Commands

R-peak detector comparison:

```powershell
python RPeakDetection\benchmark\run_comparison.py --data-dir data --datasets mitdb --out-dir Results\rpeak_comparison\mitdb
```

Layer 1 benchmark:

```powershell
python Layer1\pipeline\run_benchmark.py --data-dir data --out-dir Results\layer1_benchmark
```

Layer 2 validation:

```powershell
python Layer2\validation\run_beat_validation.py --data-dir data --datasets mit_bih_arrhythmia --out-dir Results\layer2_beat_validation
```

Bayesian optimization demo:

```powershell
python BayesOpt\run_demo.py
```

## Data and Results

ECG waveform datasets are not included in this repository. PhysioNet WFDB
datasets are expected locally under `data/`; dataset aliases and download notes
are documented in `data/README.md` and `data/dataset_registry.py`.

Generated benchmark outputs are written to `Results/`, which is ignored by Git
to avoid storing large generated files.

## Documentation

More detailed documentation is available inside each module:

- `Layer1/README.md`
- `Layer2/README.md`
- `RPeakDetection/README.md`
- `BayesOpt/README.md`
- `Safety_Supervision.md`
