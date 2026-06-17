# Master Thesis - ECG Processing

Code for an ECG safety pipeline developed for ECG-triggered cardiac stimulation
and MyoNeural Actuator (MNA) control research.

The ECG side is organized as three safety/research layers. The repository also
contains standalone R-peak detector benchmarking and a Bayesian optimization
module for MNA/LV simulation tuning.

## Repository Map

```text
ECG Processing/
  Layer1/          Fast R-peak detection and rhythm supervision
  Layer2/          Handcrafted ECG feature safety gate
  Layer3/          Learned embedding / anomaly-detection research layer
  RPeakDetection/  Standalone R-peak detector comparison benchmark
  BayesOpt/        Bayesian optimization for MNA/LV simulation parameters
  requirements.txt Python dependencies
```

Generated outputs are written locally to `Results/`, and local datasets are
expected in `data/`. These folders are intentionally ignored by Git because they
can be large.

## Installation

Create and activate a Python environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, use `python ...` in the commands below. If you do not want to
create a virtual environment, you can still run:

```powershell
python -m pip install -r requirements.txt
```

but a virtual environment is recommended so the thesis dependencies do not
modify your global Python installation.

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

Layer 2 extracts ECG features, calibrates a healthy baseline, and returns
permit/inhibit decisions. It is interpretable and is intended as the main
handcrafted safety upgrade after Layer 1.

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

Layer 3 explores learned ECG embeddings and anomaly detection. It is a research
extension and should be treated as optional until Layers 1 and 2 are validated.

Main files:

```text
Layer3/
  pipeline/     Encoder, augmentations, anomaly scores, embedding distances
  tools/        Pretraining, smoke tests, and comparison scripts
  validation/   Window-level and beat-level validation scripts
  reports/      Architecture and validation notes
```

Example smoke test:

```powershell
python Layer3\tools\smoke_test_layer3.py
```

## RPeakDetection - Detector Comparison

`RPeakDetection/` is a standalone benchmark area for comparing R-peak detection
algorithms before integrating the best approach into the layered safety
pipeline.

Compared methods include:

- the local adaptive-threshold detector from `Layer2/r_peak_detector.py`
- Hamilton, Christov, Pan-Tompkins, Engzee, and Two Average detectors from
  `py-ecg-detectors`

Example:

```powershell
python RPeakDetection\benchmark\run_comparison.py --data-dir data --datasets mitdb --out-dir Results\rpeak_comparison\mitdb
```

## BayesOpt - MNA Parameter Optimization

`BayesOpt/` contains Bayesian optimization code for tuning MNA/LV simulation
parameters such as `contraction` and `contraction_velocity`.

Main files:

```text
BayesOpt/
  api.py            High-level API to connect a simulation and run BO
  objective.py      Literature-based cardiac objective and safety filter
  optimizer.py      Bayesian optimization loop
  gp_surrogate.py   Gaussian Process surrogate model
  acquisition.py    Acquisition functions such as EI/UCB/CoolingUCB
  run_demo.py       End-to-end demo with a mock simulation
```

Example:

```powershell
python BayesOpt\run_demo.py
```

## Data

The repository does not include ECG datasets. Put downloaded PhysioNet datasets
under a local `data/` folder. Scripts accept both descriptive folder names and
old PhysioNet short IDs.

| Folder | Old ID | Dataset |
|--------|--------|---------|
| `mit_bih_arrhythmia` | `mitdb` | MIT-BIH Arrhythmia |
| `normal_sinus_rhythm` | `nsrdb` | Normal Sinus Rhythm |
| `supraventricular_arrhythmia` | `svdb` | Supraventricular Arrhythmia |
| `atrial_fibrillation` | `afdb` | Atrial Fibrillation |
| `long_term_atrial_fibrillation` | `ltafdb` | Long-Term Atrial Fibrillation |
| `malignant_ventricular_arrhythmia` | `vfdb` | Malignant Ventricular Arrhythmia |
| `creighton_vfib` | `cudb` | Creighton University VF Database |
| `noise_stress_test` | `nstdb` | MIT-BIH Noise Stress Test |
| `st_petersburg_12lead` | `incartdb` | St Petersburg INCART 12-lead records |

Example:

```powershell
python Layer1\pipeline\run_benchmark.py --data-dir data --datasets mit_bih_arrhythmia
python Layer1\pipeline\run_benchmark.py --data-dir data --datasets mitdb
```

## Results

`Results/` is not stored in Git. It is created locally when benchmarks,
validations, or plotting scripts run. This keeps the repository small while
still allowing every result to be regenerated from code and local data.
