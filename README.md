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
| `Layer2/` | active | Main interpretable ECG safety gate, including prospective 1-in-8 stimulation gating. |
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

Run a small Layer 2 prospective-cadence smoke benchmark:

```powershell
python Layer2\validation\run_cross_dataset_validation.py `
    --data-dir data `
    --out-dir Results\layer2_cadence_smoke `
    --datasets mitdb `
    --mode zero_shot `
    --feature-sets all `
    --feature-window-mode causal `
    --post-r-lookahead-s 0.08 `
    --cadence-observation-lookahead-s 0.40 `
    --cadence-min-safe-observations 6 `
    --window-limit 20 `
    --max-records-per-dataset 1 `
    --no-adaptive
```

The relevant deployment-like mode in the output CSV is:

```text
fast_causal_cadence_1of8
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

The current stimulation rule is prospective:

```text
beats 1-7: Layer 2 checks recent unstimulated beats with a longer causal lookahead
beat 8:    stimulate only if at least 6/7 passed and beat 7 passed
```

This is important because the mechanical response can arrive too soon after the
electrical R peak to wait for full same-beat morphology analysis. The 8th beat
is therefore not analyzed by Layer 2 before stimulation. At that moment, the
runtime only needs fast R-peak detection plus the already-computed trigger flag.

Main files:

```text
Layer2/
  pipeline/     Feature extraction, baseline decision gate, stimulation cadence
  validation/   Beat-synchronous, cross-dataset, and Pareto validation scripts
  viz/          Thesis plots and gate animation helpers
```

The stimulation cadence code lives in:

```text
Layer2/pipeline/stimulation_cadence.py
```

Minimal runtime pattern:

```python
from main_pipeline import ProspectiveCadenceGate, decide_layer2

cadence = ProspectiveCadenceGate(cycle_length=8, observation_beats=7)

for beat in accepted_r_peaks:
    if cadence.next_phase == 8:
        # Decision has already been made from the previous observation beats.
        cadence_decision = cadence.step(
            trigger_ok=True,
            trigger_reason="r_peak_detected",
        )
        if cadence_decision["permit"]:
            trigger_stimulation()
        continue

    # Observation beat: update safety state, but do not stimulate.
    layer2_decision, features = decide_layer2(
        window=beat.ecg_window,
        fs=beat.fs,
        calibrator=calibrator,
        r_peaks_s=beat.recent_r_peaks_s,
        focus_peak_s=beat.focus_peak_s,
    )
    cadence.step(layer2_decision)
```

Example:

```powershell
python Layer2\validation\run_beat_validation.py --data-dir data --datasets mit_bih_arrhythmia --out-dir Results\layer2_beat_validation
```

Cross-dataset causal benchmark with the new cadence mode:

```powershell
python Layer2\validation\run_cross_dataset_validation.py `
    --data-dir data `
    --out-dir Results\layer2_cadence_benchmark `
    --datasets mitdb nstdb svdb `
    --mode zero_shot `
    --feature-sets all `
    --feature-window-mode causal `
    --post-r-lookahead-s 0.08 `
    --cadence-observation-lookahead-s 0.40 `
    --cadence-min-safe-observations 6 `
    --no-adaptive
```

The benchmark writes `per_beat.csv` and `overall_summary.csv`. Rows with
`eval_mode == fast_causal_cadence_1of8` correspond only to phase-8 stimulation
opportunities, not the 7 observation beats.

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

- local self-contained causal adaptive threshold detectors in `RPeakDetection/`
- `adaptive_threshold_v2`, the current preferred real-time trigger candidate
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
parameters such as `contraction` and `contraction_velocity`. The public entry
point is `run_bayesopt`; it accepts an arbitrary `param_bounds` dictionary, so
future stimulation parameters such as pulse width, amplitude, delay, or
frequency can be added without rewriting the optimizer.

Main files:

```text
BayesOpt/
  api.py             Simple `run_bayesopt(...)` entry point
  optimizer.py       Bayesian optimization loop
  score_function.py  Baseline, safety limits, and HemodynamicScore
  run_demo.py        Minimal mock simulation example
```

`baseline_params` should normally be provided because the hemodynamic score is
interpreted relative to the unassisted or no-MNA simulation.

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
