# R-Peak Detection Algorithm Comparison

This folder is a standalone benchmark area for comparing QRS/R-peak detectors
before choosing what belongs in the real-time ECG safety pipeline.

The benchmark separates two questions:

1. How accurately does each method locate annotated beats?
2. For causal methods, how late is the live confirmation event that a controller
   could actually use?

## Folder Layout

```text
RPeakDetection/
  algorithms/
    __init__.py           Algorithm registry
    base.py               Common detector interface
    adaptive_threshold.py Wrapper around Layer2/r_peak_detector.py
    ampt.py               Causal AMPT-style simplified Pan-Tompkins baseline
    ecg_detectors.py      py-ecg-detectors batch wrappers
  benchmark/
    run_comparison.py     Main benchmark script
    metrics.py            Matching, timing, and aggregate metrics
  types.py                Uniform result format
```

## Algorithms

| Name | Source | Causal | Notes |
|---|---|---:|---|
| `adaptive_threshold` | Local `Layer2/r_peak_detector.py` | yes | Causal bandpass plus adaptive amplitude/slope thresholds. |
| `ampt` | Local AMPT-style implementation | yes | Simplified mobile/real-time Pan-Tompkins baseline inspired by Neri et al. 2023. |
| `hamilton` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `christov` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `pan_tompkins` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `engzee` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `two_average` | `py-ecg-detectors` | no | Elgendi two-average method via the library. |

The `py-ecg-detectors` methods are useful accuracy baselines, but the wrappers
are offline/batch APIs. They return all peaks after seeing the whole segment, so
they are not live latency baselines.

## Annotation Policy

R-peak detection is scored only against beat/QRS annotations. Rhythm markers
such as `+` and signal-quality markers such as `~` are ignored.

Default R-peak benchmark datasets:

- `mitdb` / `mit_bih_arrhythmia`
- `nsrdb` / `normal_sinus_rhythm`
- `svdb` / `supraventricular_arrhythmia`
- `ltafdb` / `long_term_atrial_fibrillation`
- `nstdb` / `noise_stress_test`
- `incartdb` / `st_petersburg_12lead`

Datasets skipped by default:

- `afdb`: `.atr` is rhythm-level; `.qrs` is treated as a secondary reference.
- `vfdb`: `.atr` contains rhythm-change annotations, not beat labels.
- `cudb`: PhysioNet describes the annotations as useful but non-definitive.

Use `--include-nondefault-annotations` only for exploratory stress tests.

## Timing Columns

`mean_detection_lag_ms` is the matched detected peak minus the reference
annotation. It is a benchmark alignment measure.

`mean_confirmation_delay_ms` is internal to causal detectors: confirmation
sample minus estimated peak sample.

`mean_confirmed_event_lag_ms` is the live usable event minus the reference
annotation. This is the latency-relevant metric for ECG-triggered stimulation.
It is reported only for causal detectors with explicit confirmation samples.

## Run Examples

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

Smoke test:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets mitdb `
    --record-limit 2 `
    --out-dir Results\rpeak_comparison\smoke `
    --write-per-record
```

Default validated cross-dataset benchmark:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets mitdb nsrdb svdb ltafdb nstdb incartdb `
    --out-dir Results\rpeak_comparison\validated_datasets
```

Noise stress test across all NSTDB SNR levels:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets nstdb `
    --nstdb-min-snr -6 `
    --out-dir Results\rpeak_comparison\nstdb_all_snr
```

Useful options:

- `--algorithms adaptive_threshold ampt hamilton`
- `--match-tol-ms 50`, `100`, or `150`
- `--nstdb-min-snr -6`, `0`, `6`, `12`, `18`, or `24`
- `--write-per-record`
- `--include-nondefault-annotations`

## Result Files

Results will be written locally under `Results/rpeak_comparison/...`.

| File | Purpose |
|---|---|
| `algorithm_summary.csv` | Micro-average pooled by beat count. |
| `macro_algorithm_summary.csv` | Mean of per-dataset metrics; large datasets do not dominate. |
| `per_algorithm_per_dataset.csv` | Dataset-specific performance. |
| `per_algorithm_per_snr.csv` | NSTDB breakdown by noise SNR, when NSTDB is run. |
| `timing_summary.csv` | Causal confirmation and benchmark lag metrics. |
| `per_record.csv` | Optional large debugging file. |
| `run_config.json` | Exact run configuration and skipped datasets. |
| `comparison.log` | Runtime log. |


## References

- Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng. 1985. https://doi.org/10.1109/TBME.1985.325532
- Neri L, et al. Algorithm for Mobile Platform-Based Real-Time QRS Detection. Sensors. 2023. https://www.mdpi.com/1424-8220/23/3/1625
- Wolf SM, et al. A reproducible benchmark of QRS detection algorithms across diverse ECG datasets and noise conditions. Scientific Reports. 2026. https://www.nature.com/articles/s41598-026-53724-9
