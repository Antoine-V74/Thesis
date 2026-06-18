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
    adaptive_threshold.py Self-contained causal adaptive threshold detectors
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
| `adaptive_threshold` | Local self-contained implementation | yes | Legacy causal bandpass plus adaptive amplitude/slope thresholds. |
| `adaptive_threshold_v2` | Local self-contained implementation | yes | Same detector family with a longer 160 ms refractory period to reduce duplicate detections. Current real-time candidate. |
| `ampt` | Local AMPT-style implementation | yes | Simplified mobile/real-time Pan-Tompkins baseline inspired by Neri et al. 2023. |
| `hamilton` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `christov` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `pan_tompkins` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `engzee` | `py-ecg-detectors` | no | Batch wrapper around the library implementation. |
| `two_average` | `py-ecg-detectors` | no | Elgendi two-average method via the library. |

The `py-ecg-detectors` methods are useful accuracy baselines, but the wrappers
are offline/batch APIs. They return all peaks after seeing the whole segment, so
they are not live latency baselines.

## Current Real-Time Candidate

The current preferred causal detector is `adaptive_threshold_v2`.

On the local MIT-BIH benchmark run:

```powershell
python RPeakDetection\benchmark\run_comparison.py `
    --data-dir data `
    --datasets mitdb `
    --algorithms adaptive_threshold adaptive_threshold_v2 ampt hamilton `
    --out-dir Results\rpeak_comparison\mitdb_adaptive_v2_20260618 `
    --write-per-record
```

the pooled results were:

| Algorithm | Causal | Sensitivity | PPV | F1 | Median live event lag |
|---|---:|---:|---:|---:|---:|
| `adaptive_threshold_v2` | yes | 0.9933 | 0.9510 | 0.9717 | 41.91 ms |
| `adaptive_threshold` | yes | 0.9799 | 0.8174 | 0.8913 | 42.72 ms |
| `ampt` | yes | 0.9409 | 0.9508 | 0.9458 | 149.88 ms |
| `hamilton` | no | 0.9645 | 0.9686 | 0.9666 | not live |

The improvement is mainly from reducing duplicate detections. The v2 refractory
period is more consistent with classic real-time QRS detector rules than the old
90 ms value, while still allowing fast rhythms.

This is a development result, not a final scientific claim. Since the parameter
choice was checked on MIT-BIH, final reporting should also validate on the other
default datasets, especially NSTDB and INCART.

## Report Assets

GitHub-ready summary figures and compact CSV tables are stored in:

```text
reports/rpeak_detection/
```

Regenerate them from the latest benchmark with:

```powershell
python RPeakDetection\benchmark\make_report_assets.py `
    --results-dir Results\rpeak_comparison\adaptive_v2_default_datasets_20260618 `
    --out-dir reports\rpeak_detection
```

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

- `--algorithms adaptive_threshold_v2 ampt hamilton`
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
- Hamilton PS, Tompkins WJ. Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database. IEEE Trans Biomed Eng. 1986. https://doi.org/10.1109/TBME.1986.325695
- Christov II. Real time electrocardiogram QRS detection using combined adaptive threshold. BioMedical Engineering OnLine. 2004. https://link.springer.com/article/10.1186/1475-925X-3-28
- Elgendi M. Fast QRS detection with an optimized knowledge-based method: Evaluation on 11 standard ECG databases. PLOS ONE. 2013. https://doi.org/10.1371/journal.pone.0073557
- Kim J, Shin H. Simple and robust realtime QRS detection algorithm based on spatiotemporal characteristic of the QRS complex. PLOS ONE. 2016. https://doi.org/10.1371/journal.pone.0150144
- Neri L, et al. Algorithm for Mobile Platform-Based Real-Time QRS Detection. Sensors. 2023. https://www.mdpi.com/1424-8220/23/3/1625
- Kristof F, et al. QRS detection in single-lead, telehealth electrocardiogram signals: Benchmarking open-source algorithms. PLOS Digital Health. 2024. https://doi.org/10.1371/journal.pdig.0000538
- Wolf SM, et al. A reproducible benchmark of QRS detection algorithms across diverse ECG datasets and noise conditions. Scientific Reports. 2026. https://www.nature.com/articles/s41598-026-53724-9
