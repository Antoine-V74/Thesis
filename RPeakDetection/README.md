# R-peak detection algorithm comparison

Standalone benchmark area for comparing R-peak detectors before integrating
the winner into Layer 2.

## Algorithms (initial set)

| Name | Source | Causal | Prefilter | Confirmation |
|------|--------|--------|-----------|--------------|
| `adaptive_threshold` | `Layer2/r_peak_detector.py` | yes | yes | explicit delay after peak |
| `hamilton` | py-ecg-detectors | no (batch) | no | peaks at batch output |
| `christov` | py-ecg-detectors | no (batch) | no | peaks at batch output |
| `pan_tompkins` | py-ecg-detectors | no (batch) | no | peaks at batch output |
| `engzee` | py-ecg-detectors | no (batch) | no | peaks at batch output |
| `two_average` | py-ecg-detectors | no (batch) | no | peaks at batch output |

Batch algorithms report `mean_confirmation_delay_ms = 0` because they do not
emit a separate confirmation event. Use `mean_detection_lag_ms` (matched
peak minus annotation) as the practical latency proxy for those methods.

## Run comparison

```powershell
.\.venv\Scripts\python.exe RPeakDetection\benchmark\run_comparison.py `
    --data-dir data --datasets mitdb `
    --out-dir Results\rpeak_comparison\mitdb

.\.venv\Scripts\python.exe RPeakDetection\benchmark\run_comparison.py `
    --data-dir data --datasets mitdb nstdb `
    --out-dir Results\rpeak_comparison\mitdb_nstdb_snr12
```

Smoke test (3 records):

```powershell
.\.venv\Scripts\python.exe RPeakDetection\benchmark\run_comparison.py `
    --datasets mitdb --record-limit 3 `
    --out-dir Results\rpeak_comparison\smoke
```

## Results layout (`Results/rpeak_comparison/`)

- `algorithm_summary.csv` — pooled metrics per algorithm
- `per_algorithm_per_dataset.csv` — per dataset breakdown
- `timing_summary.csv` — confirmation delay and detection lag per algorithm
- `per_record.csv` — only if `--write-per-record` (large debug grid)
- `run_config.json` — run parameters

## Dependency

```powershell
.\.venv\Scripts\python.exe -m pip install py-ecg-detectors
```
