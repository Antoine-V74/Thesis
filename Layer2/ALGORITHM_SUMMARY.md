# Layer 2 Algorithm Summary

Layer 2 is the interpretable ECG safety gate. Its job is to extract features from a
recent ECG window and RR history, compare them to a healthy baseline learned at
session start, and return **permit** or **inhibit**. It never commands stimulation
directly.

## High-Level Flow

```text
ECG window + R-peak history
  -> extract_layer2_features()
  -> calibrate_layer2()      # once, at session start on healthy data
  -> decide_layer2()         # at each trigger / rolling window
```

Main entry point: `pipeline/main_pipeline.py`

```python
feats, groups = extract_layer2_features(window, fs, r_peaks_s=peaks, species="rat")
calibrator = calibrate_layer2(baseline_features)
decision, feats = decide_layer2(window, fs, calibrator, r_peaks_s=peaks)
```

Decision logic lives in `pipeline/decision/`.

---

## Recommended Runtime Buffers

Layer 2 does not maintain buffers internally. The caller (Pynapse/XR) should
assemble:

| Buffer | Size | Used for |
|--------|------|----------|
| ECG morphology window | past 5 s + 100 ms after trigger R | `signal__*`, `morph__*` |
| RR rhythm buffer | past 30 s of accepted peaks | `rr__` scalar stats |
| Spectral HRV buffer | past 60 s of accepted peaks | `rr__hrv_*` (optional) |

Why separate horizons?

- morphology needs only a short local ECG snippet around the trigger beat
- RR stats need more beats but not the whole session
- spectral HRV needs ~50 s for human LF bands; 60 s is enough without using
  "all peaks since session start"

Reset RR buffers after gross artifact or missed-beat contamination rather than
keeping the entire session forever.

---

## Step 1: Feature Extraction

Function:

```python
extract_layer2_features(window, fs, r_peaks_s, species, focus_peak_s, ...)
```

Assembler: `pipeline/full_features.py`

### Feature groups

| Prefix | Source | Needs R-peaks? | Safety meaning |
|--------|--------|----------------|----------------|
| `signal__` | wavelets, entropy, amplitude, SQI | No | signal quality / noise / energy |
| `morph__` | beat shape around trigger | Yes | PVC / aberrant beat |
| `rr__` | RR intervals / HRV | Yes (≥2 peaks) | rhythm deterioration |

### Why R-peaks are optional in the API

The parameter `r_peaks_s` is optional so Layer 2 can still compute **signal-only**
features when peak detection fails (SQI, wavelets, energy). Without R-peaks,
`rr__` and `morph__` features are omitted.

For normal stimulation safety, **always pass accepted R-peaks**. Optional does not
mean recommended for deployment.

### Signal features (`signal_features.py` + amplitude/SQI in `full_features.py`)

Includes:

```text
wavelet log-energy and Shannon entropy per band (db4, 4 levels)
sample entropy and approximate entropy
RMS, peak-to-peak, energy, line length, zero-crossing rate
hf_noise_ratio, lf_wander_ratio (FFT SQI on filtered ECG)
```

Justification:

- wavelets separate QRS transients, slower waves, and baseline wander at
  different scales
- entropy rises for chaotic/noisy signals (VFib surrogate, noise)
- amplitude/energy catch gross signal changes
- SQI catches electrode motion and baseline wander before rhythm logic runs

Pure NumPy / PyWavelets. No deep learning. Pynapse-compatible.

### Morphology features (`morphology_features.py`)

Beat-centered features around the **trigger** R-peak (`focus_peak_s`):

```text
template_corr      correlation vs median beat in window
neighbor_corr      correlation vs other beats in window
qrs_width_ms       full-width at half-maximum
beat_amp           peak amplitude of trigger beat
amp_vs_median      trigger amplitude vs local median
post_pre_area_ratio energy asymmetry after vs before peak
```

Justification:

- isolated PVCs often have normal-ish RR context but abnormal beat shape
- template/neighbor correlation are strong PVC cues
- hard rules on morphology fire before Mahalanobis for interpretability

Designed for **beat-synchronous** gates: window centered on (or ending shortly
after) the stimulation trigger beat.

### Rhythm features (`rhythm_features.py`)

From R-peak timestamps or RR intervals:

```text
rr_mean_ms, hr_bpm, sdnn, rmssd, short/long RR fractions
optional spectral HRV: hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio, ...
```

`species` (`human`, `rat`, `pig`) sets:

- short/long RR thresholds
- LF/HF frequency bands for spectral HRV
- RR interpolation rate before FFT

Justification:

- scalar RR stats work on 30 s look-back and are real-time compatible
- spectral HRV needs long RR history (~50 s human LF); often disabled in
  beat-sync runtime (`compute_spectral_hrv=False`)

---

## Step 2: Baseline Calibration

Function:

```python
calibrate_layer2(baseline_features)  -> BaselineCalibrator
```

Implementation: `pipeline/decision/calibration.py`

Run once at session start on **healthy** ECG windows from the same animal/session.

### What calibration learns

```text
mean, std                 robust center/scale for all features
inv_cov                   inverse covariance for Mahalanobis subset
hard_rules                copied from config.py
threshold_mahalanobis     multivariate distance cutoff
threshold_max_zscore      worst single-feature z-score cutoff
threshold_signal_proxy    diagnostic cutoff for signal group
threshold_rr_proxy        diagnostic cutoff for RR group
```

### Train / validation split (70% / 30%)

Only on the **healthy baseline windows** passed to `fit()`, not on the whole
dataset:

```text
first 70% of healthy windows  -> learn mean/std/covariance
last 30% of healthy windows   -> set thresholds
```

Temporal split (not random) mimics deployment: early baseline calibrates, later
baseline validates thresholds.

Justification: avoids tuning thresholds on the exact same windows used to fit
the model.

### Robust center and scale

```text
center = median
scale  = 1.4826 * MAD
```

Justification: occasional outlier windows during "healthy" baseline recording
should not distort the baseline cloud.

### Mahalanobis subset

Not all features enter Mahalanobis. Exclusions in `decision/config.py`:

**Handled as hard rules instead:**

```text
rr__rr_count, rr__short_rr_fraction, rr__long_rr_fraction
rr__beat_coupling_ratio
morph__template_corr, morph__neighbor_corr
signal__hf_noise_ratio, signal__lf_wander_ratio, signal__raw_hf_noise_ratio
```

**Excluded because high healthy variability would widen the covariance/threshold:**

```text
other morph__ features (qrs width, amplitude ratios, ...)
```

Justification:

- some features need fixed absolute limits, not statistical modeling
- morphology varies a lot even in healthy beats; including it in Mahalanobis
  inflated healthy false inhibits during benchmark tuning

### Covariance and inversion

On z-scored healthy calibration data:

```text
covariance (Ledoit-Wolf shrinkage if sklearn available)
inv_cov = inverse(cov + ridge * I)
```

Mahalanobis at runtime:

```text
d = sqrt((x - mean)^T * inv_cov * (x - mean))
```

Justification:

- Mahalanobis detects unusual **combinations** of features, not just one feature
  at a time
- shrinkage stabilizes covariance when feature count is high relative to
  baseline window count
- ridge avoids singular matrices
- if too few windows: fall back to diagonal mode (independent z-scores)

---

## Step 3: Runtime Decision

Implementation: `pipeline/decision/gate.py` (`GateMixin` on `BaselineCalibrator`)

Two decision methods exist. Both use the same calibrated baseline.

---

### `decide()` — standard gate

Used by `decide_layer2()` in `main_pipeline.py`.

Decision order:

```text
1. hard rules
2. Mahalanobis > threshold_mahalanobis
3. max_abs_zscore > threshold_max_zscore
4. otherwise permit
```

#### 1. Hard rules

Fixed limits from `DEFAULT_HARD_RULES` in `config.py`. Checked first.

Examples:

```text
rr__rr_count < 3
rr__short_rr_fraction > 0.5
morph__template_corr < 0.55
signal__hf_noise_ratio > 0.35
rr__beat_coupling_ratio < 0.80   # FROZEN_COUPLING_THRESHOLD
```

Justification:

- absolute, interpretable vetoes
- do not depend on baseline covariance
- inhibit reason names the exact feature (`hard_rule`)

#### 2. Mahalanobis

Multivariate distance on `mahal_feature_names` subset.

Justification:

- catches subtle combined abnormalities no single feature crosses alone
- threshold set so ~99.9% of healthy validation windows pass (default quantile)

#### 3. Max z-score

```text
max_abs_zscore = max |z_i| over decision features
```

Justification:

- backup when one feature is extremely far but Mahalanobis is moderated by
  correlations
- threshold at 90th percentile of healthy validation (`FROZEN_ZSCORE_QUANTILE`)

When to use `decide()`:

- rolling-window validation
- simple runtime path when RR history is always trusted
- offline benchmarks where beat-sync reliability logic is not needed

---

### `decide_hybrid()` — beat-sync deployment gate

Used in beat-synchronous validation (`run_beat_validation.py`,
`run_cross_dataset_validation.py`).

Decision order:

```text
0. hard rules
1. signal_mahal_proxy > threshold_signal_proxy  -> inhibit (signal_not_safe)
2. current R-peak not reliable                  -> inhibit (current_r_unreliable)
3. RR history reliable:
     Mahalanobis > threshold                   -> inhibit (rr_abnormal)
     else                                      -> permit (all_safe)
4. RR history unreliable:
     n_recent_clean_beats >= warm_beats        -> permit (rewarming)
     else                                      -> inhibit (recovery)
```

Extra inputs:

```python
decide_hybrid(features, current_r_reliable, rr_history_reliable,
              n_recent_clean_beats, warm_beats)
```

`check_rr_reliability()` in `config.py` sets the two reliability flags from:

```text
current_r_reliable    enough beats in current window
rr_history_reliable   enough beats in look-back AND not dominated by
                      long/short RR fractions (missed/extra peaks)
```

Justification for hybrid mode:

- beat-triggered stimulation needs to know if **this** R-peak is usable
- a single missed beat can contaminate RR fractions for many seconds; hybrid
  avoids blocking forever on stale RR history
- signal problems (noise/morphology) are separated from rhythm problems via
  `signal_mahal_proxy` before RR logic runs
- rewarming permits stimulation after N clean beats even if RR history is still
  flagged unreliable

When to use `decide_hybrid()`:

- real-time beat-sync deployment (Pynapse trigger at R-peak)
- validation that simulates causal post-R lookahead (80–100 ms)

When **not** needed:

- offline analysis where one simple feature-vector decision is enough
- cases where you only want a single statistical gate without RR trust logic

---

### Diagnostic scores (not primary gates in `decide()`)

`score()` also returns:

```text
signal_mahal_proxy = sqrt(sum z^2 over signal__ features)
rr_mahal_proxy     = sqrt(sum z^2 over rr__ features)
```

Justification:

- simpler group-level distances for logging
- answer "was inhibit mostly signal-like or rhythm-like?"
- `signal_mahal_proxy` is used as a gate step in `decide_hybrid()`, not in
  plain `decide()`

These are **not** full separate Mahalanobis models per group (would need more
calibration data and duplicate covariance structures).

---

## Step 4: Fixed Policy (`decision/config.py`)

Not learned from data. Set by safety policy and benchmark tuning.

| Item | Role |
|------|------|
| `DEFAULT_HARD_RULES` | absolute veto limits before Mahalanobis |
| `DEFAULT_MAHAL_EXCLUDE` | features removed from Mahalanobis |
| `FROZEN_COUPLING_THRESHOLD` | PVC coupling ratio limit (0.80) |
| `FROZEN_ZSCORE_QUANTILE` | max-zscore threshold quantile (0.90) |
| `RRReliabilityConfig` | species-specific RR trust settings for hybrid mode |
| `check_rr_reliability()` | computes reliability flags for `decide_hybrid()` |

Do not change frozen thresholds without re-running cross-dataset benchmarks.

---

## Safety Interpretation

Layer 2 is conservative:

```text
missing features / NaN hard-rule features -> often treated as missing, not veto
uncertainty in peak detection -> pass R-peaks; without them only signal gate remains
hard rule hit -> inhibit with named reason
Mahalanobis / z-score exceed -> inhibit with statistical reason
hybrid rewarming -> temporary permit while RR history recovers (configurable)
```

Primary safety metric: **false permit rate** (abnormal state wrongly permitted).

Secondary: false inhibit rate (healthy state wrongly blocked).

---

## Validation, Tools, and Reports

Layer 2 has two support areas beyond `pipeline/`:

- **`validation/`** reruns the pipeline on ECG data and writes CSV metrics.
- **`viz/`** consumes those CSVs and produces thesis figures and animations.

### `validation/` — rerun pipeline on datasets

| Script | Purpose | When to use |
|--------|---------|-------------|
| `run_beat_validation.py` | one decision per R-peak, causal/centered modes | Deployment-like benchmark |
| `run_cross_dataset_validation.py` | MIT-BIH, NSTDB, SVDB, INCART, CUDB, VFDB | Generalization study |
| `run_pareto_sweep.py` | unified Pareto entry point: quick / full / posthoc | Operating-point search |
| `run_causal_lookahead_sweep.py` | sweep post-R lookahead ms | Causal delay budget study |
| `common.py` | shared helpers (filtering, scoring, calibrator fit) | imported by validation scripts |
| `pareto_quick.py` | fast 10-record MIT-BIH subset | used by `run_pareto_sweep.py quick` |
| `pareto_posthoc.py` | threshold rescaling on saved per_beat.csv | used by `run_pareto_sweep.py posthoc` |

### `viz/` — presentation figures

| Script | Output | Purpose |
|--------|--------|---------|
| `make_all.py` | all figures below | one-command generation |
| `plot_dataset_performance.py` | dataset / arrhythmia / worst-record plots | per-dataset diagnosis |
| `plot_pareto.py` | operating curves + Pareto frontier | safety trade-off slides |
| `plot_feature_auroc.py` | AUROC bars + top deviators | feature analysis |
| `animate_beat_gate.py` | gate walkthrough GIF | explain algorithm visually |

Legacy scripts from the old `tools/` and `reports/` folders are in `archive/old_tools/` and
`archive/old_reports/`.

---

## Quick Command Reference

```powershell
# Beat-sync validation (deployment-like)
.\.venv\Scripts\python.exe Layer2\validation\run_beat_validation.py `
  --data-dir data --datasets mit_bih_arrhythmia `
  --out-dir Results\layer2_beat_validation

# Cross-dataset
.\.venv\Scripts\python.exe Layer2\validation\run_cross_dataset_validation.py `
  --data-dir data --out-dir Results\cross_dataset

# All presentation figures
.\.venv\Scripts\python.exe Layer2\viz\make_all.py `
  --per-beat Results\layer2\cross_dataset_causal_100ms\per_beat.csv
```

Smoke imports:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'Layer2/pipeline'); from main_pipeline import extract_layer2_features, calibrate_layer2, decide_layer2; print('OK')"
```
