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

For the current stimulation strategy, Layer 2 is used prospectively:

```text
beats 1-7: score safety on unstimulated beats
beat 8:    stimulate only if the observation block is safe enough
```

This means the 8th beat is not analyzed before stimulation. The decision has
already been made; at the 8th R-peak the runtime only needs to detect the R peak
and apply the stored trigger/no-trigger flag.

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


| Buffer                | Size                                  | Used for                   |
| --------------------- | ------------------------------------- | -------------------------- |
| ECG morphology window | past 5 s + post-R lookahead           | `signal__*`, `morph__*`    |
| RR rhythm buffer      | past 30 s of accepted peaks           | `rr__` scalar stats        |
| Spectral HRV buffer   | past 60 s of accepted peaks           | `rr__hrv_*` (optional)     |
| Cadence safety state  | last 7 unstimulated Layer 2 decisions | 1-in-8 prospective trigger |


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


| Prefix     | Source                            | Needs R-peaks? | Safety meaning                  |
| ---------- | --------------------------------- | -------------- | ------------------------------- |
| `signal__` | wavelets, entropy, amplitude, SQI | No             | signal quality / noise / energy |
| `morph__`  | beat shape around trigger         | Yes            | PVC / aberrant beat             |
| `rr__`     | RR intervals / HRV                | Yes (≥2 peaks) | rhythm deterioration            |


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

#### Optional SQI ensemble (`compute_sqi_ensemble=True`) — artifact robustness

No single SQI is robust to every artifact, so ICD/AED and clinical wearable
practice combine complementary indices and inhibit if any one flags poor
quality (Clifford et al. 2012; Behar et al. 2013; Li et al. 2008). Off by
default so the frozen feature set / existing calibrations are unchanged; when
enabled it adds three interpretable, causal, label-free indices:

```text
signal__ksqi  kurtosis of the window. Clean ECG is highly peaked (sharp QRS);
              EMG / motion / saturation flattens it -> kurtosis drops.
signal__psqi  QRS-band (5-15 Hz) power / 5-40 Hz power. High for clean ECG,
              low when broadband noise or wander dominates.
signal__bsqi  agreement of two independent R-peak detectors (Li et al. 2008).
              Low agreement -> beat train (and every rr__ / morph__ feature)
              is untrustworthy. Supplied by the caller (needs two detectors)
              via full_features(..., bsqi=...).
```

Hard-rule limits live in `SQI_ENSEMBLE_HARD_RULES` (`decision/config.py`) and
only fire when these keys are present, so they are inert until the ensemble is
enabled. Enable the matching hard rules with
`hard_rules_with_extensions(include_sqi_ensemble=True)`.

#### SQI flip-on gate (do this before changing the frozen gate)

**Current policy:** SQI ensemble is opt-in only. Do **not** put it in
`DEFAULT_HARD_RULES` / default `decide()` until the checklist below passes.

| Step | What | Pass criterion (human MIT-BIH first) |
| ---- | ---- | ------------------------------------ |
| 1 | Run `run_artifact_stress_test.py` on real WFDB windows | Clean false-inhibit ≤ ~0.05; each artifact class detection ≥ ~0.80 (stim, lead-off, EMG, wander, powerline, saturation) |
| 2 | Decide if SQI is meaningful | If clean FI high **or** stim/lead-off detection weak → stop; do not wire into beat validation |
| 3 | Recalibrate cutoffs on a held-out clean + artifact set | Sweep `ksqi` / `psqi` / `bsqi` limits; pick the most permissive set that still meets the Step-1 budgets. Write chosen limits into `SQI_ENSEMBLE_HARD_RULES` (or a species-specific override) |
| 4 | Only then add CLI / gate wiring | `--sqi-ensemble` on `run_beat_validation.py`, then optional move into frozen `DEFAULT_HARD_RULES` after a full gold-cohort beat validation shows no unacceptable healthy FI rise |

**Calibration note:** Step 3 is required. Literature defaults (kSQI≥4, pSQI≥0.40,
bSQI≥0.80) are **shape checks only**, not deployable thresholds. Recalibrate on
human first; rat/pig need their own cutoffs (different HR / QRS bandwidth).

**Cluster order (agreed):** SQI stress test **first**. Full beat validation +
Neyman–Pearson are downstream of a meaningful SQI result — do not block on them
until Step 2 passes.

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

#### Optional onset / stability discriminators (`compute_onset_stability=True`)

Rate alone does not separate a dangerous organised VT from fast-conducted AF or
sinus tachycardia. Implantable defibrillators solve this with two
discriminators layered on top of rate-zone branching (Swerdlow et al. 1994;
Medtronic "Onset" / Boston Scientific "Stability"). Off by default; when enabled
it adds:

```text
rr__onset_accel_frac  fractional RR shortening first-half -> second-half of the
                      window. Large positive = abrupt onset (VT-like); ~0 =
                      gradual (sinus tachycardia-like).
rr__stability_ms      mean |successive RR difference|. Low = regular / stable
                      (monomorphic VT); high = irregular (AF).
rr__tachy_fraction    fraction of RR at or above the species tachy rate.
```

Rate-zone helper `classify_rate_zone(hr_bpm, species)` returns one of
`brady / normal / tachy / vt_zone / vf_zone` (species boundaries in
`_SPECIES_CONFIGS`). It decides *when* the discriminators matter: they are
informative in the tachy / VT zone, while a VF-rate is treated as dangerous
regardless of stability. Reference limits are in `ONSET_STABILITY_REFERENCE_LIMITS`
(placeholders; discriminators, not blanket vetoes).

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
threshold_mahalanobis     primary multivariate distance cutoff (Mahalanobis or kNN)
threshold_max_zscore      worst single-feature z-score cutoff
threshold_signal_proxy    diagnostic cutoff for signal group
threshold_rr_proxy        diagnostic cutoff for RR group
threshold_method          conformal (default) or healthy_quantile (legacy)
conformal_alpha           target healthy false-inhibit rate (default 10%)
calibration_outlier_frac  optional fraction of worst calibration windows removed before refit
anomaly_model             mahalanobis (default) or knn
knn_k                     neighbours for kNN scorer (default 5)
knn_calibration_vectors   z-scored Mahalanobis-subset bank for kNN distance
```

### Train / validation split (70% / 30%)

Only on the **healthy baseline windows** passed to `fit()`, not on the whole
dataset:

```text
first 70% of healthy windows  -> learn mean/std/covariance (after optional outlier pruning)
last 30% of healthy windows   -> set thresholds on held-out healthy scores
```

Temporal split (not random) mimics deployment: early baseline calibrates, later
baseline validates thresholds.

Justification: avoids tuning thresholds on the exact same windows used to fit
the model.

### Threshold selection (conformal default)

Default: **conformal threshold** on healthy validation scores (`threshold_method="conformal"`, `conformal_alpha=0.10`).

```text
sort validation primary distances (Mahalanobis or kNN)
rank = ceil((n + 1) * (1 - alpha))
threshold = score at that rank
```

Interpretation: target at most ~10% false inhibits on held-out healthy baseline
windows (same idea as Layer 3 conformal calibration).

Legacy mode: `threshold_method="healthy_quantile"` with `threshold_quantile=0.999`
(~0.1% false inhibit target on validation).

If conformal α is infeasible (too few validation windows), threshold is set to
`inf` → all beats inhibit (fail-safe).

### Optional calibration outlier pruning

When `calibration_outlier_frac > 0` (e.g. 0.05):

```text
provisional fit on calibration split
score each calibration window
drop worst fraction by primary distance
refit mean/std/covariance on kept windows only
```

Justification: occasional mislabeled or noisy windows during “healthy” baseline
should not widen the baseline cloud or inflate the conformal threshold.

### Primary scorer: Mahalanobis or kNN

Both use the same z-scored `mahal_feature_names` subset.

**Mahalanobis (default):**

```text
d = sqrt((x - mean)^T * inv_cov * (x - mean))
```

**kNN alternative** (`anomaly_model="knn"`):

```text
d = mean distance from x to k nearest calibration vectors in Mahalanobis-subset space
```

Use kNN when baseline window count is small or covariance is unstable; validation
can compare both via `--anomaly-model knn`.

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

#### 2. Primary distance (Mahalanobis or kNN)

Multivariate distance on `mahal_feature_names` subset. `decide()` uses
`primary_distance` from `score()` — Mahalanobis by default, or mean kNN distance
when `anomaly_model="knn"`.

Justification:

- catches subtle combined abnormalities no single feature crosses alone
- threshold set by conformal α on healthy validation (default 10% false inhibit)
or legacy healthy quantile (0.999)

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

### Prospective 1-in-8 stimulation cadence

Implementation: `pipeline/stimulation_cadence.py`

The current deployment assumption is that the actuator is stimulated at most
once every 8 accepted R-peaks. Layer 2 therefore does not try to classify the
candidate beat after its R-peak. Instead:

```text
1. accepted beats 1-7 are unstimulated observation beats
2. each observation beat updates Layer 2 safety state
3. accepted beat 8 is the only stimulation candidate
4. default policy permits if at least 6 of 7 were safe and beat 7 was safe
5. after beat 8, the cadence state resets and a new 7-beat observation block starts
```

This is an **X-of-Y persistence** detector — the same formalism used for
arrhythmia onset confirmation in ICDs/AEDs (Medtronic NID, Boston Scientific
duration/persistence; Swerdlow et al. 1994), but inverted for safety: instead
of requiring X-of-Y fast intervals to CONFIRM danger and shock, we require
X-of-Y SAFE observation beats to PERMIT stimulation, so uncertainty defaults to
inhibit.

```text
Y = observation_beats            (window length)
X = min_safe_observations        (required safe count)
+ optional most-recent-beat guard (require_last_observation_safe)
```

Build one directly with `ProspectiveCadenceGate.from_x_of_y(x_safe, y_observed)`.
Persistence trades a little latency (cannot permit before Y beats) for strong
rejection of single-beat artifacts and transient misdetections — the correct
trade for an inhibit-only gate.

Why this matters:

- mechanical contraction follows the electrical R-peak too quickly to wait for
full morphology/feature analysis on the same beat
- the safety decision must therefore be ready before the stimulation candidate
- the candidate beat still needs a valid fast R-peak trigger, but it is not used
for the full Layer 2 decision
- observation beats can use a longer causal post-R lookahead (default 400 ms,
capped before the next detected peak), because the decision is only needed
before the 8th beat

Validation modes using this idea:

```text
run_beat_validation.py          -> rpeak_adaptive_cadence_1of8
run_cross_dataset_validation.py -> fast_causal_cadence_1of8
```

---

## Step 4: Fixed Policy (`decision/config.py`)

Not learned from data. Set by safety policy and benchmark tuning.


| Item                        | Role                                               |
| --------------------------- | -------------------------------------------------- |
| `DEFAULT_HARD_RULES`        | absolute veto limits before Mahalanobis            |
| `DEFAULT_MAHAL_EXCLUDE`     | features removed from Mahalanobis                  |
| `FROZEN_COUPLING_THRESHOLD` | PVC coupling ratio limit (0.80)                    |
| `FROZEN_ZSCORE_QUANTILE`    | max-zscore threshold quantile (0.90)               |
| `RRReliabilityConfig`       | species-specific RR trust settings for hybrid mode |
| `check_rr_reliability()`    | computes reliability flags for `decide_hybrid()`   |
| `SQI_ENSEMBLE_HARD_RULES`   | opt-in kSQI/pSQI/bSQI limits (inert until enabled) |
| `ONSET_STABILITY_REFERENCE_LIMITS` | opt-in onset/stability reference limits     |
| `hard_rules_with_extensions()` | merge SQI-ensemble rules into DEFAULT_HARD_RULES |


Do not change frozen thresholds without re-running cross-dataset benchmarks.
The opt-in SQI-ensemble / onset-stability limits are literature-inspired
placeholders and must be recalibrated per setup before any deployment claim.

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

## Operating Point and Artifact Robustness

Two offline audits sit on top of the runtime gate. Neither changes the
label-free deployment path (conformal calibration on healthy baseline only);
they quantify what that path costs and how it fails.

### Neyman-Pearson operating point (`run_np_operating_point.py`)

The deployable threshold is set label-free (conformal α on healthy windows).
It does not, by itself, know how much danger leaks through. Using the danger
labels that exist in public datasets, this script selects the operating point
that:

```text
minimise   normal false-inhibit rate       (Type II — therapy uptime)
subject to danger false-permit rate <= budget   (Type I — the safety metric)
```

This is the classical Neyman-Pearson test: bound the dangerous error, then be
as permissive as possible under that bound (Scott & Nowak 2005). It is a
**design / reporting** tool, not a runtime calibrator — animal deployment has
no danger labels, so the chosen α must be transferred and this script measures
the danger-leakage cost of that choice on labeled data. It also emits a
**per-record worst-case danger false-permit** table: judge Layer 2 by the worst
record, not the pooled mean.

Outputs: `np_frontier.csv`, `np_operating_point.csv`, `worst_record_danger.csv`.

### Artifact / lead-off stress test (`run_artifact_stress_test.py`)

Deployment adds artifact classes public benchmarks under-represent:
stimulation pulse trains, lead-off flatline, saturation/clipping, powerline
pickup, EMG bursts, baseline wander. A false permit during any of these is a
safety failure. This script injects each artifact into clean windows and
checks the SQI gate (kSQI + pSQI + FFT ratios, plus bSQI on real data)
inhibits, reporting a per-artifact detection rate and the clean false-inhibit
rate. Prefers real WFDB windows; falls back to a synthetic surrogate (smoke
only — bSQI is disabled on the surrogate because two detectors disagree even on
clean synthetic signal).

Outputs: `artifact_detection_summary.csv`, `artifact_per_window.csv`,
`artifact_clean_baseline.csv`.

---

## Validation, Tools, and Reports

Layer 2 has two support areas beyond `pipeline/`:

- `**validation/`** reruns the pipeline on ECG data and writes CSV metrics.
- `**viz/`** consumes those CSVs and produces thesis figures and animations.

### `validation/` — rerun pipeline on datasets


| Script                            | Purpose                                                         | When to use                           |
| --------------------------------- | --------------------------------------------------------------- | ------------------------------------- |
| `run_beat_validation.py`          | beat-sync and 1-in-8 cadence modes                              | Deployment-like benchmark             |
| `run_cross_dataset_validation.py` | MIT-BIH, NSTDB, SVDB, INCART, CUDB, VFDB                        | Generalization study                  |
| `run_pareto_sweep.py`             | unified Pareto entry point: quick / full / posthoc              | Operating-point search                |
| `run_np_operating_point.py`       | Neyman-Pearson operating point + worst-record danger leak       | Offline label-aware operating point   |
| `run_artifact_stress_test.py`     | inject stim/lead-off/EMG/wander artifacts, check SQI inhibits   | Artifact-robustness audit             |
| `run_causal_lookahead_sweep.py`   | sweep post-R lookahead ms                                       | Causal delay budget study             |
| `common.py`                       | shared helpers (filtering, scoring, calibrator fit)             | imported by validation scripts        |
| `layer2_validation_utils.py`      | conformal threshold, guard region, policy metrics, coverage CSV | imported by beat validation           |
| `pareto_quick.py`                 | fast 10-record MIT-BIH subset                                   | used by `run_pareto_sweep.py quick`   |
| `pareto_posthoc.py`               | threshold rescaling on saved per_beat.csv                       | used by `run_pareto_sweep.py posthoc` |


**Beat validation outputs (enhanced):**

```text
per_beat.csv                 split, safety_group, is_healthy, primary_distance, knn, risk_family
metrics_by_safety_group.csv  DANGEROUS / BENIGN_ABNORMAL / AF_CONTEXT / NOISE policy metrics
threshold_coverage.csv       observed vs target healthy false-inhibit on test split
risk_family_breakdown.csv    signal vs rhythm vs hard-rule inhibit reasons
```

**CLI flags (calibration / scoring):**

```text
--threshold-method conformal|healthy_quantile   default: conformal
--conformal-alpha 0.10                          target healthy false-inhibit
--calibration-outlier-frac 0.0                  optional prune before refit
--anomaly-model mahalanobis|knn                 primary distance scorer
--knn-k 5
--guard-s 5.0                                   exclude post-calibration overlap from test metrics
--legacy-labels                                 use healthy/abnormal instead of safety_group policy
```

**Record split labels** (per-record calibration mode):

```text
calibration   used to fit baseline
guard         first guard_s seconds after calibration (excluded from test metrics)
test          scored for safety metrics
calibration_no_stim / guard_excluded rows logged but not counted as false permit/inhibit
```

Policy grouping reuses `Layer3/pipeline/label_grouping.py`:

```text
NORMAL           healthy baseline + expected permit
DANGEROUS        VT/VFib → must inhibit (primary safety metric)
BENIGN_ABNORMAL  isolated PVC etc. → policy-dependent
AF_CONTEXT       AF spans → configurable (default: not counted as dangerous)
NOISE            artifact → must inhibit
```

### `viz/` — presentation figures


| Script                        | Output                                    | Purpose                    |
| ----------------------------- | ----------------------------------------- | -------------------------- |
| `make_all.py`                 | all figures below                         | one-command generation     |
| `plot_dataset_performance.py` | dataset / arrhythmia / worst-record plots | per-dataset diagnosis      |
| `plot_pareto.py`              | operating curves + Pareto frontier        | safety trade-off slides    |
| `plot_feature_auroc.py`       | AUROC bars + top deviators                | feature analysis           |
| `animate_beat_gate.py`        | gate walkthrough GIF                      | explain algorithm visually |


Legacy scripts from the old `tools/` and `reports/` folders are in `archive/old_tools/` and
`archive/old_reports/`.

---

## Quick Command Reference

```powershell
# Beat-sync validation (deployment-like, conformal threshold + policy groups)
.\.venv\Scripts\python.exe Layer2\validation\run_beat_validation.py `
  --data-dir data --datasets mit_bih_arrhythmia `
  --out-dir Results\layer2_beat_validation `
  --threshold-method conformal --conformal-alpha 0.10 `
  --anomaly-model mahalanobis --guard-s 5.0

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

