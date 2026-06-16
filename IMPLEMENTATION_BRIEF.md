# IMPLEMENTATION BRIEF
*ECG Safety Supervisor — closed-loop cardiomyoplasty / MNA control*
*Last updated: May 2026*

---

## 1. Current Repository Map

### Layer 1 — Fast deterministic detector and RR supervisor

| File | Role | Rewrite? |
|---|---|---|
| `Layer1/pipeline/main_pipeline.py` | Main entry: fast causal detector + rhythm supervisor | **Do not rewrite** |
| `Layer1/pipeline/r_peak_detector.py` | Causal adaptive R-peak detector | **Do not rewrite** |
| `Layer1/pipeline/rhythm_supervisor.py` | `RRSupervisor` — refractory blanking, adaptive RR band, recovery mode | **Do not rewrite** |
| `Layer1/pipeline/reference_annotations.py` | Reference beat loading from WFDB annotations, greedy TP/FP/FN match | **Do not rewrite** |
| `Layer1/pipeline/filters.py` | Butterworth bandpass + notch filter | **Do not rewrite** |
| `Layer1/pipeline/artifact_simulation.py` | Stimulation artefact injection for stress testing | **Do not rewrite** |
| `Layer1/pipeline/run_record.py` | Command-line entry point for one record | **Do not rewrite** |
| `Layer1/pipeline/run_benchmark.py` | Full multi-dataset benchmark, CSV + log output | **Do not rewrite** |
| `Layer1/tools/summarize_benchmark.py` | Quick diagnostic summary over benchmark CSV | **Do not rewrite** |
| `Layer1/tools/animate_record.py` | Offline visualization animation | **Do not rewrite** |

### Layer 2 — Handcrafted feature safety gate

| File | Role | Rewrite? |
|---|---|---|
| `Layer2/layer2_features.py` | Wavelet features (db4/4-level, log-energy + Shannon entropy), sample/approximate entropy | Small additions needed |
| `Layer2/layer2_gate.py` | `BaselineCalibrator` — Mahalanobis + max z-score, JSON save/load | Targeted upgrades needed |

### Layer 3 — SSL / anomaly detection research layer

| File | Role | Rewrite? |
|---|---|---|
| `Layer3/layer3_encoder.py` | `ECGEncoder1D` — small 1D ResNet (~700K params), GroupNorm, projection head | **Do not rewrite** |
| `Layer3/layer3_augmentations.py` | `ECGAugmentor` — physiology-aware augmentations (3KG-inspired, single-lead) | **Do not rewrite** |
| `Layer3/layer3_anomaly.py` | `DeepSVDDHead` + `fit_svdd` + `score_windows` + `save/load_svdd` | **Do not rewrite** |
| `Layer3/layer3_pretrain.py` | NT-Xent contrastive pretraining, CLOCS-CMSC positive sampling, linear probe | **Do not rewrite** |

### Data

| Path | Contents |
|---|---|
| `data/mit_bih_arrhythmia/` | MIT-BIH Arrhythmia DB (alias: mitdb) |
| `data/supraventricular_arrhythmia/` | Supraventricular Arrhythmia DB (alias: svdb) |
| `data/malignant_ventricular_arrhythmia/` | Malignant VT/VFib DB (alias: vfdb) |
| `data/creighton_vfib/` | Creighton University VFib DB (alias: cudb) |
| `data/noise_stress_test/` | Noise Stress Test DB (alias: nstdb) |
| `data/long_term_atrial_fibrillation/` | Long-Term AF DB (alias: ltafdb) |

See `data/README.md` for the full list and aliases.

### Results

| Path | Contents |
|---|---|
| `Results/benchmark_results/` | Fixed-threshold benchmark run |
| `Results/benchmark_results_full/` | Full benchmark (fixed + auto-calibrated), per_record.csv, per_group_summary.csv, features.csv |

---

## 2. What Is Already Implemented

### Layer 1
- Candidate QRS detection: amplitude threshold with hysteresis, optional slope gate to suppress T-wave
  oversensing and slow-drift false triggers.
- RR supervisor: adaptive RR band (EMA), refractory blanking, out-of-band rejection, recovery mode,
  recalibration logic.
- Stimulation artefact injection for stress testing the detector.
- Full multi-dataset benchmark against 5 PhysioNet modality groups, fixed vs auto-calibrated threshold.
- Auto-threshold calibration from 30 s of signal (95th percentile of |signal| × fraction).
- Annotation loading and greedy TP/FP/FN matching with configurable tolerance.

**Key benchmark result (auto-calibrated threshold):**

| Group | Se | PPV | FP/h |
|---|---|---|---|
| clean_sinus | 0.872 | 0.990 | 44 |
| mixed | 0.856 | 0.965 | 120 |
| af_dominant | 0.882 | 0.983 | 66 |
| vt_vfib | 0.771 | 0.740 | 1009 |
| noisy | 0.742 | 0.805 | 855 |

The detector works well on sinus/AF. It struggles on VT/VFib (FP/h > 1000) and noisy records.
This motivates Layer 2 as the safety gate — Layer 1 cannot distinguish safe from unsafe rhythm.

### Layer 2
- Wavelet decomposition: db4, 4 levels, log-energy and Shannon entropy per band (10 features).
- Sample entropy and approximate entropy (species-portable predictability measures).
- `BaselineCalibrator`: fits mean/std and covariance on baseline windows; computes Mahalanobis
  distance and max z-score; permits/inhibits with top-3 deviating feature logging; JSON save/load.
- Diagonal-fallback covariance for small calibration sets.

### Layer 3
- `ECGEncoder1D`: 4-stage 1D ResNet, GroupNorm, global average pool, 128-dim embedding.
- `ProjectionHead`: 2-layer MLP with L2-normalized output for NT-Xent.
- `ECGAugmentor`: time warp, Gaussian noise, baseline wander, bandpass perturbation, random
  crop-repad — all with configurable probabilities.
- `NTXentLoss` (SimCLR) + CLOCS-CMSC same-record positive sampling.
- Full contrastive pretraining loop with cosine LR schedule, periodic linear probe, checkpointing.
- `DeepSVDDHead` (no-bias MLP, collapse prevention) + `fit_svdd` + `score_windows` + threshold
  from in-baseline quantile + `.pt` save/load.

---

## 3. What Is Missing (vs. Safety Supervision thesis doc)

### 3.1 Layer 1 gaps

| # | Missing component | Reference in doc |
|---|---|---|
| L1-1 | **Signal Quality Indices (SQIs)** — bSQI (dual-detector agreement), pSQI (QRS-band power fraction), basSQI (baseline wander severity). None are computed. These are the first veto line before reaching Layer 2. | §2, Clifford et al. 2012 |
| L1-2 | **Two-speed architecture for rats** — a fast candidate path for same-beat triggering and a slower confirmation path for the RR supervisor. Currently a single pipeline. Critical at 250–500 bpm. | §2 |
| L1-3 | **Post-stimulation artefact blanking** as a configurable parameter passed from the controller. The supervisor has `min_blanking_ms` but no explicit stim-pulse blanking logic wired to an external trigger. | §2 |

### 3.2 Layer 2 gaps

| # | Missing component | Reference in doc |
|---|---|---|
| L2-1 | **HRV / RR-based features** — SDNN, RMSSD, coefficient of variation of RR, short/long RR fraction, sudden RR change (successive difference > threshold). No `rhythm_features.py` or equivalent exists. `layer2_features.py` docstring references it but the file is absent. | §7.1 |
| L2-2 | **Spectral HRV** — Lomb-Scargle or periodogram LF/HF decomposition with rodent-adjusted bands (LF: 0.2–0.75 Hz, HF: 0.75–3.0 Hz for rats). | §7.1 |
| L2-3 | **Amplitude / energy features** — RMS, peak-to-peak, peak-to-mean ratio, signal energy. Mentioned in CLAUDE.md and §7.2. Not in `layer2_features.py`. | §7.2, CLAUDE.md |
| L2-4 | **QRS template correlation** — beat-to-beat similarity or correlation with a running mean QRS template. Not implemented. | §7.2, CLAUDE.md |
| L2-5 | **Separate signal-only vs RR-dependent anomaly scores** — doc explicitly requires independent logging of a morphology/noise score and a rhythm instability score. Currently a single combined Mahalanobis score. | §7.3 |
| L2-6 | **Robust location/scale (median/MAD)** — `BaselineCalibrator` uses mean/std. Doc specifies median + MAD for robustness to outlier windows. | §7.3 |
| L2-7 | **Ledoit-Wolf shrinkage covariance** — current code uses ridge-regularized sample covariance. Doc specifies shrinkage for well-conditioning with few effective samples. | §7.3, Ledoit & Wolf 2004 |
| L2-8 | **Calibration/validation split** — threshold must be set on a held-out validation portion of T2 baseline, not on the calibration set itself. Currently threshold is set on training data. | §7.3 |
| L2-9 | **Dynamic/adaptive calibration** — two modes: fixed T2 anchor (never overwritten) + slow-update adaptive that only updates under strict conditions (stim paused, RR stable, bSQI high, operator confirm). Not implemented. | §7.4 |
| L2-10 | **T0/T1/T2/T3/T4 baseline state model** — the protocol of multiple calibration states is not modeled. No calibration session management. | §6.3 |
| L2-11 | **SQI pre-gate veto** — gross failures (flatline, saturation, lead disconnect) should be caught before Mahalanobis. Not implemented. | §2 |
| L2-12 | **No Layer 2 validation script** — no script runs Layer 2 on MIT-BIH records (oracle R-peaks then own detector R-peaks) and reports false-permit / false-inhibit rates. | CLAUDE.md §12 |
| L2-13 | **No Layer 2 CSV output / logging** — gate decisions are not logged per window. | CLAUDE.md §12 |

### 3.3 Layer 3 gaps

| # | Missing component | Reference in doc |
|---|---|---|
| L3-1 | **Gaussian Mahalanobis distance in embedding space** — doc describes this as a fast no-training baseline alongside SVDD. Not implemented. | §8.3 |
| L3-2 | **Window indexer script** — `layer3_pretrain.py` requires a CSV with `record_id, signal_path, start_idx, n_samples`. No script generates this from the PhysioNet `.dat` files. | layer3_pretrain.py docstring |
| L3-3 | **Layer 3 validation pipeline** — no script evaluates the encoder on labeled anomaly data (e.g. MIT-BIH VT/VFib windows vs sinus). | §8.3, CLAUDE.md |
| L3-4 | **Continued pretraining workflow for animal ECG** — workflow described (pretrain on human → fine-tune on rat baseline) but no script or config exists. | §8.4 |

### 3.4 Integration gaps

| # | Missing component |
|---|---|
| I-1 | **No end-to-end integration script** — the three layers have no shared runner. |
| I-2 | **Layer 2 feature assembly is broken** — `layer2_features.py` produces wavelet+entropy features; `layer2_gate.py` consumes a feature dict; but no code connects Layer 1 R-peaks to a combined feature vector and feeds it to the gate. |
| I-3 | **No Pynapse / real-time interface** — no file defines the rolling buffer, the window extraction, or the safety state that a Pynapse session would consume. |

---

## 4. Layer Input / Output Specification

### Layer 1

```
Input  : raw ECG buffer (numpy 1D, float), fs (Hz)
Step 1 : causal bandpass + notch filter
Step 2 : FastCausalThresholdDetector → candidate R-peak sample indices
Step 3 : RRSupervisor → accepted R-peak sample indices + safety flag
Step 4 : [MISSING] SQI computation → bSQI, pSQI, basSQI scalars
Output : accepted_samples (array), layer1_safe (bool), sqis (dict)
```

### Layer 2

```
Input  : ECG window (numpy 1D, ~5 s), R-peak indices within window, fs (Hz)
Step 1 : [MISSING] compute HRV/RR features (SDNN, RMSSD, CV, spectral LF/HF)
Step 2 : [MISSING] compute amplitude/energy features (RMS, pk-pk, pk-mean)
Step 3 : compute wavelet features (layer2_features.py) ← EXISTS
Step 4 : compute entropy features (layer2_features.py) ← EXISTS
Step 5 : [MISSING] compute QRS template correlation
Step 6 : [MISSING] SQI pre-gate veto (reject if bSQI < threshold before scoring)
Step 7 : BaselineCalibrator.decide() → permit/inhibit + per-feature z-scores
Output : layer2_safe (bool), scores (dict), top_deviating_features (list)
Log    : per-window CSV row with timestamp, all feature values, all scores, decision
```

### Layer 3

```
Input  : ECG window (numpy 1D, 5 s), fs (Hz)
Step 1 : z-score normalize
Step 2 : ECGEncoder1D.forward() → 128-dim embedding
Step 3 : DeepSVDDHead.forward() → SVDD output
Step 4 : distance to center c → anomaly score
Output : layer3_safe (bool), anomaly_score (float)
```

### Final gate

```python
permit = layer1_safe AND layer2_safe AND (layer3_safe OR layer3_unavailable_policy)
```

`layer3_unavailable_policy` defaults to `False` (inhibit if Layer 3 is unavailable) unless
explicitly configured to be advisory-only for the current experimental phase.

---

## 5. Data Assumptions

| Assumption | Value |
|---|---|
| Human validation datasets | MIT-BIH (360 Hz), VFDB (250 Hz), CUDB (250 Hz), NSTDB (360 Hz), LTAFDB (128 Hz) |
| Rat ECG (future) | Single- or two-lead, 1–5 kHz acquisition rate, 250–500 bpm |
| Standard window | 5 s, 50% overlap (1 s stride) |
| Feature window input | 1D numpy array, float, already filtered |
| R-peaks | Sample indices within the window; caller provides either oracle (for validation) or Layer 1 output (for runtime) |
| Sampling rate | Always passed as `fs` parameter; no hardcoded Hz except in smoke tests |
| Layer 3 input | 5 s @ 250 Hz = 1250 samples; resampling needed for other rates |

---

## 6. Safety Rules (non-negotiable)

1. ML (Layers 2 and 3) can only **veto / inhibit** stimulation. They cannot command it.
2. **Uncertain = inhibit.** Missing features, NaN scores, exceptions, server timeout, low SQI, low calibration confidence, first window before baseline is fitted — all default to inhibit.
3. **False permit is the critical failure mode.** A false inhibit reduces therapy availability; a false permit risks arrhythmia induction.
4. **Layer 1 deterministic logic is never removed or bypassed.**
5. **No non-causal filtering in real-time deployment.** All offline scripts using `filtfilt` must be clearly marked as offline-only.
6. **No blind online retraining.** The adaptive calibration may update only under explicit, logged, multi-condition confirmation. It may not silently track a slow arrhythmia.
7. **Threshold is set from held-out data, not training data.**

---

## 7. First Coding Tasks (Next 1–2 Weeks)

### Week 1 — Layer 2 feature completion and gate upgrade

| Priority | Task | File(s) to create/edit |
|---|---|---|
| 1 | Create `layer2_rhythm_features.py` — SDNN, RMSSD, CV, short/long RR fraction, successive difference max, spectral HRV (Lomb-Scargle, rodent-band LF/HF) | **CREATE** `Layer2/layer2_rhythm_features.py` |
| 2 | Add amplitude/energy features to `layer2_features.py` — RMS, peak-to-peak, peak-to-mean ratio | **EDIT** `Layer2/layer2_features.py` |
| 3 | Create `layer2_full_features.py` — thin function that assembles the full feature dict from (window, r_peaks, fs): calls rhythm features + wavelet + entropy + amplitude | **CREATE** `Layer2/layer2_full_features.py` |
| 4 | Upgrade `layer2_gate.py` — (a) MAD/median location-scale, (b) calibration/validation split, (c) Ledoit-Wolf option, (d) separate signal-only and RR-dependent score fields | **EDIT** `Layer2/layer2_gate.py` |
| 5 | Create `layer1_sqi.py` — pSQI (QRS-band power / total power), basSQI (low-freq energy fraction), flatline/saturation detector; returns dict | **CREATE** `Layer1/layer1_sqi.py` |

### Week 2 — Validation and integration

| Priority | Task | File(s) to create/edit |
|---|---|---|
| 6 | Create `Layer2/layer2_validate.py` — loads MIT-BIH records, runs Layer 2 with (a) oracle R-peaks and (b) own Layer 1 detector R-peaks, reports false-permit and false-inhibit rate per record and group, writes CSV | **CREATE** `Layer2/layer2_validate.py` |
| 7 | Create `Layer3/layer3_window_index.py` — scans PhysioNet `.hea` files, extracts non-overlapping 5 s windows, writes window index CSV needed by `layer3_pretrain.py` | **CREATE** `Layer3/layer3_window_index.py` |
| 8 | Create `Layer2/layer2_session.py` — `SessionBaseline` class wrapping T0/T1/T2/T3/T4 state, fixed and adaptive calibrator instances, update-guard logic, per-window CSV logging | **CREATE** `Layer2/layer2_session.py` |

---

## 8. Files to Edit or Create (summary)

### Create (new files)

```
Layer1/layer1_sqi.py                      SQI computation (bSQI, pSQI, basSQI)
Layer2/layer2_rhythm_features.py           HRV and RR-based features
Layer2/layer2_full_features.py             Full feature assembler (rhythm + wavelet + entropy + amplitude)
Layer2/layer2_validate.py                  MIT-BIH validation — false permit/inhibit rates, CSV output
Layer2/layer2_session.py                   Session/baseline state management (T0–T4), adaptive calibration guard
Layer3/layer3_window_index.py              Window index CSV generator for SSL pretraining
```

### Edit (targeted additions, not rewrites)

```
Layer2/layer2_features.py                  Add amplitude/energy features (RMS, peak-to-peak, peak-to-mean)
Layer2/layer2_gate.py                      MAD/median, calibration/validation split, Ledoit-Wolf, dual scores
```

### Do not touch

```
Layer1/pipeline/main_pipeline.py
Layer1/pipeline/r_peak_detector.py
Layer1/pipeline/rhythm_supervisor.py
Layer1/pipeline/reference_annotations.py
Layer1/pipeline/filters.py
Layer1/pipeline/artifact_simulation.py
Layer1/pipeline/run_record.py
Layer1/pipeline/run_benchmark.py
Layer1/tools/summarize_benchmark.py
Layer3/layer3_encoder.py
Layer3/layer3_augmentations.py
Layer3/layer3_anomaly.py
Layer3/layer3_pretrain.py
```

---

## 9. Open Technical Questions

See section 10 of the main CLAUDE.md for the full list. Additional questions arising from this
implementation review:

1. **bSQI threshold**: what dual-detector agreement threshold should trigger immediate inhibition?
   (Clifford et al. suggest <0.8 for clinical alarms.)
2. **Ledoit-Wolf vs ridge**: `sklearn.covariance.LedoitWolf` is preferred per the thesis, but
   requires scikit-learn as a dependency. Acceptable? Or keep ridge-regularized NumPy fallback?
3. **Spectral HRV with few beats**: Lomb-Scargle HRV on a 5 s window at rat heart rate gives
   only ~20–40 beats. Frequency resolution is very coarse at 0.2 Hz. Is window length 5 s
   sufficient or should HRV be computed over a longer rolling window (e.g. 30 s)?
4. **Layer 2 validation metric**: should false-permit rate be evaluated at the window level or
   the beat level? Window-level is easier to compute; beat-level is more meaningful clinically.
5. **T-wave oversensing in rats**: at 400 bpm the T-wave is very close to the next P-wave.
   Is the existing slope gate sufficient, or is a dedicated T-wave suppression needed?
6. **Pynapse interface**: will Layer 2 and Layer 3 run in the Pynapse Python environment or
   as a separate server process? This determines whether PyTorch (Layer 3) is available at runtime.
