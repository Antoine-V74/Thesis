# Layer 2 Pipeline Files

Core safety path:

```text
ECG window + R-peaks
  -> extract_layer2_features()
  -> calibrate_layer2()     # session start
  -> decide_layer2()        # runtime
```

Prospective stimulation path:

```text
beats 1-7: extract features -> Layer 2 permit/inhibit -> update cadence state
beat 8:    use stored cadence state + R-peak detection -> trigger or inhibit
```

Observation beats can use a longer causal post-R lookahead because the decision
is only consumed at beat 8. The 8th beat itself remains trigger-only.

Start from `main_pipeline.py`. Use `Layer2/validation/` for dataset benchmarks.

## Feature extraction

| File | Purpose |
|------|---------|
| `main_pipeline.py` | Public API: extract, calibrate, decide |
| `stimulation_cadence.py` | Prospective 1-in-8 stimulation policy |
| `full_features.py` | Assembles all features into one dict |
| `signal_features.py` | Wavelets, entropy (no R-peaks needed) |
| `morphology_features.py` | Beat shape / template features |
| `rhythm_features.py` | RR stats and spectral HRV |

### Are R-peaks optional?

In the API, yes. In normal runtime safety, no.

- **With R-peaks:** full feature set (`signal__`, `rr__`, `morph__`)
- **Without R-peaks:** only `signal__` features (SQI, wavelets, energy, etc.)

Use R-peaks for deployment. The optional parameter exists so Layer 2 can still
compute signal-quality features when peak detection fails — but rhythm and
morphology safety checks are then unavailable.

## `decision/` folder

Baseline learning and permit/inhibit logic. Combined in `BaselineCalibrator`.

| File | Role |
|------|------|
| `config.py` | Fixed safety policy (see below) |
| `calibration.py` | Learn healthy baseline: mean/std, covariance, thresholds, save/load |
| `gate.py` | Runtime scoring and permit/inhibit methods |
| `__init__.py` | Exports `BaselineCalibrator` |

### `config.py` — fixed policy, not learned from data

| Item | What it does |
|------|--------------|
| `DEFAULT_HARD_RULES` | Absolute veto limits applied before Mahalanobis. E.g. fewer than 3 beats, too much HF noise, low template correlation. |
| `DEFAULT_MAHAL_EXCLUDE` | Features kept out of Mahalanobis because they are hard rules or too variable in healthy data. |
| `FROZEN_COUPLING_THRESHOLD` | PVC coupling ratio limit (0.80), fixed by benchmark. |
| `FROZEN_ZSCORE_QUANTILE` | Quantile for max-zscore threshold (0.90), fixed by benchmark. |
| `RRReliabilityConfig` | Species-specific settings for RR trust checks in `decide_hybrid()`. |
| `check_rr_reliability()` | Returns whether the current R-peak and RR look-back history are trustworthy. |

### `calibration.py` — session-start learning

Fits on healthy baseline windows (first 70% learn statistics, last 30% set
thresholds). Stores mean, std, inverse covariance, and permit/inhibit thresholds.

### `gate.py` — runtime decision

| Method | Use |
|--------|-----|
| `score()` | Mahalanobis, z-scores, signal/RR diagnostic proxies |
| `check_hard_rules()` | Test fixed veto limits |
| `decide()` | Standard gate: hard rules → Mahalanobis → max z-score |
| `decide_hybrid()` | Beat-sync gate with RR reliability and rewarming |

### `decide()` vs `decide_hybrid()`

**`decide()`** — default runtime gate.

```text
hard rules -> Mahalanobis -> max z-score -> permit/inhibit
```

Use when you have a feature vector and want a single safety answer.

**`decide_hybrid()`** — beat-synchronous deployment gate.

Adds checks before/around the statistical gates:

```text
hard rules
  -> signal proxy safe?
  -> current R-peak detected in window?
  -> RR history reliable?
       yes -> full Mahalanobis rhythm check
       no  -> rewarming counter (permit after N clean beats)
```

Use at each stimulation trigger when RR peak detection quality varies and
you need to separate "bad rhythm" from "unreliable RR history after artifact".

### `stimulation_cadence.py` - prospective 1-in-8 stimulation

`ProspectiveCadenceGate` implements the current therapy timing assumption:
stimulate only one accepted R-peak out of eight.

```text
observe 7 beats -> require at least 6 safe, with beat 7 safe -> candidate beat 8
```

The candidate beat is not analyzed by Layer 2 before stimulation. At beat 8 the
runtime only needs a valid R-peak trigger and the precomputed cadence decision.
