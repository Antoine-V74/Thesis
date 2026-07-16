# All Summary — ECG Safety Pipeline

Supervisor-oriented summaries for the R-peak detector and Layers 1–3. Each section
follows the same structure: motivation, core idea, method, decision logic,
project fit, and limitations.

```text
Raw ECG
  → R-peak detector (causal, adaptive)
  → Layer 1 RR supervisor (timing safety)
  → Layer 2 handcrafted feature gate (primary deploy path)
  → Layer 3 learned embedding veto (optional research)
  → Global stimulation policy
```

None of these layers commands stimulation directly. They only permit or veto.

---

# R-Peak Detector — Fast causal beat detection

## Why it is needed

Every downstream layer depends on knowing **when beats occur**. Layer 2 morphology
features, Layer 3 beat-sync windows, and Layer 1 rhythm supervision all need
reliable R-peak timestamps. The detector must therefore be:

- **causal** (uses only past and current samples at runtime),
- **fast** (suitable for a low-latency control loop),
- **conservative** (missed peaks are safer than false triggers on noise).

The R-peak detector is the **first signal-processing stage** of the pipeline. It
proposes candidate beats; Layer 1 decides which candidates are timing-plausible
enough to accept.

## Core idea

The current implementation is a **fast causal threshold detector**
(`FastCausalThresholdDetector` in `Layer1/pipeline/r_peak_detector.py`).

After bandpass filtering, the detector tracks the ECG amplitude and slope online,
 adapts noise/signal thresholds, and emits a candidate peak only after the
waveform confirms a QRS-like transient (rising edge, local maximum, confirmed
descent).

In simple terms:

```text
filtered ECG sample stream
→ adaptive amplitude + slope thresholds
→ peak tracking on rising edge
→ confirmation after descent
→ candidate R-peak
```

Important runtime detail: the system can only **act after confirmation**, not at
the ideal peak sample. Confirmation delay is part of the real-time latency budget
(median live lag ~34–72 ms depending on dataset).

## Processing steps


| Step                    | What happens                                                          |
| ----------------------- | --------------------------------------------------------------------- |
| 1. Filtering            | 5–20 Hz bandpass; optional 50/60 Hz notch (`filter_ecg`)              |
| 2. Polarity selection   | Choose positive or negative polarity from initial calibration segment |
| 3. Threshold adaptation | Online noise/signal level tracking with EMA smoothing                 |
| 4. Peak tracking        | Enter on rising edge above thresholds; track local maximum            |
| 5. Confirmation         | Emit candidate after descent, drop-from-peak, or max-width rule       |
| 6. Refractory           | Suppress re-triggers within detector refractory period (~90 ms)       |


Deployment uses **causal** filtering (`lfilter`). Offline benchmarks may use
zero-phase filtering for analysis only.

## Benchmark performance (human ECG proxy)

Results from `adaptive_threshold_v2` on PhysioNet beat-annotated datasets
(full tables in `reports/rpeak_detection/`):


| Dataset | F1 (approx.) | Median live lag | Comment                              |
| ------- | ------------ | --------------- | ------------------------------------ |
| MIT-BIH | 0.97         | ~42 ms          | Strong on development set            |
| NSRDB   | 0.97         | ~49 ms          | Mostly sinus rhythm                  |
| NSTDB   | 0.97         | ~72 ms          | Robust at SNR ≥ 12 dB                |
| INCART  | 0.92         | ~33 ms          | Some domain / lead shift             |
| LTAFDB  | 0.90         | ~39 ms          | High sensitivity; extra peaks        |
| SVDB    | 0.90         | ~45 ms          | Worst-tail records dominate failures |


Failure modes are concentrated in a small fraction of records (especially SVDB
and some INCART/LTAFDB cases): missed peaks, extra peaks on T-waves or noise, or
catastrophic morphology mismatch.

## Why this matches the project constraints


| Criterion          | Why it matches                                               |
| ------------------ | ------------------------------------------------------------ |
| Real-time          | Streaming-style causal processing                            |
| Deterministic      | No ML inference; fixed rules and adaptive thresholds         |
| Conservative bias  | Confirmation delay and refractory reduce false triggers      |
| Downstream utility | Provides `candidate_samples` for Layer 1 supervision         |
| Benchmarked        | Multi-dataset evaluation with sensitivity, PPV, F1, live lag |


## Limitations

The detector sees **timing and transient shape**, not clinical arrhythmia labels.
A beat can be detected at a plausible time while morphology is still abnormal —
that is why Layers 2 and 3 exist.

Performance degrades on very noisy ECG, unusual morphology, or domain shift
(e.g. some SVDB/INCART records). Reference annotations are used **only offline**
for benchmarking, never at runtime.

Polarity auto-selection and adaptive thresholds need a short calibration period
at session start; early samples should not be trusted for triggering.

---

# Approach 1 / Layer 1 — Fast rhythm supervision

## Why Layer 1 is needed

The R-peak detector proposes candidates quickly, but not every candidate is
timing-plausible. Noise, T-waves, missed beats, and arrhythmia can produce
extra or shifted detections.

Layer 1 adds a **deterministic RR supervisor** that asks:

```text
Is this candidate beat timing plausible and stable enough to trust for triggering?
```

Layer 1 does not command stimulation. It produces **accepted peaks** and
**trigger-eligible peaks** (`trigger_samples`) that downstream layers and the
controller may use only if all safety gates permit.

## Core idea

Layer 1 combines:

1. causal ECG filtering,
2. fast R-peak candidate detection,
3. RR-based supervisory state machine.

From stable timing, it maintains an adaptive RR reference and accepts only beats
whose intervals fall inside physiologically plausible hard limits and an adaptive
confidence band.

At the beginning of a session, the supervisor **calibrates** on early beats, then
enters **RUNNING** mode where accepted beats become trigger-eligible. If timing
becomes suspicious, it enters **RECOVERY** and stops producing trigger samples
until rhythm stabilizes.

## Supervisor modes


| Mode            | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| **WAIT_ARM**    | Ignore early candidates during filter/threshold warm-up        |
| **CALIBRATING** | Collect RR intervals; build median/EMA reference; no triggers  |
| **RUNNING**     | Accept plausible beats; emit trigger-eligible samples          |
| **RECOVERY**    | Wider plausibility band; no triggers until timing restabilizes |


Typical path after startup:

```text
WAIT_ARM → CALIBRATING → RUNNING
                ↑            ↓
                └── RECOVERY ┘
```

## Decision logic

```text
Raw ECG
        ↓
Causal filter
        ↓
R-peak candidate detector
        ↓
RR supervisor (per candidate)
        ↓
accepted_samples / trigger_samples
```

For each candidate in **RUNNING**, Layer 1 checks:


| Check                 | Example default       | Meaning                                          |
| --------------------- | --------------------- | ------------------------------------------------ |
| Refractory / blanking | 150–200 ms+           | Ignore post-beat artifacts and double detections |
| Hard RR limits        | 250–2500 ms           | Reject physiologically impossible intervals      |
| Adaptive RR band      | ±10–40% around RR EMA | Reject sudden timing changes vs recent rhythm    |


The adaptive band width (`band_frac`) tightens or widens based on recent RR
variability (robust MAD) and temporarily widens after recovery.

If accepted in RUNNING:

```text
decision = accept_running
→ append to accepted_samples
→ append to trigger_samples
→ update RR EMA
```

In RECOVERY or CALIBRATING, beats may be accepted for timing analysis but **do
not** produce trigger samples.

## Hybrid handling when timing is unstable

Layer 1 is intentionally conservative when rhythm breaks down:

- long RR → often triggers **RECOVERY**,
- repeated out-of-band intervals → reject or recover,
- during RECOVERY → **no trigger samples** until plausible timing returns.

This prevents the fast loop from continuing to fire triggers on corrupted rhythm
estimates. Downstream Layer 2 hybrid mode can apply a parallel “RR history
unreliable” policy for morphology/rhythm features.

## Why this matches the project constraints


| Criterion            | Why Layer 1 matches                                  |
| -------------------- | ---------------------------------------------------- |
| Real-time first line | Low latency, deterministic, always available         |
| Conservative         | Recovery and calibration withhold triggers           |
| No labels at runtime | Uses only timing plausibility, not arrhythmia labels |
| Complements detector | Turns raw candidates into supervised timing events   |
| Interpretable        | Every decision has a logged reason code              |


## Limitations

Layer 1 sees **timing**, not morphology or signal quality. A PVC or aberrant beat
can still arrive at a plausible RR interval and pass Layer 1.

It depends on the R-peak detector: missed or extra peaks propagate into RR
features and downstream gates.

Hard RR limits and adaptive bands are species- and policy-dependent; rat settings
may differ from human benchmark defaults.

Layer 1 is necessary but **not sufficient** for stimulation safety — Layers 2
and optionally 3 provide morphology and richer context.

---

# Approach 2 / Layer 2 — Interpretable ECG feature safety gate

## Why Layer 2 is needed

Layer 1 checks whether detected R-peaks arrive at plausible times. This is
essential for real-time stimulation, but it mostly sees timing. A beat can arrive
at a plausible RR interval while the ECG morphology is abnormal, the signal
quality is poor, or the recent ECG no longer resembles the stable baseline of the
current subject/session.

Layer 2 is designed to address this limitation. It adds a richer but still
interpretable analysis of the ECG signal itself.

The question becomes:

```text
Does the recent ECG still look compatible with the stable safe baseline of this subject/session?
```

Layer 2 does not command stimulation by itself. It only permits the beat/window to
pass to the global safety policy or vetoes it.

## Core idea

At the beginning of the relevant experimental phase, ECG is recorded during a
period assumed to be stable and safe for the current setup. This reference does
not need to be a perfectly healthy pre-experiment ECG. In animal experiments, the
relevant baseline may be a stable post-surgery, pre-stimulation, or heart-failure
state before assistance.

From this period, Layer 2 estimates a personalized baseline in feature space.

During runtime, each new beat or short analysis window is converted into a
handcrafted feature vector. This vector is compared with the stable
subject/session baseline. If the deviation is too large, Layer 2 inhibits
stimulation. If the features remain inside the calibrated safe region, Layer 2
permits the beat/window to pass.

## Feature families

Layer 2 uses handcrafted features grouped into three interpretable families.


| Feature family         | What it captures                                                                                                                      | Why it matters                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Timing / rhythm        | RR intervals, local heart rate, RR variability, short/long RR fractions, pauses, sudden rhythm changes                                | Detects rhythm instability that may not be captured by one RR interval alone |
| Morphology             | QRS width clues, beat-shape changes, amplitude ratios, template similarity, neighbor correlation, wavelet energy or entropy           | Detects changes in beat shape that Layer 1 cannot see                        |
| Noise / signal quality | Baseline wander, high-frequency noise, saturation, flatline-like behavior, signal energy, detector confidence, signal-quality indices | Prevents the system from treating unreliable ECG as safe                     |


The different feature families use different time horizons. Morphology and signal
quality can often be assessed from a short local ECG window, whereas rhythm
features require a longer history of accepted R-peaks (~30 s look-back in the
current implementation).

## Decision logic

Layer 2 converts feature evidence into a permit/veto decision using a
conservative gate.

```text
Detected R-peaks + ECG signal
        ↓
Beat/window extraction
        ↓
Handcrafted feature extraction
        ↓
Safety gate
        ↓
Layer 2 permit / veto
```

The safety gate has two main parts.

### 1. Hard safety checks

Some conditions should directly inhibit stimulation before any baseline-distance
score is trusted.

Examples include:


| Hard check           | Reason                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Poor signal quality  | If the ECG is unreliable, stimulation should not be allowed                                |
| Low morphology trust | If beat shape or template correlation is unreliable, morphology-based decisions are unsafe |
| Unstable RR history  | If rhythm history is not reliable enough, rhythm features should not be trusted            |
| Insufficient data    | If there is not enough baseline or recent history, the system should fail conservatively   |


Fixed hard rules catch extreme feature violations (e.g. `morph__template_corr < 0.55`,
`signal__hf_noise_ratio > 0.35`, excessive short/long RR fractions). Separate
RR-reliability checks decide whether rhythm-dependent features are trustworthy
enough to use.

### 2. Baseline-deviation score

If the hard checks pass, the current feature vector is compared with the
personalized baseline.

The main score is a distance from the baseline — Mahalanobis distance after
robust normalization by default, with optional kNN alternative. A max per-feature
z-score is used as a backup to catch cases where one feature becomes extreme.

In simple terms:

```text
current ECG features
→ distance from stable baseline
→ score too high?
→ inhibit
```

The threshold is set from stable baseline data, not from dangerous labels.
Operationally, the cutoff is chosen so that most held-out stable baseline windows
would still pass (conformal default: target ~10% false inhibition on baseline
validation).

## Hybrid handling when RR history is unreliable

Some Layer 2 features depend on reliable R-peaks and enough RR history. If RR
history is temporarily unreliable, the system should not blindly use rhythm
features.

In that case, Layer 2 can fall back to conservative signal and morphology checks,
or inhibit until enough clean beats rebuild a trustworthy rhythm history. After a
short sequence of clean beats, full scoring can resume.

This prevents one bad segment from permanently blocking the system while still
keeping the decision conservative during uncertainty.

## Why this matches the project constraints


| Criterion                     | Why Layer 2 matches                                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Personalized                  | The baseline is fitted from the same subject/session, not from a universal population threshold                                                  |
| No labels in real use         | In animal deployment, no beat labels are required. Human ECG labels are used only offline for benchmark validation                               |
| Interpretable                 | If Layer 2 inhibits, the reason can often be explained: rhythm instability, abnormal morphology, poor signal quality, or deviation from baseline |
| Animal-relevant               | It does not require a large labeled animal ECG dataset. Features can be adapted or rescaled across species and setups                            |
| Conservative                  | Noisy, unstable, implausible, or unfamiliar ECG should inhibit rather than permit stimulation                                                    |
| More informative than Layer 1 | It adds morphology and signal-quality information, not only RR timing                                                                            |
| Safety-oriented               | It should be evaluated mainly by false permits on dangerous rhythms, not by accuracy alone                                                       |


## Limitations

Layer 2 depends on the quality of the calibration period. If the baseline ECG is
unstable, noisy, or already unsafe, the feature baseline may be unreliable. In
that case, the system should fail conservatively and not proceed with
stimulation.

Layer 2 also depends on reliable R-peak detection for rhythm and beat-synchronous
morphology features. Missed, duplicated, or shifted R-peaks can make some
features unreliable.

Some rhythms may require explicit safety policy decisions. For example, isolated
PVCs, fusion beats, or benign ectopy may be abnormal relative to baseline, but
their stimulation meaning must be defined with supervisors.

Finally, Layer 2 is limited by the handcrafted features chosen. It may miss
complex ECG changes involving subtle combinations of morphology, rhythm, and
noise. This motivates Layer 3, where a learned representation may capture richer
ECG structure while still using a personalized baseline.

---

# Approach 3 / Layer 3 — Learned embedding safety veto

## Why Layer 3 is needed

Layer 2 compares handcrafted features to a session baseline. This is
interpretable and well suited to deployment, but it is limited by the features we
chose. Some ECG changes may appear only as subtle combinations of morphology,
rhythm, and noise that are hard to capture with fixed descriptors.

Layer 3 is designed to test whether a **learned representation** can add safety
value beyond Layer 2, while keeping the same deployment philosophy: compare new
ECG to a stable subject/session baseline, and inhibit when the signal no longer
looks compatible with that reference.

The question becomes:

```text
Does this beat still look compatible with the stable baseline of this subject/session,
when viewed through a learned encoder?
```

Layer 3 does not command stimulation by itself. It only permits the beat/window to
pass to the global safety policy or vetoes it. It is an **optional research
extension**, not the primary deployment gate.

## Core idea

At the beginning of the relevant experimental phase, ECG is recorded during a
period assumed to be stable and safe for the current setup — same logic as Layer
2. In animal experiments, the relevant baseline may be a stable post-surgery,
pre-stimulation, or chronic state before assistance.

From this period, Layer 3 builds a **personalized healthy reference in embedding
space**.

During runtime, each new beat or short ECG window is passed through a small
encoder and converted into a compact embedding vector. This embedding is compared
with the stable subject/session baseline. If the deviation is too large, Layer 3
inhibits stimulation. If the embedding remains inside the calibrated safe region,
Layer 3 permits the beat/window to pass.

An encoder may be pretrained offline on unlabeled ECG, but the **safety boundary
itself is always learned per subject/session** from stable baseline data, not from
arrhythmia labels.

## Representation design

Layer 3 does not engineer explicit feature families like Layer 2. Instead, it
relies on a learned summary of the local ECG waveform.


| Component                  | What it captures                                                              | Why it matters                                                                   |
| -------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Beat-centred ECG window    | Local morphology and short-context waveform structure around the trigger beat | Stimulation safety depends on the current beat, not a long mixed window          |
| Encoder (`ECGEncoder1D`)   | A 128-dimensional embedding summarising the window                            | Compresses complex waveform information into a compact representation            |
| Healthy embedding baseline | Cloud of embeddings from stable baseline ECG for this subject/session         | Defines what “normal for this animal/session” looks like in representation space |
| Distance score             | How far the current embedding is from that healthy cloud                      | Converts representation difference into a safety score                           |


The preferred runtime setting is a **short beat-sync window** (~1 s), with causal
timing and a small post-R lookahead (50–150 ms) to approximate stimulation delay.

## Phase 1 encoder comparison

Layer 3 research compares **representation methods** under a **fixed downstream
safety scorer**. This isolates encoder quality from threshold tuning.


| Arm    | Method                                        | Role                                             | Implementation status           |
| ------ | --------------------------------------------- | ------------------------------------------------ | ------------------------------- |
| **A0** | Layer 2 handcrafted features                  | Control floor — same scorer, handcrafted input   | Implemented                     |
| **A**  | NT-Xent contrastive SSL                       | Default learned encoder (SimCLR-style)           | Implemented                     |
| **A1** | VICReg non-contrastive SSL                    | Alternative without negatives or reconstruction  | Implemented                     |
| **B**  | MAE + subject-contrastive (ZEROSHOT-inspired) | Masked reconstruction + record/subject alignment | Implemented                     |
| **C**  | Multi-lead upper bound (3KG/ST-MEM)           | Theoretical upper bound if multi-lead data exist | Not implemented — appendix only |


**Fair comparison rule:** all arms use the same session calibration, Mahalanobis or
kNN distance, and conformal thresholding. Only the representation changes.

In practice:

- **A0** is evaluated directly inside the Phase 1 harness (`--phase1-eval`).
- **A / A1 / B** are trained as different encoder checkpoints
(`pretrain_encoder.py --ssl-objective ntxent|vicreg|mae_subject_contrastive`),
then validated with the same beat-sync pipeline and different `--checkpoint`.
- After pretraining, only the **encoder weights** are kept; projection heads,
decoders, and expanders are discarded.

## Decision logic

Layer 3 converts embedding evidence into a permit/veto decision using a
conservative one-class anomaly gate.

```text
ECG window
        ↓
Robust normalization
        ↓
Encoder
        ↓
Embedding
        ↓
Distance to healthy baseline
        ↓
Threshold
        ↓
Layer 3 permit / veto
```

The safety gate has two main parts.

### 1. Conservative pre-checks

Some conditions should directly inhibit stimulation before any embedding score is
trusted.


| Check                            | Reason                                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Insufficient healthy calibration | If there is not enough stable baseline data, the safe region cannot be defined reliably                  |
| Calibration / guard windows      | Windows used for baseline fitting or overlap periods should never be treated as runtime permits          |
| Missing or invalid score         | If encoding fails or the score is NaN, the system should fail conservatively                             |
| Encoder / server failure         | If Layer 3 is unavailable, the safe policy is to inhibit or ignore Layer 3 according to deployment rules |


Unlike Layer 2, Layer 3 does not rely on a large table of handcrafted hard rules.
Most of the safety decision comes from departure from the learned healthy
embedding cloud.

### 2. Baseline-deviation score

If the pre-checks pass, the current embedding is compared with the personalized
healthy baseline.


| Scorer                             | How it works                                                   | Why it is used                                                                    |
| ---------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Mahalanobis distance** (default) | Multivariate distance after robust normalization of embeddings | Detects unusual combinations of learned features, not just one isolated dimension |
| **kNN distance** (alternative)     | Mean distance to the nearest healthy calibration embeddings    | Useful when the healthy baseline is irregular or covariance is unstable           |


In simple terms:

```text
current embedding
→ distance from stable baseline
→ score too high?
→ inhibit
```

The threshold is set from stable baseline data, not from dangerous labels.
Operationally, the cutoff is chosen so that most held-out stable baseline
embeddings would still pass. By default, this corresponds to a target of roughly
10% false inhibition on known-stable baseline data.

**Important distinction:** thresholding controls how strict Layer 3 is on stable
baseline ECG. Whether dangerous rhythms fall outside that region is evaluated
separately, primarily through **false permit rate** on labeled dangerous groups
in offline validation.

## Relation to Layer 2 and deployment

Layer 3 is **independent from Layer 2 at scoring time**:

```text
Layer 2: handcrafted features → baseline gate
Layer 3: learned embedding    → baseline distance
```

The useful research question is:

> Among beats Layer 2 already permits, does Layer 3 veto additional abnormal ones?

In deployment, Layer 3 may run on an **external server** while the fast loop
keeps Layer 1 and Layer 2. If the server is slow or unavailable, the system
should not become permissive by default.

For the prospective 1-in-8 stimulation policy, the same logic applies as Layer 2:
safety is built from prior observation beats, and the stimulation decision is based
on recent safe context rather than full analysis of the candidate beat at trigger
time.

## Offline validation (human proxy)

Human PhysioNet labels are used **only for offline validation**, not at
deployment. Beats/windows are mapped to safety groups:


| Group               | Meaning                         | Evaluation expectation            |
| ------------------- | ------------------------------- | --------------------------------- |
| **NORMAL**          | Stable sinus baseline           | Should permit                     |
| **DANGEROUS**       | VT, VFib, flutter spans         | **Must inhibit** — primary metric |
| **BENIGN_ABNORMAL** | Isolated PVC, APC, etc.         | Policy-dependent                  |
| **NOISE**           | Artifact / untrustworthy signal | Must inhibit                      |
| **AF_CONTEXT**      | AF spans                        | Configurable policy               |


Primary metric: **false permit rate on DANGEROUS groups**.  
Secondary metric: **false inhibit rate on NORMAL**.

## Why this matches the project constraints


| Criterion                       | Why Layer 3 matches                                                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Personalized                    | The embedding baseline and threshold are fitted from the same subject/session, not from a universal population model                |
| No labels in real use           | In animal deployment, no beat labels are required. Human ECG labels are used only offline for benchmark validation                  |
| Potentially richer than Layer 2 | A learned encoder may capture subtle waveform structure not covered by handcrafted features                                         |
| Animal-relevant                 | The encoder can be pretrained on unlabeled ECG; session calibration still uses healthy baseline recordings from the animal          |
| Conservative                    | Unfamiliar, poorly calibrated, or unavailable Layer 3 states should inhibit rather than permit stimulation                          |
| Research-oriented               | Phase 1 arms test whether learned representations improve safety beyond Layer 2 without changing the baseline-comparison philosophy |
| Safety-oriented                 | It should be evaluated mainly by false permits on dangerous rhythms, not by accuracy alone                                          |
| Comparable to Layer 2           | A0 uses the same downstream scorer, enabling a direct control comparison                                                            |


## Limitations

Layer 3 depends on the quality of the calibration period. If the baseline ECG is
unstable, noisy, or already unsafe, the embedding baseline may be unreliable. In
that case, the system should fail conservatively.

Layer 3 is also **less interpretable than Layer 2**. If it inhibits, the main
explanation is usually “embedding too far from baseline,” not a named feature
family such as noise ratio or template correlation.

It depends on acceptable windowing and, in true runtime, acceptable R-peak
alignment for beat-sync encoding. Human benchmark results are a proxy only; rat
deployment will require per-session recalibration.

Layer 3 is not a supervised arrhythmia classifier. It detects departure from a
healthy baseline, not VT/VFib labels directly. Some rhythms, such as isolated PVCs
or AF spans, may require explicit policy decisions with supervisors, as in Layer
2.

Phase 1 arms **A**, **A1**, and **B** are implemented in code, but full cluster
comparison across seeds may still be in progress. Arm **C** (multi-lead upper
bound) remains an appendix goal and is not implemented.

Finally, Layer 3 is heavier than Layer 2 in compute and latency. It is therefore
framed as an optional learned veto, not the first real-time safety gate.

---

# Cross-layer summary


| Layer           | Input                 | Core question                               | Deploy role                    |
| --------------- | --------------------- | ------------------------------------------- | ------------------------------ |
| R-peak detector | Filtered ECG stream   | Where is the QRS?                           | Always-on beat proposals       |
| Layer 1         | Detector candidates   | Is timing plausible?                        | Fast deterministic veto        |
| Layer 2         | Features + RR history | Do handcrafted features match baseline?     | **Primary interpretable gate** |
| Layer 3         | Encoder embedding     | Does learned representation match baseline? | Optional research veto         |


**Combined safety rule:**

```text
permit stimulation only if all required layers permit
uncertainty at any layer → inhibit
```

**Primary evaluation metric (Layers 2 and 3):** false permit rate on dangerous
rhythms. Secondary: false inhibit on stable baseline.

---

*Related docs: `reports/ECG_SAFETY_PIPELINE_ARCHITECTURE.md`, `Layer1/ALGORITHM_SUMMARY.md`, `Layer2/ALGORITHM_SUMMARY.md`, `Layer3/ALGORITHM_SUMMARY.md`, `reports/rpeak_detection/README.md`*