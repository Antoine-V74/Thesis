---
marp: true
theme: default
paginate: true
size: 16:9
---

# Layer 3: Learned ECG Embedding Safety Veto

ECG-triggered stimulation safety detector for closed-loop cardiomyoplasty / muscle-powered cardiac assist

---

# Clinical and Engineering Problem

The controller must decide, for each candidate R peak:

```text
Should stimulation be permitted or inhibited?
```

This is not a clinical arrhythmia classifier.

It is a stimulation safety gate.

---

# System Safety Rule

Final stimulation rule:

```text
Permit stimulation only if:
Layer 1 trigger is reliable
AND Layer 2 permits
AND Layer 3 permits, if enabled.
```

Layer 3 cannot command stimulation.

It can only veto stimulation.

---

# Safety Assumptions

- Uncertainty means inhibit.
- Runtime failure means inhibit.
- Calibration periods never stimulate.
- No oracle annotations are used at runtime.
- Human MIT-BIH validation is proxy validation.
- Animal deployment requires prospective animal ECG validation.

---

# Layer 3 Objective

Layer 3 asks:

```text
Does this stimulation opportunity look sufficiently close
to calibrated healthy ECG?
```

If yes:

```text
Layer 3 permits.
```

If no, uncertain, or failed:

```text
Layer 3 inhibits.
```

---

# Layer 3 Pipeline

```text
Layer 1 trigger / annotated beat
        ↓
Beat-synchronous ECG window
        ↓
1D ResNet encoder
        ↓
128-dimensional embedding
        ↓
Healthy baseline model
        ↓
Mahalanobis anomaly score
        ↓
permit / inhibit
```

---

# Why Beat-Synchronous Validation?

The therapy decision is made around a beat:

```text
one trigger → one safety decision
```

So the main metric should also be beat-synchronous:

```text
one beat / trigger
one ECG window
one anomaly score
one permit/inhibit decision
```

Fixed window validation remains useful for debugging, but it is not the main therapy metric.

---

# Why Not 5 s Windows As Main Metric?

Long windows include many beats.

The current beat morphology can be diluted by surrounding normal beats.

One abnormal event can also contaminate many overlapping windows.

For stimulation gating, a shorter beat-focused window is more relevant.

---

# Current Preferred Window

Current preferred Layer 3 validation:

```text
1 s beat-synchronous window
Layer 1 adaptive trigger mode
causal window
limited post-R lookahead
```

This focuses the encoder on local morphology:

- QRS width
- QRS slope
- polarity
- fusion morphology
- early post-R morphology

---

# Causal Window and Lookahead

Strict real-time simulation:

```text
--causal-window
```

The window uses ECG available up to the trigger.

Because stimulation may occur after the R peak, we also evaluate:

```text
--lookahead-ms 50
--lookahead-ms 100
--lookahead-ms 150
```

This is an offline stimulation-latency simulation, not zero-latency causal detection.

---

# Why Test 50 ms Lookahead?

The application may need to decide shortly after the R peak.

50 ms is more conservative than 150 ms.

It tests:

```text
How much safety can Layer 3 provide
with only early post-R morphology?
```

This is closer to a practical controller timing constraint.

---

# Why A 1D ResNet Encoder?

ECG is a 1D waveform.

Safety-relevant information is local morphology:

- QRS slope
- QRS duration
- notches
- polarity
- post-R shape

1D convolutions are well suited to learning these short waveform patterns.

---

# Why ResNet Specifically?

A ResNet has skip connections:

```text
input → convolution block → + input → next block
```

Skip connections make training more stable.

They let the network learn refinements instead of completely transforming the signal at every layer.

For ECG, this helps preserve useful low-level morphology while adding higher-level shape features.

---

# Why Small 1D ResNet?

This is a safety gate, not a large diagnostic classifier.

We want the model to be:

- lightweight
- stable
- CPU-compatible
- not too data-hungry
- easier to embed later

A small 1D ResNet is more expressive than handcrafted features alone, but much smaller than a transformer.

---

# Why Not Transformer, LSTM, Or MLP?

Transformer:

- heavier
- more data-hungry
- likely overkill for 1 s ECG windows

LSTM:

- better for long sequence modeling
- less direct for local beat morphology

Fully connected network:

- weak inductive bias for local waveform patterns
- more sensitive to sample alignment

---

# Self-Supervised Pretraining

Layer 3 first learns an ECG representation without training a clinical rhythm classifier.

Current method:

```text
SimCLR / InfoNCE-style contrastive SSL
```

Two augmented views of the same ECG window should have similar embeddings.

Different windows in the batch help prevent collapse.

---

# Why Contrastive SSL?

Layer 3 later uses distances in embedding space.

Contrastive SSL is useful because it explicitly shapes this space:

```text
same window under safe perturbations → close
different windows → not collapsed together
```

This makes the embedding more useful for downstream anomaly scoring.

---

# Why Not Supervised Classification?

The runtime task is not:

```text
Which arrhythmia class is this?
```

The runtime task is:

```text
Is this stimulation opportunity safe enough?
```

Supervised MIT-BIH rhythm labels may not transfer to animal ECG.

Supervised classifiers can be studied offline, but they should not define the deployable Layer 3 safety gate.

---

# Why Not Autoencoder As Primary Layer 3?

Autoencoders are valid ECG anomaly baselines, but they have a dangerous failure mode.

They can reconstruct structured abnormal beats too well.

Reconstruction error may focus on:

- amplitude mismatch
- noise
- baseline wander
- timing jitter

rather than safety-relevant morphology.

---

# Autoencoder Failure Mode

A clean PVC or fusion beat may reconstruct well:

```text
abnormal beat → low reconstruction error → false permit
```

A noisy healthy beat may reconstruct poorly:

```text
healthy beat → high reconstruction error → false inhibit
```

For a stimulation safety gate, this is not the desired failure mode.

---

# Why Healthy-Only SSL?

All-window SSL trains on both normal and abnormal ECG.

This can teach the encoder that abnormal beats from the same record are normal variations.

Healthy-only SSL trains only on healthy-labeled windows:

```text
--healthy-only
```

Rationale:

```text
Learn normal ECG morphology first,
then detect deviations from it.
```

---

# Healthy-Only SSL Caveat

Healthy-only MIT-BIH SSL uses human labels for an offline ablation.

It does not prove transfer to animals.

Animal deployment still requires:

- per-session animal calibration
- prospective animal ECG validation
- testing with stimulation artifacts and surgical noise

---

# Embedding-Space Anomaly Detection

After SSL:

```text
ECG window → encoder → 128-dimensional embedding
```

Then Layer 3 fits a healthy baseline:

```text
healthy calibration embeddings → mean + covariance
```

Each test beat receives an anomaly score.

---

# Why Mahalanobis Distance?

Healthy ECG embeddings do not vary equally in every direction.

ECG features are correlated:

- amplitude and slope
- QRS duration and area
- polarity and lead placement
- patient/session morphology

Mahalanobis distance accounts for the shape of the healthy embedding cloud.

---

# Euclidean vs Mahalanobis

Euclidean distance asks:

```text
How far from the healthy mean?
```

Mahalanobis distance asks:

```text
How unusual relative to normal healthy variation?
```

This is better for per-record ECG calibration because normal variation is correlated.

---

# Threshold Quantile

Threshold is set using held-out healthy calibration scores.

Example:

```text
--threshold-quantile 0.90
```

means:

```text
set threshold at the 90th percentile
of healthy calibration anomaly scores
```

Decision:

```text
score <= threshold → permit
score > threshold  → inhibit
```

---

# Threshold Interpretation

`0.90` does not mean 90% accuracy.

It means:

```text
approximately 90% of held-out healthy calibration beats
fall below the threshold
```

Lower quantile:

```text
stricter, fewer false permits, more false inhibits
```

Higher quantile:

```text
more permissive, more healthy availability, more false-permit risk
```

---

# Per-Record Calibration

Per-record calibration better matches deployment:

```text
calibrate on this animal/session
then detect deviations within this animal/session
```

Validation split:

```text
early healthy segment → fit baseline
held-out healthy segment → set threshold
guard region → prevent overlap leakage
later segment → test
```

Abnormal labels are used only for offline metrics.

---

# Metrics

Safety-gate metrics:

```text
healthy permit
false inhibit
abnormal inhibit
false permit
```

Main safety metric:

```text
false permit on abnormal beats
```

Therapy availability metric:

```text
healthy permit
```

AUROC/AUPRC are secondary score-separation metrics.

---

# Current Best Result

Configuration:

```text
healthy-only SSL
1 s beat-synchronous window
Layer 1 adaptive trigger mode
150 ms post-R lookahead
threshold quantile 0.90
per-record calibration
```

Result:

```text
healthy permit:   81.5%
abnormal inhibit: 95.8%
false permit:      4.2%
AUROC:             0.975
AUPRC:             0.951
```

---

# Comparison To All-Window SSL

Previous all-window SSL result:

```text
healthy permit:   80.6%
abnormal inhibit: 94.9%
false permit:      5.1%
```

Healthy-only SSL:

```text
healthy permit:   81.5%
abnormal inhibit: 95.8%
false permit:      4.2%
```

This supports the hypothesis that abnormal-window contamination during SSL can hurt anomaly detection.

---

# Remaining Failure Mode

Fusion beats remain difficult.

Observed label-specific result:

```text
F false permit: approximately 42.7%
```

Physiological explanation:

```text
fusion beat = normal conduction + ventricular ectopic activation
```

So part of the beat can genuinely resemble healthy morphology.

---

# Why Layer 3 Should Be A Veto

Layer 3 is not reliable enough to replace deterministic rules.

It should be combined with Layer 2:

```text
combined permit = Layer 2 permit AND Layer 3 permit
combined inhibit = either layer inhibits
```

This preserves safety:

```text
ML can remove stimulation opportunities,
but cannot command stimulation.
```

---

# Non-Contrastive SSL Opportunity

Non-contrastive SSL methods such as VICReg and Barlow Twins are interesting ablations.

Reason:

```text
they avoid explicit negative pairs
and may produce better-conditioned embeddings
for Mahalanobis scoring
```

This is scientifically promising, but not the primary Layer 3 yet.

---

# Why VICReg Is Interesting

VICReg has three terms:

```text
invariance: two augmented views should match
variance: embeddings should not collapse
covariance: dimensions should not be redundant
```

The covariance term is especially relevant because Mahalanobis scoring depends on covariance estimation.

Hypothesis:

```text
VICReg may improve covariance conditioning and anomaly-score stability.
```

---

# Deep SVDD Ablation

Deep SVDD is another healthy-baseline anomaly detector:

```text
healthy embeddings → compact hypersphere
distance from center → anomaly score
```

It is aligned with the safety-gate idea, but more fragile than Mahalanobis.

Recommended role:

```text
ablation, not primary method
```

---

# Current Interpretation

Layer 3 is best described as:

```text
a learned healthy-morphology anomaly veto
```

not:

```text
a clinical arrhythmia classifier
```

Its purpose is to improve the safety-availability tradeoff when combined with Layer 2.

---

# Limitations

- MIT-BIH is human ECG proxy validation.
- Animal ECG morphology may differ substantially.
- Layer 1 adaptive mode still uses an offline validation filter helper.
- Post-R lookahead must match real stimulation timing.
- Fusion beats remain challenging.
- Prospective animal experiments are required before deployment claims.

---

# Next Experiments

Immediate:

- Evaluate 50 ms lookahead.
- Compare Layer 2 alone vs Layer 3 alone vs combined veto.
- Analyze false permits by label and record.

Optional ablations:

- Deep SVDD healthy-baseline detector.
- VICReg or Barlow Twins non-contrastive SSL.
- Autoencoder baseline for comparison.

---

# Take-Home Message

Layer 3 provides a learned anomaly veto around each candidate stimulation trigger.

The current best design uses:

```text
healthy-only contrastive SSL
1D ResNet ECG encoder
per-record Mahalanobis baseline
healthy calibration threshold
beat-synchronous validation
Layer 2 AND Layer 3 veto logic
```

This is a safety gate, not a diagnostic classifier.

