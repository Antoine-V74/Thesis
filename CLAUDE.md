# CLAUDE.md

Working notes for AI coding assistants. Last updated May 2026.

Read this before writing code. This file explains the current project intent, folder structure, priorities, safety rules, and what should not be changed unless benchmark results justify it.

---

## 0. Project intent

This project develops an ECG safety pipeline for an ECG-triggered cardiac stimulation system in the context of closed-loop cardiomyoplasty / MyoNeural Actuator (MNA) control.

The software should answer one operational question:

> Is the current ECG/rhythm state safe enough to permit stimulation, or should stimulation be inhibited?

The classifier or anomaly detector must **never command stimulation directly**. It can only permit the rest of the controller to continue, or veto stimulation.

The project is now organized in three layers:

```text
Layer 1 — Fast deterministic R-peak detection and benchmarking
Layer 2 — Handcrafted ECG feature safety gate
Layer 3 — Self-supervised / anomaly-detection research layer
```

The final animal application is expected to use rat ECG. Human PhysioNet datasets are used first for benchmarking, method development, and validation of safety logic.

---

## 1. Current folder structure

Current visible project layout:

```text
ECG Processing/
    data/                         Local ECG datasets and downloaded resources
    Layer1/                       R-peak detector and first benchmark layer
    Layer2/                       Handcrafted feature safety gate
    Layer3/                       SSL / anomaly-detection research layer
    CLAUDE.md                     This project-context file for AI assistants
```

Layer-specific layout:

```text
Layer1/
    pipeline/                     Fast causal detector + RR supervisor
        main_pipeline.py          Main entry point (run_layer1, layer1_r_peaks)
        r_peak_detector.py        Fast causal R-peak detector
        rhythm_supervisor.py      RRSupervisor state machine
        run_benchmark.py          Multi-dataset benchmark script
    tools/                        Optional analysis, plots, and animations
    archive/                      Explored alternatives (adaptive, Pan-Tompkins, etc.)
    _bootstrap.py                 Import path helper for Layer 2/3

Layer2/
    pipeline/                     Core feature extraction and gate logic
        full_features.py          Feature assembler
        gate.py                   Baseline calibration and permit/inhibit gate
    validation/                   Window, beat, and cross-dataset validation
    tools/                        Post-hoc result analysis helpers
    reports/                      Plots and slide generation
    archive/                      Scratch and old experiments
    _bootstrap.py                 Import path helper

Layer3/
    layer3_anomaly.py             Anomaly detector / Deep SVDD-style head
    layer3_augmentations.py       Physiology-aware ECG augmentations
    layer3_encoder.py             Small ECG encoder for SSL/anomaly features
    layer3_pretrain.py            SSL pretraining loop / contrastive pretraining
```

This structure is coherent: Layer 1 contains the fast detector and previous benchmark infrastructure, Layer 2 contains the interpretable feature-based safety supervisor, and Layer 3 contains the heavier ML research components.

Recommended small improvements:

- Add a short `README.md` inside each `Layer1/`, `Layer2/`, and `Layer3/` folder.
- Add a `requirements.txt` or `environment.yml` at the project root.
- Keep generated benchmark outputs out of Git unless they are final thesis results.
- If Python imports become difficult, consider renaming folders to lowercase (`layer1`, `layer2`, `layer3`) later. For now the current naming is readable and acceptable on Windows.

---

## 2. Current priority

The immediate priority is to prepare a clean supervisor discussion and then validate the pipeline layer by layer.

Do **not** jump directly to a large Transformer or diffusion model. The current priority is:

```text
1. Freeze and summarize Layer 1 detector performance.
2. Validate Layer 2 on human ECG first.
3. Use Layer 2 as the likely first rat-deployment safety gate.
4. Treat Layer 3 as a research / server-side extension until proven useful.
```

The key scientific question for the next phase is:

> Can ECG research from signal processing and ML improve the current naive R-peak/rhythm safety logic, and by how much, while preserving real-time safety?

---

## 3. Safety rules

These rules override model performance and convenience.

1. **The software can only inhibit stimulation.** It must never independently command stimulation.
2. **Uncertainty means inhibit.** Missing data, low confidence, server timeout, exceptions, or ambiguous ECG should all default to inhibit.
3. **False permit is the main safety metric.** A false permit means stimulation would be allowed during an unsafe state.
4. **False inhibit is secondary.** It reduces therapy availability but is safer than a false permit.
5. **Fast deterministic logic remains the first line of defense.** ML can refine or veto, not replace the hard safety layer until validated.
6. **No non-causal filtering in real-time deployment.** Offline analysis can use non-causal tools, but runtime code must remain causal or explicitly marked as offline-only.
7. **The external server is optional and advisory.** If the server fails, the system inhibits.

For rhythm classification, the binary safety aggregation is:

```python
inhibit_prob = P(vt) + P(vfib) + P(noise)
permit = inhibit_prob < threshold
```

For anomaly detection, the equivalent rule is:

```python
permit = anomaly_score < calibrated_threshold
```

---

## 4. Layer 1 — Fast detector and RR supervisor

### Purpose

Layer 1 is the low-latency deterministic detector/supervisor.

It should:

- filter ECG causally;
- detect candidate R-peaks quickly;
- apply refractory and blanking rules;
- reject impossible RR intervals;
- inhibit immediately during gross rhythm or signal-quality failures.

Layer 1 is not expected to be perfect. Its goal is to be fast, conservative, and deterministic.

### Current role

Layer 1 is the baseline. It is used to:

- quantify how well the existing detector works;
- identify outlier records / failure modes;
- provide candidate R-peaks to Layer 2;
- provide immediate safety decisions before slower ML analysis returns.

### What not to do

Do not replace the R-peak detector entirely with a deep learning model yet. A learned R-peak detector may be future work, but it adds latency, validation burden, and failure modes.

A more reasonable extension is an ML or feature-based **QRS confirmer** that checks candidate peaks after Layer 1 proposes them.

---

## 5. Layer 2 — Handcrafted ECG feature safety gate

### Purpose

Layer 2 is the main interpretable safety upgrade.

It uses ECG features computed on rolling windows and/or beat-centered windows to decide whether the current signal still looks close to a healthy baseline.

Typical features:

```text
RR features:
    mean RR, RR variability, short/long RR fraction, sudden RR changes

Energy features:
    RMS, peak-to-peak amplitude, peak-to-mean ratio, signal energy

Spectral / wavelet features:
    energy in frequency bands, QRS-band energy, low-frequency drift,
    high-frequency noise, wavelet-band energy

Morphology features:
    QRS template correlation, beat-to-beat similarity, QRS width proxy,
    local waveform consistency around candidate R-peaks
```

### Operating principle

At the beginning of a stable/healthy recording period:

```text
healthy baseline ECG
    -> rolling windows
    -> feature vectors
    -> estimate baseline distribution
```

During runtime:

```text
current ECG window
    -> compute same features
    -> compare to baseline
    -> if too far from baseline, inhibit
```

Possible distance scores:

```text
robust per-feature z-score
Mahalanobis distance
one-class SVM / Isolation Forest / Deep SVDD later if useful
```

Start simple:

```text
Layer 2 v1 = robust z-score + Mahalanobis distance
```

### Validation plan

Validate first on labeled human ECG:

```text
healthy/sinus windows should remain close to baseline
VT/VFib/noise windows should be far from baseline
AF can be treated according to the experiment's safety policy
```

Then adapt to rats using per-rat/session calibration. Do not transfer fixed human thresholds to rats.

### Rat deployment logic

For rats, Layer 2 should be calibrated per animal/session:

```text
record 2-5 min healthy baseline
estimate normal RR / morphology / wavelet / energy ranges
inhibit when current window deviates too much
```

This is likely the most realistic first animal-stage safety model because initial rat data may be healthy and unlabeled.

---

## 6. Layer 3 — SSL / anomaly-detection research layer

### Purpose

Layer 3 explores whether unlabeled ECG can improve safety through learned representations.

This layer is not the first real-time deployment target. It is a research extension that may run offline or on an external server.

### Current idea

Pretrain an ECG encoder with self-supervised learning, then use healthy rat ECG to learn a personalized safe baseline in embedding space.

Conceptual pipeline:

```text
human ECG + unlabeled rat ECG
    -> SSL pretraining / contrastive or reconstruction learning
    -> ECG encoder

healthy rat baseline ECG
    -> embeddings
    -> one-class / distance-based anomaly model

new ECG window
    -> embedding
    -> distance to healthy baseline
    -> abnormal or uncertain = inhibit
```

Possible methods to test:

```text
contrastive learning with physiology-aware augmentations
masked reconstruction / non-contrastive learning
Deep SVDD on embeddings
simple Gaussian or Mahalanobis distance in embedding space
```

### Deployment model

Layer 3 may run on an external Python server:

```text
Pynapse sends latest 4-5 s ECG window + R-peak timestamps
server returns anomaly score / morphology confidence / refined R-peaks
Pynapse updates safety state
server failure = inhibit
```

Layer 3 must not be required for the immediate safety loop.

---

## 7. Real-time architecture

Recommended architecture:

```text
TDT/Synapse/Pynapse fast loop
    -> causal filtering
    -> candidate R-peak detection
    -> RR supervisor
    -> immediate inhibit if suspicious

Pynapse rolling buffer
    -> stores last 4-5 s ECG
    -> stores recent R-peak timestamps
    -> computes Layer 2 features
    -> optionally sends window to Layer 3 server

External server, optional
    -> refined R-peak detection a posteriori
    -> morphology/anomaly analysis
    -> slow safety-state update
```

Decision rule:

```text
permit = Layer1_safe AND Layer2_safe AND Layer3_safe_or_unavailable_policy
```

In practice, if Layer 3 is unavailable or uncertain:

```text
Layer3_safe = False
```

unless explicitly running in a mode where Layer 3 is disabled and the system relies only on Layers 1 and 2.

---

## 8. Data and validation conventions

### Human ECG validation

Use human ECG datasets to validate the method before rat experiments because labels are available.

Evaluate:

```text
false permit rate
false inhibit rate
false inhibit at false permit <= 1%, 0.5%, 0.1%
VT/VFib/noise recall
worst-record and worst-fold behavior
```

Do not report only accuracy or macro-F1.

## 9. Literature mapping

Most relevant papers and how they map to the project:

```text
Lightweight UAD filters:
    Most directly relevant to Layer 2/3 safety filtering.
    Use anomaly detector upstream of classifier to reject OOD/noisy ECG.
    Deep SVDD is especially interesting for lightweight deployment.

NERULA:
    Relevant to Layer 3.
    Single-lead SSL using masked / inverse-masked reconstruction and non-contrastive learning.
    Useful if rat ECG is single-lead and unlabeled.

PhysioCLR:
    Relevant to Layer 3 augmentations and positive-pair selection.
    Use physiology-aware similarity and ECG-preserving augmentations.

VAE-BiLSTM-MHA anomaly paper:
    Useful conceptually for healthy-only anomaly detection.
    Too heavy for first implementation; borrow reconstruction-error thresholding.

3KG:
    Useful conceptually for ECG-specific augmentations.
    Less directly useful unless multi-lead ECG is available.

MetaVA:
    Relevant later if labeled ventricular arrhythmia data is available.
    Useful for personalization / fast adaptation ideas.

DiffECG:
    Interesting future direction for label-efficient personalization.
    Too heavy for immediate implementation.
```

---

## 10. What not to do now

Do not:

- deploy the human-trained supervised classifier directly in rats;
- assume human thresholds transfer to rats;
- start with a large Transformer/diffusion model before Layer 2 is validated;
- remove the deterministic Layer 1 supervisor;
- rely only on R-triggered inference, because VFib/noise may destroy reliable R-peaks;
- use a server response as the only safety signal;
- report only mean performance without worst-record analysis;
- treat healthy-only rat data as if it proves arrhythmia sensitivity.

---

## 11. Suggested near-term tasks

### For the next supervisor meeting

Prepare a clear summary of:

```text
Layer 1: fast detector and RR supervisor
Layer 2: handcrafted ECG feature baseline/anomaly gate
Layer 3: SSL/anomaly research extension
```

Ask supervisor:

```text
1. Can we access healthy rat ECG before the main experiments?
2. Will rat ECG be single-lead or multi-lead?
3. Under which anesthesia / condition will baseline ECG be recorded?
4. Can we obtain any abnormal or induced arrhythmia segments later?
5. Is an external server acceptable for non-critical morphology/anomaly analysis?
6. What false-permit budget should we target for preliminary animal experiments?
7. Can we discuss Ripple technology and how it fits the acquisition/control stack?
```

### Coding priorities

```text
1. Cleanly document Layer 1 results.
2. Run Layer 2 smoke tests on human ECG windows.
3. Implement baseline calibration in Layer 2:
       robust z-score + Mahalanobis distance.
4. Validate Layer 2 against labeled human abnormal/noise windows.
5. Add rat-baseline workflow once healthy rat ECG is available.
6. Keep Layer 3 as optional/offline until Layer 2 results justify more complexity.
```

---

## 12. Verification checklist

Before considering a code change done:

```bash
# Layer 1 benchmark / diagnosis
python Layer1/tools/summarize_benchmark.py

# Layer 2 smoke tests
python Layer2/pipeline/signal_features.py
python Layer2/pipeline/full_features.py
python Layer2/pipeline/main_pipeline.py  # import smoke: from decision import BaselineCalibrator

# Layer 3 smoke tests
python Layer3/layer3_encoder.py
python Layer3/layer3_augmentations.py
python Layer3/layer3_anomaly.py
```

If a script uses generated data or local paths, document the expected input path at the top of the file.

---

## 13. Principle for future changes

Prefer the smallest interpretable change that solves the measured safety problem.

Synthetic tests are only smoke tests. Real-data benchmarks and safety metrics decide the architecture.
