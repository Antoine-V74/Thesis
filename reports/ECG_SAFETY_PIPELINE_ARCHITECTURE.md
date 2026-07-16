# ECG Safety Pipeline — Layers 1 to 3

**Master Thesis — ECG-triggered cardiac stimulation / MyoNeural Actuator (MNA)**

One-page architecture reference for supervisors and thesis writing.

---

## Operational question

> Is the current ECG/rhythm state safe enough to permit stimulation, or should stimulation be inhibited?

The software is **inhibit-only**: it can permit the controller to continue or **veto** stimulation. It never commands stimulation directly.

---

## Three-layer stack (overview)

```mermaid
flowchart TB
    subgraph Input
        ECG[Raw ECG stream]
    end

    subgraph L1["Layer 1 — Deterministic timing (always on)"]
        F1[Bandpass + notch filter]
        D1[R-peak candidate detector]
        RR[RR supervisor state machine]
        F1 --> D1 --> RR
    end

    subgraph L2["Layer 2 — Handcrafted feature gate (primary deploy path)"]
        FE[Feature extraction: signal / morphology / RR]
        BL2[Healthy baseline calibration]
        G2[Robust z-score + Mahalanobis gate]
        FE --> BL2 --> G2
    end

    subgraph L3["Layer 3 — Learned embedding veto (optional / research)"]
        ENC[ECGEncoder1D → 128-d embedding]
        BL3[Per-session healthy embedding baseline]
        SC[Mahalanobis or kNN distance]
        TH[Conformal threshold]
        ENC --> BL3 --> SC --> TH
    end

    ECG --> F1
    RR -->|accepted trigger beats| FE
    RR -->|beat-sync window| ENC

    RR --> V1{L1 safe?}
    G2 --> V2{L2 safe?}
    TH --> V3{L3 safe or disabled?}

    V1 -->|no| INHIBIT[INHIBIT stimulation]
    V2 -->|no| INHIBIT
    V3 -->|no| INHIBIT
    V1 -->|yes| V2
    V2 -->|yes| V3
    V3 -->|yes| PERMIT[PERMIT stimulation]

    UNK[Uncertainty / failure / missing data] --> INHIBIT
```



**Final rule:**

```text
permit = Layer1_safe AND Layer2_safe AND (Layer3_safe OR Layer3_disabled)
```

Any uncertainty → **inhibit** (fail-safe).

---

## Layer roles at a glance


| Layer       | Role                                          | Input                | Output                                 | Deployment                    |
| ----------- | --------------------------------------------- | -------------------- | -------------------------------------- | ----------------------------- |
| **Layer 1** | Fast R-peak detection + RR/rhythm supervision | Raw ECG              | Accepted peaks, trigger-eligible beats | Always on, embedded           |
| **Layer 2** | Interpretable morphology/rhythm feature gate  | ECG window + R-peaks | permit / inhibit                       | Primary rat path              |
| **Layer 3** | Learned embedding anomaly veto                | Beat-sync ECG window | permit / inhibit                       | Optional server-side research |


---

## Layer 1 — R-peak detection and rhythm supervision

Layer 1 is the fast deterministic timing layer. It does **not** command stimulation;
it certifies which R-peaks are timing-reliable enough to pass to Layers 2 and 3.

Implementation: `Layer1/pipeline/main_pipeline.py` → `run_layer1()`

### Main idea

Layer 1 = **causal R-peak detection** + **RR supervisor with EMA rhythm reference**.

- Track successive R-peaks and measure **RR intervals** (time between consecutive peaks).
- Maintain an **exponential moving average (EMA)** of RR as the live rhythm reference.
- Accept or reject each new candidate peak if it arrives **too early or too late**
relative to that reference (adaptive acceptance band).
- Candidates come from a separate adaptive detector; the supervisor does not trust
raw detector output without RR validation.

### How it works (two stages)

```mermaid
flowchart TB
    subgraph stage1 [Stage 1: R-peak detection]
        RAW[Raw ECG] --> FILT["Bandpass 5-20 Hz + notch"]
        FILT --> DET["Adaptive amplitude + slope detector"]
        DET --> CAND[Candidate R-peaks]
    end

    subgraph stage2 [Stage 2: RR supervisor]
        CAND --> FSM["State machine: WAIT_ARM → CALIBRATING → RUNNING ↔ RECOVERY"]
        FSM --> EMA["RR reference: median then EMA"]
        EMA --> BAND["Adaptive acceptance band around EMA"]
        BAND --> ACC[Accepted peaks]
        BAND --> TRG[Trigger-eligible beats in RUNNING]
    end
```



#### Stage 1 — Fast causal R-peak detector

File: `Layer1/pipeline/r_peak_detector.py`

- **Causal only** at runtime: past/current samples only (no `filtfilt`, no oracle).
- **5–20 Hz bandpass** + optional 50/60 Hz notch → emphasizes QRS energy.
- **Adaptive thresholds**: amplitude and slope envelopes track noise vs signal online.
- **Peak tracking**: enter on rising edge above thresholds; emit peak after confirmed
descent (real-time confirmation delay is part of the latency budget).
- **Detector refractory** (~90 ms): suppress double detections on one QRS complex.
- Output: **candidate** peak timestamps — not yet trusted for stimulation.

#### Stage 2 — RR supervisor

File: `Layer1/pipeline/rhythm_supervisor.py`

**Four modes:**


| Mode          | What happens                                                           |
| ------------- | ---------------------------------------------------------------------- |
| `WAIT_ARM`    | Ignore first ~2 s (filter and detector warm-up)                        |
| `CALIBRATING` | Collect ~10 RR intervals; median → EMA warm-up; **no triggers**        |
| `RUNNING`     | Normal operation; accepted beats → `trigger_samples` for downstream    |
| `RECOVERY`    | After suspicious timing; wider band; no triggers until rhythm rebuilds |


**RUNNING acceptance checks (in order):**

1. Not inside **blanking/refractory** (post-beat protection; longer after stimulation).
2. **Hard RR limits**: 250–2500 ms (reject physiologically impossible rates).
3. **Adaptive band** around EMA: width from recent stable RR variability (robust MAD),
  clipped to roughly 10–40% of the current reference.

If RR is out of band → reject candidate. Repeated instability → **RECOVERY**
(conservative: no stimulation triggers until recalibrated via CALIBRATING → RUNNING).

```mermaid
stateDiagram-v2
    direction LR
    WAIT_ARM --> CALIBRATING: start_time_reached
    CALIBRATING --> RUNNING: enough_stable_RRs
    RUNNING --> RUNNING: accept_in_band_beat
    RUNNING --> RECOVERY: suspicious_timing
    RECOVERY --> CALIBRATING: plausible_RRs_rebuilt
```



Key defaults: `rr_ema_alpha=0.20`, `calibration_rr_count=10`, `rr_min_ms=250`,
`rr_max_ms=2500`.

### How it meets expectations

- **Causal, deterministic, conservative, very low latency** — suitable for embedded
deployment (TDT, Ripple, or custom hardware).
- **Hard physiological rules** reject impossible heart rates before any soft band logic.
- **Post-beat blanking** prevents double triggers on the same beat or immediately
after stimulation artifacts.
- **No ML, no server** — first line of defense always available even if Layer 2/3 fail.

### Limitations (motivates Layer 2 and 3)

- Effective at detecting **suspicious RR timing** or impossible beat intervals.
- Cannot judge whether the **ECG waveform shape** (morphology) is abnormal.
- Cannot assess **signal quality** (noise, lead-off, saturation) beyond gross timing failure.
- A perfectly timed PVC or fusion beat may pass Layer 1 → Layer 2/3 handle morphology.

### Outputs passed downstream

From `run_layer1()`:


| Output                 | Meaning                                               |
| ---------------------- | ----------------------------------------------------- |
| `candidate_samples`    | Raw detector peaks (before supervisor)                |
| `accepted_samples`     | Beats passing RR supervisor                           |
| `trigger_samples`      | **RUNNING-mode** beats eligible for stimulation logic |
| `supervisor.decisions` | Full decision log for debugging and plots             |


Filtered ECG and accepted R-peak timestamps are passed to Layer 2 (feature extraction)
and Layer 3 (beat-sync windows).

---

## Layer 2 — Handcrafted feature safety gate

**Purpose:** Compare current ECG features to a **healthy baseline** learned at session start. Primary interpretable safety upgrade for animal deployment.

```mermaid
flowchart TB
    WIN[ECG window + R-peak history] --> FEAT[Feature extraction]
    FEAT --> SIG[signal__: wavelets, entropy, SQI, energy]
    FEAT --> MOR[morph__: QRS shape around trigger]
    FEAT --> RR[rr__: interval stats / HRV]

    CAL[Healthy calibration windows] --> PRUNE[Optional outlier prune]
    PRUNE --> BASE[Baseline distribution]
    BASE --> GATE[Hard rules + primary distance]
    SIG --> GATE
    MOR --> GATE
    RR --> GATE

    VAL[Healthy validation split] --> THR[Conformal threshold α=10%]
    GATE --> SCORE[Mahalanobis or kNN distance]
    SCORE --> THR
    THR --> DEC{score ≤ threshold?}
    DEC -->|yes| P[permit]
    DEC -->|no| I[inhibit]
```



**Calibration defaults (aligned with Layer 3):**


| Setting                    | Default       | Role                                                |
| -------------------------- | ------------- | --------------------------------------------------- |
| `threshold_method`         | `conformal`   | Held-out healthy validation scores                  |
| `conformal_alpha`          | 0.10          | Target ~10% healthy false inhibit                   |
| `calibration_outlier_frac` | 0.0           | Optional worst-window prune before refit            |
| `anomaly_model`            | `mahalanobis` | Primary distance; `knn` alternative                 |
| `guard_s`                  | 5.0 s         | Post-calibration overlap excluded from test metrics |


**Validation policy groups** (shared with Layer 3 via `label_grouping.py`):


| Group             | Examples     | Safety expectation       |
| ----------------- | ------------ | ------------------------ |
| `NORMAL`          | NSR baseline | permit                   |
| `DANGEROUS`       | VT, VFib     | inhibit (primary metric) |
| `BENIGN_ABNORMAL` | isolated PVC | policy-dependent         |
| `AF_CONTEXT`      | AF spans     | configurable             |
| `NOISE`           | artifact     | inhibit                  |


**Feature groups:**


| Prefix     | Needs R-peaks? | Safety meaning                |
| ---------- | -------------- | ----------------------------- |
| `signal__` | No             | noise, energy, signal quality |
| `morph__`  | Yes            | PVC, aberrant beat shape      |
| `rr__`     | Yes (≥2 peaks) | rhythm deterioration          |


**Prospective 1-in-8 policy (rat stimulation strategy):**

- Beats 1–7: observe unstimulated beats, score safety
- Beat 8: stimulate only if the observation block is safe enough
- The 8th beat is not re-analysed before firing — decision is already stored

**Main entry:** `Layer2/pipeline/main_pipeline.py` → `extract_layer2_features()`, `calibrate_layer2()`, `decide_layer2()`

---

## Layer 3 — Learned embedding anomaly veto

**Purpose:** Research extension. Pretrain a small 1D ResNet encoder with self-supervised learning, then score beat morphology as distance from a **personalized healthy embedding baseline**. Optional veto — not required for the immediate safety loop.

```mermaid
flowchart TB
    subgraph Pretrain["Offline pretraining (optional)"]
        U[Unlabeled ECG windows] --> AUG[Physiology-aware augmentations]
        AUG --> SSL[SSL objective: NT-Xent / VICReg / MAE+contrastive]
        SSL --> ENCW[ECGEncoder1D weights]
    end

    subgraph Runtime["Runtime scoring"]
        BWIN[Beat-sync ECG window ~1 s] --> ENC[ECGEncoder1D]
        ENC --> EMB[128-d embedding]
        HCAL[Healthy calibration embeddings] --> MD[Mahalanobis or kNN]
        EMB --> MD
        MD --> THR[Conformal threshold α=10%]
        THR --> D3{permit / inhibit}
    end

    ENCW -.->|frozen weights| ENC
```



**Phase 1 encoder comparison arms** (fixed downstream scorer isolates representation quality):


| Arm | Method                                        |
| --- | --------------------------------------------- |
| A0  | Layer 2 handcrafted features (control)        |
| A   | NT-Xent contrastive SSL                       |
| A1  | VICReg non-contrastive SSL                    |
| B   | MAE + subject-contrastive (ZEROSHOT-inspired) |
| C   | Multi-lead upper bound (appendix)             |


**Primary safety metric:** false-permit rate on **DANGEROUS** beats (VT/VFib/noise), with Wilson confidence intervals. Secondary: healthy-permit (therapy availability).

**Main entry:** `Layer3/validation/run_beat_validation.py`

---

## Real-time architecture (deployment view)

```mermaid
sequenceDiagram
    participant HW as Acquisition (TDT/Synapse/Pynapse)
    participant L1 as Layer 1 (embedded)
    participant L2 as Layer 2 (Pynapse)
    participant L3 as Layer 3 server (optional)
    participant STIM as Stimulation controller

    HW->>L1: ECG samples (streaming)
    L1->>L1: filter + detect + RR supervise
    L1->>L2: accepted R-peak + rolling ECG buffer
    L2->>L2: extract features vs baseline
    L2->>STIM: permit / inhibit

    opt Layer 3 enabled
        L1->>L3: beat window + timestamps
        L3->>L3: encode + Mahalanobis score
        L3->>STIM: permit / inhibit (veto)
    end

    Note over L3: Server timeout or failure → inhibit
    Note over STIM: Stimulate only if all enabled layers permit
```



**Buffers the caller must maintain:**


| Buffer              | Typical size            | Layer  |
| ------------------- | ----------------------- | ------ |
| ECG morphology      | ~5 s + post-R lookahead | L2     |
| RR history          | ~30 s accepted peaks    | L2     |
| Beat-sync window    | ~1 s @ 125 Hz           | L3     |
| Healthy calibration | session start, 2–5 min  | L2, L3 |


---

## Safety rules (all layers)

1. Software can only **inhibit** — never command stimulation autonomously
2. **Uncertainty → inhibit** (missing data, NaN score, failed calibration, server timeout)
3. **False permit** is the primary safety error; false inhibit is secondary
4. Layer 1 deterministic logic remains the first line of defense
5. Non-causal filtering is offline-only; runtime must be causal or explicitly labelled
6. Human PhysioNet validation is **proxy** validation; rat deployment needs per-session calibration

---

## Validation status


| Layer   | Human ECG validation               | Rat deployment                    |
| ------- | ---------------------------------- | --------------------------------- |
| Layer 1 | Multi-dataset R-peak benchmark     | Causal mode ready                 |
| Layer 2 | Beat/window validation on MIT-BIH  | Primary planned deploy path       |
| Layer 3 | Beat-sync Phase 1 arms (A0/A/A1/B) | Research; session calibration TBD |


---

## Where to read more


| Topic                    | Document                                          |
| ------------------------ | ------------------------------------------------- |
| Layer 1 algorithm        | `Layer1/ALGORITHM_SUMMARY.md`                     |
| Layer 2 algorithm        | `Layer2/ALGORITHM_SUMMARY.md`                     |
| Layer 3 algorithm        | `Layer3/ALGORITHM_SUMMARY.md`                     |
| Layer 3 design rationale | `Layer3/reports/LAYER3_ARCHITECTURE_RATIONALE.md` |
| Layer 3 doc index        | `Layer3/reports/README.md`                        |
| Supervisor slides        | `Layer3/reports/LAYER3_SUPERVISOR_DECK.pptx`      |
| Project safety rules     | `CLAUDE.md`                                       |


---

*Generated for thesis documentation. Mermaid diagrams render in GitHub, VS Code, Obsidian, and Marp.*