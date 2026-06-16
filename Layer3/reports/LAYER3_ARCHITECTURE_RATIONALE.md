# Layer 3 Architecture and Design Rationale

## Purpose

Layer 3 is a learned ECG embedding anomaly detector used as an additional stimulation safety veto.

It is not a clinical arrhythmia classifier. It does not diagnose rhythms and it must never command stimulation by itself. Its only role is to decide whether a candidate stimulation opportunity looks sufficiently close to the calibrated healthy ECG baseline.

Final therapy rule:

```text
Permit stimulation only if:
Layer 1 trigger is reliable
AND Layer 2 permits
AND Layer 3 permits, if Layer 3 is enabled.
```

Uncertainty, calibration periods, model failure, missing data, or runtime failure must inhibit stimulation.

Human MIT-BIH validation is proxy validation only. Animal deployment requires per-session animal ECG calibration and prospective animal validation.

## High-Level Pipeline

```text
ECG record
  ↓
Layer 1 trigger or oracle beat annotation
  ↓
Beat-synchronous ECG window extraction
  ↓
1D ResNet ECG encoder
  ↓
128-dimensional embedding
  ↓
Per-record healthy baseline model
  ↓
Mahalanobis anomaly score
  ↓
Threshold from held-out healthy calibration data
  ↓
Layer 3 permit / inhibit
  ↓
Optional Layer 2 AND Layer 3 combined veto
```

## Beat-Synchronous Evaluation

The main Layer 3 evaluation is beat-synchronous validation. For each accepted trigger or annotated beat, the pipeline produces exactly one Layer 3 decision:

```text
one beat / trigger -> one ECG window -> one anomaly score -> one permit/inhibit decision
```

This matches the therapy problem better than fixed sliding-window validation because the real controller decides whether stimulation should be allowed after a candidate R peak.

Window-level validation is still useful for debugging learned representations, but it is not the main therapy metric. A single abnormal beat can contaminate many overlapping fixed windows, making window-level validation pessimistic and less directly interpretable as stimulation availability.

## Trigger Modes

Layer 3 beat-synchronous validation supports two conceptual trigger modes.

### Oracle Mode

```text
--mode oracle
```

Oracle mode uses MIT-BIH annotated beat locations. This is useful as an offline upper-bound analysis because it removes Layer 1 timing and detection errors.

It is not runtime deployable. A real controller does not have access to oracle annotations.

### Layer 1 Adaptive Gated Mode

```text
--mode layer1_adaptive_gated
```

This mode scores accepted Layer 1 triggers, making the validation closer to the intended closed-loop architecture.

Oracle annotations are used only after detection to assign offline metric labels. They are not used to make runtime decisions.

Important caveat: the current Layer 1 helper uses the existing offline zero-phase validation filter. Therefore, this is a runtime-style trigger comparison, but not yet a final embedded deployment trace.

## ECG Window Choice

Layer 3 originally supported longer windows such as 5 s. These are useful for research and debugging, but they performed poorly for the stimulation-gate task because the morphology of the current beat can be diluted by surrounding beats.

The current preferred beat-synchronous configuration uses short windows:

```text
--window-s 1
```

The rationale is that stimulation safety depends strongly on local beat morphology:

```text
QRS width
QRS slope
polarity
fusion morphology
post-R morphology
early repolarization behavior
```

A 1 s window focuses the embedding on the beat being evaluated rather than long patient or record context.

## Causality and Lookahead

The stricter real-time simulation uses:

```text
--causal-window
```

This makes the window use only ECG available up to the trigger time.

However, the intended cardiomyoplasty stimulation may occur after the R peak, for example around 100-150 ms after R. Therefore, limited post-R lookahead is scientifically relevant:

```text
--lookahead-ms 50
--lookahead-ms 100
--lookahead-ms 150
```

This should be interpreted as an offline stimulation-latency simulation:

```text
R peak detected
wait a short physiologically meaningful interval
evaluate safety
permit or inhibit stimulation before the intended stimulation time
```

It must not be described as zero-latency causal detection. Centered windows and post-R lookahead are non-causal unless the final device timing explicitly allows that delay.

## Encoder Architecture

Layer 3 uses a lightweight 1D residual convolutional encoder:

```text
ECG waveform -> 1D ResNet encoder -> 128-dimensional embedding
```

The 1D ResNet was selected because ECG is a one-dimensional waveform and the safety-relevant information is local morphology.

Convolutional filters can learn local waveform patterns such as:

```text
sharp QRS slopes
wide ventricular deflections
inverted polarity
notches
post-QRS morphology
baseline/noise patterns
```

Residual connections stabilize training and help preserve low-level morphology while allowing deeper shape features to be learned.

The model is intentionally small because Layer 3 is a safety gate, not a large diagnostic classifier. The architecture should remain computationally modest and compatible with future embedded or near-real-time use.

Alternatives were less appropriate for the current goal:

```text
Transformers: heavier and more data-hungry than needed for short ECG windows.
LSTMs: better suited to longer sequence modeling than local beat morphology.
Fully connected networks: weaker inductive bias for local waveform patterns.
```

## Self-Supervised Pretraining

The encoder is pretrained using self-supervised contrastive learning. The model sees augmented views of ECG windows and learns embeddings that are stable under realistic waveform perturbations.

Two pretraining strategies are useful:

```text
All-window SSL
Healthy-only SSL
```

All-window SSL learns from all windows, including abnormal beats. This can be useful for generic ECG representation learning, but it may be harmful for anomaly detection because the model can learn to place normal and abnormal beats from the same record close together.

Healthy-only SSL trains the encoder only on healthy-labeled MIT-BIH windows:

```text
--healthy-only
```

The rationale is that an anomaly detector should learn the structure of healthy ECG morphology rather than normalize abnormal morphology.

This is an offline human ECG ablation because MIT-BIH labels are used to select healthy pretraining windows. It does not prove animal generalization. Animal deployment still requires animal-specific baseline calibration and prospective validation.

## Healthy Baseline and Mahalanobis Scoring

After pretraining, each beat window is mapped to a 128-dimensional embedding. Layer 3 then fits a healthy baseline using calibration embeddings only.

For each record/session:

```text
healthy calibration embeddings -> mean vector + covariance matrix
```

Each test embedding receives a Mahalanobis anomaly score:

```text
score = distance from healthy baseline, accounting for healthy covariance
```

Mahalanobis distance is preferred over simple Euclidean distance because healthy ECG variation is correlated. For example:

```text
QRS amplitude and slope vary together.
QRS duration and waveform area vary together.
Lead placement affects amplitude, polarity, and morphology together.
Patient/session morphology shifts multiple embedding dimensions together.
```

Euclidean distance treats all directions equally and effectively assumes a circular healthy cloud. Mahalanobis distance accounts for the shape of the healthy cloud. Variation along normal healthy directions is penalized less than variation in unusual directions.

## Threshold Selection

The Layer 3 threshold is selected using held-out healthy calibration scores.

For example:

```text
--threshold-quantile 0.90
```

means:

```text
Set the threshold at the 90th percentile of healthy calibration anomaly scores.
```

Decision rule:

```text
anomaly_score <= threshold -> permit
anomaly_score > threshold  -> inhibit
```

This does not mean 90% accuracy. It means the safety gate is calibrated so that approximately 90% of held-out healthy calibration beats would be permitted and the most unusual 10% of healthy calibration beats would be inhibited.

Lower threshold quantiles are stricter:

```text
0.90: stricter, lower false permit risk, lower healthy permit.
0.95: more permissive, higher healthy permit, higher false permit risk.
0.99: very permissive, generally less appropriate for safety-first gating.
```

Because the system follows the rule uncertainty equals inhibit, `0.90` is a reasonable safety-first threshold to evaluate.

## Calibration and Test Separation

Strict separation is required because this is a safety detector.

Per-record calibration is preferred:

```text
early healthy segment -> fit healthy baseline
held-out healthy calibration segment -> set threshold
guard region -> avoid overlapping-window leakage
later segment -> test
```

Abnormal labels are not used to fit the deployable unsupervised baseline or threshold. They are used only for offline validation metrics.

The guard region prevents near-identical overlapping ECG windows from appearing in both calibration and test splits.

Calibration periods never trigger stimulation. They are reported as calibration/no-stimulation periods.

## Metrics

Layer 3 reports stimulation-gate metrics rather than clinical classifier metrics:

```text
healthy permit = healthy beats/windows permitted / total healthy beats/windows
false inhibit = healthy beats/windows inhibited / total healthy beats/windows
abnormal inhibit = abnormal beats/windows inhibited / total abnormal beats/windows
false permit = abnormal beats/windows permitted / total abnormal beats/windows
```

For this application, false permit on abnormal beats is the primary safety concern. Healthy permit quantifies therapy availability.

AUROC and AUPRC are useful secondary metrics for understanding score separation, but the final decision is a thresholded permit/inhibit gate.

## Combined Layer 2 and Layer 3 Veto

Layer 3 should be evaluated both standalone and as an additional veto on Layer 2.

The combined rule is:

```text
combined permit = Layer 2 permit AND Layer 3 permit
combined inhibit = Layer 2 inhibits OR Layer 3 inhibits OR either layer fails
```

This preserves the safety architecture:

```text
No ML layer commands stimulation by itself.
Layer 3 can only remove stimulation opportunities.
```

Layer 2 and Layer 3 are complementary. Layer 2 provides interpretable RR and morphology features, while Layer 3 provides learned morphology anomaly sensitivity.

Fusion beats remain difficult for Layer 3 because they contain both normal and ventricular activation components. This supports using Layer 3 as a veto alongside Layer 2 rather than as a standalone controller.

## Current Best Configuration Under Evaluation

The strongest current Layer 3 configuration is:

```text
healthy-only SSL pretraining
1 s beat-synchronous window
Layer 1 adaptive gated trigger mode
causal window with limited post-R lookahead
per-record healthy Mahalanobis baseline
threshold quantile 0.90
```

Example command:

```bash
python Layer3/layer3_validate_beat_sync.py \
  --data-dir data \
  --datasets mitdb \
  --checkpoint Results/layer3_pretrain_healthy_only_100ep_cuda_w1s/encoder_last.pt \
  --out-dir Results/layer3_validation/beat_sync_layer1_adaptive_healthy_ssl_q090_w1s_look50 \
  --mode layer1_adaptive_gated \
  --causal-window \
  --lookahead-ms 50 \
  --window-s 1 \
  --threshold-quantile 0.90 \
  --per-record-calibration \
  --device cuda \
  --seed 0
```

The 50 ms lookahead condition is especially relevant for the intended application because it is closer to a practical delay between R-peak detection and stimulation decision.

## Scientific Interpretation

Layer 3 should be interpreted as:

```text
a learned healthy-morphology anomaly veto for stimulation safety
```

not as:

```text
a clinical rhythm classifier
```

The central question is:

```text
Does the current stimulation opportunity resemble calibrated healthy ECG closely enough to allow stimulation?
```

If yes, Layer 3 permits. If no, uncertain, or failed, Layer 3 inhibits.

