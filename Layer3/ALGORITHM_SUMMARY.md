# Layer 3 Algorithm Summary

Layer 3 is a learned ECG embedding anomaly detector used as an optional
stimulation safety veto. Its job is to embed a recent ECG window, compare the
embedding to a healthy baseline learned at session start, and return
**permit** or **inhibit**. It never commands stimulation directly.

Layer 3 is exploratory relative to Layers 1 and 2. The primary validated safety
path remains the deterministic timing layer plus the handcrafted feature gate.

## High-Level Flow

```text
ECG window
  -> ECGEncoder1D
  -> 128-dim embedding
  -> healthy-baseline distance model (Mahalanobis or kNN)
  -> permit / inhibit
```

Optional upstream step before deployment:

```text
unlabeled ECG windows (8 s @ 125 Hz)
  -> physiology-aware augmentations
  -> SSL pretraining (NT-Xent / VICReg / masked+subject)
  -> shared ECGEncoder1D weights
  -> per-session healthy Mahalanobis/kNN (ZEROSHOT-style)
```

Layer 2 and Layer 3 remain independent; final permit is AND across layers.
Project therapy cadence (1-in-8 observe/stimulate) applies to the whole stack.

Main runtime files:


| File                                       | Role                                        |
| ------------------------------------------ | ------------------------------------------- |
| `pipeline/layer3_encoder.py`               | 1D ResNet encoder + projection head for SSL |
| `pipeline/layer3_augmentations.py`         | Safe contrastive augmentations              |
| `pipeline/layer3_embedding_mahalanobis.py` | Primary Mahalanobis / kNN baseline models   |
| `pipeline/layer3_anomaly.py`               | Optional Deep SVDD ablation head            |
| `tools/pretrain_encoder.py`                | SSL pretraining loop                        |
| `validation/run_beat_validation.py`        | Main beat-sync validation                   |
| `validation/run_window_validation.py`      | Window-level validation / debugging         |


There is no single `main_pipeline.py` wrapper yet. Validation scripts and the
anomaly modules above are the current public entry points.

---

## Recommended Runtime Buffers

Layer 3 does not maintain buffers internally. The caller should provide:


| Buffer                              | Size                                                       | Used for                       |
| ----------------------------------- | ---------------------------------------------------------- | ------------------------------ |
| ECG veto window                     | **8 s** beat-sync @ 125 Hz (primary); 1 s morphology ablation | encoder input               |
| Healthy calibration embeddings      | per-record fit + val windows                               | baseline mean/cov or kNN bank  |
| Optional pretrained encoder weights | one checkpoint file                                        | shared across sessions         |


Preferred beat-sync setting (Layer 3 independent of Layer 2):

```text
window_s = 8.0          # primary: rhythm + morphology in the waveform
# window_s = 1.0        # optional morphology-only ablation
causal_window = True
lookahead_ms = 50 to 150   # stimulation-latency simulation, not zero-latency detection
target_fs = 125
```

Why **8 s** primary (not 1 s)?

- Layer 2 and Layer 3 are **independent** vetoes; Layer 3 must not borrow RR features from Layer 2
- An 8 s strip already contains several RR intervals → rhythm information is in the signal
- Matches SSL pretrain scale (8 s @ 125 Hz); avoids train/eval window mismatch
- 1 s remains useful as a morphology-only ablation
- Dual-scale (1 s + 8 s concat) is a future improvement, not required for the first campaign

Reset calibration if the healthy baseline segment becomes contaminated by PVC,
noise bursts, or missed-beat artifacts.

---

## Step 1: ECG Encoding

Implementation: `pipeline/layer3_encoder.py`

### Architecture

`ECGEncoder1D` is a small 1D residual CNN:

```text
input:  (B, 1, T)     default T = 5 s @ 250 Hz = 1250 samples
stem:   Conv1d 1 -> 32
stages: 4 downsample stages, 2 ResBlocks each
pool:   global average pool
head:   Linear -> 128-dim embedding
```

Design choices:


| Choice            | Reason                                                               |
| ----------------- | -------------------------------------------------------------------- |
| 1D ResNet         | ECG is a 1D waveform; convolutions capture local QRS/T morphology    |
| GroupNorm         | stable with small inference batches and patient-balanced SSL batches |
| ~700K params      | small enough for near-real-time / server-side use                    |
| 128-dim embedding | compact baseline model in embedding space                            |


`ProjectionHead` is used only during SSL pretraining and discarded afterward
(SimCLR / SupCon practice).

### Window preprocessing at runtime

Validation utilities (`validation/layer3_validation_utils.py`) apply:

```text
robust median/MAD normalization per window
optional resampling to target_fs (default 250 Hz)
```

Each window is passed to the encoder as `(1, T)`.

---

## Step 2: SSL Pretraining (Optional)

Implementation: `tools/pretrain_encoder.py`

Pretraining is **not required** for smoke tests, but it is the intended way to
obtain a useful shared encoder before animal/session calibration.

### Positive-pair strategies


| Mode                     | Positive pair                              | Meaning                                |
| ------------------------ | ------------------------------------------ | -------------------------------------- |
| `same_window` (default)  | two augmented views of the same window     | invariance to noise, wander, mild crop |
| `same_record` (ablation) | two different windows from the same record | CLOCS-style record invariance          |


Default is `same_window` because `same_record` can pull healthy and abnormal
beats from the same recording too close together for a safety veto.

### Augmentations

Implementation: `pipeline/layer3_augmentations.py`

Included:

```text
Gaussian noise (SNR-controlled)
baseline wander (low-frequency sinusoid)
random crop + repad
```

Available but disabled by default in the safe veto preset:

```text
time warp
bandpass cutoff perturbation
```

Excluded on purpose:

```text
polarity inversion
aggressive masking
strong amplitude rescaling
random permutation
```

Justification: augmentations must preserve ECG semantics (sinus vs VT/VFib vs
noise) while adding nuisance variability the encoder should ignore.

### Loss and training loop

```text
NT-Xent (SimCLR) on L2-normalized projection vectors
AdamW + cosine LR schedule
linear warmup for first epochs
optional periodic linear-probe accuracy on labeled windows
checkpoints saved as encoder_state_dict
```

Input index: CSV from `tools/build_window_index.py` with columns:

```text
record_id, signal_path, start_idx, n_samples
```

---

## Step 3: Healthy Baseline in Embedding Space

Primary implementation: `pipeline/layer3_embedding_mahalanobis.py`

Optional ablation: `pipeline/layer3_anomaly.py`

Calibration is **per record / per session**, matching the Layer 2 philosophy:
each animal/session gets its own healthy embedding cloud and threshold.

### Main models


| Model                          | Score                                         | When to use                              |
| ------------------------------ | --------------------------------------------- | ---------------------------------------- |
| `EmbeddingMahalanobisBaseline` | sqrt Mahalanobis distance to healthy mean/cov | default transparent baseline             |
| `EmbeddingKNNBaseline`         | mean distance to k nearest healthy embeddings | multimodal or irregular healthy baseline |
| `DeepSVDDHead`                 | L2 distance to learned center c               | optional ablation only                   |


Deep SVDD is **not** the primary runtime framing. Use it only if it wins an
explicit ablation against Mahalanobis / kNN.

### What calibration learns

For Mahalanobis:

```text
mean_                 healthy embedding center
cov_ / precision_     shrunk covariance and inverse
threshold_            distance cutoff from held-out healthy validation scores
```

For kNN:

```text
embeddings_           stored healthy calibration embeddings
threshold_            kNN distance cutoff from held-out healthy validation scores
```

### Train / validation split inside a record

Both window and beat validation use temporal healthy-window ordering:

```text
early healthy windows  -> fit
next healthy windows   -> val (threshold setting)
later windows/beats    -> test
```

Additional safeguards:

```text
calibration_excluded   windows before/at end of calibration segment
guard_excluded         buffer after calibration to avoid window overlap
calibration_no_stim    never counted as runtime permits
```

If healthy calibration is insufficient:

```text
decision = inhibit
status   = insufficient_healthy_calibration
```

### Robust calibration pruning

Both Mahalanobis and kNN support `fit_robust()`:

```text
1. fit provisional baseline on all healthy calibration embeddings
2. score calibration embeddings
3. remove worst outlier_frac
4. refit on kept embeddings
```

Justification: occasional PVC/noise in an otherwise healthy calibration segment
should not distort the personalized baseline cloud.

### Threshold setting

Thresholds are set from **held-out healthy validation scores**, not from abnormal
labels. Two methods are available via `--threshold-method`:

```text
conformal        (default) split-conformal upper-tail threshold with a stated
                 healthy false-inhibit budget alpha (--conformal-alpha, default 0.10).
                 If alpha is infeasible for the healthy calibration size, the
                 record fails safe to inhibit (uncertainty -> inhibit).

healthy_quantile (legacy) threshold = quantile(healthy_val_scores, threshold_quantile),
                 default threshold_quantile = 0.99.
```

Conformal is preferred because it gives a distribution-free, *stated* bound on
the healthy false-inhibit (therapy-loss) rate under exchangeability, instead of a
bare quantile. Both validation scripts write `threshold_coverage.csv` reporting
the achieved healthy false-inhibit rate (with Wilson CIs) against the target, per
record and overall. Phase 1 tables always report both methods regardless of the
main-decision `--threshold-method`.

This is unsupervised calibration suitable for deployment. Any tuning that uses
abnormal-beat labels is offline analysis only.

---

## Step 4: Runtime Decision

### Mahalanobis / kNN path

```text
embedding = encoder(window)
score     = baseline.score(embedding)
permit    = score <= threshold
inhibit   = score > threshold
```

Decision helper in Deep SVDD ablation:

```python
decide_window(score, threshold)
```

Output fields include:

```text
permit
inhibit
anomaly_score
threshold
score_over_threshold_ratio
```

### Conservative rules

```text
missing/NaN score                -> inhibit
calibration / guard rows         -> calibration_no_stim
insufficient healthy calibration -> inhibit
encoder/checkpoint failure       -> inhibit
```

Primary safety metric: **false permit rate** on abnormal beats/windows.

Secondary: false inhibit rate on healthy beats/windows.

---

## Step 5: Deep SVDD Ablation (Optional)

Implementation: `pipeline/layer3_anomaly.py`

Deep SVDD learns a small bias-free MLP head on top of frozen encoder embeddings:

```text
healthy windows -> encoder (frozen) -> head -> minimize ||g(z) - c||^2
test score      = ||g(f(x)) - c||_2
decision        = inhibit if score > threshold
```

Why it exists:

- lightweight nonlinear boundary in embedding space
- useful comparison against linear Mahalanobis / kNN baselines

Why it is not primary:

- harder to interpret than distance-to-healthy-cloud models
- requires separate head training per session
- current project framing treats Mahalanobis/kNN as the main deployable path

Persistence:

```python
save_svdd(path, head, center, threshold, config)
load_svdd(path)
```

---

## Validation, Tools, and Reports

Layer 3 has three support areas beyond `pipeline/`:

- `**tools/**` build indexes, pretrain, smoke-test, compare with Layer 2
- `**validation/**` rerun embedding + anomaly veto on PhysioNet data
- `**reports/**` architecture notes and validation write-ups

### `tools/`


| Script                     | Purpose                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `build_window_index.py`    | Build CSV index of ECG windows + optional `.npy` signal cache |
| `pretrain_encoder.py`      | NT-Xent SSL pretraining on window index                       |
| `smoke_test_layer3.py`     | End-to-end synthetic wiring check                             |
| `compare_layer2_layer3.py` | Compare Layer 2 and Layer 3 beat-sync decisions               |


### `validation/`


| Script                       | Purpose                                            | When to use                     |
| ---------------------------- | -------------------------------------------------- | ------------------------------- |
| `run_beat_validation.py`     | beat-sync oracle or Layer-1-triggered validation   | main therapy-relevant benchmark |
| `run_window_validation.py`   | fixed/overlapping window validation                | representation debugging        |
| `layer3_validation_utils.py` | encoder loading, encoding, metrics, safety helpers | imported by validation scripts  |


Beat-sync modes:


| Mode                    | Trigger source            | Deployable?                       |
| ----------------------- | ------------------------- | --------------------------------- |
| `oracle`                | MIT-BIH beat annotations  | no — upper-bound offline analysis |
| `layer1_adaptive_gated` | accepted Layer 1 triggers | closer to intended runtime path   |


Window modes:


| Setting           | Meaning                                   |
| ----------------- | ----------------------------------------- |
| centered window   | offline only                              |
| `--causal-window` | uses only past + limited post-R lookahead |
| `--lookahead-ms`  | simulates stimulation delay after R peak  |


Primary metrics written by validation:

```text
metrics_overall.csv
metrics_by_record.csv
metrics_by_label.csv
false_permits_detail.csv
per_window.csv / per_beat.csv
thresholds.csv
threshold_coverage.csv
run_config.json
```

Restrict decision metrics to:

```text
split == "test"
decision in {"permit", "inhibit"}
```

### `reports/`

Start at `[reports/README.md](reports/README.md)` for the full map. Active files:


| File                                          | Contents                                                                            |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| `reports/README.md`                           | documentation index (start here if lost)                                            |
| `reports/LAYER3_ARCHITECTURE_RATIONALE.md`    | design rationale and deployment framing                                             |
| `reports/LAYER3_ARCHITECTURE_IMPROVEMENTS.md` | prioritised backlog of candidate architecture changes                               |
| `reports/README_LAYER3_VALIDATION.md`         | validation workflow notes                                                           |
| `reports/LAYER3_SUPERVISOR_DECK.pptx`         | supervisor presentation (regenerate via `tools/generate_layer3_supervisor_deck.py`) |
| `reports/LAYER3_REVIEW_AND_OPEN_ISSUES.md`    | scientific caveats and fixed issues                                                 |
| `reports/ZEROSHOT_CLUSTER_RUN_NOTES.md`       | cluster run commands                                                                |
| `reports/VICREG_A1_IMPLEMENTATION_PLAN.md`    | A1 arm spec (implemented)                                                           |


---

## Relation to Layer 2

Layer 3 is independent from Layer 2 at scoring time:

```text
Layer 2 -> handcrafted features + baseline gate
Layer 3 -> learned embedding + baseline distance
```

For combined analysis, `tools/compare_layer2_layer3.py` evaluates:

```text
combined_permit = Layer2_permit AND Layer3_permit
```

The useful research question is conditional:

```text
Among beats Layer 2 already permits, does Layer 3 veto additional abnormal beats?
```

That does **not** mean the runtime must multiply the layers into one equation.
Each layer produces its own veto decision upstream of the controller.

---

## Safety Interpretation

Layer 3 is conservative:

```text
calibration rows never permit stimulation
uncertainty -> inhibit
human MIT-BIH labels are proxy validation only
animal deployment requires per-session animal calibration
oracle beat mode is an offline upper bound, not runtime code
```

Do not describe Layer 3 as:

```text
a clinical arrhythmia classifier
a replacement for Layer 1 or Layer 2
proof of animal safety by itself
```

Do describe Layer 3 as:

```text
an optional learned veto on healthy-baseline similarity in embedding space
a research extension that may run offline or on an external server
```

---

## Quick Command Reference

Build a window index:

```powershell
python Layer3\tools\build_window_index.py `
  --data-dir data `
  --datasets mitdb `
  --out-csv Results\layer3\index\windows.csv
```

Optional SSL pretraining:

```powershell
python Layer3\tools\pretrain_encoder.py `
  --window-index Results\layer3\index\windows.csv `
  --epochs 100 `
  --checkpoint-dir Results\layer3\checkpoints
```

Beat-sync validation (main benchmark):

```powershell
python Layer3\validation\run_beat_validation.py `
  --data-dir data `
  --datasets mitdb `
  --checkpoint Results\layer3\checkpoints\encoder_last.pt `
  --out-dir Results\layer3\beat_validation `
  --mode oracle `
  --window-s 1.0 `
  --causal-window `
  --lookahead-ms 100
```

Window-level validation:

```powershell
python Layer3\validation\run_window_validation.py `
  --data-dir data `
  --datasets mitdb `
  --window-index Results\layer3\index\windows.csv `
  --checkpoint Results\layer3\checkpoints\encoder_last.pt `
  --out-dir Results\layer3\window_validation `
  --anomaly-model mahalanobis
```

End-to-end smoke test:

```powershell
python Layer3\tools\smoke_test_layer3.py
```

Compare with Layer 2:

```powershell
python Layer3\tools\compare_layer2_layer3.py `
  --layer2-csv Results\layer2_beat_validation\per_beat.csv `
  --layer3-csv Results\layer3\beat_validation\per_beat.csv `
  --out-dir Results\layer3\layer2_vs_layer3
```

Smoke imports:

```powershell
python Layer3\pipeline\layer3_encoder.py
python Layer3\pipeline\layer3_augmentations.py
python Layer3\pipeline\layer3_embedding_mahalanobis.py
python Layer3\pipeline\layer3_anomaly.py
```

