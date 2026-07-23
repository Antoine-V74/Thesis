# Layer 3 — Complete status (Arm A: NT-Xent)

**Purpose:** One place that explains **Arm A** in the same depth as the A0
status doc. A is the first SSL encoder arm: SimCLR-style contrastive pretraining
(NT-Xent), then the frozen personalized Mahalanobis/kNN veto.

**Date:** July 2026  
**Sibling docs:**
- A0 control → `LAYER3_COMPLETE_STATUS_A0_FIRST.md`
- **L2 vs A0 vs A cluster compare** → `LAYER3_L2_A0_A_COMPARE.md` + `cluster_jobs/`
- A1 (VICReg) → `VICREG_A1_IMPLEMENTATION_PLAN.md` (+ §10 of A0 status)
- B → `LAYER3_COMPLETE_STATUS_B.md`
- B1 → `LAYER3_COMPLETE_STATUS_B1.md` (+ design freeze `LAYER3_ARM_B_B1_SPEC.md`)
- C → `LAYER3_COMPLETE_STATUS_C.md` (+ design freeze `LAYER3_ARM_C_SUPERVISED_SPEC.md`)
- Commands → `ZEROSHOT_CLUSTER_RUN_NOTES.md` (§2 train A, §0b / §5 eval)
- Metrics guide → `LAYER3_PHASE1_OUTPUT_METRICS.md`

---

## 0. One-sentence framing for A

> Arm A learns an ECG embedding by pulling two augmented views of the **same
> window** together and pushing other windows in the batch apart (NT-Xent).
> Safety still comes only from a **per-record healthy baseline** in that
> embedding space — not from contrastive labels or disease supervision.

```text
Question answered by A:
  "Under the frozen personalized anomaly protocol, do NT-Xent SSL embeddings
   beat A0 (Layer 2 handcrafted features) on false-permit / availability?"
```

---

## 1. What Arm A is

| Item | Definition |
| --- | --- |
| **Name** | A — NT-Xent / SimCLR-family contrastive SSL |
| **CLI** | `--ssl-objective ntxent` (default in `pretrain_encoder.py`) |
| **Papers** | Chen et al. 2020 (SimCLR); CLOCS-inspired pairing options; ECG augs in the spirit of 3KG / PhysioCLR |
| **Input** | Unlabeled 8 s @ 125 Hz windows from the window index |
| **Positive pair (default)** | Two **safe augmentations** of the **same window** (`--positive-mode same_window`) |
| **Encoder** | `ECGEncoder1D` → 128-d embedding |
| **Training-only head** | `ProjectionHead` → projection space for NT-Xent; **discarded** after pretrain |
| **Runtime** | Frozen encoder → optional L2 + PCA → Mahalanobis/kNN → conformal threshold |
| **What it is not** | Not a VT/VF classifier; not full CLOCS (CMSC/CMLC/CMPC); not Arm B (no masking) |

**Place in the ladder:**

```text
A0 = handcrafted features + same scorer     (control floor)
A  = NT-Xent contrastive SSL                (this document)
A1 = VICReg non-contrastive SSL             (same views, no negatives)
B  = masked recon + same-window consistency (different hypothesis)
B1 = masked + subject contrastive           (ablation)
```

---

## 2. How A is trained (code path)

Entry: `Layer3/tools/pretrain_encoder.py`.

```text
1. Load window index CSV
     (record_id, signal_path, start_idx, n_samples, …)
2. Optional filters:
     --exclude-records-csv  → hold gold eval records out of SSL
     --healthy-only         → only is_healthy_window=True (fails if column missing)
3. ContrastiveECGDataset:
     load cached .npy slice
     robust median/MAD normalize (same as Phase 1 eval)
     if ntxent: apply ECGAugmentor twice → views (a, b)
4. Model:
     EncoderWithProjection(ECGEncoder1D + ProjectionHead)
5. Loss:
     NT-Xent on L2-normalized projections, temperature τ (default 0.1)
6. Save:
     encoder_last.pt / checkpoints (encoder weights for eval)
     pretrain_records.json (which records were seen)
```

### 2.1 NT-Xent in one picture

```text
window x
  → aug₁(x) → encoder → proj → z₁
  → aug₂(x) → encoder → proj → z₂

Loss: pull z₁ and z₂ together;
      push z₁ away from all other projections in the batch
      (and same for z₂).
```

Temperature `τ` softens/hardens the softmax over similarities.

### 2.2 Positive-pair modes

| Mode | Flag | Meaning | Use |
| --- | --- | --- | --- |
| **same_window** (default) | `--positive-mode same_window` | Two augs of one window | **Primary for A** — safer for anomaly detection |
| **same_record** | `--positive-mode same_record` | Two windows from same record | CLOCS-style ablation — can pull healthy + abnormal from one record together |

Do **not** use `same_record` as the primary A claim unless you label it as an ablation.

### 2.3 Augmentations (physiology-aware, “safe” preset)

From `layer3_augmentations.py` / `AugmentConfig` (defaults):

| Aug | Default p | Role |
| --- | --- | --- |
| Gaussian noise | 0.7 | Sensor / muscle noise |
| Baseline wander | 0.5 | Slow drift (needs correct `--augment-fs`) |
| Random crop + repad | 0.1 | Mild temporal jitter |
| Time warp | 0.0 (off) | Aggressive; off by default |
| Bandpass jitter | 0.0 (off) | Off by default (non-causal filtfilt if enabled) |

**Critical:** pass `--augment-fs 125` so wander/bandpass match the cached window rate
(protocol is 125 Hz, not 250).

Excluded on purpose: polarity inversion, aggressive masking, strong amplitude
chaos — they can destroy the sinus-vs-danger semantics you care about later.

### 2.4 Normalization

Per-window **robust median/MAD** in the dataset — aligned with
`layer3_validation_utils.robust_normalize_window` at eval time. Do not silently
switch back to mean/std z-score on one side only.

---

## 3. How A is evaluated (Phase 1)

Same script as A0, but `layer3` arm uses **A’s checkpoint embeddings**.

```text
1. Build beat table (oracle or layer1_adaptive_gated)
2. Load 8 s beat-sync windows @ 125 Hz → encode with A checkpoint
3. Per record:
     early healthy → fit / val (+ guard)
     fit Mahalanobis/kNN on embeddings (optional L2 + PCA)
     conformal α=0.10 on healthy val scores
4. Also compute A0 features if --phase1-arms a0,layer3
5. Write phase1_* CSVs (bootstrap CI, strata, CAV, …)
```

**Always** use `--no-random-fallback` and check `encoder_info.json`:
`checkpoint_loaded: true`.

### 3.1 Recommended joint pilot (A0 vs A)

This is the main one-seed MIT-BIH pilot:

```bash
# --- Pretrain A ---
python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/ntxent_mitbih_seed0 \
  --ssl-objective ntxent \
  --positive-mode same_window \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --seed 0 --device cuda

# --- Phase 1: A0 + A ---
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/ntxent_mitbih_seed0/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_ntxent_seed0_8s \
  --mode oracle \
  --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --l2-normalize-embeddings --pca-dim 32 \
  --phase1-eval --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --no-random-fallback \
  --device cuda
```

In the outputs, `arm=layer3` rows are **Arm A** for this folder.
`arm=a0_layer2_features` is the control.

Full multi-seed / multi-dataset index: see `ZEROSHOT_CLUSTER_RUN_NOTES.md` §1–2, §5.

---

## 4. Protocol freeze (what A inherits)

Identical decision protocol to A0 / A1 / B — only the representation changes.

| Setting | Frozen value |
| --- | --- |
| Pretrain / L3 window | **8 s @ 125 Hz** |
| Eval cohort (pilot) | 13 MIT-BIH gold transition records |
| Positive mode (primary) | `same_window` |
| SSL objective | `ntxent` |
| Projection | Used in train only; discarded at eval |
| Embedding dim | 128 |
| Eval preprocess | `--l2-normalize-embeddings --pca-dim 32` (protocol default) |
| Scorers | Mahalanobis + kNN |
| Threshold | Conformal α = 0.10 (healthy val only) |
| Primary metric | False-permit on DANGEROUS + **record-cluster bootstrap CI** |
| Disjoint split (preferred) | `--exclude-records-csv` gold list at pretrain |

---

## 5. Outputs to read for A (same Phase 1 files)

See `LAYER3_PHASE1_OUTPUT_METRICS.md` for every column. For A specifically:

| Priority | File | What you learn about A |
| --- | --- | --- |
| 1 | `encoder_info.json` | Checkpoint actually loaded? |
| 2 | `phase1_metrics_bootstrap.csv` | A’s false-permit ± record CI (`arm=layer3`) |
| 3 | `phase1_metrics_overall.csv` | A vs A0 rates / AUROC |
| 4 | `phase1_cav_l2_l3.csv` | Does A catch A0 false permits? |
| 5 | `phase1_metrics_by_danger_subtype.csv` | Where A fails (rhythm / morph / noise) |
| 6 | `phase1_metrics_by_record.csv` | Record-level instability |
| 7 | `phase1_thresholds.csv` | Fit/val sizes, conformal status, PCA meta |
| 8 | `pretrain_records.json` (in ckpt dir) | Did A see the gold eval records? |

**Ignore for A claims:** random-encoder pollution only matters if checkpoint
failed — with `--no-random-fallback` the job should die instead.

Non-Phase-1 files (`per_beat.csv`, `metrics_overall.csv`) are the **same**
encoder path as `arm=layer3` under the main threshold method — useful, but the
pre-registered multi-arm tables are the `phase1_*` files.

---

## 6. A-specific caveats / footguns

### 6.1 Must have a real checkpoint

Unlike A0, A is meaningless without pretrained weights. Always:

```text
--no-random-fallback
encoder_info.json → checkpoint_loaded: true
```

### 6.2 Negatives can hurt the healthy cloud

NT-Xent pushes *other batch windows* away. In ECG, many “negatives” are also
healthy sinus → the loss can **spread** the healthy manifold, which is the
opposite of what Mahalanobis wants (tight healthy cloud). That is exactly why
**A1 (VICReg)** exists as the next arm.

### 6.3 `same_record` positives are dangerous for AD

`--positive-mode same_record` can treat VT and sinus from the same patient as a
positive pair → collapses danger toward healthy. Primary A = `same_window` only.

### 6.4 Mixed vs healthy-only pretrain

| Setting | Meaning |
| --- | --- |
| Default (all windows) | Broad representation learning — OK for A |
| `--healthy-only` | Safer morphology prior; ablation / sensitivity check |

`--healthy-only` **fails closed** if `is_healthy_window` is missing from the index.

### 6.5 Pretrain / eval leakage

Without `--exclude-records-csv`, A can pretrain on the same 13 gold MIT-BIH
records you evaluate → optimistic story. Prefer record-disjoint; check
`pretrain_records.json`.

### 6.6 Augmentor sampling rate

Must match cache: `--augment-fs 125`. Wrong fs → wrong wander frequency.

### 6.7 Signal cache provenance

If you ever built `.npy` caches at 250 Hz, rebuild with
`--overwrite-signal-cache` when switching to 125 Hz.

### 6.8 A vs A0 input mismatch (framing)

| | A0 | A |
| --- | --- | --- |
| Observation | L2 features (≈5 s morph + 30 s RR) | 8 s waveform embedding @ 125 Hz |
| Scorer | Same Mahalanobis/kNN + conformal | Same |

Fair as **system / representation** comparison. Not “identical input, only the
encoder changed.” Say that explicitly in the thesis.

### 6.9 PCA / L2 on embeddings

Protocol uses L2 + PCA(32) before the scorer for SSL arms. A0 does **not** get
that embedding preprocess. Compare both Mahalanobis and kNN before crowning A.

---

## 7. What “good A” looks like (go / no-go after one seed)

**Engineering sanity (must pass before multi-seed):**

1. Pretrain finishes; loss decreases; checkpoint loads.
2. All (or explained) gold records score; few `alpha_infeasible` / insufficient fit.
3. Bootstrap CI non-degenerate.
4. Not all-permit or all-inhibit everywhere.

**Scientific read (compare to A0 in same out-dir):**

| Pattern | Interpretation |
| --- | --- |
| A false-permit ≪ A0 (CI separates) | SSL representation helps |
| A ≈ A0 | Contrastive SSL no clear gain on this cohort |
| A worse than A0 | Negatives / augs may hurt; try A1 or healthy-only A |
| High CAV | A catches danger A0 misses → complementarity |
| Low CAV, high score correlation | Near-redundant with A0 |

Only then unlock multi-seed A and arm **A1**.

---

## 8. A vs A1 vs B (so you don’t confuse them)

| | A | A1 | B |
| --- | --- | --- | --- |
| Objective | NT-Xent | VICReg | Masked recon + consistency |
| Views | Two augs of full window | Same | Two **masks** of same window |
| Negatives | Yes | No | No |
| Reconstruction | No | No | Yes |
| CLI | `ntxent` | `vicreg` | `mae_consistency` |

A and A1 share the **same data/augmentation pipeline**; only the loss changes.
B changes the learning signal (masking).

---

## 9. Code / file map for A

| Piece | Path |
| --- | --- |
| Pretrain driver | `Layer3/tools/pretrain_encoder.py` |
| Encoder + projection | `Layer3/pipeline/layer3_encoder.py` |
| Augmentations | `Layer3/pipeline/layer3_augmentations.py` |
| Window index | `Layer3/tools/build_window_index.py` |
| Phase 1 eval | `Layer3/validation/run_beat_validation.py` + `layer3_phase1_eval.py` |
| Scorer | `Layer3/pipeline/layer3_embedding_mahalanobis.py` |
| Cluster commands | `Layer3/reports/ZEROSHOT_CLUSTER_RUN_NOTES.md` §2, §5 |
| This status | `Layer3/reports/LAYER3_COMPLETE_STATUS_A.md` |

---

## 10. Cluster verdict for A

| Check | Status |
| --- | --- |
| A code path implemented | Yes |
| Footguns fixed (fs, norm, healthy-only, exclude-records, checkpoint fail-closed) | Yes |
| Run after A0 protocol sanity | Recommended |
| One-seed MIT-BIH A0+A pilot | **GO** |
| Multi-seed / A1 / B | After one-seed A looks sane |

**Order:** A0 (optional alone) → **pretrain A + Phase 1 `a0,layer3`** → inspect
bootstrap / CAV → then A1 / B.

---

## 11. Thesis paragraph (Arm A)

> Arm A pretrained a 1D CNN encoder with NT-Xent contrastive learning on 8 s
> ECG windows at 125 Hz, using two physiology-aware augmentations of the same
> window as positives. The projection head was discarded after training. For
> each MIT-BIH gold transition record we fit a healthy-only Mahalanobis/kNN
> baseline in embedding space and set a conformal threshold with healthy
> false-inhibit budget α = 0.10. We compare false-permit rates on DANGEROUS
> beats against Arm A0 (Layer 2 features, same scorer), with record-cluster
> bootstrap confidence intervals and conditional added value (CAV). Arm A tests
> whether a standard contrastive ECG representation improves a personalized
> stimulation-safety veto; it is not trained with arrhythmia labels and does
> not command stimulation.

---

*Update this file when one-seed A numbers land or when A-specific defaults change.*
