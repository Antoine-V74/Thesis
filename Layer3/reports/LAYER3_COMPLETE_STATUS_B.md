# Layer 3 — Complete status (Arm B: masked recon + same-window consistency)

**Purpose:** One place that explains **Arm B** at the same depth as the A0 and A
status docs: what it is, how it trains, how it is evaluated, caveats / footguns,
and when to run it on the cluster.

**Date:** July 2026  
**Canonical design spec:** [`LAYER3_ARM_B_B1_SPEC.md`](LAYER3_ARM_B_B1_SPEC.md)  
**Code:** `Layer3/pipeline/layer3_masked_ssl.py` (`MaskedConsistencyModel`)  
**Driver:** `Layer3/tools/pretrain_encoder.py` (`--ssl-objective mae_consistency`)

**Sibling docs:**
- A0 → `LAYER3_COMPLETE_STATUS_A0_FIRST.md`
- A → `LAYER3_COMPLETE_STATUS_A.md`
- A1 → `VICREG_A1_IMPLEMENTATION_PLAN.md`
- B1 (ablation) → `LAYER3_COMPLETE_STATUS_B1.md` (+ short §8 below)
- C → `LAYER3_COMPLETE_STATUS_C.md` (+ design freeze `LAYER3_ARM_C_SUPERVISED_SPEC.md`)
- Commands → `ZEROSHOT_CLUSTER_RUN_NOTES.md` §4 / §5
- Metrics → `LAYER3_PHASE1_OUTPUT_METRICS.md`

---

## 0. One-sentence framing for B

> Arm B learns ECG structure by **reconstructing masked patches** and keeps
> embeddings informative with a **non-contrastive consistency** term on two
> independent masks of the **same window**. Safety still comes only from a
> per-record healthy baseline in encoder embedding space — never from
> reconstruction error.

```text
Question answered by B:
  "Does masked reconstruction + same-window consistency give a better
   personalized anomaly veto than A0 / A / A1 under the same scorer?"
```

---

## 1. What Arm B is

| Item | Definition |
| --- | --- |
| **Name** | B — masked reconstruction + non-contrastive same-window consistency |
| **CLI** | `--ssl-objective mae_consistency` |
| **Papers (spirit)** | NERULA (masked + non-contrastive); masked-restoration AD literature; VICReg terms for anti-collapse |
| **Input** | 8 s @ 125 Hz windows (same index as A/A1) |
| **Views** | Two **independent patch masks** of the **same** window (not two augmentations) |
| **Encoder** | Same `ECGEncoder1D` → 128-d |
| **Training-only heads** | Conv decoder + VICReg expander — **both discarded** after pretrain |
| **Runtime** | Identical to A: frozen encoder → L2/PCA → Mahalanobis/kNN → conformal |
| **Preferred pretrain data** | **`--healthy-only`** (see caveats) |
| **What it is not** | Not A1 (A1 = VICReg on full augs, no recon); not B1 (B1 = subject contrastive); not recon-error AD |

**Place in the ladder:**

```text
A0 = handcrafted features + same scorer
A  = NT-Xent (negatives)
A1 = VICReg on full augmented windows (no mask, no recon)
B  = masked recon + same-window consistency   ← this document (primary masked arm)
B1 = masked recon + subject/record contrastive (ablation only)
```

### Why B was redesigned (important)

Older idea: “mask + subject contrastive on mixed records.”  
That can pull **healthy and VT from the same patient** into one region → bad for
anomaly detection.

**New primary B:** invariance unit = **window**, never subject.  
Two masks of the same strip should embed similarly; different windows from one
record are **not** forced together.

---

## 2. How B is trained (code path)

Entry: `pretrain_encoder.py` with `mae_consistency`.

```text
1. Load window index
2. Prefer:
     --healthy-only
     --exclude-records-csv <gold>
     --augment-fs 125   (unused for masking path augs, but keep for consistency)
3. Dataset: load slice → robust median/MAD normalize
     (NO ECGAugmentor for mae_*; views come from masks inside the model)
4. Model: MaskedConsistencyModel(encoder + decoder + expander)
5. Loss: recon_loss + consistency_lambda * VICReg(expander(z_a), expander(z_b))
6. Save encoder weights; discard decoder + expander
   Write pretrain_records.json
```

### 2.1 Forward pass (one picture)

```text
x
  → mask_a, mask_b          # independent patch masks (ratio 0.75, patch 25 samp)
  → z_a = enc(x with mask_a zeros)
  → z_b = enc(x with mask_b zeros)
  → recon_a, recon_b = decoder(z_*)
  → recon_loss = MSE on masked samples (avg of two views)
  → cons_loss  = VICReg(expander(z_a), expander(z_b))
  → total      = recon_loss + λ * cons_loss
```

Defaults: `mask_ratio=0.75`, `mask_patch_size=25` (200 ms at 125 Hz),
`consistency_lambda=1.0`, expander `512,512,512`.

### 2.2 Distinct from A1

| | A1 | B |
| --- | --- | --- |
| Views | Two **augmented full** windows | Two **masked** views of one window |
| Reconstruction | No | Yes |
| Consistency | VICReg on expander | VICReg-style on expander of masked embeddings |
| Hypothesis | Neg-free SSL on augs | Structure via inpainting + mask invariance |

Saying “B is just VICReg” is wrong.

### 2.3 Logging / collapse monitors

During pretrain, watch:

- `reconstruction_loss`, `consistency_loss`
- `embedding_std` (variance term — collapse if → 0)
- `mask_fraction`

A low recon loss alone is **not** success. Success = better Phase 1 false-permit
/ healthier embedding geometry vs A0/A/A1.

---

## 3. How B is evaluated (Phase 1)

Same as Arm A — only the checkpoint changes:

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/mae_consistency_mitbih_seed0/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_mae_consistency_seed0_8s \
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

In outputs, `arm=layer3` = **B** for that folder. Compare to NT-Xent / VICReg folders.

**Never** use decoder reconstruction error as the anomaly score.

---

## 4. Protocol freeze (B inherits the same decision protocol)

| Setting | Value |
| --- | --- |
| Window | 8 s @ 125 Hz |
| Eval cohort (pilot) | 13 MIT-BIH gold |
| Objective | `mae_consistency` |
| Pretrain data (primary) | `--healthy-only` |
| Disjoint split | `--exclude-records-csv` gold |
| Mask | ratio 0.75, patch 25 |
| Expander | `512,512,512` (train only) |
| Eval | L2 + PCA(32) + Mahalanobis/kNN + conformal α=0.10 |
| Primary metric | False-permit DANGEROUS + record-cluster bootstrap CI |

---

## 5. Recommended cluster sequence for B

**Do not** start B before A0 + one-seed A look sane.

```text
1. A0 / A pilot OK
2. (optional) A1
3. Build/reuse 8 s @ 125 Hz window index (must have is_healthy_window)
4. Pretrain B with --healthy-only + --exclude-records-csv
5. Phase 1 a0,layer3 with B checkpoint
6. Later: B1 ablation if you need the subject-term contrast
```

### Pretrain B (primary)

```bash
python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/mae_consistency_mitbih_seed0 \
  --ssl-objective mae_consistency \
  --healthy-only \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --mask-ratio 0.75 \
  --mask-patch-size 25 \
  --consistency-lambda 1.0 \
  --vicreg-expander-dims 512,512,512 \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --num-workers 4 --seed 0 --device cuda
```

Expect a `[WARN]` if you omit `--healthy-only` — that is intentional.

---

## 6. Outputs to read for B

Same Phase 1 files as A (`LAYER3_PHASE1_OUTPUT_METRICS.md`):

| Priority | File | Role for B |
| --- | --- | --- |
| 1 | `encoder_info.json` | `checkpoint_loaded: true` |
| 2 | `phase1_metrics_bootstrap.csv` | Headline false-permit CI (`arm=layer3`) |
| 3 | `phase1_metrics_overall.csv` | B vs A0 |
| 4 | `phase1_cav_l2_l3.csv` | Does B catch A0 false permits? |
| 5 | `phase1_metrics_by_danger_subtype.csv` | Rhythm vs morph (recon may help morph) |
| 6 | `pretrain_records.json` | Gold excluded? Healthy-only used? |

Also compare **across folders**: NT-Xent (A) vs VICReg (A1) vs this B out-dir.

---

## 7. Caveats / footguns (B-specific)

### 7.1 Healthy-only is preferred for B (not optional fluff)

Reconstruction on **mixed** data can teach the model to reconstruct VT/noise well
→ pathology looks “normal” in embedding space.  
Primary B = `--healthy-only`. Mixed = deliberate robustness ablation.

`--healthy-only` **fails closed** if `is_healthy_window` is missing from the index
(rebuild index with current `build_window_index.py`).

### 7.2 Never score with reconstruction error

Decoder is training-only. Runtime = encoder distance only. Claiming “MAE anomaly
score” would be a different method and is **out of protocol**.

### 7.3 Do not confuse B with A1

Same VICReg *terms*, different views and presence of recon. Thesis must say so.

### 7.4 Subject contrastive is B1, not B

If you want Yu/ZEROSHOT-style subject positives, that is **`mae_subject_contrastive`**
(B1 ablation), preferably healthy-only. Do not call it primary B.

### 7.5 Checkpoint / random encoder

Always `--no-random-fallback`. Partial key mismatch now fails closed.

### 7.6 Pretrain / eval leakage

Use `--exclude-records-csv` gold list; check `pretrain_records.json`.

### 7.7 Augmentor fs

B does not apply the ECGAugmentor in the dataset path (`apply_augmentations=False`
for mae objectives). Masking is inside the model. Still keep index / eval at 125 Hz;
use `--overwrite-signal-cache` if you ever built caches at 250 Hz.

### 7.8 Batch size / collapse

VICReg variance/cov want a decent batch (protocol 256). Watch `embedding_std`.

### 7.9 A0 vs B input mismatch (same framing as A)

A0 = L2 features (5 s morph + 30 s RR). B = 8 s waveform embedding.  
Matched **scorer**, not matched **input**. Fair as representation/system compare;
do not claim identical observation.

### 7.10 Patch size tied to fs

`mask_patch_size=25` ≈ 200 ms **at 125 Hz**. If you change `target_fs`, revisit patch size.

### 7.11 Order of arms

B is heavier scientifically and computationally than A. Run after A0+A pilot.

---

## 8. B1 ablation (pointer)

Full B1 status (train, eval, caveats, thesis paragraph):
[`LAYER3_COMPLETE_STATUS_B1.md`](LAYER3_COMPLETE_STATUS_B1.md).

| Item | B1 |
| --- | --- |
| CLI | `--ssl-objective mae_subject_contrastive` |
| Extra term | Subject/record NT-Xent on embeddings (+ mask recon) |
| Risk | Can align healthy + abnormal from same record |
| Prefer | `--healthy-only` |
| Role | Report help vs hurt vs primary B |

---

## 9. What “good B” looks like

**Engineering:** pretrain finishes; `embedding_std` not collapsed; checkpoint loads;
Phase 1 bootstrap non-degenerate.

**Scientific (vs A0 / A / A1 in same protocol):**

| Pattern | Read |
| --- | --- |
| B false-permit ≪ A and A0 | Masked+consistency helps |
| B ≈ A | Objective family matters little on this cohort |
| B worse than A, especially on morph danger | Recon/healthy-only settings need inspection |
| High CAV vs A0 | Complementary veto |
| B1 worse than B | Supports “no subject contrastive as primary” |

---

## 10. Code / file map

| Piece | Path |
| --- | --- |
| B / B1 models | `Layer3/pipeline/layer3_masked_ssl.py` |
| VICReg loss (shared) | `Layer3/pipeline/layer3_vicreg.py` |
| Pretrain driver | `Layer3/tools/pretrain_encoder.py` |
| Design freeze | `Layer3/reports/LAYER3_ARM_B_B1_SPEC.md` |
| This status | `Layer3/reports/LAYER3_COMPLETE_STATUS_B.md` |

---

## 11. Cluster verdict for B

| Check | Status |
| --- | --- |
| B code implemented | Yes (`mae_consistency`) |
| B1 ablation implemented | Yes |
| Healthy-only fail-closed | Yes |
| Run after A0 + A pilot | **Required** |
| Primary pretrain flags | `--healthy-only` + exclude gold |
| Multi-seed B | After one-seed B looks sane |

---

## 12. Thesis paragraph (Arm B)

> Arm B pretrained the same 1D CNN encoder with masked ECG reconstruction plus a
> non-contrastive consistency loss between two independently masked views of each
> 8 s window. The invariance unit is the window, not the subject, so healthy and
> abnormal segments from one recording are not forced together. The decoder and
> expander were discarded after training; permit/inhibit used only personalized
> Mahalanobis/kNN distance in encoder space with a conformal healthy
> false-inhibit budget. Pretraining preferred healthy-only windows to avoid
> learning to reconstruct pathology. Subject/record contrastive masking (B1)
> was retained only as an ablation.

---

*Update when one-seed B numbers land. Keep B1 results labeled ablation.*
