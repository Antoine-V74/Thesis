# Layer 3 — Complete status (Arm B1: masked recon + subject contrastive)

**Purpose:** One place that explains **Arm B1** at the same depth as the A0 / A / B
status docs. B1 is an **ablation only** — not the primary masked arm.

**Date:** July 2026  
**Canonical design spec:** [`LAYER3_ARM_B_B1_SPEC.md`](LAYER3_ARM_B_B1_SPEC.md) §3  
**Code:** `Layer3/pipeline/layer3_masked_ssl.py` (`MaskedSubjectContrastiveModel`)  
**Driver:** `Layer3/tools/pretrain_encoder.py` (`--ssl-objective mae_subject_contrastive`)

**Sibling docs:**
- B (primary masked arm) → `LAYER3_COMPLETE_STATUS_B.md`
- A0 → `LAYER3_COMPLETE_STATUS_A0_FIRST.md`
- A → `LAYER3_COMPLETE_STATUS_A.md`
- A1 → `VICREG_A1_IMPLEMENTATION_PLAN.md`
- C → `LAYER3_COMPLETE_STATUS_C.md`
- Commands → `ZEROSHOT_CLUSTER_RUN_NOTES.md` §4 / §5
- Metrics → `LAYER3_PHASE1_OUTPUT_METRICS.md`

---

## 0. One-sentence framing for B1

> Arm B1 learns ECG structure by **reconstructing masked patches** and adds a
> **subject/record contrastive** term (Yu / ZEROSHOT-style): windows that share
> a record id are pulled together. It exists to test whether that subject term
> helps or **hurts** personalized anomaly detection versus primary B.

```text
Question answered by B1:
  "Does adding subject/record contrastive to masked recon improve or degrade
   the personalized veto vs B (same-window consistency, no subject positives)?"
```

**Role in the thesis:** ablation / negative-control hypothesis — not a deployment
candidate unless it clearly wins and the safety argument holds.

---

## 1. What Arm B1 is

| Item | Definition |
| --- | --- |
| **Name** | B1 — masked reconstruction + subject/record contrastive |
| **CLI** | `--ssl-objective mae_subject_contrastive` |
| **Papers (spirit)** | Yu ZEROSHOT subject term; masked-reconstruction SSL |
| **Input** | 8 s @ 125 Hz windows (same index as B) |
| **Views / labels** | One masked view per window + **subject_id** from `--subject-col` (default `record_id`) |
| **Encoder** | Same `ECGEncoder1D` → 128-d |
| **Training-only heads** | Conv decoder + `ProjectionHead` — **both discarded** after pretrain |
| **Runtime** | Identical to A/B: frozen encoder → L2/PCA → Mahalanobis/kNN → conformal |
| **Preferred pretrain data** | **`--healthy-only`** (even more important than for B) |
| **What it is not** | Not primary B; not A (no same-window augs as the main pair); not recon-error AD |

**Place in the ladder:**

```text
A0 = handcrafted features + same scorer
A  = NT-Xent (window augs + batch negatives)
A1 = VICReg on full augmented windows
B  = masked recon + same-window consistency     ← PRIMARY masked arm
B1 = masked recon + subject/record contrastive  ← this document (ABLATION)
```

### Why B1 was demoted

Subject/record positives assert: “windows from the same recording should be
close.” On **mixed** ECG that can pull **sinus and VT from one patient** into
the same region → bad for anomaly detection.

Primary B therefore uses **window-level** consistency only. B1 keeps the old
ZEROSHOT-style subject term so we can **measure** the cost/benefit.

---

## 2. How B1 is trained (code path)

Entry: `pretrain_encoder.py` with `mae_subject_contrastive`.

```text
1. Load window index (must have subject column, default record_id)
2. Prefer:
     --healthy-only
     --exclude-records-csv <gold>
3. Dataset: load slice → robust median/MAD normalize
     return_subject_id=True  (batch is a, p, subject_ids)
     NO ECGAugmentor (apply_augmentations=False for mae_*)
4. Model: MaskedSubjectContrastiveModel(encoder + decoder + ProjectionHead)
5. Loss: recon_loss + subject_contrastive_lambda * SubjectContrastiveLoss(proj, ids)
6. Save encoder weights; discard decoder + projection
   Write pretrain_records.json
```

### 2.1 Forward pass (one picture)

```text
x, subject_ids
  → mask                    # one patch mask (ratio 0.75, patch 25)
  → z = enc(x⊙¬mask)
  → recon = decoder(z)
  → recon_loss = MSE(recon, x | mask)
  → proj = ProjectionHead(z)          # train-only
  → subj_loss = NT-Xent-style among batch items
                with same subject_id as positives
  → total = recon_loss + λ_subj * subj_loss
```

Defaults: `mask_ratio=0.75`, `mask_patch_size=25`,
`subject_contrastive_lambda=0.30`, `subject_col=record_id`, temperature `0.1`.

### 2.2 Distinct from B and from A

| | B (primary) | B1 (ablation) | A |
| --- | --- | --- | --- |
| Masked recon | Yes | Yes | No |
| Extra term | Same-window VICReg consistency | Subject/record contrastive | Window-aug NT-Xent |
| Needs subject ids | No | **Yes** | No (unless `same_record` positive mode) |
| Invariance unit | Window | **Record / subject** | Window (default) |
| Thesis role | Primary masked | Ablation | Primary contrastive |

### 2.3 Logging

Watch during pretrain:

- `reconstruction_loss`, `subject_contrastive_loss`
- `mask_fraction`

There is no VICReg `embedding_std` monitor on B1 (that is B’s expander path).
Judge B1 by Phase 1 geometry / false-permit vs B, not by recon loss alone.

---

## 3. How B1 is evaluated (Phase 1)

Same protocol as B — only the checkpoint path changes:

```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data \
  --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/mae_subject_contrastive_mitbih_seed0/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_mae_subject_contrastive_seed0_8s \
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

In outputs, `arm=layer3` = **B1** for that folder. Always compare to the
**primary B** folder (`mae_consistency_*`), not only to A0.

**Never** use decoder reconstruction error as the anomaly score.

---

## 4. Protocol freeze (B1 inherits the same decision protocol)

| Setting | Value |
| --- | --- |
| Window | 8 s @ 125 Hz |
| Eval cohort (pilot) | 13 MIT-BIH gold |
| Objective | `mae_subject_contrastive` |
| Pretrain data (preferred) | `--healthy-only` |
| Disjoint split | `--exclude-records-csv` gold |
| Mask | ratio 0.75, patch 25 |
| Subject column | `record_id` (default) |
| `subject_contrastive_lambda` | 0.30 |
| Eval | L2 + PCA(32) + Mahalanobis/kNN + conformal α=0.10 |
| Primary metric | False-permit DANGEROUS + record-cluster bootstrap CI |
| Reporting label | Always **“B1 ablation”**, never “primary B” |

---

## 5. Recommended cluster sequence for B1

```text
1. A0 + A pilot OK
2. Primary B one-seed (healthy-only) OK enough to compare
3. Then B1 one-seed with --healthy-only + exclude gold
4. Phase 1 a0,layer3 with B1 checkpoint
5. Report B1 vs B (and vs A0) — help or hurt?
6. Optional: mixed-data B1 as a deliberate “worst case” ablation
```

B1 is **not** in the first pilot. Do not delay A0-vs-A for B1.

### Pretrain B1 (ablation)

```bash
python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/mae_subject_contrastive_mitbih_seed0 \
  --ssl-objective mae_subject_contrastive \
  --healthy-only \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --mask-ratio 0.75 \
  --mask-patch-size 25 \
  --subject-contrastive-lambda 0.30 \
  --subject-col record_id \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --num-workers 4 --seed 0 --device cuda
```

Expect `[WARN]` if you omit `--healthy-only` — intentional.

---

## 6. Outputs to read for B1

Same Phase 1 files as B (`LAYER3_PHASE1_OUTPUT_METRICS.md`):

| Priority | File | Role for B1 |
| --- | --- | --- |
| 1 | `encoder_info.json` | `checkpoint_loaded: true` |
| 2 | `phase1_metrics_bootstrap.csv` | Headline false-permit CI (`arm=layer3`) |
| 3 | `phase1_metrics_overall.csv` | B1 vs A0 in this folder |
| 4 | Compare folders | **B1 out-dir vs B (`mae_consistency`) out-dir** |
| 5 | `phase1_cav_l2_l3.csv` | Complementary veto vs A0 |
| 6 | `phase1_metrics_by_danger_subtype.csv` | Where subject term helps/hurts |
| 7 | `pretrain_records.json` | Healthy-only? Gold excluded? |

Headline comparison for the thesis is **B1 vs B**, not B1 alone.

---

## 7. Caveats / footguns (B1-specific)

### 7.1 Ablation, not primary

Do not present B1 as the main masked method. Spec and code both mark it
`ABLATION`. Primary masked = B (`mae_consistency`).

### 7.2 Subject positives can poison anomaly geometry

Same `record_id` → pulled together. On mixed data that includes VT/VF/noise in
that record → healthy baseline cloud may absorb danger. Even with healthy-only,
you are still asserting **record-level** invariance across different healthy
windows (less dangerous, but a different hypothesis than B).

### 7.3 Healthy-only is strongly preferred

Without `--healthy-only`, B1 is the **worst-case** design we demoted. Use mixed
only as a labeled ablation and say so in the text.

`--healthy-only` fails closed if `is_healthy_window` is missing.

### 7.4 Subject column must exist and be stable

Default `--subject-col record_id`. Wrong column → nonsense positives. MIT-BIH
pilot: one record ≈ one subject; multi-subject corpora need a real subject id.

### 7.5 Batch composition matters

Subject contrastive needs **multiple windows per subject in a batch** to form
positives. Very small batches or one-window-per-record sampling weaken the term
(effective λ → 0). Prefer protocol batch 256.

### 7.6 Never score with reconstruction error

Decoder is training-only. Runtime = encoder distance only.

### 7.7 Do not confuse with A’s `same_record` positive mode

Arm A can use `--positive-mode same_record` (also an ablation). B1 combines
**masking + recon + subject contrastive** in one objective. Different arm.

### 7.8 Checkpoint / random encoder

Always `--no-random-fallback`.

### 7.9 Pretrain / eval leakage

`--exclude-records-csv` gold; check `pretrain_records.json`.

### 7.10 A0 vs B1 input mismatch

Same framing as A/B: A0 = L2 features; B1 = 8 s waveform embedding. Matched
scorer, not matched input.

### 7.11 Patch size tied to fs

`mask_patch_size=25` ≈ 200 ms at 125 Hz.

### 7.12 Underpowered fine ranking

13 MIT-BIH gold records: treat B1 as a **robustness / mechanism check** vs B,
not a high-precision rank of SSL families.

---

## 8. What “good B1” looks like (interpretation)

**Engineering:** pretrain finishes; checkpoint loads; Phase 1 non-degenerate.

**Scientific (relative to B):**

| Pattern | Read |
| --- | --- |
| B1 ≪ B on false-permit (worse) | Supports demoting subject contrastive — report as expected risk |
| B1 ≈ B | Subject term adds little under healthy-only |
| B1 ≫ B (better) | Unexpected win — inspect carefully; still argue safety of subject positives |
| Mixed B1 ≪ healthy-only B1 | Confirms pathology-normalization / subject-poisoning story |
| High CAV vs A0 but worse than B | May still complement A0; primary masked stays B |

Either help or hurt is publishable **if** framed as the ablation question.

---

## 9. Code / file map

| Piece | Path |
| --- | --- |
| B1 model | `Layer3/pipeline/layer3_masked_ssl.py` (`MaskedSubjectContrastiveModel`) |
| Subject loss | same file (`SubjectContrastiveLoss`) |
| Pretrain driver | `Layer3/tools/pretrain_encoder.py` |
| Design freeze | `Layer3/reports/LAYER3_ARM_B_B1_SPEC.md` |
| Primary B status | `Layer3/reports/LAYER3_COMPLETE_STATUS_B.md` |
| This status | `Layer3/reports/LAYER3_COMPLETE_STATUS_B1.md` |

---

## 10. Cluster verdict for B1

| Check | Status |
| --- | --- |
| B1 code implemented | Yes (`mae_subject_contrastive`) |
| Marked ablation in CLI help | Yes |
| Healthy-only warn without flag | Yes |
| Run after primary B one-seed | **Required** |
| Primary pretrain flags | `--healthy-only` + exclude gold |
| Multi-seed B1 | Only if one-seed B1 vs B is worth refining |

---

## 11. Thesis paragraph (Arm B1)

> As an ablation of the masked family, Arm B1 retained masked reconstruction but
> replaced same-window consistency with a subject/record contrastive term in the
> spirit of ZEROSHOT: embeddings from the same recording were pulled together.
> Because that invariance can align healthy and abnormal windows when pathology
> is present in the pretraining corpus, B1 was not treated as the primary masked
> method. It was preferably pretrained on healthy-only windows and compared to
> primary B under the same personalized Mahalanobis/kNN + conformal protocol to
> test whether the subject term helped or hurt danger separation.

---

*Update when one-seed B1 numbers land. Always label results “B1 ablation.”*
