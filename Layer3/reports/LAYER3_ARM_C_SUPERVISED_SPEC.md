# Layer 3 — Arm C specification (supervised-pretrained embedding)

**Canonical definition** of the supervised representation arm. Arm C answers the
question the SSL arms (A / A1 / B / B1) do **not**: *if we use the labels we
already have in public ECG to shape the embedding, does the learned
representation beat A0 handcrafted features under the same label-free deployment
scorer?*

Last updated: July 2026.
**Complete status (deep dive):** [`LAYER3_COMPLETE_STATUS_C.md`](LAYER3_COMPLETE_STATUS_C.md)  
Companion: `LAYER3_COMPLETE_STATUS_A.md` (A / NT-Xent), `LAYER3_ARM_B_B1_SPEC.md`
(masked family), `ZEROSHOT_CLUSTER_RUN_NOTES.md` (eval commands),
`LAYER3_PHASE1_OUTPUT_METRICS.md` (metrics).

Code target: `Layer3/pipeline/layer3_supervised.py` (**implemented**);
driver `Layer3/tools/pretrain_encoder.py` (`--ssl-objective supcon`).
Variants C-ce / C-oe remain deferred (spec only).

---

## 0. The key idea (why Arm C exists)

There are **two** label questions at **two** times:

| Time | Labels available? | Consequence |
| --- | --- | --- |
| **Pretraining** (offline, public ECG) | **Yes** — abundant beat/rhythm labels | We *can* use them to shape the encoder |
| **Deployment / calibration** (sensor on patient) | **No** — only the patient's own assumed-healthy window | Decision rule **must** stay one-class |

The SSL arms assumed "label-free at deployment" implies "label-free at
pretraining." **It does not.** Arm C uses public labels **only at pretraining**,
then freezes the encoder and calibrates a **label-free** per-record healthy
Mahalanobis/kNN veto — identical deployment contract to A0 / A / A1 / B / B1.

> Arm C separates the *representation* (trained with labels) from the *decision*
> (calibrated without labels). Deployment never sees a danger label.

This is also the ZEROSHOT lesson turned into an experiment: their **supervised**
linear model (~0.964) slightly beat their **unsupervised** SSL (~0.954). Arm C
is the supervised representation ceiling for our safety veto.

---

## 1. Place in the arm ladder

| Arm | Objective flag | Representation trained with… |
| --- | -------------- | ---------------------------- |
| **A0** | *(no encoder)* | Handcrafted Layer 2 features (control) |
| **A** | `ntxent` | Self-supervised contrastive (no labels) |
| **A1** | `vicreg` | Self-supervised non-contrastive (no labels) |
| **B** | `mae_consistency` | Masked recon + same-window consistency (no labels) |
| **B1** | `mae_subject_contrastive` | Masked recon + subject contrastive (no labels) |
| **C (this doc)** | `supcon` | **Supervised contrastive on public safety labels** |
| **C1** ([ladder](LAYER3_ARM_C_LADDER_SPEC.md)) | `supcon_oe` | SupCon + outlier exposure (danger pushed from healthy centre) |
| **C2** ([ladder](LAYER3_ARM_C_LADDER_SPEC.md)) | `deepsad` | SVDD compact normal + outlier exposure (AD-native geometry) |
| **C3** ([ladder](LAYER3_ARM_C_LADDER_SPEC.md)) | `supcon_hybrid` | SupCon + SVDD + outlier exposure (multi-term, single-scale) |
| **C-ce** (variant) | `supervised_ce` | Supervised cross-entropy encoder, head discarded |

All arms feed the **same** frozen personalized Mahalanobis/kNN + conformal
scorer. Only the representation changes. C1/C2/C3 are **exploratory** improvements
on C — see [`LAYER3_ARM_C_LADDER_SPEC.md`](LAYER3_ARM_C_LADDER_SPEC.md).

---

## 2. Arm C (primary variant) — Supervised Contrastive (SupCon)

### Idea
Use public labels so that, in embedding space, **beats of the same safety class
cluster** and **danger is pushed away from normal** — the geometry our distance
veto wants — *before* any patient calibration.

Reference: **Khosla et al. 2020 (Supervised Contrastive Learning)**.

### Label source (already in the window index)
`safety_group` column, values observed on MIT-BIH 8 s index:

```text
NORMAL           17225
BENIGN_ABNORMAL  19101
AF_CONTEXT        4306
NOISE             1831
DANGEROUS          689
```

Default class map for SupCon positives (configurable via `--label-map`):

```text
normal   := NORMAL
unsafe   := DANGEROUS + NOISE          # the classes the veto must separate
benign   := BENIGN_ABNORMAL            # kept as own class (not "normal", not "unsafe")
drop     := AF_CONTEXT                  # ambiguous policy; exclude from SupCon by default
```

Rationale: we do **not** fold benign ectopy into "normal" (it is a don't-care
for false-permit) and we do **not** fold it into "unsafe" (it is not dangerous).
Keeping it as its own class avoids teaching the encoder a wrong 2-way boundary.

### Forward pass
```text
x (B,1,T)
 → aug_a, aug_b            # two SAFE augmentations of each window (as in Arm A)
 → z_a = enc(aug_a); z_b = enc(aug_b)
 → p_a = proj(z_a); p_b = proj(z_b)      # projection head, discarded after pretrain
 → SupCon loss over the 2B projections:
      positives = other views with the SAME safety label
      negatives = all other labels
```

Projection head is discarded after pretraining. Downstream scores come **only**
from encoder embedding distance — never the projection, never a class logit.

### Deployment contract (unchanged, critical)
- Encoder frozen.
- Per-record **healthy-only** calibration (Mahalanobis mean/cov or kNN).
- Conformal healthy-side threshold, `alpha = 0.10`.
- **No safety labels used at deployment.** Labels touched only in pretraining.

---

## 3. Arm C-ce (variant) — supervised cross-entropy encoder

Simplest possible "did labels help the representation?" control:

```text
enc → linear head → cross-entropy on safety_group (class-weighted)
```

Discard the head after training; keep the encoder. Same frozen scorer downstream.
Cheaper than SupCon, weaker geometry; report only if SupCon is promising or as a
sanity baseline. SupCon is the primary C.

---

## 3b. Arm C-oe (merged variant) — outlier-exposed compact normal

This is the "merge the good ideas" variant. It fuses three literatures that all
point the same way for a *label-free-deployment* veto:

- **Deep SVDD** (compact hypersphere for normal) — but SVDD alone is unsupervised
  and can collapse.
- **Deep SAD / Outlier Exposure** (use the *few* labelled danger/noise windows we
  have as explicit outliers to push away from the normal centre).
- **Two-stage one-class** (learn representation, then fit the one-class scorer) —
  which is exactly our frozen-encoder + Mahalanobis/kNN deployment.

### Objective (pretraining only)
```text
normal windows   → pulled toward a fixed centre c   (SVDD compactness)
unsafe windows   → pushed to have LARGE distance from c   (outlier exposure / Deep SAD)
benign windows   → mild / ignored (don't-care for false-permit)
```
i.e. `loss = mean_normal ||z-c||²  −  η · mean_unsafe log(||z-c||² )` (Deep-SAD
form), optionally added to the SupCon term as a regulariser.

### Why it is well-matched
It shapes **exactly** the geometry the deployment veto reads (compact healthy
cloud, danger far outside) while (a) using the scarce danger labels efficiently
as *outliers* rather than as a balanced class, and (b) never requiring danger
labels at deployment — calibration still fits the patient's own healthy centre.

Report C-oe only if C (SupCon) is promising; it is the most literature-aligned
"AD-shaped supervised representation." Keep the same frozen scorer downstream.

---

## 4. Leakage & fairness rules (must hold)

1. **Exclude gold eval records from pretraining** with `--exclude-records-csv`
   (same list as every other arm). Supervised labels make leakage worse if the
   eval records are seen — enforce it and check `pretrain_records.json`.
2. **Same encoder, same augmentations, same fixed scorer** as A/A1. Only the
   loss + label usage change, so any delta is attributable to the objective.
3. **Deployment label-free** — calibration/threshold use healthy calibration
   beats only. If this is violated the arm is disqualified.
4. **Report false-permit DANGEROUS + record-cluster bootstrap CI + CAV vs A0**,
   not accuracy.

---

## 5. Protocol freeze (Arm C inherits the pilot protocol)

| Setting | Value |
| --- | --- |
| Window | 8 s @ 125 Hz |
| Eval cohort | 13 MIT-BIH gold |
| Objective | `supcon` (primary), `supervised_ce` (variant) |
| Label column | `safety_group` (pretrain only) |
| Class map | normal / unsafe(=DANGEROUS+NOISE) / benign; drop AF_CONTEXT |
| Disjoint split | `--exclude-records-csv` gold |
| Augmentations | same safe set as Arm A, `--augment-fs 125` |
| Encoder | `ECGEncoder1D` → 128-d, frozen after pretrain |
| Deployment scorer | L2 + PCA(32) + Mahalanobis/kNN + conformal α=0.10 |
| Primary metric | False-permit DANGEROUS + bootstrap CI + CAV vs A0 |

---

## 6. Commands (after §7 code lands)

### Pretrain Arm C (SupCon)
```bash
python Layer3/tools/pretrain_encoder.py \
  --window-index Results/layer3/window_index/layer3_windows_mitbih_8s_125hz.csv \
  --checkpoint-dir Results/layer3/pretrain/supcon_mitbih_seed0_100ep_8s_goldexcluded \
  --ssl-objective supcon \
  --label-col safety_group \
  --label-map "NORMAL=normal,DANGEROUS=unsafe,NOISE=unsafe,BENIGN_ABNORMAL=benign,AF_CONTEXT=drop" \
  --exclude-records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --augment-fs 125 \
  --supcon-temperature 0.1 \
  --epochs 100 --batch-size 256 --lr 3e-4 \
  --num-workers 4 --seed 0 --device cuda
```

### Phase 1 A0 + C (same scorer as every arm)
```bash
python Layer3/validation/run_beat_validation.py \
  --data-dir data --datasets mit_bih_arrhythmia \
  --records-csv Layer3/reports/pilot_lists/pilot_primary_mitbih_gold.csv \
  --checkpoint Results/layer3/pretrain/supcon_mitbih_seed0_100ep_8s_goldexcluded/encoder_last.pt \
  --out-dir Results/layer3/validation/pilot_mitbih_supcon_seed0_100ep_8s \
  --mode oracle --window-s 8 --target-fs 125 \
  --causal-window --lookahead-ms 100 \
  --per-record-calibration --guard-s 8 \
  --l2-normalize-embeddings --pca-dim 32 \
  --phase1-eval --phase1-arms a0,layer3 \
  --phase1-scorers mahalanobis,knn \
  --threshold-method conformal --conformal-alpha 0.10 \
  --no-random-fallback --device cuda
```

Cluster wrappers: `cluster_jobs/07a_pretrain_arm_c_supcon.sh`,
`cluster_jobs/07b_phase1_a0_plus_c.sh`.

---

## 7. Code status (primary Arm C)

**Implemented** (primary `supcon` only; C-ce / C-oe deferred):

1. `Layer3/pipeline/layer3_supervised.py` — `SupConLoss`, `parse_label_map`, `apply_label_map`
2. `ContrastiveECGDataset` — `return_label` → `(a, b, label_id)`
3. `pretrain_encoder.py` — `--ssl-objective supcon`, `--label-col`, `--label-map`,
   `--supcon-temperature`; augmentations on; encoder-only checkpoint
4. `pretrain_records.json` — `labels_used_in_pretraining_only`, `head_discarded`,
   `label_map`, `class_to_id`
5. Fail-closed on missing label column / empty mapped class; stratified
   `--max-windows` sampling for smoke
6. Class-balanced training via sqrt-tempered inverse-frequency `WeightedRandomSampler`
   (supcon path only); provenance logs `class_balanced_sampler`, `epochs`,
   `batch_size`, `seed`

No change to the eval/scorer path — Arm C reuses the frozen-encoder Phase 1
exactly like A/A1/B.

---

## 8. What "good Arm C" looks like

**Engineering:** pretrain converges; checkpoint loads; Phase 1 non-degenerate;
`pretrain_records.json` shows gold excluded and labels pretrain-only.

**Scientific (vs A0 and the SSL arms, same scorer):**

| Pattern | Read |
| --- | --- |
| C false-permit ≪ A0 and SSL arms | Labels shape a better veto space — the one case where ML clearly adds value |
| C ≈ A0 | Even supervised representation barely beats features — strong, publishable null (matches ZEROSHOT) |
| C > SSL but ≈ A0 | Representation learning helps only marginally; personalization is the lever |
| High CAV vs A0 | C complements A0 even if headline rates are close |
| C-ce ≈ C | SupCon geometry not needed; plain supervised encoder enough |

Either outcome is a clean thesis result **because** the deployment contract
(label-free calibration) is identical across arms.

---

## 9. Caveats / footguns (Arm C specific)

1. **Human-danger labels are human/in-domain.** For human deployment this is
   fine (in-domain). For **pig** there is a modest species gap; for the (now
   deferred) rat target the supervised boundary would likely not transfer — say
   so. Arm C's strongest claim is **human-in-domain**.
2. **Class imbalance / tiny DANGEROUS.** Only ~689 DANGEROUS windows (fewer after
   excluding gold; `unsafe` ≈ 1646 after mapping). Training uses a **sqrt-tempered
   inverse-frequency `WeightedRandomSampler`** (implemented) so `unsafe` appears in
   every batch instead of ~5% natural prevalence; still report that the supervised
   danger signal is scarce and that the sampler tempers rather than fully balances.
3. **`safety_group` policy is fixed by the label map.** Folding NOISE into
   "unsafe" and dropping AF_CONTEXT are choices — do not change them silently
   (same rule as AF/SVTA policy elsewhere).
4. **Deployment stays label-free.** If anyone calibrates the threshold with
   danger labels, the arm no longer answers the deployment question.
5. **Never score with class logits.** Runtime = encoder distance only, like every
   arm. The classifier head is a pretraining device, discarded.
6. **Leakage is worse with labels.** Enforce `--exclude-records-csv`; verify.

---

## 10. Thesis paragraph (Arm C)

> To test whether the abundant labels in public ECG could improve the learned
> representation without changing the label-free deployment contract, Arm C
> pretrained the shared encoder with a supervised contrastive loss over safety
> classes (normal / unsafe / benign), then discarded the projection head and used
> the same per-record healthy Mahalanobis/kNN + conformal veto as all other arms.
> Labels were used only during pretraining; deployment calibration never sees a
> danger label. This isolates the value of a label-informed representation and
> provides the supervised ceiling against the self-supervised arms and the A0
> handcrafted-feature control.

---

---

## 11. Related work (citations) and merged design

Citation format matches `Literature_Review_Outline.md`:
**Author et al., Year** — *Venue*. One-sentence relevance.

### 11a. Supervised / label-informed representation learning
- **Khosla et al., 2020** — *NeurIPS*. Supervised Contrastive Learning; same-label
  windows cluster, other labels repel — the core loss for primary Arm C.
- **Chen et al., 2020 (SimCLR)** — *ICML*. NT-Xent contrastive base that SupCon
  extends with labels; shared augmentation/backbone story with Arm A.
- **Kiyasseh et al., 2021 (CLOCS)** — *ICML*. Patient/space/time contrastive ECG;
  motivates label- and subject-aware pretraining under our fixed scorer.
- **Gopal et al., 2021 (3KG)** — *ML4H*. Physiology-aware ECG augmentations; keeps
  Arm C's augmentations safe (no sinus↔danger blurring).

### 11b. Supervised ECG classification (the "ceiling" references)
- **Hannun et al., 2019** — *Nature Medicine*. Cardiologist-level single-lead
  arrhythmia DNN; the supervised ceiling Arm C's encoder approximates.
- **Ribeiro et al., 2020** — *Nature Communications*. Deep 12-lead ECG diagnosis;
  evidence labelled ECG representations are strong in-domain.
- **Hong et al., 2020** — *Computers in Biology and Medicine*. Systematic review of
  deep learning for ECG; situates supervised vs self-supervised choices.

### 11c. Semi-supervised & outlier-exposed anomaly detection (the merge)
- **Görnitz et al., 2013** — *JAIR*. Toward supervised anomaly detection; a few
  labelled anomalies sharpen the normal boundary — rationale for C-oe.
- **Hendrycks et al., 2019 (Outlier Exposure)** — *ICLR*. Known anomalies at
  training improve AD; here our labelled DANGEROUS/NOISE act as outlier exposure.
- **Ruff et al., 2018 (Deep SVDD)** — *ICML*. Compact hypersphere for normal;
  the compactness term in C-oe.
- **Ruff et al., 2020 (Deep SAD)** — *ICLR*. Semi-supervised deep AD using labelled
  normal + labelled anomalies + unlabelled; the exact form C-oe adopts.
- **Ruff et al., 2021** — *Proceedings of the IEEE*. Unifying review of deep/shallow
  AD; frames "representation then one-class scorer" as the general recipe.

### 11d. Two-stage: pretrained representation → one-class scorer (our deployment)
- **Sohn et al., 2021** — *ICLR*. Learn a representation, then a one-class
  classifier; validates Arm C's freeze-then-calibrate split.
- **Reiss et al., 2021 (PANDA)** — *CVPR*. Pretrained features + nearest-neighbour
  AD; direct support for our frozen-encoder + kNN/Mahalanobis veto.
- **Perera & Patel, 2019** — *IEEE TIP*. Deep features for one-class
  classification; label-informed features improve one-class scoring.

### 11e. ECG anomaly detection precedents (same scorer family)
- **Carrera et al., 2019** — *Pattern Recognition*. Per-user normal model with
  online HR adaptation; label-free deployment precedent (also §7a of the review).
- **Jiang et al., 2024** — *arXiv:2404.04935*. Normal-only SSL ECG AD at clinical
  scale; feasibility of embedding-space ECG AD.
- **Yu, 2026 (ZEROSHOT)** — *Research Square*. SSL + per-subject healthy
  Mahalanobis; the supervised-vs-unsupervised near-tie that motivates Arm C.

### 11f. The merged design in one sentence
> Arm C combines **supervised-contrastive / outlier-exposed** representation
> learning (Khosla 2020; Hendrycks 2019; Ruff 2018/2020) with a **two-stage,
> frozen-encoder one-class deployment** (Sohn 2021; Reiss 2021) and **per-record
> healthy calibration** (Carrera 2019; Yu 2026): public labels shape a compact
> healthy / distant-danger geometry offline, and the patient's own unlabelled
> baseline sets the veto threshold online.

New references to add to `Literature_Review_Outline.md` §9b (one line each):
Khosla 2020, Ruff 2018/2020/2021, Sohn 2021, Reiss 2021, Hendrycks 2019,
Görnitz 2013, Perera & Patel 2019, Ribeiro 2020, Hong 2020.

---

*Implement §7, then run `07a`/`07b` and slot the result into the A0/A/A1/B/B1
comparison table. Report false-permit + CAV, not accuracy.*
