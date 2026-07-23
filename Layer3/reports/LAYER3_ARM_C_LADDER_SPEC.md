# Layer 3 — Arm C ladder (C1 / C2 / C3) specification

**Canonical definition** of the three exploratory improvements on Arm C (SupCon).
Each rung keeps the **identical label-free deployment contract** as A0 / A / A1 / B /
B1 / C — only the *pretraining objective* changes, so any delta is attributable to the
representation.

Last updated: July 2026.
Parent: [`LAYER3_ARM_C_SUPERVISED_SPEC.md`](LAYER3_ARM_C_SUPERVISED_SPEC.md) (Arm C baseline).
Code: `Layer3/pipeline/layer3_supervised.py` (loss primitives) + `Layer3/tools/pretrain_encoder.py`
(`--ssl-objective supcon_oe|deepsad|supcon_hybrid`). **Implemented + locally smoke-verified.**

---

## 0. Guardrail (read first)

> **This ladder is EXPLORATORY.** It is tuned/inspected on the same 13 MIT-BIH gold
> records the primary campaign already saw, so improvements here are **p-hacking-risk
> until confirmed on an untouched cohort** (Creighton, or a held-out MIT-BIH split).
> **A0 (handcrafted features) remains the deployable baseline.** Report false-permit +
> CAV vs A0, not accuracy. A clean *negative* (no rung beats A0 / adds CAV) is a valid,
> publishable result and the signal to stop chasing Layer 3.

Why this ladder exists: the SSL arms + Arm C did not beat A0 on false-permit. The AD
literature (Deep SVDD, Deep SAD, Outlier Exposure) says the representation should be
shaped into the *exact geometry the veto reads* — a **compact healthy cloud with danger
pushed far out** — rather than a classification-shaped space. The ladder tests whether
that AD-native shaping closes the gap to A0.

---

## 1. The ladder (increasing complexity)

All three shape the **encoder embedding `h`** (the vector the per-record Mahalanobis/kNN
veto actually reads), using a **frozen center `c`** = mean of NORMAL embeddings computed
once before training (`--center-init-windows`, default 2048) and never updated. They
reuse Arm C's class-balanced `WeightedRandomSampler`, gold exclusion, projection/head
discard, and encoder-only checkpoint.

| Rung | `--ssl-objective` | Loss | Complexity | Literature |
| --- | --- | --- | --- | --- |
| **C** (baseline) | `supcon` | `SupCon(z)` | — | Khosla 2020 |
| **C1** | `supcon_oe` | `SupCon(z) + oe_weight · OE(h)` | Low | + Hendrycks OE 2019 |
| **C2** | `deepsad` | `svdd_weight · compact(h) + oe_weight · OE(h)` | Medium | Ruff SVDD 2018 + Deep SAD 2020 |
| **C3** | `supcon_hybrid` | `SupCon(z) + svdd_weight · compact(h) + oe_weight · OE(h)` | High | Khosla + Ruff |

### Loss primitives (`Layer3/pipeline/layer3_supervised.py`)
- `svdd_compactness_loss(h, labels, center, normal_id)` → mean ‖h−c‖² over **normal**
  (pull healthy toward c). Grad-safe 0 if no normal in batch.
- `outlier_exposure_loss(h, labels, center, unsafe_id, eps=1.0)` → mean 1/(‖h−c‖²+eps)
  over **unsafe** (push danger away from c; bounded in (0, 1/eps]). Grad-safe 0 if no
  unsafe in batch.

`unsafe` = DANGEROUS + NOISE; `normal` = NORMAL; `benign` = BENIGN_ABNORMAL (ignored by
the SVDD/OE terms — a don't-care for false-permit); AF_CONTEXT dropped. Same label map as
Arm C.

---

## 2. Deployment contract (unchanged, critical)
- Encoder frozen after pretraining; **projection head discarded** (encoder-only checkpoint).
- Per-record **healthy-only** calibration (Mahalanobis/kNN) + conformal α = 0.10.
- **No safety labels at deployment.** Labels touched only in pretraining.
- Center `c` is a *pretraining* device only; the deployment veto uses the per-record
  healthy calibration, never `c`.

---

## 3. Protocol (inherits the Arm C pilot freeze)

| Setting | Value |
| --- | --- |
| Window | 8 s @ 125 Hz (single scale) |
| Eval cohort | 13 MIT-BIH gold |
| Class map | normal / unsafe(=DANGEROUS+NOISE) / benign; drop AF_CONTEXT |
| Disjoint split | `--exclude-records-csv` gold (verified: 48−13=35 pretrain records) |
| Sampler | sqrt-tempered inverse-frequency `WeightedRandomSampler` |
| Encoder | `ECGEncoder1D` → 128-d, frozen after pretrain |
| Deployment scorer | L2 + PCA(32) + Mahalanobis/kNN + conformal α=0.10 |
| Primary metric | False-permit DANGEROUS + bootstrap CI + CAV vs A0 |

---

## 4. Commands

Cluster wrappers (run on GPU; `--device cuda`):

```bash
# C1
bash Layer3/reports/cluster_jobs/14a_pretrain_arm_c1_supcon_oe.sh
bash Layer3/reports/cluster_jobs/14b_phase1_a0_plus_c1.sh
# C2
bash Layer3/reports/cluster_jobs/15a_pretrain_arm_c2_deepsad.sh
bash Layer3/reports/cluster_jobs/15b_phase1_a0_plus_c2.sh
# C3
bash Layer3/reports/cluster_jobs/16a_pretrain_arm_c3_hybrid.sh
bash Layer3/reports/cluster_jobs/16b_phase1_a0_plus_c3.sh
```

All scripts are env-overridable: `OE_WEIGHT`, `SVDD_WEIGHT`, `SUPCON_TEMPERATURE`,
`CENTER_INIT_WINDOWS`, `EPOCHS`, `BATCH_SIZE`, `SEED`, `DEVICE`, `CKPT_DIR`, `OUT_DIR`.

---

## 5. Post-run checks (each rung)
- `pretrain_records.json`: `labels_used_in_pretraining_only=true`, gold excluded,
  `class_balanced_sampler`, and `oe_weight`/`svdd_weight`/`center_init_windows` logged.
- `encoder_info.json`: `checkpoint_loaded=true`.
- Pull `phase1_metrics_bootstrap.csv`, `phase1_cav_l2_l3.csv`, `phase1_metrics_by_record.csv`,
  danger-subtype split. **Question: does the rung catch A0's misses (CAV), or just shift
  the tradeoff?**

---

## 6. Sweep tail ("more to try tomorrow")

Run only if a rung looks promising vs A0. All via the same 14/15/16 scripts + env vars.
Keep the deployment scorer fixed; sweep one axis at a time.

| Axis | Values | How |
| --- | --- | --- |
| OE weight η | 0.5, 1.0, 2.0 | `OE_WEIGHT=... bash 14a...` (C1/C2/C3) |
| SVDD weight | 0.5, 1.0 | `SVDD_WEIGHT=... bash 15a...` (C2/C3) |
| Center windows | 1024, 2048, 4096 | `CENTER_INIT_WINDOWS=...` |
| SupCon temperature | 0.07, 0.1, 0.2 | `SUPCON_TEMPERATURE=...` (C1/C3) |
| Scorer | mahalanobis vs knn | already both in Phase 1 output; read separately |
| PCA dim | 16, 32, 64 | edit `--pca-dim` in the 14b/15b/16b script |

Because the 13 gold records are already inspected, every sweep result is **exploratory
until confirmed on an untouched cohort** — do not present sweep winners as preregistered.

---

## 7. What "good" looks like (per rung, vs A0 / C / SSL arms, same scorer)

| Pattern | Read |
| --- | --- |
| Cx false-permit ≪ A0 | AD-shaped supervised representation finally beats features — the one case ML clearly adds value (confirm on untouched cohort before claiming) |
| Cx ≈ A0 | Even AD-shaped supervised representation ties features — strong publishable null; personalization is the lever |
| High CAV vs A0 | Cx complements A0 even if headline rates tie → keep as complementary veto |
| C2/C3 collapse (all scores ≈ equal) | SVDD hypersphere collapse — try larger `--center-init-windows`, lower `svdd_weight`, or rely on OE to counteract |

---

## 8. Caveats (ladder-specific)
1. **SVDD collapse risk.** With a fixed center and a trainable encoder, C2/C3 can drive all
   embeddings toward `c`. The OE term counteracts this (pushes unsafe out), and the encoder
   is frozen + re-scored per-record, but watch the `svdd`/`oe` logs and embedding spread.
2. **Scarce danger signal.** `unsafe` ≈ 1646 windows after gold exclusion; the sampler
   tempers but does not fully balance. OE uses those windows as outliers, which is more
   efficient than treating them as a balanced class — but the signal is still small.
3. **Same fairness rules as Arm C** (§4 of the parent spec): gold excluded, same encoder /
   augmentations / frozen scorer, deployment label-free, report false-permit + CAV.
4. **Human-in-domain only.** Supervised danger labels are human/in-domain; do not transfer
   the learned boundary to rat. This is a ceiling study, not a deployment path.

---

## 9. Related work
- **Ruff et al., 2018 (Deep SVDD)** — *ICML*. Compact hypersphere for normal; the `compact(h)` term.
- **Ruff et al., 2020 (Deep SAD)** — *ICLR*. Semi-supervised AD with labelled normal + anomalies; the C2 form.
- **Hendrycks et al., 2019 (Outlier Exposure)** — *ICLR*. Known anomalies at training sharpen the normal boundary; the `OE(h)` term.
- **Khosla et al., 2020 (SupCon)** — *NeurIPS*. Supervised contrastive; the `SupCon(z)` term (C / C1 / C3).
- **Ruff et al., 2021** — *Proc. IEEE*. Unifying AD review; "representation then one-class scorer" = our deployment.

---

*Run 14/15/16, fill the A0/C/C1/C2/C3 comparison row-block in
`LAYER3_L2_A0_A_COMPARE.md`, and report false-permit + CAV — not accuracy.*
