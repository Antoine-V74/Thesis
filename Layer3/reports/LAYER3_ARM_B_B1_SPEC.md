# Layer 3 — Arm B / B1 specification (masked-reconstruction family)

**Canonical definition** of the masked SSL arms. Freezes the redesigned **B**
and the demoted **B1** so the other Layer 3 docs stop describing the older
"mask + subject contrastive on mixed records" as primary.

Last updated: July 2026.
Companion: `LAYER3_COMPLETE_STATUS_B.md` (full B status + caveats),
`LAYER3_COMPLETE_STATUS_B1.md` (full B1 ablation status + caveats),
`LAYER3_CONSOLIDATED_SUMMARY.md` (arm map), `ZEROSHOT_CLUSTER_RUN_NOTES.md`
(§4 commands), `VICREG_A1_IMPLEMENTATION_PLAN.md` (A1).

Code: `Layer3/pipeline/layer3_masked_ssl.py` (`MaskedConsistencyModel` = B,
`MaskedSubjectContrastiveModel` = B1); driver `Layer3/tools/pretrain_encoder.py`
(`--ssl-objective mae_consistency` = B, `mae_subject_contrastive` = B1).

---

## 1. The full arm ladder (context)

| Arm | Objective flag | One-line meaning |
| --- | -------------- | ---------------- |
| **A0** | *(no encoder — Layer 2 features)* | Handcrafted-feature control under the **same** scorer |
| **A** | `ntxent` | NT-Xent / SimCLR contrastive (**with** negatives) |
| **A1** | `vicreg` | VICReg non-contrastive on augmented **full** windows (no reconstruction) |
| **B** (primary) | `mae_consistency` | Masked reconstruction **+ non-contrastive same-window consistency** |
| **B1** (ablation) | `mae_subject_contrastive` | Masked reconstruction **+ subject/record contrastive** (Yu / ZEROSHOT-style) |

The three learning ideas are covered without redundancy:
`negatives (A) | no-negatives (A1) | reconstruction + same-window consistency (B)`,
with **B1** isolating the effect of a subject/record contrastive term.
All arms feed the **same** personalized Mahalanobis/kNN + conformal scorer.

> **Do not** put VICReg on every arm. A1 is VICReg on full augmented windows;
> B's consistency term acts on **two masked views of the same window** and is
> paired with a reconstruction loss. They are different hypotheses.

---

## 2. B (primary) — masked recon + non-contrastive same-window consistency

### Idea
Learn an ECG representation where **healthy beats form a tight personalized
cloud and abnormal beats fall outside it**, without ever asserting that all
windows from one recording are close.

- **Masked reconstruction** forces the encoder to learn ECG structure
  (QRS/T, local dynamics) — spirit of NERULA (Manimaran et al. 2024) and
  Jiang et al. 2024 masked-restoration anomaly detection.
- **Non-contrastive same-window consistency** (VICReg-style invariance +
  variance + covariance) aligns the embeddings of **two independent masks of
  the same window**, keeping them informative and anti-collapsed.

### The invariance unit is the WINDOW, never the subject
This is the whole point of the redesign. Two masked views of *the same 8 s
window* should embed similarly. We never force two *different* windows from the
same record together, so a patient's sinus and their VT are **not** pulled into
the same region. Subject/record-level positives (B1) can do exactly that, which
is unsafe for anomaly detection.

### Forward pass (see `MaskedConsistencyModel`)
```text
x (B,1,T)
 → mask_a, mask_b            # two independent patch masks of the SAME window
 → z_a = enc(x⊙¬mask_a);  z_b = enc(x⊙¬mask_b)
 → recon_a, recon_b via training-only decoder
 → recon_loss   = ½·(MSE(recon_a, x | mask_a) + MSE(recon_b, x | mask_b))
 → p_a,p_b      = expander(z_a), expander(z_b)      # discarded after pretraining
 → cons_loss    = VICReg(p_a, p_b)                  # invariance+variance+covariance
 → total        = recon_loss + consistency_lambda · cons_loss
```
Decoder **and** expander are discarded after pretraining. Downstream anomaly
scores come **only** from encoder embedding distance — never reconstruction
error, never any loss term.

### Distinct from A1
A1 = VICReg on two **augmented full** windows, **no** reconstruction, **no**
masking. B = reconstruction + consistency on two **masked** views. Same anti-
collapse machinery, different objective.

### Pretraining data — healthy-only is the PREFERRED PRIMARY for B
Unlike A/A1, B reconstructs the signal. On **mixed** (healthy+abnormal) data the
decoder can learn to reconstruct pathology well, which risks **normalizing
danger** (abnormal beats become "reconstructible" and thus look normal to the
encoder). Therefore:

| Arm | Primary pretraining data | Robustness ablation |
| --- | ------------------------ | ------------------- |
| A (`ntxent`) | Mixed unlabeled | `--healthy-only` |
| A1 (`vicreg`) | Mixed unlabeled | `--healthy-only` |
| **B (`mae_consistency`)** | **`--healthy-only` preferred** | mixed (note the caveat) |
| **B1 (`mae_subject_contrastive`)** | `--healthy-only` if run at all | mixed = worst case |

The driver prints a `[WARN]` when B is run without `--healthy-only`.

### Sanity check beyond loss
Report **embedding geometry**, not just reconstruction loss: does B produce a
**tighter healthy cloud / larger danger distance** than A0 under the fixed
scorer? A good reconstruction loss with a poorly-conditioned embedding space is
a failure for our Mahalanobis/kNN head. Track `embedding_std` (collapse monitor)
during pretraining and healthy-compactness at eval.

---

## 3. B1 (ablation) — masked recon + subject/record contrastive

Kept **only as an ablation** (`MaskedSubjectContrastiveModel`). It borrows Yu's
ZEROSHOT subject term: positives share a subject/record id. This directly tests
the hypothesis we suspect is harmful for AD — that subject-level alignment pulls
normal and abnormal windows from one record together.

- Run preferably with `--healthy-only` (so same-subject positives are at least
  all baseline morphology).
- Expected finding to report either way: whether the subject term **helps or
  hurts** the personalized danger separation vs the primary B.

Thesis sentence:

> Because subject/record contrastive positives can accidentally pull abnormal
> and healthy windows from the same recording closer together, the primary
> masked arm (B) uses reconstruction plus non-contrastive same-window
> consistency. Subject contrastive pretraining (B1) is retained only as an
> ablation, preferably healthy-only.

---

## 4. Commands

Build the 8 s / 125 Hz window index first (see `ZEROSHOT_CLUSTER_RUN_NOTES.md` §1).

### B (primary) — healthy-only preferred
```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/mae_consistency_seed${SEED} \
    --ssl-objective mae_consistency \
    --healthy-only \
    --mask-ratio 0.75 --mask-patch-size 25 \
    --consistency-lambda 1.0 \
    --vicreg-expander-dims 512,512,512 \
    --epochs 100 --batch-size 256 --lr 3e-4 \
    --num-workers 4 --seed ${SEED} --device cuda
done
```

### B1 (ablation) — subject contrastive, healthy-only
```bash
for SEED in 0 1 2; do
  python Layer3/tools/pretrain_encoder.py \
    --window-index Results/layer3/window_index/layer3_windows_8s_125hz.csv \
    --checkpoint-dir Results/layer3/pretrain/mae_subject_contrastive_seed${SEED} \
    --ssl-objective mae_subject_contrastive \
    --healthy-only \
    --mask-ratio 0.75 --mask-patch-size 25 \
    --subject-contrastive-lambda 0.3 --subject-col record_id \
    --epochs 100 --batch-size 256 --lr 3e-4 \
    --num-workers 4 --seed ${SEED} --device cuda
done
```

Phase 1 evaluation is identical for both (same fixed scorer): pass the resulting
`encoder_last.pt` to `run_beat_validation.py --phase1-arms a0,layer3` exactly as
in the cluster notes §5. The decoder/expander are not needed at eval.

---

## 5. Pilot ordering (resource-aware)

B/B1 are **not** in the first pilot. Order stays:

```text
§0 gate  →  A0 vs A (NT-Xent) one-seed MIT-BIH pilot  →  inspect
        →  only then multi-seed A / A1 / B (+ B1 ablation)
```

Freeze this B/B1 definition now; do **not** let it delay the A0-vs-A pilot.
Underpowered to finely rank A vs A1 vs B on 13 MIT-BIH gold records
(MDD ~10 pp) — frame the campaign as **A0 vs learned representation** plus an
SSL-family robustness sweep, not a fine ranking.

---

## 6. Smoke test (local, no CUDA needed for compile)

```bash
python -m py_compile Layer3/pipeline/layer3_masked_ssl.py Layer3/tools/pretrain_encoder.py
# torch-dependent forward/backward smoke (run where torch is installed, e.g. cluster):
python Layer3/pipeline/layer3_masked_ssl.py   # asserts finite loss + mask fraction for B and B1
```
