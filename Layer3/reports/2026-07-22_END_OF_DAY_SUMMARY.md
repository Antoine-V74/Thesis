# 22/07 — Summary (end of the day)

**Date:** Tuesday 22 July 2026 (end of day)
**Scope:** Everything discussed after the "the deterministic + handcrafted stack
is the more suitable approach, and you *proved* it" turn — i.e. the L3 strategy
review, the "what else could L3 have been" question, the human/pig reframing, the
two-label-questions insight that produced **Arm C**, and today's follow-ups
(A/B translation intent, windows, ZEROSHOT-SSL critique, calibration validity).

This is a **discussion/decision record** for the supervisor meeting, not a code
doc. Canonical technical docs it points to:
- `LAYER3_COMPLETE_STATUS_C.md` (Arm C deep dive)
- `LAYER3_ARM_C_SUPERVISED_SPEC.md` (Arm C design freeze)
- `LAYER3_CONSOLIDATED_SUMMARY.md` (medium overview)
- `LAYER3_SUPERVISOR_SUMMARY.md` (2-page brief)

---

## 0. TL;DR — decisions made today

1. **L1 + L2 remain the deployable core.** Learned Layer 3 is exploratory. This
   is now stated as a *result we proved*, not an assumption.
2. **A0 beating SSL is an in-domain finding.** It does **not** by itself kill the
   SSL-for-transfer rationale, because the transfer case was never tested.
3. **The pivotal insight:** "label-free at **deployment**" does **not** imply
   "label-free at **pretraining**." SSL threw away public labels we actually have.
4. **→ Arm C created and implemented:** supervised-contrastive (SupCon) pretrain
   on public labels, then the *same* label-free per-record healthy calibration.
5. **Calibration stays valid — and mandatory — for Arm C** (see §7). It is the
   invariant decision rule across all arms; it is what keeps Arm C fair.
6. **Windows unchanged:** Arm C uses the frozen 8 s @ 125 Hz protocol like every
   arm. 1 s is a later shared ablation, not a C-specific choice.
7. **Honest open gaps** are a cross-setup **transfer test** and **stim-artefact
   robustness**, not "a bigger model."

---

## 1. The core reframing: in-domain vs transfer

- The MIT-BIH A0-vs-SSL pilot is **within-domain** (train MIT-BIH, test MIT-BIH).
  Handcrafted features naturally win there — there is no domain shift to punish
  them.
- The reason SSL was chosen was **transfer**: pretrain on public human ECG →
  per-subject calibration → deploy on a different setup / electrode / species.
  That hypothesis was **never tested** by the in-domain pilot.

> Honest statement: *"On in-domain human ECG, learned representations add nothing
> over features. The cross-setup transfer question SSL was chosen for is still
> open."* — not "SSL failed."

Also flagged: early runs were short (~5–8 min). Bank the negative as
**exploratory short-run**, not "SSL fails."

---

## 2. Thesis scope grounding (from the PDFs)

Reading `Summary_ECG_Jane.pdf`, `ECG_Personalized_Anomaly_Detection_v5.pdf`,
`Layer3-2.0.pdf`, `Literature_Review_Outline.md` reframed the whole question:

- The thesis is an **ECG-triggered closed-loop cardiac-assist control + safety**
  project (MyoNeural Actuator / cardiomyoplasty for heart failure), on
  **TDT/Ripple real-time hardware**, deploying to **rat**.
- ECG safety is **one module packaged before the pressure-feedback / TDT
  controller** work — not "compare SSL encoders on ECG."
- Human PhysioNet data is an explicit **proxy**; deployment reality is rat,
  single-lead, anaesthetised, with a **stimulation artefact injected by the
  device itself**.
- A **T0–T4 calibration protocol** already exists; **T2 (stable pre-stimulation
  baseline)** is the personalization reference.

Consequences:
- **Heavy/advanced ML is disqualified for the real target, not just "not now":**
  rat RR ≈ 120–240 ms on TDT hardware — a latency-heavy encoder cannot make a
  per-beat permit decision in a fraction of a rat cardiac cycle.
- **The SSL-for-transfer instinct is already implemented — as T2 calibration,**
  not as the encoder. The transferable asset is *re-fit the healthy reference on
  the current animal/session*. That is exactly why A0 (features + the same
  personalized scorer) already competes.

---

## 3. Human / pig-only lens (rats set aside)

What **changes** if we imagine only human/pig stimulation:
- **Latency argument disappears** (human RR ≈ 600–1000 ms; measured encode
  ≈ 0.169 ms/beat). A real encoder fits comfortably in the cardiac cycle.
- **Human data stops being a proxy** — MIT-BIH etc. are *in-domain*. The
  human→rat domain shift that justified transfer-robust representations
  evaporates for human (shrinks to a modest gap for pig).
- **Labels are abundant (human)** → supervised methods become fully viable.

What does **not** change:
- A0 still beat SSL in-domain (now *more* damning — can't excuse with "it'll
  transfer to rat").
- Personalization > representation (the ZEROSHOT lesson).
- Danger is still **gross-rhythm** (VT/VF/noise) — hand-engineerable = L1/L2 turf.
- Stim-artefact problem is species-independent.
- Safety framing (false-permit-first, inhibit-only, conformal caveat) unchanged.

**Implication:** for pure **human**, the naturally best-matched *advanced* method
is not SSL anomaly detection — it is an **ICD/AED-style supervised, real-time
shockable-rhythm discriminator personalized to the patient's baseline**, which is
**what Layer 2 already approximates**. Real-time danger classifiers already exist
(ICD/AED rate + onset + stability + morphology/wavelet discrimination;
VF detectors — Amann 2005 review; Hannun 2019 as a supervised ceiling). They
**validate L1+L2**, they don't replace it.

SSL/AD's honest surviving niche: **pig transfer** (human train → pig deploy +
calibrate on the pig's own baseline) and **unknown-danger advisory veto**.

---

## 4. The pivotal insight — two "no labels", only one is true

This is the turn that produced Arm C.

| Time | Labels? | Binds what |
| --- | --- | --- |
| **Pretraining** (offline, public ECG) | **Yes** (abundant) | the *representation* — free to use labels |
| **Deployment / calibration** (sensor on patient) | **No** (only assumed-healthy window) | the *decision rule* — must stay one-class |

The reasoning "deployment is label-free → use anomaly detection" is **correct for
the decision rule**. But the extra step —

> "label-free at deployment" ⟹ "label-free (self-supervised) at pretraining"

— is **false**. The label-free constraint binds the *decision*, not the
*representation*. SSL **discarded usable public labels** for no necessity.

**Better-matched design (= Arm C):**
> Supervised / supervised-contrastive pretraining on public labels
> → freeze encoder → per-patient **label-free** one-class calibration.

It uses the labels we have, respects label-free deployment, and produces an
embedding where danger already sits far from normal — the geometry the
Mahalanobis/kNN veto wants. ZEROSHOT supports this: their **supervised** linear
model (~0.964) slightly **beat** their SSL (~0.954).

The L3 question was therefore too narrow:
- **As run:** "Does *self-supervised* representation beat features?" → No (A0 won).
- **What matters:** "Does *any* representation trained on the labels we have beat
  features, when deployed with label-free calibration?" → **Arm C tests this.**

---

## 5. Why ZEROSHOT used SSL (correcting a common conflation)

A widely-stated explanation ("they used SSL because there were no labels at
deployment") is **half wrong**. Zero labels *for this patient* justifies the
**one-class head**, not SSL for the **encoder**. You can pretrain the encoder
supervised on *other* subjects and still deploy label-free.

The **actual** reasons ZEROSHOT used SSL:
1. Scale to large **unlabeled** corpora (cross-subject labels are expensive).
2. Foundation-model framing (one encoder, personalize only the head).
3. A scientific claim: near-supervised anomaly performance with zero labels.

And critically, ZEROSHOT itself showed **supervised ≈ (slightly better than)
SSL** — which is the entire basis for Arm C. The **transferable** idea is
**personalized baseline calibration**, not the SSL encoder.

---

## 6. Fine-tuning the encoder vs calibrating the scorer

The idea "pretrain SSL → fine-tune the encoder on the small stable baseline →
then test" is **related but riskier** than the ZEROSHOT pattern, and for a safety
gate the risks dominate:

- **Data volume:** a 2–5 min baseline cannot fine-tune a ~700K-param encoder
  without overfitting. At most adapt **normalization affine params / a small
  adapter**.
- **Healthy-only adaptation cannot teach danger in the new domain.** It
  re-centres "normal" but can silently **shrink the healthy↔danger margin** — it
  fixes the easy half and can damage the half that matters.
- **Auditability:** per-session encoder fine-tuning gives every deployment a
  different model with no fixed operating characteristics — bad for an
  inhibit-only device.

**Safer form (what the thesis uses):** freeze encoder → per-session healthy
baseline calibration; treat encoder adaptation as a **guarded future ablation**
(norm/adapter only) with drift + contamination checks.

---

## 7. Why calibration is still valid for Arm C  *(requested)*

**Yes — calibration is not only valid for Arm C, it is required, and it is what
keeps Arm C honest.** Three points:

**7.1 Pretraining and calibration answer different questions.**

| Stage | Role | Depends on how encoder was trained? |
| --- | --- | --- |
| Pretrain (SupCon, labels, offline) | shapes the embedding *geometry* | — |
| **Calibration (per-record healthy baseline)** | finds *where this subject's normal sits* + sets the threshold | **No** |

The encoder gives a coordinate system; calibration finds **where this subject's
healthy cloud lives in it** and how far is "too far." Supervised pretraining does
**not** give a transferable global threshold — distances are
subject/electrode-specific. So calibration is as necessary for C as for A0/A/B.

**7.2 Calibration is what makes Arm C *fair*, not cheating.**
Arm C changes **only the representation**. The decision rule stays the identical
**label-free** per-record healthy Mahalanobis/kNN + conformal used by every arm:
- labels touch the **encoder** (offline),
- **no danger label** touches the **threshold** (deploy).

If one scored with SupCon logits or the class boundary at runtime, C would become
a deployed supervised classifier — a different method that violates the deploy
contract and is disqualified. **Runtime = encoder distance only.**

**7.3 Honest caveat (true for every arm, incl. C).**
Calibration controls the **healthy false-inhibit budget** (≈α), **not**
danger-side performance. SupCon may make the healthy cloud tighter/better-
conditioned for Mahalanobis (a plausible plus), but it optimises separation on
the *training* danger distribution — a novel danger morphology at deploy could
still land near normal. Calibration can't fix that; it never sees danger. Danger
performance is **measured** (false-permit + CAV), never assumed.

**One-line version:**
> Freeze the SupCon-pretrained encoder → per-record healthy Mahalanobis/kNN +
> conformal α → distance-only permit/inhibit. Same contract as every arm; labels
> only shaped the space, never the threshold.

---

## 8. What Arm C is (and its status)

- **Objective:** Supervised Contrastive (SupCon; Khosla 2020) on the
  `safety_group` labels already in the window index.
- **Default class map:** `normal` = NORMAL; `unsafe` = DANGEROUS+NOISE;
  `benign` = BENIGN_ABNORMAL; `drop` = AF_CONTEXT.
- **Deployment contract:** identical to all arms — frozen encoder, per-record
  healthy calibration, conformal α=0.10, **no labels at deploy**, projection head
  discarded, never score with logits.
- **Protocol:** 8 s @ 125 Hz, 13 MIT-BIH gold, exclude gold from pretrain.
- **Status:** **implemented** (`Layer3/pipeline/layer3_supervised.py` +
  `--ssl-objective supcon` in `pretrain_encoder.py`); local 1-epoch smoke passed;
  **gold cluster pilot pending** (`cluster_jobs/07a`, `07b`).
- **Variants deferred:** C-ce (supervised cross-entropy) and C-oe (Deep SAD /
  outlier exposure) — spec only, run only if primary C looks promising.

Expected outcome (honest): C is the **best shot** at "ML adds value," but per the
ZEROSHOT pattern + the A0 result it may still only **tie A0** — and *"even a
supervised representation barely beats handcrafted features"* is itself a clean,
publishable finding, precisely because the deploy contract is identical.

---

## 9. What L3 could have been (alternatives *not* taken)

None is likely to beat A0 on this data, but some are better-matched / more
informative than four SSL variants:

| Tier | Option | Why it was a candidate | Verdict |
| --- | --- | --- | --- |
| 1 | **Deep SVDD** (compact healthy hypersphere) | Matches "tight healthy manifold for a strict veto" better than recon/contrastive | Cleaner AD arm, likely still ≈ A0; deferred as C-oe idea |
| 1 | **HR-adaptive / drift-aware normal (Carrera-style)** | Non-stationary healthy baseline tracking HR; the real bridge to rat | Most defensible L3 upgrade; strong future direction |
| 2 | **Supervised ceiling** (labels → best possible learned detector) | Measures whether *any* learning beats A0 | **Chosen** — became Arm C (SupCon) |
| 2 | **Open-set / OOD classifier** | Known danger + "none of the above" | Bridges rules and AD; backlog |
| 3 | **Sequence models (TCN/temporal transformer)** | Rhythm dynamics | Low marginal value (RR/L2 already carry rhythm) |
| 3 | **ECG foundation models** | Modern, large pretrained encoders | Heavy, human-domain; ZEROSHOT says won't move the needle vs personalization; appendix at best |

The single most informative experiment still *missing*: a **cross-setup transfer
test** (pretrain/fit on dataset A → calibrate + evaluate on dataset B / different
lead) — the honest stand-in for human→rat and the actual test of the SSL
rationale.

---

## 10. A/B arms and the pig/rat translation story

- It is fair to say **A / A1 / B / B1 were chosen with animal translation in
  mind** — a label-free encoder + per-session healthy calibration is the only
  deploy contract that could later move to pig/rat (no danger labels on the
  animal). This is **design intent**, not a demonstrated transfer result.
- It is **not** fair to claim A/B *achieved* pig/rat generalization; Phase 1 only
  tests in-domain human MIT-BIH under the fixed scorer.
- The **baseline-stable fitting idea must be kept** — it is the spine: the only
  transferable deploy contract, what makes Arm C fair, and consistent with
  Layer 2's personalization. What can still be ablated: Mahalanobis vs kNN, PCA,
  conformal α, baseline length/cleanliness.

---

## 11. Windows (why Arm C is 8 s, not "1 s/8 s special")

The arm ladder varies **only the representation objective**, not the window:
- **8 s @ 125 Hz** is the frozen primary for **all** arms, incl. C.
- **1 s** is a later **shared** morphology ablation across A0/SSL/C, run after the
  8 s gold pilot — not a C-specific schedule.
- Changing objective *and* window at once would confound the comparison.

---

## 12. Honest open gaps (bigger than "a better algorithm")

1. **Cross-setup transfer test** — the experiment SSL was built for; still unrun.
2. **Stimulation-artefact / lead-off robustness** — the device injects the very
   artefact that breaks ECG sensing; currently no citation or test. For an MNA
   safety thesis this is a larger gap than any encoder choice.
3. **Drift-aware baseline (Carrera-style)** — matters most for rat rate drift.

---

## 13. Next steps

```text
When the cluster is available:
  1. Locked gold pilot: A0 / A / A1 / B (+B1 ablation) on MIT-BIH gold, 8 s.
  2. Arm C: 07a (SupCon pretrain, exclude gold) → 07b (Phase 1 a0,layer3).
  3. Fill the compare table: false-permit DANGEROUS + record-bootstrap CI + CAV vs A0.
  4. Oracle vs layer1_adaptive_gated for the pipeline-relevant claim.
  5. Only if C helps → C-ce / C-oe variants.

Higher-value than a new arm:
  - Cross-dataset transfer test (pretrain A → calibrate+eval B).
  - Stim-artefact / lead-off stress test.
```

---

## 14. One-paragraph framing for supervisors

> Layer 1 + Layer 2 are the deployable ECG-safety core, in the established
> ICD/AED tradition of deterministic rate/morphology danger discrimination; we
> *proved* rather than assumed this by showing that, in-domain, handcrafted
> features under a personalized healthy-baseline scorer (A0) are not beaten by
> self-supervised encoders. Because "no labels at deployment" constrains only the
> decision rule and not the representation, we added Arm C: a supervised-
> contrastive encoder trained on the public labels we already have, deployed under
> the identical label-free per-record healthy calibration. Arm C is the supervised
> ceiling for "can learning add value?"; calibration remains the invariant,
> auditable decision rule across all arms. The genuinely open questions are
> cross-setup transfer and stimulation-artefact robustness, not model capacity.

---

*End of 22/07 summary.*
