# Literature Review Outline — uniform citations

## Format de citation (identique partout)

**Dans l’outline et le texte :**

```text
**Auteur et al., Année** — *Revue*. Une phrase de pertinence.
```

**Dans la bibliographie finale :**

```text
Auteur AB, Auteur CD, Auteur EF. Titre complet de l’article. Nom de la revue. Année;Volume(Numéro):Pages. doi:...
```

- Toujours **Auteur et al., Année** en tête (pas le titre seul).
- Toujours **une phrase** après le tiret — pas de puce sans explication en §9.
- Préprints : indiquer *bioRxiv* / *arXiv* + DOI.

---

## Verdict global

**§1–5 :** logique clinique claire.  
**§6 :** bon pivot « classifieurs insuffisants ».  
**§7 :** bonne liste AD compacte — renommer (voir ci-dessous).  
**§8 :** bon contenu features, mais **manque le paragraphe « personalized + online + T2 »** qui justifie Layer 2.  
**§9 :** bonne structure ; ajouter **CLOCS** et **une phrase par paper**.

**Corrections urgentes :**

- **Thireau et al., 2008** → sortir de §6c ; mettre en **§8a** (HRV rat), pas « generalization ».
- **§2** : deux fois « b. » → renommer le 3e en **2c** (Mario).
- **§7** : titre = liste AD, pas « From classification to… » (le lien §6→§7→§8 va dans **1 phrase de transition** en début de §8).
- **§8** : ajouter **Clifford et al., 2012** (SQI) et **Carrera / de Chazal / T2** en §8d (justification personalized).
- **§6b** : une seule revue DL suffit (Ebrahimi **ou** overview 2017–2023).
- **Jiang** cité en §7 et §9 : OK — §7 = AD ; §9b = même logique en embedding space.

---

## 1. Mechanical assistance for advanced heart failure

**Main idea:** End-stage HF motivates mechanical assist; soft robotics and artificial muscles explore less rigid, biologically inspired actuation.

### 1a. Why mechanical assist and why soft / muscle-based approaches

- **Weymann et al., 2023** — *Advanced Materials*. Review of artificial muscles and soft robotic devices for end-stage HF; compression/twisting assist, electrothermal and biohybrid actuators.
- **Roche et al., 2017** — *Science Translational Medicine*. Soft robotic sleeve synchronized to native beat; no blood contact; porcine HF model.

---

## 2. Dynamic cardiomyoplasty as historical precedent

**Main idea:** Muscle-wrap assist with R-wave sync was tried clinically; limitations motivate MNA and strict ECG safety.

### 2a. Promising but challenging idea

- **Carpentier & Chachques, 1985** — *The Lancet*. First successful clinical dynamic cardiomyoplasty.
- **Salmons & Jarvis, 2009** — *Basic and Applied Myology*. Historical review; fatigue, timing, limited haemodynamic benefit.

### 2b. Clinical procedure and limitations

- **Badhwar et al., 1998** — book chapter *Dynamic Cardiomyoplasty* (verify editors/venue). Clinical overview, selection, outcomes.
- **Geddes et al., 1993** — *Pacing and Clinical Electrophysiology*. Stimulation timing relative to QRS strongly affects haemodynamic gain.

### 2c. Conditioned muscle loses power and velocity

- **Barbiero et al., 1999** — *Basic and Applied Myology*. Demand dynamic cardiomyoplasty; conditioned muscle loses shortening velocity and power.

---

## 3. MyoNeural Actuator (MNA)

**Main idea:** MNA improves fatigue resistance and controllable muscle recruitment; it does not solve sensing, timing, control, or safety by itself.

### 3a. Fatigue-resistant controllable biological actuation

- **Song et al., 2025** — *bioRxiv*. Sensory reinnervation MNA; improved fatigue resistance; rodent force tracking.

---

## 4. ECG-synchronized cardiac assistance

**Main idea:** Assist must be triggered from native electrical rhythm; ECG is a control signal, not only a diagnostic trace.

### 4a. R-wave / electrogram triggering

- **Roche et al., 2017** — *Science Translational Medicine*. Sleeve actuation synchronized to cardiac cycle.
- **Vasilyev et al., 2020** — *Journal of Medical Devices*. Epicardial electrogram R-wave thresholding for soft robotic VAD sync; refractory blanking.

---

## 5. R-peak / QRS detection

**Main idea:** Fast causal detection (Layer 1) before any safety gate.

### 5a. Classic causal QRS pipeline

- **Pan & Tompkins, 1985** — *IEEE TBME*. Real-time QRS detection baseline.

### 5b. Wavelet / morphology-aware detection

- **Martínez et al., 2004** — *IEEE TBME*. Wavelet delineation; atypical QRS morphology.

### 5c. Adaptive thresholding

- **Hamilton, 2002** — *Computers in Cardiology*. Open-source ECG analysis refinements.

**Optional:** **Clifford et al., 2012** — *J Electrocardiol*. SQIs — often better here or §8e as bridge L1→L2.

---

## 6. Supervised ECG classification and its limits

**Main idea:** Strong on human benchmarks; weak when platform, patient, setup, or state change — motivates AD (§7) not a bigger classifier.

### 6a. Classical heartbeat classification

- **de Chazal et al., 2004** — *IEEE TBME*. RR + morphology; inter-patient evaluation; per-subject normalization.
- **Sivapalan et al., 2022 (ANNet)** — *IEEE TBCAS*. Supervised edge ECG anomaly detection; useful contrast but label-dependent, not our unsupervised session-baseline veto.

### 6b. Deep learning arrhythmia detection

- **Ebrahimi et al., 2020** — *Expert Systems with Applications: X*. DL ECG arrhythmia review.
- **Ansari et al., 2023** — *Frontiers in Physiology*. Overview of DL for ECG arrhythmia detection and classification (2017–2023).

### 6c. Cross-dataset generalization

- **Ballas & Diou, 2023** — *IEEE TETCI*. Domain shift across datasets.
- **Li et al., 2026** — *Scientific Reports*. Cross-dataset cardiac signal degradation; public benchmarks do not reproduce experimental lead placement, hardware, or stimulation artefacts.

---

## 7. Anomaly detection for ECG and biosignals (literature list)

**Main idea:** Short survey of AD papers — feasibility of « model normal, score deviation ». **No design choice yet.**

**Transition (1 sentence end of §7):** *These works show AD is viable; Section 8 explains why stimulation safety requires personalized, online, session-baseline monitoring (Layer 2), and Section 9 extends the same scorer to SSL embeddings (Layer 3).*

### 7a. Distance / baseline scoring

- **Carrera et al., 2019** — *Pattern Recognition*. Per-user dictionary of normal beats; sparse distance; online wearable monitoring.
- **Thill et al., 2019** — *ICPRAM*. LSTM forecast + Mahalanobis on prediction residuals; MIT-BIH.

### 7b. Reconstruction and hybrid scores

- **Kapsecker & Jonas, 2025** — *PLOS Digital Health*. Federated linear AE; reconstruction + embedding distance; on-device personalization.

### 7c. Self-supervised + anomaly

- **Jiang et al., 2024** — *arXiv:2404.04935*. Normal-only SSL ECG; masked restoration; clinical-scale AD.

---

## 8. Interpretable personalized feature monitoring (Layer 2)

**Main idea:** After §6–§7: AD must be **personalized** (subject/setup), **online** (runtime windows), anchored to **T2 baseline** (operating HF state). Layer 2 = handcrafted features + Mahalanobis/kNN.

**Bridge paragraph (paste in thesis):**  
*Population classifiers (§6) and generic AD (§7) are insufficient for closed-loop stimulation: the normal reference must be fit on the stable pre-stimulation baseline of the current session (T2), updated only under verified safe conditions, and computed online beat-by-beat or window-by-window. Layer 2 implements this with interpretable features.*

### 8a. Rhythm and HRV features

- **Task Force, 1996** — *Circulation* / *European Heart Journal*. HRV standards (SDNN, RMSSD, spectral bands).
- **Thireau et al., 2008** — *Experimental Physiology*. Rodent HRV frequency bands for rat RR features.

### 8b. Morphology and signal-level features

- **de Chazal et al., 2004** — *IEEE TBME*. Morphology + interval features (also §6a).
- **Martínez et al., 2004** — *IEEE TBME*. Wavelet morphology (also §5b).
- **Richman & Moorman, 2000** — *Am J Physiol*. Sample / approximate entropy.

### 8c. Personalized monitoring precedent

- **Carrera et al., 2019** — *Pattern Recognition*. Per-user normal model; online HR adaptation (design link from §7a).

### 8d. T2 safe operating baseline (thesis)

- **Thesis protocol (T0–T4).** Calibrate on T2 (stable HF pre-stimulation), not T0 or population norms.

### 8e. Signal quality hard gate (SQI ensemble)

No single SQI is robust to every artifact, so we combine complementary indices
and inhibit if any one flags poor quality (implemented as the opt-in
`compute_sqi_ensemble` path: kSQI, pSQI, bSQI + FFT hf/lf ratios).

- **Clifford et al., 2012** — *J Electrocardiol*. SQIs; inhibit before anomaly scoring.
- **Behar et al., 2013** — *IEEE TBME*. ECG SQI ensemble for false-alarm reduction; motivates combining multiple complementary indices rather than one.
- **Li et al., 2008** — *Physiological Measurement*. bSQI: agreement between two independent QRS detectors as a robust signal-quality index; used here to flag untrustworthy beat trains before rr__/morph__ features.

### 8f. Onset & stability rate discriminators (ICD/AED lineage)

Decades of implantable-defibrillator engineering already discriminate dangerous
organised VT from SVT/AF at the same rate using rate-zone branching plus onset
and stability criteria — a handful of interpretable features. Layer 2 adopts
the same discriminators (opt-in `compute_onset_stability`: onset_accel_frac,
stability_ms, tachy_fraction; `classify_rate_zone`).

- **Swerdlow et al., 1994** — *Circulation*. Onset and stability SVT/VT discriminators for implantable defibrillators; interpretable rate-based safety logic.
- **Task Force, 1996** — *Circulation*. HRV standards underpinning the RR variability features reused as stability (also §8a).

### 8g. Operating point and persistence (Neyman-Pearson + X-of-Y)

The runtime threshold is fit label-free on the healthy baseline; the operating
point and its danger-leakage cost are audited offline against a
Neyman-Pearson bound, and the 1-in-8 cadence is formalised as an inverted
ICD-style X-of-Y persistence rule.

- **Scott & Nowak, 2005** — *IEEE Trans. Information Theory*. Neyman-Pearson classification: bound the (dangerous) Type-I error, minimise the other; frames the false-permit-budgeted operating point.
- **Swerdlow et al., 1994** — *Circulation*. Sustained-rate / duration (X-of-Y) detection criteria; the safety gate inverts this to require persistent SAFE beats before permitting (also §8f).

---

## 9. SSL representations and embedding-space anomaly scoring (Layer 3)

**Main idea:** Same AD logic as Layer 2, but embeddings from SSL pretraining replace handcrafted features. Compare encoders (A / A1 / B) under **fixed** Mahalanobis/kNN scorer.

### 9a. Self-supervised ECG representation learning

- **Kiyasseh et al., 2021 (CLOCS)** — *ICML*. Contrastive learning across space, time, and patients; motivates patient-aware contrastive pretraining (Arm A).
- **Gopal et al., 2021 (3KG)** — *ML4H*. Physiology-aware augmentations for contrastive ECG; constrains augmentation design.
- **Qin et al., 2026 (CoRe-ECG)** — *arXiv:2604.11359*. Unified contrastive + reconstructive SSL; frequency augmentation + spatio-temporal masking; hybrid pretraining reference.
- **Manimaran et al., 2024 (NERULA)** — *arXiv:2405.19348*. Single-lead dual-pathway: masked + inverse-masked non-contrastive + reconstruction; motivates Arm B masking.
- **Chen et al., 2020 (SimCLR)** — *ICML*. NT-Xent contrastive framework; methods cite for Arm A baseline.
- **Bardes et al., 2022 (VICReg)** — *ICLR*. Non-contrastive invariance + variance/covariance; Arm A1.

### 9b. Applying AD in embedding space (same scorer as Layer 2)

- **Yu, 2026** — *Research Square* preprint. MAE + subject-contrastive SSL → per-subject Mahalanobis on healthy iEEG; zero labels; template for Arm B + scoring pipeline.
- **Jiang et al., 2024** — *arXiv:2404.04935*. Normal-only SSL ECG AD at clinical scale; proves ECG embedding AD is feasible (also §7c).
- **Kapsecker & Jonas, 2025** — *PLOS Digital Health*. Embedding distance + personalization on single-lead mobile ECG (also §7b).

---

## 10. Synthesis (optional short section)

**Main idea:** No prior work combines layered permit/inhibit, T2 personalization, SSL encoder comparison under fixed scorer, and policy-aware false-permit metrics for ECG-triggered stimulation.

---

## Informations manquantes — merci de confirmer


| #   | Ce que tu as écrit               | J’ai besoin de                                                                     |
| --- | -------------------------------- | ---------------------------------------------------------------------------------- |
| 1   | Salmons, cardiomyoplasty review  | **Salmons & Jarvis, 2009** — titre exact + revue (prob. *Artificial Organs*) ?     |
| 2   | Badhwar, Dynamic Cardiomyoplasty | Éditeurs / année / chapitre complet ?                                              |
| 3   | Mario et al., 1999               | Titre exact (*Demand dynamic cardiomyoplasty* ?) + revue                           |
| 4   | Ebrahimi 2020 review             | Titre exact + revue (confirmé ?)                                                   |
| 5   | Overview 2017–2023               | **Auteurs + titre complet + revue** (tu cites les deux surveys — garder un seul ?) |
| 6   | Task Force 1996                  | OK tel quel (Malik et al., 1996) — tu veux « Task Force » ou auteurs ?             |
| 7   | CoRe-ECG                         | Confirmé : **Qin et al., 2026**, arXiv — as-tu une version publiée ?               |
| 8   | NERULA                           | Confirmé : **Manimaran et al., 2024**, arXiv — conférence/journal final ?          |
| 9   | Vasilyev                         | Confirmé **Vasilyev et al., 2020**, *J Med Devices* — c’est bien ton PDF ?         |


---

## Papers à ajouter (suggestions)


| Paper                       | Section      | Why                                |
| --------------------------- | ------------ | ---------------------------------- |
| **Clifford et al., 2012**   | §8e          | SQI — déjà suggéré                 |
| **Behar et al., 2013**      | §8e          | Ensemble SQI (kSQI/pSQI ensemble)  |
| **Li et al., 2008**         | §8e          | bSQI two-detector agreement        |
| **Swerdlow et al., 1994**   | §8f          | ICD onset/stability discriminators |
| **Scott & Nowak, 2005**     | §8g          | Neyman-Pearson operating point     |
| **Richman & Moorman, 2000** | §8b          | Entropie — dans Safety_Supervision |
| **Hannun et al., 2019**     | §6b          | Si tu retires une des deux reviews |
| **Li et al., 2026**         | §6c          | Si pas déjà dans ton Word          |
| **Ruff / Ibrahim**          | §9b optional | Deep SVDD ablation — une ligne max |


