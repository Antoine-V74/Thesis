# ECG-Based Safety Supervision for Closed-Loop Cardiac Assist: From Real-Time QRS Detection to Personalised Anomaly Monitoring

*Master’s Thesis Literature Review*

## 1. ECG as a Control Signal in Cardiac Assist

Electrocardiography occupies a dual role in medicine. In its conventional diagnostic application, the ECG is interpreted offline by a clinician to characterise cardiac function, identify arrhythmias, and guide treatment decisions. In closed-loop cardiac assist, however, the ECG serves an entirely different purpose: it becomes a real-time control signal. The distinction is fundamental. A diagnostic system may tolerate seconds of processing latency, uncertain outputs on ambiguous signals, and occasional misclassification corrected by clinical oversight. A closed-loop stimulation system can tolerate none of these: it must make a binary decision — permit or inhibit stimulation — on every cardiac cycle, with latency measured in milliseconds, in conditions it may never have encountered during development, and with consequences for patient safety if it is wrong.

This review is motivated by the specific requirements of an ECG-synchronised stimulation system designed for use with a MyoNeural Actuator (MNA) in a cardiomyoplasty-inspired cardiac assist application \[Song et al. 2025\]. In this system, a skeletal muscle actuator is wrapped around the ventricle and stimulated in synchrony with the cardiac cycle to augment haemodynamic output. The ECG provides two things: the timing reference for stimulation (the R-peak triggering each contraction) and the safety signal (the rhythm assessment determining whether stimulation is safe at all). Three sub-problems follow directly from this framing.

The first is *timing accuracy*: R-peak detection must be fast, precise, and robust under the degraded signal conditions of an operating theatre or animal laboratory, where electrode motion, anaesthetic agents, respiration-induced baseline wander, EMG contamination from skeletal muscle activation, and stimulation artefacts all corrupt the recording. The second is *safety gating*: the system must inhibit stimulation when the underlying rhythm is unsafe, even when no labelled example of that specific arrhythmia has ever been observed from this animal. The third is *adaptive calibration*: signal characteristics and rhythm change throughout an experiment as a function of anaesthetic depth, haemodynamic state, electrode contact, and the progression of heart failure, so any fixed threshold or population-derived norm will drift out of validity.

None of these requirements are well served by the dominant paradigm in ECG research, which seeks ever-higher classification accuracy on large, clean, human-labelled benchmark datasets. The remainder of this review builds the case for an alternative approach: a layered, personalised safety supervisor combining deterministic QRS timing, interpretable feature-based anomaly detection, and self-supervised representation learning.

> This review argues that safe ECG-synchronised stimulation cannot be reduced to offline arrhythmia classification. Because the deployment setting involves severe domain shift, scarce labelled animal data, and safety-critical timing constraints, the appropriate architecture is a conservative personalised supervisor: deterministic R/QRS detection for low-latency timing, interpretable baseline-calibrated features for safety monitoring, and self-supervised anomaly detection as an additional learned veto layer.

## 2. Classical Real-Time QRS Detection

The problem of automatic R-peak detection has a long history, and the benchmark against which subsequent methods have been measured for four decades remains the algorithm of Pan and Tompkins \[Pan & Tompkins 1985\]. Their approach applies a sequence of deterministic digital signal-processing steps to the raw ECG: a bandpass filter isolating the QRS frequency content, a differentiator to emphasise slope, a squaring operation to make all values positive and amplify large peaks, a moving-window integrator to obtain a smooth envelope, and an adaptive dual-threshold scheme that tracks the noise and signal levels and applies refractory blanking after each detected event. Variants and improvements have been proposed over subsequent decades, most notably the thresholding refinements of Hamilton \[Hamilton 2002\], and these remain the engineering backbone of most real-time ECG monitoring systems.

**Latency considerations for rodent preparations.** Pan-Tompkins-style detection operates with a confirmation latency of approximately 80–100 milliseconds, arising primarily from the moving-window integration step. In human ECG at 60–80 beats per minute, this latency represents a small fraction of the RR interval. In the rat at 250–500 beats per minute, where RR intervals span only 120–240 milliseconds, the same confirmation delay consumes up to half the cardiac cycle. For same-beat stimulation triggering, this is problematic: the R-peak must be identified and acted upon in tens of milliseconds, not hundreds. The appropriate design in the rodent context therefore separates two functions: a fast candidate detector that produces a provisional R-peak timestamp with minimal latency for same-beat stimulation triggering, and a slower confirmation step — applied in the window immediately following each beat — that updates the RR supervisor and informs the safety gate for subsequent beats. The candidate detection does not need to be definitive; the confirmation step resolves ambiguity before the system makes a downstream safety decision.

Wavelet-based QRS delineation represents a parallel line of development. Rather than filtering for QRS specifically, wavelet decomposition of the ECG signal produces a multi-resolution representation that isolates different morphological features at different scales. The approach introduced by Martínez and colleagues \[Martínez et al. 2004\] uses the dyadic wavelet transform to delineate the onset, peak, and offset of P, QRS, and T waves simultaneously, enabling robust detection even when QRS morphology is atypical. This is directly relevant to closed-loop stimulation applications, where pathological states such as bundle branch block, ventricular hypertrophy, and ischaemia may alter QRS morphology substantially from the narrow, sharp morphology that threshold-based detectors are optimised to find.

**Signal quality as a bridge between detection and safety.** Signal quality assessment is a complementary and often underappreciated component of robust real-time ECG analysis. An R-peak detector that reports a peak but cannot assess whether that peak is trustworthy provides incomplete information for a safety-critical application. Clifford and colleagues formalised a suite of signal quality indices (SQIs) for ECG monitoring \[Clifford et al. 2012\]:

- *bSQI* — the agreement between two independent QRS detectors applied to the same signal. High disagreement indicates an unreliable or ambiguous ECG, even if each individual detector returns a result.
- *iSQI* — the inter-lead agreement in multi-lead recordings. Inconsistency between leads indicates artefact rather than true cardiac events.
- *pSQI* — the proportion of power in the QRS frequency band relative to total power. A low pSQI indicates that the QRS content has been overwhelmed by broadband noise or baseline wander.
- *basSQI* — a measure of baseline wander severity, derived from low-frequency content.

These SQIs serve as a critical bridge between Layer 1 (detection) and Layer 2 (safety gating): a low bSQI directly indicates that the R-peak timestamps should not be trusted, independent of any downstream rhythm analysis. Gross failures such as lead disconnection, signal saturation, and flatlines are appropriately caught at the SQI level — before they reach the Mahalanobis-distance computation of Layer 2 — since they produce extreme but diagnostically meaningless feature values. For the closed-loop stimulation context, the conservative rule is: low SQI triggers immediate inhibition and freezes further recalibration until signal quality recovers.

Several failure modes are specific to the stimulation context. T-wave oversensing occurs when the detector misidentifies the T-wave as an R-peak, resulting in a falsely shortened RR interval and inappropriate stimulation in the vulnerable period. Stimulation artefact is another important concern: the electrical pulse delivered to the skeletal muscle actuator produces a large transient in the ECG channel. Without explicit blanking in the milliseconds around each stimulus, the artefact can trigger a cascade of false detections. High-frequency EMG contamination from the contracting skeletal muscle overlaps the QRS frequency band and cannot be removed by the standard bandpass filter without also attenuating the QRS itself. These failure modes do not appear in standard ECG benchmark datasets and cannot be addressed by models trained on them; they must be handled explicitly in the signal-processing pipeline.

## 3. Machine Learning for ECG Arrhythmia Classification

The application of deep learning to ECG classification has produced remarkable results over the past decade. The landmark study of Hannun and colleagues \[Hannun et al. 2019\] demonstrated that a convolutional neural network trained on a large dataset of single-lead ambulatory ECG recordings could classify fourteen arrhythmia types at a level comparable to the average cardiologist, with narrower confidence intervals on several categories. Ribeiro and colleagues \[Ribeiro et al. 2020\] showed that deep learning models scale effectively to very large datasets: their study used over two million 12-lead ECG recordings from a Brazilian population to train a model identifying six abnormalities, demonstrating both the scalability of the approach and the influence of population characteristics on model performance. Systematic benchmarking on the publicly available PTB-XL dataset \[Strodthoff et al. 2021\] has established that modern architectures — residual CNNs, LSTM networks, and Transformer-based models — all achieve competitive performance on standard multi-label classification tasks, with task difficulty varying substantially by label type.

Competition benchmarks have played an important role in catalysing progress. The PhysioNet/Computing in Cardiology Challenge of 2017 \[Clifford et al. 2017\], focused on AF detection from short single-lead recordings, attracted hundreds of teams and established that machine learning methods substantially outperform traditional approaches on this specific task. The 2021 Challenge \[Reyna et al. 2021\], which extended the problem to 12-lead multilabel classification across twenty-seven rhythm types and multiple international datasets, provided the most comprehensive benchmarking exercise to date and highlighted substantial performance variation across recording sources.

Despite this progress, all of these systems share an implicit assumption that is difficult to satisfy in the stimulation context: that training and deployment occur within the same or closely related data distributions. The models are trained on large, carefully annotated datasets from human subjects, with standard lead configurations, at known sampling rates, and with expert-verified labels. Whether the same performance is maintained when any of these conditions change — as they inevitably do in a rat electrophysiology experiment — is a separate question, and one the benchmark paradigm is not designed to answer.

## 4. The Translation Gap: Domain Shift and Generalisation

The gap between benchmark performance and real-world deployment performance in ECG systems has been documented empirically and analysed theoretically. The canonical demonstration of the inter-patient generalisation problem was provided by de Chazal and colleagues \[de Chazal et al. 2004\], whose study of heartbeat classification established that performance metrics derived from randomly splitting data across patients dramatically overestimate the performance achievable on new, previously unseen patients. When the training and test sets are separated by patient identity — the correct evaluation for any deployment scenario — accuracy on minority arrhythmia classes falls substantially. The authors showed that per-patient normalisation of RR intervals and morphological features is necessary to achieve reasonable inter-patient performance, and this finding has shaped ECG machine learning evaluation practice ever since.

The cross-dataset problem is more severe. Ballas and Diou \[Ballas & Diou 2023\] specifically evaluated domain generalisation across ECG and EEG classification tasks, finding that models trained on one dataset degrade significantly when evaluated on another, and that standard domain adaptation methods provide only partial mitigation. Li and colleagues \[Li et al. 2026\], whose domain-generalisation framework is directly relevant to the present work, quantified this degradation and demonstrated that methods achieving strong within-dataset performance can lose eight percentage points or more of accuracy when transferred to a different source. Their analysis attributes the degradation to differences in acquisition equipment, subject populations, sampling conditions, and annotation conventions — all of which change between public benchmark datasets.

For the application considered in this thesis, the domain shifts are more severe than any studied in the literature cited above. Five distinct shifts stack on top of one another: *species shift* — human ECG morphology, heart rate, and frequency content differ fundamentally from rat ECG, where the resting heart rate is five times higher and QRS morphology is substantially different; *lead shift* — from twelve-lead clinical recordings to two-lead or fewer animal preparations, possibly including epicardial or cuff electrodes; *acquisition shift* — from hospital-grade ECG machines to research-grade streaming acquisition systems such as TDT/Pynapse; *state shift* — from awake ambulatory human subjects to anaesthetised animals in haemodynamically compromised states; and *artefact shift* — the deliberate introduction of stimulation artefacts into the signal by the very device the ECG is controlling. Stimulation artefacts are of particular importance: they appear synchronously with each stimulation event, directly within the physiologically active window of the cardiac cycle, and are not represented in any standard ECG classification dataset.

The combined effect of these shifts is that any model trained on public human ECG datasets and evaluated by conventional benchmark metrics provides essentially no guarantee of performance in an acute rat experiment. This is not a criticism of the models or the benchmarking paradigm — both are appropriate for their intended use — but it is a fundamental constraint on the applicability of supervised arrhythmia classification for the present application. The appropriate response is not to seek a more powerful model, but to reformulate the problem in a way that does not require cross-domain label transfer.

## 5. The Application Domain: Cardiomyoplasty and Closed-Loop Cardiac Assist

### 5.1 Dynamic Cardiomyoplasty

The idea of using a skeletal muscle to assist the failing heart has a history spanning more than three decades. Carpentier and Chachques introduced dynamic cardiomyoplasty in 1985 \[Carpentier & Chachques 1985\], wrapping the latissimus dorsi muscle around the ventricles and stimulating it in synchrony with the cardiac cycle to augment ventricular contraction. The technique generated substantial clinical interest in the late 1980s and 1990s, and several clinical trials were conducted. However, the procedure ultimately showed only limited and inconsistent haemodynamic benefit, and it did not achieve widespread clinical adoption \[Annals of Thoracic Surgery 1999\].

The lessons from cardiomyoplasty’s limited success directly motivate the design choices in the present work. Three limitations were central. First, native skeletal muscle fatigues rapidly under continuous electrical stimulation — the non-uniform axonal composition of motor nerves leads to preferential recruitment of fast-twitch, fatigue-prone motor units \[Song et al. 2025\]. A conditioning protocol requiring weeks of progressive stimulation was needed to convert the muscle to a more fatigue-resistant phenotype. Second, stimulation timing relative to the cardiac cycle was critical but suboptimally addressed in early clinical implementations. Third, electrical coupling between the stimulated skeletal muscle and the adjacent myocardium posed a risk of arrhythmia induction — a concern that makes a robust ECG safety gate all the more essential in any revival of the approach.

### 5.2 Stimulation Timing Optimisation

The importance of precise stimulation timing was established through modelling and acute animal experiments. Levin and colleagues \[Levin et al. 1991\] used a mathematical haemodynamic model to show that skeletal muscle contraction arriving approximately 50–75 milliseconds after the QRS complex maximises mechanical synchrony with ventricular ejection, providing up to 40% improvement in cardiac output and arterial pressure over the heart-failure baseline. Earlier or later onset substantially reduces or eliminates the haemodynamic benefit. Geddes and colleagues \[Geddes et al. 1993\] confirmed this timing sensitivity in acute canine experiments, systematically varying the stimulus train onset from the R-wave to the end of the isovolumic period and demonstrating the pronounced dependence of haemodynamic outcome on trigger timing.

The problem of adaptive timing optimisation in synchronised cardiac devices has been addressed in the context of cardiac resynchronisation therapy (CRT), where the atrioventricular (AV) delay and interventricular (VV) interval must be tuned to maximise stroke volume. Whinnett and colleagues \[Whinnett et al. 2006\] showed that the haemodynamic response to AV and VV timing follows a consistent and approximately Gaussian landscape, but that the optimum varies between patients and changes over time as cardiac remodelling progresses. The patent of Rom \[Rom 2013, US 8,396,550\] proposed a Q-learning based adaptive controller for online CRT timing optimisation, exploiting a probabilistic replacement scheme that uses an internally maintained Q-table to substitute for noisy haemodynamic sensor readings, providing noise robustness superior to direct gradient ascent. These principles — adaptive timing optimisation, probabilistic noise handling, and multi-state safety fallback — inform the control architecture proposed in this thesis.

### 5.3 The MyoNeural Actuator

The MyoNeural Actuator (MNA) technology \[Song et al. 2025\] addresses the primary limitation of classical cardiomyoplasty — muscle fatigue — through an engineered regenerative approach. By denervating the native motor nerve of a skeletal muscle and reinnervating it with a pure sensory nerve, the MNA redirects volitional control from the CNS to a computer-controlled functional electrical stimulation system. The uniform axonal size distribution of sensory nerves, compared to the mixed-diameter motor nerve, avoids preferential recruitment of fast-twitch fibres and distributes activation more evenly across motor units. The result is a 260% improvement in sustained force production time compared to native muscle under continuous stimulation, with qualitatively different — logarithmic rather than exponential — fatigue dynamics. The MNA additionally provides reversible neural isolation from the CNS through an electrical nerve block. Closed-loop force control demonstrated in the rodent model achieves high-linearity tracking across force targets from 30% to 70% of maximum, motivating the use of force level as the outer control variable in the haemodynamic optimisation framework.

### 5.4 Soft Robotic Cardiac Assist and Haemodynamic Sensing

Parallel development in soft robotic cardiac assist has advanced the concept of organ-in-the-loop control. Payne and colleagues \[Payne et al. 2017\] demonstrated soft pneumatic ventricular assist devices anchored to the interventricular septum and applying force to the free wall, capable of assisting both the left and right ventricles independently, with synchronisation derived from an epicardial electrogram. The biohybrid robotic right ventricle of Singh and colleagues \[Singh et al. 2023\] provided the first experimentally validated platform for in vitro characterisation of RV assist interventions, including haemodynamic pressure and flow sensing instrumentation. These systems establish the broader context for the present work and provide validated approaches for haemodynamic feedback sensing — in particular the use of arterial pressure waveforms, dP/dt_max, and pulse pressure to assess assist efficacy beat-by-beat.

## 6. Label Scarcity and Baseline Definition in Animal Electrophysiology

### 6.1 The Absence of Labelled Animal Arrhythmia Data

The translation of supervised ECG classification to an animal experimental context confronts a practical barrier that is rarely discussed in the benchmarking literature: labelled animal arrhythmia data are essentially unavailable at scale. The large human ECG datasets that enable deep learning have no analogs for rat or porcine electrophysiology. An acute rat experiment may yield hours of ECG from a small number of animals, with rhythm annotations possible only for the episodes that are deliberately induced or that happen to occur and be recognised in real time. Building a supervised arrhythmia classifier for this context would require a labelled rat-specific dataset that does not exist and cannot be assembled within the timeline of a master’s thesis project.

### 6.2 Rat ECG Characteristics

Rats have a resting heart rate of 250–500 beats per minute, giving RR intervals of 120–240 milliseconds. This compressed cardiac cycle affects the relative duration of all cardiac events: the QRS complex, though brief, represents a larger fraction of the cardiac cycle than in humans; T-waves may overlap with the beginning of the following P-wave at high heart rates; and the frequency content of ECG waveforms is shifted upward by approximately a factor of five relative to human values. HRV analysis in rats requires correspondingly adjusted spectral bands. Whereas human HRV is conventionally decomposed into a low-frequency band of 0.04–0.15 Hz and a high-frequency band of 0.15–0.40 Hz, rodent HRV studies use a low-frequency band of approximately 0.2–0.75 Hz and a high-frequency band of approximately 0.75–3.0 Hz \[Thireau et al. 2006\]. Wavelet-based feature extraction using scales calibrated to human ECG will not directly transfer without rescaling. Standard population-level arrhythmia detection thresholds derived from human data are similarly inapplicable.

### 6.3 The Stable Safe Operating Baseline

Given the impossibility of labelled arrhythmia supervision, the safety monitoring problem must be reformulated in terms that are solvable with the data actually available. The appropriate reformulation is: rather than asking “does this ECG match a known dangerous pattern?” the system should ask “does this ECG still resemble the verified safe baseline established for this animal at this stage of the experiment?”

This reformulation introduces a critical conceptual distinction. The relevant reference is not the *healthy* pre-experiment ECG of the animal — that baseline may be substantially different from the ECG during heart-failure experiments, and calibrating to it would cause constant inhibition during the very state in which stimulation is needed. The relevant baseline is the *stable safe operating baseline* established at the stage of the experiment when stimulation will occur. A practical experimental protocol tracks multiple baseline states:

- **T0**: healthy pre-heart-failure baseline (scientific reference; characterises the animal’s starting state)
- **T1**: post-surgery / post-instrumentation baseline (accounts for immediate surgical effects)
- **T2**: stable heart-failure baseline before stimulation (the primary safety calibration reference)
- **T3**: stimulation trials (runtime; compared against T2)
- **T4**: recovery, post-stimulation, or stress episodes (validation data for the safety system)

For safety gating during stimulation trials, **T2 is the critical calibration reference**. The Layer 2 and Layer 3 anomaly boundaries should be fit on T2 windows, not on T0. If the T2 ECG itself is not stable enough for calibration, stimulation should not proceed — a fact the calibration system can enforce by failing to converge on a stable baseline distribution.

## 7. Interpretable Personalised Feature-Based Monitoring

The second layer of the proposed safety supervisor operates on a compact feature vector capturing rhythm regularity and signal morphology, calibrated to the T2 stable operating baseline of the specific animal at the start of the stimulation phase. Its interpretability is a specific advantage in a safety-critical application where every inhibit decision must be explainable.

### 7.1 Heart Rate Variability Features

Heart rate variability analysis provides a family of well-validated features sensitive to autonomic function, rhythm regularity, and pathological states. Time-domain metrics including the standard deviation of normal-to-normal RR intervals (SDNN), the root mean square of successive RR differences (RMSSD), and the coefficient of variation of RR intervals have established reference ranges in rodent physiology \[Thireau et al. 2006\]. SDNN reflects global HRV and is reduced in heart failure; RMSSD is sensitive to parasympathetic tone and is further reduced under anaesthesia; sharp increases in these values may indicate rhythm deterioration or artefact. Spectral analysis of the RR sequence via periodogram or Lomb-Scargle methods decomposes HRV into low-frequency and high-frequency components, providing additional sensitivity to autonomic imbalance. The practical use of spectral HRV in the rat context requires the adjusted frequency bands described in Section 6.2.

### 7.2 Wavelet and Signal-Level Features

The discrete wavelet transform applied to the raw ECG window provides multi-resolution analysis capturing both morphological and rhythmic information. Using the Daubechies db4 wavelet at four decomposition levels, the energy and Shannon entropy of each detail coefficient band represent signal content at multiple frequency scales \[Martínez et al. 2004\]. Energy in mid-frequency bands (spanning the QRS frequency range) is sensitive to QRS morphology changes; energy in high-frequency bands is sensitive to noise and stimulation artefact; energy in low-frequency bands captures baseline wander. Nonlinear entropy features offer complementary sensitivity. Sample entropy \[Richman & Moorman 2000\] measures signal predictability without morphological assumptions and is therefore transferable across species: a healthy organised cardiac rhythm produces low entropy; ventricular fibrillation produces high entropy; flatlines or saturated amplifiers produce near-zero entropy.

### 7.3 Baseline Calibration, Robust Scoring, and Threshold Setting

The key design choice of the second safety layer is per-session, per-animal baseline calibration on the T2 stable operating baseline. The feature vector is computed on rolling windows of clean pre-stimulation ECG. To establish the baseline distribution robustly, several statistical choices matter.

*Robust location and scale estimates* — median and median absolute deviation (MAD) — are preferred over mean and standard deviation for the per-feature z-scores, as they are insensitive to the occasional outlier window that passes SQI checks but contains a transient artefact. For the multivariate anomaly score, the Mahalanobis distance requires a covariance matrix estimate. With the correlated, non-independent windows typical of a 5-second window with 1-second stride (where adjacent windows share 80% of their data), the effective sample size is far smaller than the window count. A shrinkage covariance estimator \[Ledoit & Wolf 2004\] is therefore preferred over the sample covariance, as it regularises towards a diagonal structure and remains well-conditioned when the number of effective samples is modest relative to the feature dimension.

*Separate signal-only and RR-dependent anomaly scores* should be computed and logged independently. A signal-only score based on wavelet energy, entropy, and amplitude features detects morphological and noise-related deviations; an RR-dependent score based on SDNN, RMSSD, and spectral features detects rhythm instability. Separating these provides diagnostic value: an inhibit triggered by the signal-only score points to electrode contact or artefact issues, while an inhibit triggered by the RR score points to rhythm deterioration.

*Threshold setting* uses the held-out segment approach: the T2 baseline is split into a calibration portion (used to fit the mean, MAD, and covariance) and a validation portion (used to set the anomaly threshold from the empirical distribution). The threshold is set at a high quantile of the validation-set anomaly scores — typically the 99th or 99.9th percentile — so that the false-inhibit rate on the safe baseline is controlled rather than assumed.

### 7.4 Dynamic Calibration and Drift Management

Signal characteristics evolve throughout an experiment. Electrode contact changes with animal movement, anaesthetic depth shifts RR interval statistics, haemodynamic loading changes with stimulation, and the EMG contribution from the activated MNA may contaminate the ECG. A fixed calibration from T2 alone may therefore become stale, particularly in long experiments.

Two calibration modes address this:

- **Fixed baseline**: the T2 calibration, retained permanently as the anchor reference. This is never overwritten or discarded, and its anomaly score is always computed in parallel.
- **Adaptive baseline**: a slow-update version that incorporates ECG from verified stable periods after T2, expanding the baseline distribution as the experiment proceeds.

Critically, the adaptive baseline updates only when a strict set of conditions is simultaneously satisfied: stimulation is paused or inhibited, RR intervals are stable and within the expected range, the bSQI is high, QRS detection confidence is high, and the operator has confirmed a safe state. Under any other condition, the adaptive baseline is frozen. This rule prevents the dangerous failure mode in which the baseline drifts to follow a slowly developing arrhythmia, silencing the safety alarm precisely when it is most needed.

Every recalibration event is logged with a timestamp, the gate conditions at the time, and the statistical distance between the old and new baseline distributions. If the new baseline differs substantially from the fixed T2 reference, this divergence is flagged for operator attention rather than accepted silently.

## 8. Self-Supervised and Unsupervised ECG Representation Learning

The third layer of the safety supervisor extends the personalised anomaly detection approach into the embedding space of a learned ECG encoder. The motivation is that handcrafted features may miss complex deviations from the safe baseline that involve combinations of morphological and temporal changes not captured by any single feature. A learned encoder pretrained on a large corpus of human ECG data provides a richer representation space.

### 8.1 Contrastive Self-Supervised Pretraining

Self-supervised contrastive learning provides a framework for training ECG encoders on large unlabeled datasets by constructing positive pairs — windows that should produce similar embeddings — and negative pairs — windows that should produce dissimilar embeddings — without requiring explicit labels. The CLOCS framework of Kiyasseh and colleagues \[Kiyasseh et al. 2021\] extends this to cardiac signals by exploiting three natural sources of positive pairs: temporal proximity within a recording, spatial proximity across leads, and patient identity. Training with these constraints produces representations invariant to within-patient variation while sensitive to clinically meaningful inter-patient differences.

The choice of augmentations for contrastive learning is particularly critical for ECG, where many transformations that appear innocuous in other domains carry specific physiological meaning. The 3KG framework of Gopal and colleagues \[Gopal et al. 2021\] addressed this by designing augmentations grounded in cardiac electrophysiology: three-dimensional rotations of the cardiac electrical axis, temporal warping that preserves RR intervals, mild amplitude scaling, and additive noise. Their analysis showed that physiologically invalid augmentations — including polarity inversion and aggressive frequency masking — degrade downstream task performance, providing empirical support for the principle that ECG-specific domain knowledge should constrain augmentation design.

When partial labels are available on the human pretraining corpus, supervised contrastive learning \[Khosla et al. 2020\] extends the contrastive framework by designating all same-class windows across patients as positive pairs. This strengthens the semantic organisation of the embedding space, placing all sinus rhythm representations close together and arrhythmic representations in distinct clusters, even when they originate from different patients and datasets.

### 8.2 Masked Reconstruction Pretraining

An alternative pretraining paradigm trains the encoder to reconstruct masked portions of the input ECG rather than to distinguish positive from negative pairs. The ST-MEM framework \[Na et al. 2024\] applies this approach to 12-lead ECG by segmenting recordings into spatio-temporal patches, masking a high proportion of patches, and training the model to reconstruct the missing content. This approach avoids the augmentation sensitivity problem entirely, since there are no augmentation choices to make: the masking pattern is the only source of stochasticity. ST-MEM explicitly learns the spatio-temporal structure of multi-lead ECG, capturing both morphological content within leads and the relationships between leads, and it has been shown to work well in reduced-lead settings through lead dropout — which is relevant for an application where the number of available leads is variable.

### 8.3 Anomaly Detection in Embedding Space

Once a pretrained encoder is available, the anomaly detection problem reduces to fitting a model of the stable operating baseline in embedding space. Three methods are considered in this work.

*Gaussian Mahalanobis distance* fits a multivariate Gaussian to the embeddings of T2 baseline windows and scores new windows by their Mahalanobis distance from the fitted mean. This approach is fast, parameter-free given the encoder, and requires only a moderate number of baseline windows.

*Deep SVDD* \[Ruff et al. 2018\] trains a small additional network without bias terms to map baseline embeddings to a hypersphere centred at a fixed point, minimising the mean squared distance. At deployment, the anomaly score is the Euclidean distance from the mapped embedding to the sphere centre. The absence of bias terms specifically prevents the “hypersphere collapse” failure mode in which the network learns a constant mapping. Application of Deep SVDD-style one-class learning to ECG anomaly filtering has been explored in a lightweight form suitable for resource-constrained deployment in the ECG robustness literature \[Ibrahim et al. 2025\]. Ibrahim and colleagues specifically benchmarked lightweight unsupervised anomaly-detection filters, including Deep SVDD, reconstruction-based models, masked anomaly detection, normalizing flows, and diffusion models, and reported that Deep SVDD offered the strongest performance-efficiency trade-off under a parameter budget of approximately 512k parameters.

*Reconstruction-error methods* use an autoencoder or variational autoencoder trained on normal ECG to assign anomaly scores based on reconstruction fidelity \[Atamny et al. 2023\]. Atamny and colleagues compared AE, VAE, diffusion models, normalizing flows, partial-shift VAE, and Gaussian mixture models for ECG outlier detection, reporting that VAE-based reconstruction error achieved meaningful anomaly detection performance on PTB-XL and CPSC. In this thesis, reconstruction-error methods are therefore treated as a baseline for Layer 3 rather than the main deployment strategy, because the primary architecture aims to use a pretrained SSL encoder followed by animal/session-specific anomaly scoring.

### 8.4 From Human Pretraining to Animal Deployment

The practical workflow connecting human-data pretraining to animal deployment proceeds in stages. First, the encoder is pretrained on the combined available human ECG corpus using NT-Xent contrastive loss \[Chen et al. 2020\] with same-record positive sampling and physiologically-informed augmentations. Second, when healthy animal ECG is available, continued pretraining on the animal data adapts the encoder toward animal-relevant signal characteristics without discarding human-derived knowledge. Third, the anomaly head is fitted on the embeddings of the T2 stable operating baseline windows recorded before stimulation, and its threshold is set from the empirical distribution of baseline anomaly scores.

**Layer 3 is a veto layer, not a rhythm classifier.** Its output is not “this is VT” or “this is VF” — it is “this window is distant from the learned safe operating baseline in embedding space.” This distinction is important for two reasons. First, it avoids the unsolvable problem of cross-species label transfer: the system never needs to know what specific arrhythmia a rat is experiencing; it only needs to know that the rhythm has deviated from the verified safe state. Second, it sets honest expectations for validation: the system can be shown to inhibit on any deviation from the established normal, including novel artefacts or previously unseen rhythm perturbations, without requiring exhaustive coverage of every possible pathological morphology.

**Validation gap.** Layer 3 cannot be fully validated on pathological rat ECG in the absence of labelled pathological data. Validation of the encoder on human benchmark anomaly detection tasks provides evidence of encoder quality but does not directly demonstrate deployment performance. End-to-end system validation — that Layer 3 actually fires on rat VT or VF when they occur — requires pharmacologically induced episodes or stim-induced arrhythmic events with retrospective annotation. Planning for such validation data collection must be integrated into the experimental protocol from the outset.

## 9. Synthesis and Research Gaps

The three-layer architecture proposed in this thesis follows from the analysis above. Each layer addresses a distinct failure mode that the other layers cannot handle.

The first layer provides the speed and timing precision required for within-beat stimulation triggering, and catches gross failures such as lead disconnection and signal saturation through hard SQI thresholds. Its deterministic, interpretable structure ensures predictable behaviour and minimal latency. It cannot, however, distinguish safe from unsafe rhythm — its job is to find R-peaks reliably, not to assess rhythm safety.

The second layer provides per-session, per-animal calibration against the T2 stable operating baseline, using interpretable features that transfer across species without model retraining. Its explainable output supports real-time debugging and retrospective audit, and its dynamic calibration framework accommodates the signal drift that is inevitable in long experiments. It requires no labelled abnormal data. Its limitation is that handcrafted features may not capture all relevant deviations, particularly complex combinations of morphological changes.

The third layer addresses this limitation through a learned embedding that captures richer structure, enabling detection of anomalies invisible to the handcrafted feature set. It is a research extension and additional veto layer, not a replacement for Layers 1 and 2. Its validation pathway on animal data is the most open problem identified in this review.

The three layers fail differently: the first by missing subtle rhythm abnormalities, the second by missing morphologically complex deviations, the third by uncharacterised sensitivity on out-of-distribution inputs. Their conjunction is more robust than any single layer. The conservative decision rule — stimulation permitted only if all active layers concur — ensures that degradation of any component defaults to inhibition rather than inappropriate stimulation. This conservative design posture, in which the burden of proof lies with the permit signal rather than the inhibit signal, is the central safety philosophy of the proposed system.

## References

\[1\] Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*. 1985;32(3):230–236.

\[2\] Hamilton PS. Open source ECG analysis. In: *Computers in Cardiology*; 2002:101–104.

\[3\] Martínez JP, Almeida R, Olmos S, Rocha AP, Laguna P. A wavelet-based ECG delineator: evaluation on standard databases. *IEEE Transactions on Biomedical Engineering*. 2004;51(4):570–581.

\[4\] Clifford GD, Behar J, Li Q, Rezek I. Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms from wearable sensors. *Journal of Electrocardiology*. 2012;45(6):596–601.

\[5\] de Chazal P, O’Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical Engineering*. 2004;51(7):1196–1206.

\[6\] Hannun AY, Rajpurkar P, Haghpanahi M, et al. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*. 2019;25(1):65–69.

\[7\] Ribeiro AH, Ribeiro MH, Paixão GMM, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*. 2020;11(1):1760.

\[8\] Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning for ECG analysis: benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*. 2021;25(5):1519–1528.

\[9\] Clifford GD, Liu C, Moody B, et al. AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. In: *Computers in Cardiology*; 2017.

\[10\] Reyna MA, Alday EAP, Gu A, et al. Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021. In: *Computers in Cardiology*; 2021.

\[11\] Ballas A, Diou C. Towards domain generalization for ECG and EEG classification: algorithms and benchmarks. *IEEE Transactions on Emerging Topics in Computational Intelligence*. 2023;8(1):44–54.

\[12\] Li R, Aierken Y, Xu Y, Liu J, Tang Y. Research on cross-dataset cardiac signal domain generalization and feature interpretability. *Scientific Reports*. 2026;16:3138.

\[13\] Carpentier A, Chachques JC. Myocardial substitution with a stimulated skeletal muscle: first successful clinical case. *The Lancet*. 1985;325(8440):1267.

\[14\] Levin HR, Tsitlik JE, Halperin HR. Optimization of the timing of skeletal to cardiac muscle contraction during dynamic cardiomyoplasty: analysis using a mathematical model. *Journal of Cardiac Surgery*. 1991;6(1 Suppl):236–244.

\[15\] Geddes LA, Wessale JL, Badylak SF, et al. The importance of timing muscle contraction in dynamic cardiomyoplasty. *Pacing and Clinical Electrophysiology*. 1993;16(6):1241–1249.

\[16\] Song H, Herrera-Arcos G, Friedman GN, et al. A fatigue-resistant myoneural actuator for implantable biohybrid systems. *bioRxiv*. 2025. doi:10.1101/2025.03.14.642606.

\[17\] Payne CJ, Wamala I, Abah C, et al. Soft robotic ventricular assist device with septal bracing for therapy of heart failure. *Science Robotics*. 2017;2(12):eaan6736.

\[18\] Singh D, Saeed MY, Quevedo-Moreno D, et al. Robotic right ventricle is a biohybrid platform that simulates right ventricular function in (patho)physiological conditions and intervention. *Nature Cardiovascular Research*. 2023;2:1310–1326.

\[19\] Whinnett ZI, Davies JER, Willson K, et al. Haemodynamic effects of changes in AV and VV delay in cardiac resynchronisation therapy show a consistent pattern. *Heart*. 2006;92(12):1665–1670.

\[20\] Rom R. Optimal cardiac pacing with Q learning. US Patent 8,396,550 B2. 2013.

\[21\] Thireau J, Zhang BL, Poisson D, Babuty D. Heart rate variability in mice: a theoretical and practical guide. *Experimental Physiology*. 2008;93(1):83–94. \[Note: while the primary data in this reference concern mice, the spectral band recommendations are widely adopted in the rodent HRV literature and are applicable to rats with appropriate HR-scaled adjustment. A rat-specific validation should be confirmed in the experimental protocol.\]

\[22\] Kiyasseh D, Zhu T, Clifton DA. CLOCS: Contrastive learning of cardiac signals across space, time, and patients. In: *Proceedings of the 38th International Conference on Machine Learning*; 2021.

\[23\] Gopal B, Han R, Raghupathi G, Ng AY, Deb-Chatterji M, Rajpurkar P. 3KG: Contrastive learning of 12-lead electrocardiograms using physiologically-inspired augmentations. In: *Machine Learning for Health (ML4H)*; 2021.

\[24\] Na H, Park J, Tae M, Joo CM. Guiding masked representation learning to capture spatio-temporal relationship of electrocardiogram. In: *International Conference on Learning Representations (ICLR)*; 2024.

\[25\] Ruff L, Vandermeulen R, Goernitz N, et al. Deep one-class classification. In: *Proceedings of the 35th International Conference on Machine Learning*; 2018.

\[26\] Khosla P, Tian P, Wang X, et al. Supervised contrastive learning. In: *Advances in Neural Information Processing Systems*; 2020.

\[27\] Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology — Heart and Circulatory Physiology*. 2000;278(6):H2039–H2049.

\[28\] Dynamic cardiomyoplasty: at the crossroads. *Annals of Thoracic Surgery*. 1999;68(2):750–755.

\[29\] Atamny O, Saguner A, Abaecherli R, Konukoglu E. Outlier Detection in ECG. In: *2023 Computing in Cardiology (CinC)*. Vol. 50. 2023:1–4. doi:10.22489/CinC.2023.038.

\[30\] Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. In: *Proceedings of the 37th International Conference on Machine Learning*; 2020.

\[31\] Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*. 2004;88(2):365–411.
\[32\] Ibrahim MFR, Meijer M, Schlaefer A, Stelldinger P. Enhancing ECG Classification Robustness with Lightweight Unsupervised Anomaly Detection Filters. arXiv:2510.26501. 2025. doi:10.48550/arXiv.2510.26501.
