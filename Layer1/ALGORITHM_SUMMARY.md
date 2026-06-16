# Layer 1 Algorithm Summary

Layer 1 is the fast deterministic ECG safety layer. Its job is to detect plausible R-peaks, supervise RR timing, and decide which beats are stable enough to be passed onward as trigger-eligible events. It does not command stimulation directly; it only produces accepted peaks and trigger samples that the rest of the safety stack can use.

## High-Level Flow

```text
raw ECG
  -> filter_ecg()
  -> fast causal R-peak candidate detector
  -> RR supervisor state machine
  -> Layer1Result
```

The main entry point is `run_layer1()` in `main_pipeline.py`.

```python
result = run_layer1(raw_ecg, fs)
```

It returns a `Layer1Result` containing:

```text
raw                original ECG
filt               filtered ECG
candidate_samples  detector candidates before RR supervision
accepted_samples   beats accepted by the RR supervisor
trigger_samples    accepted RUNNING-mode beats, eligible for downstream triggering
supervisor         full RR supervisor state and decision log
metrics            optional benchmark metrics if reference annotations were passed
```

## Step 1: ECG Filtering

Function:

```python
filter_ecg(raw, fs, mode="zero_phase" or "causal")
```

The filter path applies:

```text
1. 5-20 Hz Butterworth bandpass
2. optional 50 Hz notch
3. optional 60 Hz notch
```

The bandpass emphasizes QRS energy and reduces slow baseline wander and high-frequency noise. The notch filters reduce mains interference when the sampling rate allows it.

Two filtering modes exist:

```text
zero_phase  offline/benchmark mode, uses filtfilt, no phase delay, uses future samples
causal      deployment-style mode, uses lfilter, only past/current samples
```

For real-time reasoning, use `filter_mode="causal"`. Causal filtering can shift or deform the QRS waveform; this is expected and is part of the real-time delay budget.

## Step 2: Candidate R-Peak Detection

Function:

```python
detect_candidates(filt, fs)
```

This calls the fast causal threshold detector in `r_peak_detector.py`.

The detector is designed as a streaming-style detector:

```text
causal filtered ECG
  -> adaptive amplitude threshold
  -> adaptive slope threshold
  -> peak tracking
  -> confirmation after descent
```

Important defaults:

```text
calibration_s = 2.0
detector_refractory_ms = 90.0
threshold_frac = 0.35
slope_ma_ms = 12.0
descent_confirm_ms = 4.0
peak_drop_frac = 0.18
max_peak_width_ms = 90.0
```

### Detector Logic

At startup, the detector uses the first calibration period to estimate initial noise and signal levels. It also chooses polarity automatically:

```text
positive polarity if positive excursions dominate
negative polarity if negative excursions dominate
```

For each new sample, it computes:

```text
amplitude threshold
slope-envelope threshold
```

A sample can enter a candidate peak only when:

```text
filtered amplitude >= amplitude threshold
slope envelope >= slope threshold
sample is on a rising edge
outside detector refractory period
```

Once inside a candidate peak, the detector tracks the maximum sample. It emits a candidate R-peak when the waveform has been held long enough and either:

```text
the signal dropped enough from the peak
or the signal descended for enough samples
or the candidate became too wide
```

The detector output includes both:

```text
peak_samples          estimated R-peak sample indices
confirmation_samples  samples where the detector knew enough to emit the peak
confirmation_delays   confirmation_samples - peak_samples, in ms
```

This distinction matters: in real time, the system can only act after confirmation, not at the ideal peak sample.

## Step 3: RR Supervisor

Function:

```python
run_supervisor(candidate_samples, fs)
```

The detector candidates are not trusted directly. Each candidate is passed to:

```python
RRSupervisor.process_candidate(sample, fs)
```

The supervisor checks whether the beat timing is physiologically plausible and stable enough. It maintains a state machine with four modes:

```text
WAIT_ARM
CALIBRATING
RUNNING
RECOVERY
```

## Supervisor Modes

### WAIT_ARM

The supervisor ignores early detector candidates until:

```text
time >= calibration_start_s
```

Default:

```text
calibration_start_s = 2.0
```

This avoids trusting detections during initial filter and threshold warm-up.

### CALIBRATING

The supervisor collects plausible RR intervals but does not create trigger samples.

During calibration it:

```text
stores accepted RR intervals
uses the median RR as an initial reference
warms up an exponential moving average (EMA)
starts post-beat refractory protection after accepted beats
```

Default:

```text
calibration_rr_count = 10
ema_warmup_count = 4
rr_ema_alpha = 0.20
```

After enough calibration intervals, the supervisor enters `RUNNING`.

### RUNNING

This is the normal operating mode. A candidate can be accepted only if it passes:

```text
1. not inside blanking/refractory protection
2. hard RR minimum/maximum limits
3. adaptive RR band around the current RR reference
```

If accepted:

```text
decision = accept_running
candidate is appended to accepted_samples
candidate is appended to trigger_samples
RR EMA is updated
recent stable RR history is updated
post-beat protection starts
mode stays RUNNING
```

`trigger_samples` are the beats eligible for downstream triggering. The Layer 1 code itself does not command stimulation.

### RECOVERY

The supervisor enters recovery after suspicious timing, especially long RR intervals or repeated unstable short/early intervals.

In recovery, it:

```text
does not produce trigger samples
uses a wider plausibility band
waits for a small number of plausible RR intervals
then returns to CALIBRATING
```

Default:

```text
recovery_low_frac = 0.50
recovery_high_frac = 1.80
recovery_needed_count = 2
```

The path back to normal operation is conservative:

```text
RECOVERY -> CALIBRATING -> RUNNING
```

## RR Acceptance Rules In RUNNING

For each candidate:

```text
rr_candidate = candidate_time - last_accepted_beat_time
```

First, hard physiological limits are applied:

```text
rr_min_ms <= rr_candidate <= rr_max_ms
```

Default:

```text
rr_min_ms = 250.0
rr_max_ms = 2500.0
```

Then the adaptive RR band is applied:

```text
low <= rr_candidate <= high
```

where:

```text
rr_ref = current RR EMA, or median calibration RR if EMA is not ready
band_frac = current_confidence_frac()
low = rr_ref * (1 - band_frac)
high = rr_ref * (1 + band_frac)
```

Example:

```text
rr_ref = 800 ms
band_frac = 0.40
accepted band = 480 ms to 1120 ms
```

## How `band_frac` Is Computed

`band_frac` is the fractional width of the adaptive RR acceptance band.

If there are fewer than four recent stable RR intervals:

```text
band_frac = default_confidence_frac
```

Default:

```text
default_confidence_frac = 0.40
```

Once enough recent stable RR intervals exist, the supervisor estimates robust RR variability:

```text
median_rr = median(recent_stable_rrs)
MAD = median(abs(recent_stable_rrs - median_rr))
robust_sigma = 1.4826 * MAD
band_frac = adaptive_band_mad_scale * robust_sigma / rr_ref
```

Default:

```text
adaptive_band_mad_scale = 3.0
```

Then it clips the value:

```text
min_confidence_frac <= band_frac <= max_confidence_frac
```

Defaults:

```text
min_confidence_frac = 0.10
max_confidence_frac = 0.40
```

After recovery, the band can temporarily widen and then taper back:

```text
recovery_warm_beats = 5
recovery_band_multiplier = 2.0
```

This prevents immediate re-entry into recovery while the timing reference is being rebuilt.

## Blanking And Post-Beat Protection

After an accepted beat, the supervisor ignores candidates during a protection window.

If no stimulation is assumed:

```text
protection_ms = hard_refractory_ms
```

Default:

```text
hard_refractory_ms = 200.0
```

If a beat is trigger-eligible in `RUNNING`, the supervisor uses a longer post-stimulation protection window:

```text
protection_ms = max(
    hard_refractory_ms,
    min_blanking_ms,
    blanking_fraction * rr_ref
)
```

Defaults:

```text
min_blanking_ms = 150.0
blanking_fraction = 0.50
```

This avoids double detection, post-beat artifacts, and stimulation-artifact contamination.

## Decision Labels

The supervisor logs every event as a `Decision`. Common labels:

```text
ignored_wait_arm             early candidate ignored during startup
first_beat                   first timing anchor
accept_calibration           beat accepted during calibration, no trigger
calibration_to_running       enough calibration beats collected
accept_running               beat accepted in RUNNING, trigger-eligible
skip_refractory              candidate ignored during refractory protection
skip_post_stim_protection    candidate ignored during post-trigger protection
reject_short                 RR below hard minimum
reject_long                  RR above hard maximum
reject_out_of_band_low       RR below adaptive band
reject_out_of_band_high      RR above adaptive band
enter_recovery               supervisor enters recovery mode
recovery_accept              plausible beat accepted during recovery
recovery_to_calibration      enough recovery beats collected
```

These labels are useful in plots and in the streaming replay tool.

## Real-Time Replay Tool

Use:

```powershell
.\.venv\Scripts\python.exe Layer1\tools\stream_record.py --record data\mit_bih_arrhythmia\100
```

This tool reveals the ECG progressively and reruns Layer 1 only on data available up to the current time using causal filtering. It is meant to answer:

```text
What would the system have known and done moment by moment?
```

Useful faster replay:

```powershell
.\.venv\Scripts\python.exe Layer1\tools\stream_record.py --record data\mit_bih_arrhythmia\203 --window-s 5 --step-ms 100 --speed 4
```

## Offline Benchmarking

Use:

```powershell
.\.venv\Scripts\python.exe Layer1\pipeline\run_benchmark.py --data-dir data --out-dir Results\layer1_benchmark
```

Benchmarking can pass reference annotations into `run_layer1()`. These annotations are used only after detection to compute metrics such as sensitivity, positive predictive value, F1, false positives, and jitter. They are not used by the detector or supervisor logic.

## Safety Interpretation

Layer 1 is conservative:

```text
uncertain timing -> reject candidate or enter recovery
recovery -> no trigger samples
calibration -> accepted peaks but no trigger samples
running -> trigger samples only if RR timing remains plausible and stable
```

The output should be interpreted as:

```text
candidate_samples = detector proposals
accepted_samples  = plausible supervised R-peaks
trigger_samples   = supervised beats during stable RUNNING mode
```

`trigger_samples` are not a stimulation command. They are timing events that may be used by downstream control only if all higher-level safety gates also permit.
