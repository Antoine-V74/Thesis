# Layer 3 review and open issues

## July 2026 code audit — fixed

- **Pretrain/eval normalization** now identical (per-window robust median/MAD in
  both `pretrain_encoder.ContrastiveECGDataset` and `read_wfdb_window`). NB: the
  window cache is whole-record normalized, but per-window robust norm is affine-
  equivariant so it cancels the global scaling → pretrain and eval inputs match.
- **Augmentor sampling rate** now set from `--augment-fs` (default 125 Hz) so
  baseline-wander/bandpass are calibrated to the real fs; augmentor is also seeded.
- **`--healthy-only`** fails closed if `is_healthy_window` is absent (no silent mix).
- **Checkpoint loading**: `build_encoder` now sets `checkpoint_loaded` only when no
  encoder keys are missing, and under `--no-random-fallback` it RAISES on a missing
  file, missing state dict, or missing keys (previously it silently ran a random
  encoder). This makes `--no-random-fallback` trustworthy.
- **`--exclude-records-csv`** matches only fully-qualified `dataset/record`
  (no bare-record collisions); writes `pretrain_records.json` provenance.
- **Phase 1 analysis** now auto-writes record-cluster bootstrap CI, per-record and
  danger-subtype-stratified false-permit, and A0↔L3 CAV/correlation.

## July 2026 code audit — open (tracked, not yet fixed)

- **Mahalanobis robust pruning** (`layer3_embedding_mahalanobis.fit_robust`) scores
  calibration points in-sample, not leave-one-out (kNN path already uses LOO). A
  contaminated "healthy" fit point can survive pruning. Low risk with clean gold
  records; align with LOO later.
- **Legacy metrics** (`metrics_legacy_healthy_vs_abnormal.csv`) treat
  `~is_healthy_window` as abnormal, so BENIGN_ABNORMAL (e.g. isolated PVC) counts as
  abnormal. Policy metrics and Phase 1 are correct; the legacy table is advisory only.
- **`SBR` (sinus bradycardia) maps to NORMAL** in `label_grouping` → can enter
  healthy calibration. Confirm this is intended policy for the safety baseline.
- **Default (non-`--per-record-calibration`) window validation** uses a global
  record split; the deployment-shaped per-record path needs the flag. Beat
  validation is always per-record, so the pilot is unaffected. Consider defaulting
  window validation to per-record.
- **Signal cache provenance**: `cache_record_signal` trusts an existing `.npy` on a
  cache hit (assumes current `--target-fs`/lead). Rebuild with
  `--overwrite-signal-cache` if fs/lead change; a provenance sidecar would be safer.
- **Trigger mode (`layer1_adaptive_gated`)** detects R-peaks with a zero-phase
  (`filtfilt`) filter over the whole record → non-causal by construction. Already
  labeled offline-only in code/outputs; do not cite as a real-time causal trace.

## Fixed in cleanup pass

- Direct script execution now uses `Layer3/_bootstrap.py` so tools, validation
scripts, and pipeline modules can import each other from the repository root.
- `run_beat_validation.py` now imports `fit_score_one_group` from the current
`run_window_validation.py` module instead of the removed `layer3_validate.py`.
- `pretrain_encoder.py`, `run_window_validation.py`, `run_beat_validation.py`,
`build_window_index.py`, and `compare_layer2_layer3.py` all reach `--help`.
- `run_window_validation.py` and `run_beat_validation.py` now pass the resolved
device (`cpu` or `cuda`) into `encode_windows()` instead of the literal string
`auto`.
- `tools/smoke_test_layer3.py` now calls the current file layout and passes
end-to-end on synthetic WFDB data.
- `compare_layer2_layer3.py` now filters Layer 3 to `split == test` by default.
Use `--layer3-eval-split all` only for conservative no-stim accounting that
intentionally counts calibration and guard rows as inhibits.
- Documentation command paths were updated from the old flat Layer 3 filenames
to the current `tools/`, `validation/`, and `pipeline/` layout.

## Reporting and scientific caveats

- Do not mix `test` metrics with all-row metrics. All-row comparison includes
calibration/guard rows that are forced to no-stim, which can make abnormal
inhibition look better and healthy availability look worse.
- Oracle beat mode uses MIT-BIH annotations to choose beats. This is an offline
upper-bound analysis, not a deployable runtime mode.
- Centered windows include future ECG samples. They should be labelled
non-causal/offline in figures and tables.
- `layer1_adaptive_gated` currently reproduces the offline zero-phase Layer 1
validation trigger path. It is closer to runtime than oracle mode, but still
not embedded causal firmware.
- `--lookahead-ms` includes post-trigger samples. It is a latency simulation only
if stimulation would occur after that delay; it is not zero-latency detection.
- Human MIT-BIH validation is proxy validation. It does not prove animal or pig
generalization; animal use still needs per-session baseline calibration and
prospective validation.
- Healthy-only SSL pretraining uses beat labels to select healthy windows. This
is valid as an offline ablation, but should not be described as deployable
unsupervised calibration.
- If a checkpoint is missing or incompatible, random fallback can produce smoke
test metrics. Final reported runs should use `--no-random-fallback` and should
verify `encoder_info.json` says `checkpoint_loaded: true` with no unexpected
key mismatch.
- Layer 1 trigger mode labels unmatched triggers as non-healthy for conservative
reporting. This mixes Layer 1 detection errors into the Layer 3 abnormal bucket,
so it is useful for gate-level safety but not a pure Layer 3 morphology score.
- Per-record Mahalanobis calibration needs enough early healthy beats/windows.
Records with insufficient healthy calibration become conservative inhibits and
can reduce availability.
- The Mahalanobis model uses a 128-dimensional embedding with empirical
covariance shrinkage. It is stable enough for validation, but thresholds should
be inspected per record because small calibration sets can be sensitive.
- Fusion beats remain a weak case because they contain both normal and abnormal
activation components. Layer 3 should be framed as an added veto, not a
standalone arrhythmia classifier.
- Window length, causal mode, threshold quantile, and lookahead change the safety
and availability tradeoff. Figures should group runs by these settings so
offline improvements are not presented as deployable gains.
- Existing result folders were generated before this cleanup. Treat them as
historical artifacts until key tables are regenerated with the fixed scripts.