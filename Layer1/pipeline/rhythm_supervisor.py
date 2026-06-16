
"""
RR-based supervisory logic.

This module decides which detector candidates are accepted, which are
rejected, when stimulation is allowed, when to hold post-beat protection,
and when to enter recovery / recalibration modes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SupervisorConfig:
    """
    Configuration for the supervisory state machine.
    """
    calibration_start_s: float = 2.0
    calibration_rr_count: int = 10
    ema_warmup_count: int = 4

    rr_min_ms: float = 250.0
    rr_max_ms: float = 2500.0

    rr_ema_alpha: float = 0.20
    default_confidence_frac: float = 0.40
    min_confidence_frac: float = 0.10
    max_confidence_frac: float = 0.40
    adaptive_band_history_len: int = 10
    adaptive_band_mad_scale: float = 3.0
    unstable_limit: int = 5

    blanking_fraction: float = 0.50
    min_blanking_ms: float = 150.0
    hard_refractory_ms: float = 200.0

    recovery_low_frac: float = 0.50
    recovery_high_frac: float = 1.80
    recovery_needed_count: int = 2

    # Post-recovery warm-up: widen the RR acceptance band for the first N beats
    # after returning to RUNNING, then tighten back linearly.
    recovery_warm_beats: int = 5
    recovery_band_multiplier: float = 2.0


@dataclass
class Decision:
    """
    Log entry for one supervisor decision.

    This object is used both for debugging and for the plotting layer.
    """
    t_ms: float
    sample: int
    decision: str
    rr_candidate_ms: float
    rr_ref_ms: float
    blanking_ms: float
    mode: str
    band_frac: float = np.nan
    protection_kind: str = "none"


@dataclass
class SupervisorState:
    """
    Mutable state for the supervisory state machine.
    """
    mode: str = "WAIT_ARM"  # WAIT_ARM, CALIBRATING, RUNNING, RECOVERY

    last_anchor_ms: Optional[float] = None
    recovery_anchor_ms: Optional[float] = None

    rr_ema_ms: Optional[float] = None
    calibration_rrs: List[float] = field(default_factory=list)
    recovery_rrs: List[float] = field(default_factory=list)
    recent_stable_rrs: List[float] = field(default_factory=list)

    blanking_until_ms: float = -np.inf
    protection_kind: str = "none"
    unstable_count: int = 0
    high_side_bad_count: int = 0

    accepted_samples: List[int] = field(default_factory=list)
    trigger_samples: List[int] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)

    # Post-recovery warm-up counter (set when re-entering RUNNING after recovery)
    warm_beats_remaining: int = 0

    recovery_entry_times_ms: List[float] = field(default_factory=list)
    recalibration_times_ms: List[float] = field(default_factory=list)

    n_skip_blanking: int = 0
    n_skip_refractory: int = 0
    n_skip_post_stim: int = 0
    n_reject_short: int = 0
    n_reject_long: int = 0
    n_reject_out_of_band: int = 0
    n_recovery_entries: int = 0
    n_recalibrations: int = 0


class RRSupervisor:
    """
    Supervisor that uses RR timing to accept/reject detector candidates.

    Modes
    -----
    WAIT_ARM:
        Ignore candidates until the configured calibration start time.
    CALIBRATING:
        Collect accepted RR intervals without stimulation.
        Start from a robust median reference, then warm the EMA after a few
        accepted RR intervals.
    RUNNING:
        Apply adaptive bounds around the RR reference and trigger
        stimulation after accepted beats.
    RECOVERY:
        Wait for plausible beats to rebuild timing before returning to
        calibration.
    """

    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.state = SupervisorState()

    def current_rr_ref(self) -> float:
        """
        Current timing reference used by the supervisor.
        """
        state = self.state
        if state.rr_ema_ms is not None:
            return state.rr_ema_ms
        if len(state.calibration_rrs) > 0:
            return float(np.median(state.calibration_rrs))
        return 1000.0

    def _append_recent_stable_rr(self, rr_ms: float) -> None:
        """
        Keep a short rolling history of accepted RR values for adaptive-band
        estimation.
        """
        state = self.state
        state.recent_stable_rrs.append(float(rr_ms))
        keep = self.cfg.adaptive_band_history_len
        if len(state.recent_stable_rrs) > keep:
            state.recent_stable_rrs = state.recent_stable_rrs[-keep:]

    def current_confidence_frac(self) -> float:
        """
        Compute the fractional width of the RR acceptance band.

        During warm-up after recovery, the band is widened by recovery_band_multiplier
        and tapers linearly back to normal over recovery_warm_beats accepted beats.
        """
        state = self.state
        rr_ref = self.current_rr_ref()

        window = state.recent_stable_rrs[-self.cfg.adaptive_band_history_len :]
        if len(window) < 4:
            base_frac = self.cfg.default_confidence_frac
        else:
            window_arr = np.asarray(window, dtype=float)
            med = float(np.median(window_arr))
            mad = float(np.median(np.abs(window_arr - med)))
            robust_sigma = 1.4826 * mad
            base_frac = self.cfg.adaptive_band_mad_scale * robust_sigma / max(rr_ref, 1e-6)
            base_frac = float(np.clip(
                base_frac, self.cfg.min_confidence_frac, self.cfg.max_confidence_frac))

        # Post-recovery widening: taper from multiplier → 1.0 over warm_beats
        warm = state.warm_beats_remaining
        if warm > 0:
            total = max(1, self.cfg.recovery_warm_beats)
            taper = 1.0 + (self.cfg.recovery_band_multiplier - 1.0) * (warm / total)
            base_frac = float(np.clip(
                base_frac * taper,
                self.cfg.min_confidence_frac,
                self.cfg.max_confidence_frac * self.cfg.recovery_band_multiplier,
            ))
        return base_frac

    def current_band(self) -> Tuple[float, float]:
        """
        Current acceptance band used in RUNNING mode.
        """
        rr_ref = self.current_rr_ref()
        frac = self.current_confidence_frac()
        return rr_ref * (1.0 - frac), rr_ref * (1.0 + frac)

    def current_recovery_band(self) -> Tuple[float, float]:
        """
        Wider band used during RECOVERY mode.
        """
        rr_ref = self.current_rr_ref()
        return rr_ref * self.cfg.recovery_low_frac, rr_ref * self.cfg.recovery_high_frac

    def start_postbeat_protection(
        self,
        now_ms: float,
        rr_ref_ms: float,
        did_stimulate: bool,
    ) -> Tuple[float, str]:
        """
        Start a protection interval after an accepted beat.

        If the system stimulated, post-stimulation protection is longer. If
        no stimulation occurred, only a shorter hard refractory is applied.
        """
        if did_stimulate:
            protection_ms = max(
                self.cfg.hard_refractory_ms,
                self.cfg.min_blanking_ms,
                self.cfg.blanking_fraction * rr_ref_ms,
            )
            protection_kind = "post_stim"
        else:
            protection_ms = self.cfg.hard_refractory_ms
            protection_kind = "refractory"

        self.state.blanking_until_ms = now_ms + protection_ms
        self.state.protection_kind = protection_kind
        return protection_ms, protection_kind

    def enter_recovery(self, now_ms: float, sample_idx: int) -> None:
        """
        Move the supervisor into RECOVERY mode.
        """
        state = self.state
        state.mode = "RECOVERY"
        state.recovery_anchor_ms = now_ms
        state.recovery_rrs = []
        state.unstable_count = 0
        state.high_side_bad_count = 0
        state.blanking_until_ms = -np.inf
        state.protection_kind = "none"
        state.n_recovery_entries += 1
        state.recovery_entry_times_ms.append(now_ms)

        state.decisions.append(
            Decision(
                t_ms=now_ms,
                sample=sample_idx,
                decision="enter_recovery",
                rr_candidate_ms=0.0,
                rr_ref_ms=self.current_rr_ref(),
                blanking_ms=0.0,
                mode=state.mode,
            )
        )

    def process_candidate(self, sample_idx: int, fs: float) -> None:
        """
        Process one detector candidate.

        This is the core state-machine entry point. It:
        - handles arm time
        - applies blanking / refractory logic
        - collects calibration RR intervals
        - runs acceptance-band logic in RUNNING mode
        - manages RECOVERY / recalibration
        """
        now_ms = 1000.0 * sample_idx / fs
        state = self.state

        # WAIT_ARM: ignore candidates until the calibration start time.
        if state.mode == "WAIT_ARM":
            if now_ms >= 1000.0 * self.cfg.calibration_start_s:
                state.mode = "CALIBRATING"
            else:
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="ignored_wait_arm",
                        rr_candidate_ms=0.0,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
                return

        # POST-BEAT PROTECTION: the detector still sees the event, but the
        # supervisor deliberately ignores it during the protected window.
        if now_ms < state.blanking_until_ms:
            state.n_skip_blanking += 1
            rr_val = 0.0 if state.last_anchor_ms is None else now_ms - state.last_anchor_ms

            if state.protection_kind == "post_stim":
                state.n_skip_post_stim += 1
                skip_name = "skip_post_stim_protection"
            else:
                state.n_skip_refractory += 1
                skip_name = "skip_refractory"

            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision=skip_name,
                    rr_candidate_ms=rr_val,
                    rr_ref_ms=self.current_rr_ref(),
                    blanking_ms=state.blanking_until_ms - now_ms,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                    protection_kind=state.protection_kind,
                )
            )
            return

        # RECOVERY: accept only plausible beats to re-establish timing.
        if state.mode == "RECOVERY":
            if state.recovery_anchor_ms is None:
                state.recovery_anchor_ms = now_ms
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="recovery_first_anchor",
                        rr_candidate_ms=0.0,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
                return

            rr_recovery = now_ms - state.recovery_anchor_ms

            if rr_recovery < self.cfg.rr_min_ms:
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="recovery_reject_short",
                        rr_candidate_ms=rr_recovery,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
                return

            if rr_recovery > self.cfg.rr_max_ms:
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="recovery_reanchor_long",
                        rr_candidate_ms=rr_recovery,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
                state.recovery_anchor_ms = now_ms
                state.recovery_rrs = []
                return

            low_rec, high_rec = self.current_recovery_band()
            if not (low_rec <= rr_recovery <= high_rec):
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="recovery_reject_band",
                        rr_candidate_ms=rr_recovery,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
                if rr_recovery > high_rec:
                    state.recovery_anchor_ms = now_ms
                    state.recovery_rrs = []
                return

            state.recovery_rrs.append(rr_recovery)
            state.recovery_anchor_ms = now_ms
            rr_ref = self.current_rr_ref()
            protection_ms, protection_kind = self.start_postbeat_protection(
                now_ms,
                rr_ref,
                did_stimulate=False,
            )

            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="recovery_accept",
                    rr_candidate_ms=rr_recovery,
                    rr_ref_ms=rr_ref,
                    blanking_ms=protection_ms,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                    protection_kind=protection_kind,
                )
            )

            if len(state.recovery_rrs) >= self.cfg.recovery_needed_count:
                state.mode = "CALIBRATING"
                state.calibration_rrs = list(state.recovery_rrs)
                state.recent_stable_rrs = list(state.recovery_rrs)[-self.cfg.adaptive_band_history_len :]
                state.recovery_rrs = []
                state.last_anchor_ms = now_ms
                state.rr_ema_ms = None
                state.recovery_anchor_ms = None
                state.unstable_count = 0
                state.high_side_bad_count = 0
                state.n_recalibrations += 1
                state.recalibration_times_ms.append(now_ms)

                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="recovery_to_calibration",
                        rr_candidate_ms=rr_recovery,
                        rr_ref_ms=self.current_rr_ref(),
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
            return

        # FIRST ACCEPTABLE BEAT in CALIBRATING or RUNNING.
        if state.last_anchor_ms is None:
            state.last_anchor_ms = now_ms
            rr_ref = self.current_rr_ref()
            protection_ms, protection_kind = self.start_postbeat_protection(
                now_ms,
                rr_ref,
                did_stimulate=False,
            )

            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="first_beat",
                    rr_candidate_ms=0.0,
                    rr_ref_ms=rr_ref,
                    blanking_ms=protection_ms,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                    protection_kind=protection_kind,
                )
            )
            return

        rr_candidate = now_ms - state.last_anchor_ms

        # Hard physiological lower bound.
        if rr_candidate < self.cfg.rr_min_ms:
            state.n_reject_short += 1
            state.unstable_count += 1
            state.high_side_bad_count = 0

            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="reject_short",
                    rr_candidate_ms=rr_candidate,
                    rr_ref_ms=self.current_rr_ref(),
                    blanking_ms=0.0,
                    mode=state.mode,
                )
            )

            if state.unstable_count >= self.cfg.unstable_limit:
                self.enter_recovery(now_ms, sample_idx)
            return

        # Hard physiological upper bound.
        if rr_candidate > self.cfg.rr_max_ms:
            state.n_reject_long += 1
            state.unstable_count += 1
            state.high_side_bad_count += 1

            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="reject_long",
                    rr_candidate_ms=rr_candidate,
                    rr_ref_ms=self.current_rr_ref(),
                    blanking_ms=0.0,
                    mode=state.mode,
                )
            )

            self.enter_recovery(now_ms, sample_idx)
            return

        # CALIBRATING:
        # - no stimulation
        # - robust median startup
        # - EMA warm-up after a few accepted beats
        # - switch to RUNNING after calibration_rr_count accepted intervals
        if state.mode == "CALIBRATING":
            state.calibration_rrs.append(rr_candidate)
            state.last_anchor_ms = now_ms
            state.unstable_count = 0
            state.high_side_bad_count = 0
            self._append_recent_stable_rr(rr_candidate)

            if len(state.calibration_rrs) >= self.cfg.ema_warmup_count:
                if state.rr_ema_ms is None:
                    state.rr_ema_ms = float(np.median(state.calibration_rrs))
                else:
                    alpha = self.cfg.rr_ema_alpha
                    state.rr_ema_ms = (1.0 - alpha) * state.rr_ema_ms + alpha * rr_candidate

            rr_ref = self.current_rr_ref()
            protection_ms, protection_kind = self.start_postbeat_protection(
                now_ms,
                rr_ref,
                did_stimulate=False,
            )

            state.accepted_samples.append(sample_idx)
            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="accept_calibration",
                    rr_candidate_ms=rr_candidate,
                    rr_ref_ms=rr_ref,
                    blanking_ms=protection_ms,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                    protection_kind=protection_kind,
                )
            )

            if len(state.calibration_rrs) >= self.cfg.calibration_rr_count:
                if state.rr_ema_ms is None:
                    state.rr_ema_ms = float(np.median(state.calibration_rrs))
                state.mode = "RUNNING"
                state.unstable_count = 0
                state.high_side_bad_count = 0
                # Arm post-recovery warm-up only when this is a recalibration
                # (not the very first startup), identified by n_recalibrations > 0.
                if state.n_recalibrations > 0:
                    state.warm_beats_remaining = self.cfg.recovery_warm_beats
                state.decisions.append(
                    Decision(
                        t_ms=now_ms,
                        sample=sample_idx,
                        decision="calibration_to_running",
                        rr_candidate_ms=rr_candidate,
                        rr_ref_ms=state.rr_ema_ms,
                        blanking_ms=0.0,
                        mode=state.mode,
                    )
                )
            return

        # RUNNING: apply adaptive band around the RR reference.
        low, high = self.current_band()

        if rr_candidate < low:
            state.n_reject_out_of_band += 1
            state.unstable_count += 1
            state.high_side_bad_count = 0
            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="reject_out_of_band_low",
                    rr_candidate_ms=rr_candidate,
                    rr_ref_ms=self.current_rr_ref(),
                    blanking_ms=0.0,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                )
            )
            if state.unstable_count >= self.cfg.unstable_limit:
                self.enter_recovery(now_ms, sample_idx)
            return

        if rr_candidate > high:
            state.n_reject_out_of_band += 1
            state.unstable_count += 1
            state.high_side_bad_count += 1
            state.decisions.append(
                Decision(
                    t_ms=now_ms,
                    sample=sample_idx,
                    decision="reject_out_of_band_high",
                    rr_candidate_ms=rr_candidate,
                    rr_ref_ms=self.current_rr_ref(),
                    blanking_ms=0.0,
                    mode=state.mode,
                    band_frac=self.current_confidence_frac(),
                )
            )
            self.enter_recovery(now_ms, sample_idx)
            return

        # ACCEPT RUNNING:
        # Update EMA, log accepted beat, and allow stimulation.
        if state.rr_ema_ms is None:
            state.rr_ema_ms = rr_candidate
        else:
            alpha = self.cfg.rr_ema_alpha
            state.rr_ema_ms = (1.0 - alpha) * state.rr_ema_ms + alpha * rr_candidate

        state.last_anchor_ms = now_ms
        state.unstable_count = 0
        state.high_side_bad_count = 0
        if state.warm_beats_remaining > 0:
            state.warm_beats_remaining -= 1

        rr_ref = self.current_rr_ref()
        self._append_recent_stable_rr(rr_candidate)
        protection_ms, protection_kind = self.start_postbeat_protection(
            now_ms,
            rr_ref,
            did_stimulate=True,
        )

        state.accepted_samples.append(sample_idx)
        state.trigger_samples.append(sample_idx)
        state.decisions.append(
            Decision(
                t_ms=now_ms,
                sample=sample_idx,
                decision="accept_running",
                rr_candidate_ms=rr_candidate,
                rr_ref_ms=rr_ref,
                blanking_ms=protection_ms,
                mode=state.mode,
                band_frac=self.current_confidence_frac(),
                protection_kind=protection_kind,
            )
        )
