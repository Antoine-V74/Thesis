"""
Prospective stimulation cadence for Layer 2.

The intended deployment policy is:
    - observe a block of unstimulated beats with the Layer 2 safety gate
    - stimulate only the next cadence beat if the observation block was safe
    - do not use the candidate beat itself to decide whether to stimulate it

Default: observe 7 beats, then make beat 8 the only stimulation opportunity.
By default the policy now allows one observation beat to fail, but requires the
most recent observation beat to be safe. This avoids blocking therapy because
of one isolated noisy observation while still requiring a safe state just before
the candidate beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple


def _decision_ok(decision: Optional[Mapping[str, object]]) -> Tuple[bool, str]:
    """Return (permit, reason) from a Layer 2 decision-like mapping."""
    if decision is None:
        return False, "missing_layer2_decision"
    return bool(decision.get("permit", False)), str(decision.get("reason", ""))


@dataclass
class ProspectiveCadenceGate:
    """
    Track a fixed beat cadence and permit only pre-approved stimulation beats.

    The class is intentionally small and stateless with respect to ECG samples.
    The caller still owns feature extraction and Layer 2 scoring.
    """
    cycle_length: int = 8
    observation_beats: Optional[int] = None
    min_safe_observations: Optional[int] = None
    require_last_observation_safe: bool = True
    require_trigger_ok: bool = True
    _phase: int = 0
    _observations: List[Tuple[bool, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cycle_length < 2:
            raise ValueError("cycle_length must be at least 2")
        if self.observation_beats is None:
            self.observation_beats = self.cycle_length - 1
        if not 1 <= self.observation_beats < self.cycle_length:
            raise ValueError("observation_beats must be in [1, cycle_length)")
        if self.min_safe_observations is None:
            self.min_safe_observations = max(1, int(self.observation_beats) - 1)
        if not 1 <= self.min_safe_observations <= int(self.observation_beats):
            raise ValueError("min_safe_observations must be in [1, observation_beats]")

    @property
    def next_phase(self) -> int:
        """Phase of the next accepted R-peak, using 1..cycle_length numbering."""
        return 1 if self._phase >= self.cycle_length else self._phase + 1

    def reset(self) -> None:
        """Restart the observation cycle."""
        self._phase = 0
        self._observations.clear()

    def step(
        self,
        safety_decision: Optional[Mapping[str, object]] = None,
        *,
        trigger_ok: bool = True,
        trigger_reason: str = "r_peak_detected",
    ) -> Dict[str, object]:
        """
        Advance one accepted R-peak and return the cadence decision.

        For observation beats, pass the Layer 2 decision for that beat.
        For the stimulation candidate beat, pass no safety decision; only
        trigger_ok is used, because the candidate beat must not be analyzed
        before deciding whether to stimulate it.
        """
        phase = self.next_phase
        self._phase = phase
        is_candidate = phase == self.cycle_length

        if not is_candidate:
            safe, reason = _decision_ok(safety_decision)
            self._observations.append((safe, reason))
            self._observations = self._observations[-int(self.observation_beats):]
            return self._result(
                phase=phase,
                is_candidate=False,
                permit=False,
                reason="observation_beat_no_stimulation",
                trigger_ok=bool(trigger_ok),
                trigger_reason=trigger_reason,
                current_beat_used_for_layer2=True,
            )

        observations = self._observations[-int(self.observation_beats):]
        n_observed = len(observations)
        n_safe = sum(1 for safe, _reason in observations if safe)
        enough_history = n_observed >= int(self.observation_beats)
        last_observation_safe = bool(observations[-1][0]) if observations else False
        observation_block_safe = bool(
            enough_history
            and n_safe >= int(self.min_safe_observations)
            and (last_observation_safe or not self.require_last_observation_safe)
        )
        trigger_pass = bool(trigger_ok) if self.require_trigger_ok else True

        permit = bool(observation_block_safe and trigger_pass)
        if permit:
            reason = "cadence_permit"
        elif not enough_history:
            reason = "cadence_not_warmed"
        elif self.require_last_observation_safe and not last_observation_safe:
            reason = "last_observation_inhibit"
        elif not observation_block_safe:
            reason = "previous_observation_block_inhibit"
        else:
            reason = trigger_reason or "trigger_veto"

        result = self._result(
            phase=phase,
            is_candidate=True,
            permit=permit,
            reason=reason,
            trigger_ok=bool(trigger_ok),
            trigger_reason=trigger_reason,
            current_beat_used_for_layer2=False,
        )

        self.reset()
        return result

    def _result(
        self,
        *,
        phase: int,
        is_candidate: bool,
        permit: bool,
        reason: str,
        trigger_ok: bool,
        trigger_reason: str,
        current_beat_used_for_layer2: bool,
    ) -> Dict[str, object]:
        observations = self._observations[-int(self.observation_beats):]
        n_observed = len(observations)
        n_safe = sum(1 for safe, _reason in observations if safe)
        obs_reasons = [reason for _safe, reason in observations if reason]
        return {
            "cadence_cycle_length": self.cycle_length,
            "cadence_observation_beats": int(self.observation_beats),
            "cadence_min_safe_observations": int(self.min_safe_observations),
            "cadence_require_last_observation_safe": self.require_last_observation_safe,
            "cadence_phase": phase,
            "cadence_is_stimulation_beat": is_candidate,
            "cadence_current_beat_used_for_layer2": current_beat_used_for_layer2,
            "cadence_observed_beats": n_observed,
            "cadence_observed_safe_beats": n_safe,
            "cadence_observed_unsafe_beats": n_observed - n_safe,
            "cadence_required_safe_beats": int(self.min_safe_observations),
            "cadence_last_observation_safe": (
                bool(observations[-1][0]) if observations else False
            ),
            "cadence_trigger_ok": trigger_ok,
            "cadence_trigger_reason": trigger_reason,
            "cadence_reason": reason,
            "cadence_observation_reasons": "|".join(obs_reasons),
            "permit": permit,
            "inhibit": not permit,
            "reason": reason,
        }


__all__ = ["ProspectiveCadenceGate"]
