#!/usr/bin/env python3
"""
Danger-grouped safety labels for Layer 3 evaluation.

Why this module exists
----------------------
The Layer 3 anomaly veto is validated on human ECG as a proxy for the eventual
animal stimulation-safety gate. For that validation to mean anything, every
window/beat must be mapped to a *safety* group, not just a beat symbol:

    NORMAL          should permit (healthy sinus baseline)
    DANGEROUS       must inhibit  (VT / VF / flutter / asystole / etc.)
    BENIGN_ABNORMAL don't-care    (isolated ectopy: reported, not penalized)
    NOISE           must inhibit  (signal not trustworthy; fail-safe test)
    AF_CONTEXT      policy-defined (clinical call via AF_TREATED_AS)
    UNLABELED       no annotation evidence (excluded from danger metrics)

The crucial point, confirmed by scanning the actual PhysioNet files
(`Layer3/tools/scan_annotations.py`):

* vfdb is 100% rhythm-level annotations (every annotation symbol is '+'); it has
  NO beat symbols at all. DANGEROUS labels there come ONLY from rhythm-span
  aux_note tokens, never from beat symbols.
* cudb marks ventricular flutter/fibrillation episodes with the standalone '['
  (episode start) and ']' (episode end) symbols, around otherwise-'N' beats.
* nstdb carries clean (copied) beat annotations; its noise lives in the SIGNAL,
  so NOISE there is assigned by dataset identity, not by annotation.

This module turns WFDB annotations (samples, symbols, aux_note) into safety
groups, using rhythm SPANS (not just beat symbols) for the dangerous datasets.

Safety note: nothing here commands stimulation. It only produces evaluation
labels. The mapping tables below are the single auditable source of truth; edit
them in one place to change a clinical assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Group constants
# ---------------------------------------------------------------------------

NORMAL = "NORMAL"
DANGEROUS = "DANGEROUS"
BENIGN_ABNORMAL = "BENIGN_ABNORMAL"
NOISE = "NOISE"
AF_CONTEXT = "AF_CONTEXT"
UNLABELED = "UNLABELED"

ALL_GROUPS = (NORMAL, DANGEROUS, BENIGN_ABNORMAL, NOISE, AF_CONTEXT, UNLABELED)

# Resolution order when a window/beat has evidence for several groups at once.
# Higher index = wins. DANGEROUS must always win (never mask a danger by calling
# it noisy/benign). NOISE beats AF/benign/normal because an untrustworthy signal
# cannot be certified normal. UNLABELED is the weakest.
GROUP_PRECEDENCE: Dict[str, int] = {
    UNLABELED: 0,
    NORMAL: 1,
    BENIGN_ABNORMAL: 2,
    AF_CONTEXT: 3,
    NOISE: 4,
    DANGEROUS: 5,
}


def higher_precedence(a: str, b: str) -> str:
    """Return the group that wins under GROUP_PRECEDENCE."""
    return a if GROUP_PRECEDENCE.get(a, 0) >= GROUP_PRECEDENCE.get(b, 0) else b


def resolve_groups(groups: Sequence[str]) -> str:
    out = UNLABELED
    for g in groups:
        if g is None:
            continue
        out = higher_precedence(out, g)
    return out


# ---------------------------------------------------------------------------
# Beat-symbol -> group  (single annotated beats, NOT rhythm spans)
# ---------------------------------------------------------------------------
# Only 'N' is baseline-safe (matches DEFAULT_NORMAL_SYMBOLS in the validation
# utils). '!' is the ventricular-flutter-wave beat symbol and is dangerous on its
# own. '~' is a signal-quality (noise) marker. '[' / ']' are handled as span
# delimiters, not here.

SYMBOL_GROUP: Dict[str, str] = {
    # Normal sinus beat (the only baseline-safe beat symbol by decision).
    "N": NORMAL,
    # Dangerous beat-level symbol: ventricular flutter wave.
    "!": DANGEROUS,
    # Signal-quality / noise marker.
    "~": NOISE,
    # Benign-abnormal ectopic / non-sinus beats (isolated, outside danger spans).
    "V": BENIGN_ABNORMAL,   # premature ventricular contraction
    "A": BENIGN_ABNORMAL,   # atrial premature beat
    "S": BENIGN_ABNORMAL,   # supraventricular premature beat
    "a": BENIGN_ABNORMAL,   # aberrated atrial premature beat
    "J": BENIGN_ABNORMAL,   # nodal (junctional) premature beat
    "j": BENIGN_ABNORMAL,   # nodal (junctional) escape beat
    "F": BENIGN_ABNORMAL,   # fusion of ventricular and normal beat
    "E": BENIGN_ABNORMAL,   # ventricular escape beat (single beat; the *rhythm* VER is dangerous)
    "/": BENIGN_ABNORMAL,   # paced beat
    "f": BENIGN_ABNORMAL,   # fusion of paced and normal beat
    "Q": BENIGN_ABNORMAL,   # unclassifiable beat
    # Bundle-branch / escape "normal-ish" beats are NOT baseline-safe by decision
    # (normal_set = N only), but they are clearly not dangerous -> benign.
    "L": BENIGN_ABNORMAL,   # left bundle branch block beat
    "R": BENIGN_ABNORMAL,   # right bundle branch block beat
    "e": BENIGN_ABNORMAL,   # atrial escape beat
    "n": BENIGN_ABNORMAL,   # supraventricular escape beat
    "B": BENIGN_ABNORMAL,   # bundle branch block beat (unspecified)
    "r": BENIGN_ABNORMAL,   # R-on-T premature ventricular contraction
}

# Symbols that are not beats and contribute no group on their own.
NON_BEAT_SYMBOLS = frozenset({"+", "|", '"', "=", "x", "(", ")", "[", "]", "?"})


def symbol_group(symbol: str) -> Optional[str]:
    """Group implied by a single beat/quality symbol, or None if non-beat."""
    return SYMBOL_GROUP.get(str(symbol))


# ---------------------------------------------------------------------------
# Rhythm aux_note token -> group  (spans introduced by '+' annotations)
# ---------------------------------------------------------------------------
# Tokens are normalized by stripping a leading '(' and surrounding whitespace and
# uppercasing. All tokens below were observed in the downloaded datasets; see
# Results/layer3/annotation_scan/annotation_aux_tokens_by_dataset.csv.

RHYTHM_TOKEN_GROUP: Dict[str, str] = {
    # --- Normal / baseline-safe rhythms ---
    "N": NORMAL,        # normal sinus rhythm
    "NSR": NORMAL,      # normal sinus rhythm
    "SBR": NORMAL,      # sinus bradycardia (by decision: normal-ish, not penalized)
    # --- Dangerous ventricular / absent rhythms (primary safety target) ---
    "VT": DANGEROUS,    # ventricular tachycardia
    "VFL": DANGEROUS,   # ventricular flutter
    "VF": DANGEROUS,    # ventricular fibrillation
    "VFIB": DANGEROUS,  # ventricular fibrillation (alt token)
    "IVR": DANGEROUS,   # idioventricular rhythm
    "VER": DANGEROUS,   # ventricular escape rhythm
    "ASYS": DANGEROUS,  # asystole
    "HGEA": DANGEROUS,  # high-grade ventricular ectopic activity
    "SVTA": DANGEROUS,  # supraventricular tachyarrhythmia (by decision: dangerous)
    # --- Benign-abnormal rhythms (reported, not penalized) ---
    "B": BENIGN_ABNORMAL,    # ventricular bigeminy
    "T": BENIGN_ABNORMAL,    # ventricular trigeminy
    "AB": BENIGN_ABNORMAL,   # atrial bigeminy
    "NOD": BENIGN_ABNORMAL,  # nodal (junctional) rhythm
    "P": BENIGN_ABNORMAL,    # paced rhythm
    "PM": BENIGN_ABNORMAL,   # pacemaker
    "BI": BENIGN_ABNORMAL,   # first-degree heart block
    "BII": BENIGN_ABNORMAL,  # second-degree heart block
    "PREX": BENIGN_ABNORMAL, # pre-excitation (WPW)
    # --- Noise ---
    "NOISE": NOISE,
    # --- Atrial fibrillation / flutter context (policy via AF_TREATED_AS) ---
    "AFIB": AF_CONTEXT,
    "AFL": AF_CONTEXT,
    "AF": AF_CONTEXT,
    "WPWAF": AF_CONTEXT,  # WPW with atrial fibrillation -> treat as AF context
}

# Tokens that are administrative / not rhythm states; ignored for grouping.
IGNORED_AUX_TOKENS = frozenset({
    "MISSB",   # missed beat(s) marker
    "PSE",     # pause
    "TS",      # tape slippage / timing
    "M", "MB", # measurement markers
    "AUX",
})

# Datasets whose windows are NOISE by identity (noise injected into the signal,
# not the annotation). Both the descriptive folder name and PhysioNet id.
NOISE_DATASETS = frozenset({"noise_stress_test", "nstdb"})


def normalize_rhythm_token(aux: str) -> str:
    """Canonicalize a raw aux_note string to a comparable rhythm token."""
    token = str(aux).strip().replace("\x00", "").replace("\x01", "")
    token = token.strip()
    if token.startswith("("):
        token = token[1:]
    # Some files append stray text after the token; keep the leading alpha run.
    token = token.strip().upper()
    # Keep only leading token chars (letters/digits) before any space.
    cut = ""
    for ch in token:
        if ch.isalnum():
            cut += ch
        else:
            break
    return cut


def rhythm_token_group(aux: str) -> Optional[str]:
    """Group for a normalized rhythm aux token, or None if unknown/ignored."""
    tok = normalize_rhythm_token(aux)
    if not tok or tok in IGNORED_AUX_TOKENS:
        return None
    return RHYTHM_TOKEN_GROUP.get(tok)


# ---------------------------------------------------------------------------
# AF policy
# ---------------------------------------------------------------------------

AF_TREATED_AS_DEFAULT = "inhibit"  # one-line clinical switch: inhibit|permit|exclude

# Safety expectation per group, used by metrics to decide what "correct" means.
#   permit_expected  : a permit here is correct; an inhibit is a false_inhibit
#   inhibit_expected : an inhibit here is correct; a permit is a false_permit
#   dont_care        : reported but neither penalized nor rewarded
#   ignore           : excluded from metrics entirely
PERMIT_EXPECTED = "permit_expected"
INHIBIT_EXPECTED = "inhibit_expected"
DONT_CARE = "dont_care"
IGNORE = "ignore"


def safety_expectation(group: str, af_treated_as: str = AF_TREATED_AS_DEFAULT) -> str:
    """Map a safety group to what a correct gate should do."""
    if group == NORMAL:
        return PERMIT_EXPECTED
    if group in (DANGEROUS, NOISE):
        return INHIBIT_EXPECTED
    if group == BENIGN_ABNORMAL:
        return DONT_CARE
    if group == AF_CONTEXT:
        mode = str(af_treated_as).strip().lower()
        if mode == "permit":
            return PERMIT_EXPECTED
        if mode == "exclude":
            return IGNORE
        return INHIBIT_EXPECTED  # default: inhibit
    return IGNORE  # UNLABELED and anything unexpected


# ---------------------------------------------------------------------------
# Rhythm spans
# ---------------------------------------------------------------------------

@dataclass
class RhythmSpan:
    start: int          # inclusive sample
    end: int            # exclusive sample
    group: str
    token: str          # raw/normalized token or "[]" for bracket episodes


def build_rhythm_spans(
    ann_samples: Sequence[int],
    ann_symbols: Sequence[str],
    ann_aux: Sequence[str],
    sig_len: int,
) -> List[RhythmSpan]:
    """Construct rhythm spans from '+' aux tokens and '[' / ']' bracket episodes.

    '+' annotations introduce a rhythm that holds until the next '+' (or end of
    record). '[' opens a ventricular flutter/fibrillation episode that closes at
    the matching ']' (or end of record if unmatched). Bracket episodes are always
    DANGEROUS.
    """
    samples = np.asarray(ann_samples, dtype=np.int64)
    n = len(samples)
    spans: List[RhythmSpan] = []

    # --- '+' rhythm-change spans ---
    plus_idx = [i for i in range(n) if str(ann_symbols[i]) == "+"]
    for k, i in enumerate(plus_idx):
        grp = rhythm_token_group(ann_aux[i]) if i < len(ann_aux) else None
        if grp is None:
            continue
        start = int(samples[i])
        if k + 1 < len(plus_idx):
            end = int(samples[plus_idx[k + 1]])
        else:
            end = int(sig_len)
        if end > start:
            spans.append(RhythmSpan(start, end, grp, normalize_rhythm_token(ann_aux[i])))

    # --- '[' ... ']' bracket episodes (VF/flutter) ---
    open_start: Optional[int] = None
    for i in range(n):
        sym = str(ann_symbols[i])
        if sym == "[":
            open_start = int(samples[i])
        elif sym == "]" and open_start is not None:
            end = int(samples[i]) + 1
            if end > open_start:
                spans.append(RhythmSpan(open_start, end, DANGEROUS, "[]"))
            open_start = None
    if open_start is not None:  # unmatched '[' runs to end of record
        spans.append(RhythmSpan(open_start, int(sig_len), DANGEROUS, "[]"))

    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def spans_overlapping(spans: Sequence[RhythmSpan], start: int, end: int) -> List[RhythmSpan]:
    """Spans overlapping the half-open interval [start, end)."""
    out = []
    for s in spans:
        if s.start < end and start < s.end:
            out.append(s)
    return out


def span_group_at(spans: Sequence[RhythmSpan], sample: int) -> Optional[str]:
    """Highest-precedence span group covering a single sample."""
    grp: Optional[str] = None
    for s in spans:
        if s.start <= sample < s.end:
            grp = s.group if grp is None else higher_precedence(grp, s.group)
    return grp


# ---------------------------------------------------------------------------
# Public grouping API
# ---------------------------------------------------------------------------

def group_for_beat(
    sample: int,
    symbol: str,
    spans: Sequence[RhythmSpan],
    dataset: Optional[str] = None,
) -> str:
    """Resolve the safety group for a single annotated beat/trigger sample."""
    candidates: List[str] = []
    if dataset is not None and dataset in NOISE_DATASETS:
        candidates.append(NOISE)
    sg = symbol_group(symbol)
    if sg is not None:
        candidates.append(sg)
    span_g = span_group_at(spans, int(sample))
    if span_g is not None:
        candidates.append(span_g)
    return resolve_groups(candidates) if candidates else UNLABELED


def group_for_window(
    start: int,
    end: int,
    spans: Sequence[RhythmSpan],
    ann_samples: Sequence[int],
    ann_symbols: Sequence[str],
    dataset: Optional[str] = None,
) -> str:
    """Resolve the safety group for a window covering [start, end) samples."""
    candidates: List[str] = []
    if dataset is not None and dataset in NOISE_DATASETS:
        candidates.append(NOISE)
    for s in spans_overlapping(spans, int(start), int(end)):
        candidates.append(s.group)
    samples = np.asarray(ann_samples, dtype=np.int64)
    lo = int(np.searchsorted(samples, start, side="left"))
    hi = int(np.searchsorted(samples, end, side="right"))
    for i in range(lo, hi):
        sg = symbol_group(ann_symbols[i])
        if sg is not None:
            candidates.append(sg)
    return resolve_groups(candidates) if candidates else UNLABELED


# ---------------------------------------------------------------------------
# Real-data tally (for the prompt's "print group counts per dataset" check)
# ---------------------------------------------------------------------------

def tally_dataset_beat_groups(
    data_dir: str,
    dataset: str,
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
) -> Dict[str, int]:
    """Count per-beat group labels across a dataset's records (oracle beats)."""
    import wfdb  # local import; only needed for the data scan
    from pathlib import Path
    from collections import Counter

    ds_dir = Path(data_dir) / dataset
    counts: Counter = Counter()
    heas = sorted(ds_dir.rglob("*.hea"))
    if max_records is not None:
        heas = heas[: int(max_records)]
    for hea in heas:
        rec = str(hea.with_suffix(""))
        try:
            ann = wfdb.rdann(rec, ann_ext)
            hdr = wfdb.rdheader(rec)
        except Exception:
            continue
        sig_len = int(hdr.sig_len)
        samples = np.asarray(ann.sample, dtype=np.int64)
        symbols = [str(s) for s in ann.symbol]
        aux = [str(a) for a in (ann.aux_note if getattr(ann, "aux_note", None) is not None else [""] * len(symbols))]
        spans = build_rhythm_spans(samples, symbols, aux, sig_len)
        for i, sym in enumerate(symbols):
            # Count only real beats (skip rhythm/comment/bracket markers) so the
            # tally reflects scoreable beats; spans still influence the group.
            if symbol_group(sym) is None:
                continue
            counts[group_for_beat(int(samples[i]), sym, spans, dataset=dataset)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Smoke test + optional real-data scan
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    # Synthetic annotations: a normal stretch, a VT span, a noisy beat, brackets.
    fs = 250
    samples = [0, 250, 500, 510, 750, 1000, 1250, 1500, 1750]
    symbols = ["+", "N", "N", "~", "+", "N", "[", "N", "]"]
    aux =     ["(N", "", "", "", "(VT", "", "", "", ""]
    sig_len = 2000
    spans = build_rhythm_spans(samples, symbols, aux, sig_len)

    groups = {(s.group, s.token) for s in spans}
    assert (DANGEROUS, "VT") in groups, groups
    assert (NORMAL, "N") in groups, groups
    assert (DANGEROUS, "[]") in groups, groups

    # Beat at 250 is in the (N span -> NORMAL.
    assert group_for_beat(250, "N", spans) == NORMAL
    # The '~' beat at 510 -> NOISE.
    assert group_for_beat(510, "~", spans) == NOISE
    # Beat at 1000 falls inside the (VT span -> DANGEROUS even though symbol is N.
    assert group_for_beat(1000, "N", spans) == DANGEROUS
    # Beat at 1500 is inside the [ ... ] episode -> DANGEROUS.
    assert group_for_beat(1500, "N", spans) == DANGEROUS

    # A V beat in a normal span is benign; a V beat in a VT span is dangerous.
    assert group_for_beat(250, "V", spans) == BENIGN_ABNORMAL
    assert group_for_beat(1000, "V", spans) == DANGEROUS

    # Window covering the VT span is dangerous.
    assert group_for_window(900, 1100, spans, samples, symbols) == DANGEROUS

    # Token normalization.
    assert normalize_rhythm_token("(VFIB") == "VFIB"
    assert rhythm_token_group("(AFIB") == AF_CONTEXT
    assert rhythm_token_group("(VT") == DANGEROUS

    # AF policy switch.
    assert safety_expectation(AF_CONTEXT, "inhibit") == INHIBIT_EXPECTED
    assert safety_expectation(AF_CONTEXT, "permit") == PERMIT_EXPECTED
    assert safety_expectation(AF_CONTEXT, "exclude") == IGNORE
    assert safety_expectation(DANGEROUS) == INHIBIT_EXPECTED
    assert safety_expectation(NORMAL) == PERMIT_EXPECTED

    # nstdb identity -> NOISE regardless of (clean copied) beat symbol.
    assert group_for_beat(250, "N", [], dataset="noise_stress_test") == NOISE

    print("label_grouping smoke test: OK")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Layer 3 safety label grouping (smoke test / data scan).")
    p.add_argument("--scan-data", default=None, help="If set, tally per-beat group counts per dataset from this data dir.")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--ann-ext", default="atr")
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    _smoke_test()

    if args.scan_data:
        datasets = args.datasets or [
            "mit_bih_arrhythmia", "normal_sinus_rhythm", "supraventricular_arrhythmia",
            "long_term_atrial_fibrillation", "noise_stress_test", "st_petersburg_12lead",
            "atrial_fibrillation", "malignant_ventricular_arrhythmia", "creighton_vfib",
        ]
        print(f"\nPer-beat group counts (ann='{args.ann_ext}'):")
        header = f"{'dataset':32s} " + " ".join(f"{g:>16s}" for g in ALL_GROUPS)
        print(header)
        for ds in datasets:
            counts = tally_dataset_beat_groups(args.scan_data, ds, args.ann_ext, args.max_records)
            row = f"{ds:32s} " + " ".join(f"{counts.get(g, 0):16d}" for g in ALL_GROUPS)
            print(row)


if __name__ == "__main__":
    main()
