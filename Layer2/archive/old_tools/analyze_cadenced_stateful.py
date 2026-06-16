"""
Analyze cadence-aware Layer 2 carryover policies from per-beat benchmark output.

This script does not rerun feature extraction.  It consumes per_beat.csv files
from causal lookahead sweeps and simulates stimulation policies that only fire
every N beats (for example 1:4 or 1:8).  The goal is to separate:

  - how much post-R lookahead is needed for strong same-beat full Layer 2 stats
  - whether a previous-beat risk state can be made less blunt by using recent
    history over the non-stimulated beats between scheduled pulses
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from compare_oracle_adaptive import nstdb_snr


METRIC_COLS = [
    "n_healthy",
    "n_abnormal",
    "healthy_permit",
    "abnormal_inhibit",
    "false_permit",
]


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False})


def _metrics(df: pd.DataFrame, permit: pd.Series) -> Dict[str, float]:
    healthy = df["label"].eq("healthy")
    abnormal = df["label"].eq("abnormal_v")
    n_h = int(healthy.sum())
    n_a = int(abnormal.sum())
    hp = float(permit[healthy].mean()) if n_h else np.nan
    fp = float(permit[abnormal].mean()) if n_a else np.nan
    return {
        "n_healthy": n_h,
        "n_abnormal": n_a,
        "healthy_permit": round(hp, 4) if np.isfinite(hp) else np.nan,
        "abnormal_inhibit": round(1.0 - fp, 4) if np.isfinite(fp) else np.nan,
        "false_permit": round(fp, 4) if np.isfinite(fp) else np.nan,
    }


def _filter_nstdb_snr(df: pd.DataFrame, min_snr: int | None) -> pd.DataFrame:
    if min_snr is None or df.empty or "dataset" not in df.columns:
        return df
    is_nstdb = df["dataset"].eq("nstdb")
    snr = df["record"].map(nstdb_snr)
    keep = ~is_nstdb | snr.ge(min_snr).fillna(False)
    return df[keep].copy()


def _load_base(per_beat_path: Path, nstdb_min_snr: int | None) -> pd.DataFrame:
    df = pd.read_csv(per_beat_path, dtype={"record": str}, low_memory=False)
    df = _filter_nstdb_snr(df, nstdb_min_snr)
    base = df[
        (df["benchmark_mode"] == "zero_shot")
        & (df["feature_set"] == "all")
        & (df["eval_mode"] == "fast_causal_nextbeat")
        & (df["label"].isin(["healthy", "abnormal_v", "svt"]))
    ].copy()
    if base.empty:
        return base

    base["permit"] = _as_bool(base["permit"])
    base["same_beat_fast_ok"] = _as_bool(base["same_beat_fast_ok"])
    base["current_full_layer2_permit"] = _as_bool(base["current_full_layer2_permit"])
    base["beat_time_s"] = pd.to_numeric(base["beat_time_s"], errors="coerce")
    base = base.dropna(subset=["beat_time_s"]).sort_values(
        ["dataset", "record", "beat_time_s"]
    )
    base["full_fail"] = ~base["current_full_layer2_permit"]
    base["full_reason"] = base["current_full_layer2_reason"].fillna("").astype(str)
    base["hard_rule"] = base["hard_rule_violated"].fillna("").astype(str)
    base["persistent_hard_fail"] = (
        base["full_reason"].eq("hard_rule")
        & (
            base["hard_rule"].str.startswith("signal__")
            | base["hard_rule"].str.startswith("rr__rr_count")
            | base["hard_rule"].str.startswith("rr__short_rr_fraction")
            | base["hard_rule"].str.startswith("rr__long_rr_fraction")
        )
    )
    base["soft_fail"] = base["full_fail"] & ~base["persistent_hard_fail"]
    return base


def _history_count(group: pd.DataFrame, col: str, history_beats: int) -> pd.Series:
    return (
        group[col]
        .astype(int)
        .shift(1)
        .rolling(history_beats, min_periods=1)
        .sum()
        .fillna(0)
    )


def simulate_history_policies(
    base: pd.DataFrame,
    lookahead_ms: int,
    cadences: Iterable[int],
    max_history: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if base.empty:
        return pd.DataFrame()

    work = base.copy()
    grouped = work.groupby(["dataset", "record"], group_keys=False)
    for h in range(1, max_history + 1):
        work[f"prev_full_fail_h{h}"] = grouped.apply(
            lambda g, h=h: _history_count(g, "full_fail", h)
        )
        work[f"prev_persistent_hard_h{h}"] = grouped.apply(
            lambda g, h=h: _history_count(g, "persistent_hard_fail", h)
        )

    for dataset, ds in work.groupby("dataset"):
        same_fast = ds["same_beat_fast_ok"].fillna(False)
        rows.append({
            "lookahead_ms": lookahead_ms,
            "dataset": dataset,
            "cadence": 1,
            "phase": "all",
            "policy": "same_beat_fast_only",
            **_metrics(ds, same_fast),
        })

        nextbeat_permit = same_fast & (ds["prev_full_fail_h1"].eq(0))
        rows.append({
            "lookahead_ms": lookahead_ms,
            "dataset": dataset,
            "cadence": 1,
            "phase": "all",
            "policy": "previous_beat_must_pass",
            **_metrics(ds, nextbeat_permit),
        })

        for cadence in cadences:
            for history in sorted(set([1, 2, cadence, min(max_history, cadence * 2)])):
                if history > max_history:
                    continue
                for fail_threshold in range(1, history + 1):
                    permit = (
                        same_fast
                        & ds[f"prev_persistent_hard_h{history}"].eq(0)
                        & ds[f"prev_full_fail_h{history}"].lt(fail_threshold)
                    )
                    rows.append({
                        "lookahead_ms": lookahead_ms,
                        "dataset": dataset,
                        "cadence": cadence,
                        "phase": "all",
                        "history_beats": history,
                        "fail_threshold": fail_threshold,
                        "policy": f"history{history}_fail_lt_{fail_threshold}",
                        **_metrics(ds, permit),
                    })

                    # Phase sensitivity: scheduled beats only, one pulse every
                    # cadence beats.  Metrics are reported per phase so the user
                    # can see whether a lucky / unlucky schedule start matters.
                    for phase in range(cadence):
                        scheduled = ds.groupby("record", group_keys=False).apply(
                            lambda g, phase=phase, cadence=cadence: g.iloc[phase::cadence]
                        )
                        if scheduled.empty:
                            continue
                        sched_permit = permit.loc[scheduled.index]
                        rows.append({
                            "lookahead_ms": lookahead_ms,
                            "dataset": dataset,
                            "cadence": cadence,
                            "phase": phase,
                            "history_beats": history,
                            "fail_threshold": fail_threshold,
                            "policy": f"history{history}_fail_lt_{fail_threshold}",
                            **_metrics(scheduled, sched_permit),
                        })

    return pd.DataFrame(rows)


def transition_stats(base: pd.DataFrame, lookahead_ms: int, max_lag: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if base.empty:
        return pd.DataFrame()

    for (dataset, record), g in base.groupby(["dataset", "record"]):
        g = g.sort_values("beat_time_s").reset_index(drop=True)
        labels = g["label"].to_numpy()
        full_fail = g["full_fail"].to_numpy()
        for lag in range(1, max_lag + 1):
            if len(g) <= lag:
                continue
            prev_label = labels[:-lag]
            future_label = labels[lag:]
            prev_fail = full_fail[:-lag]
            prev_abnormal = np.isin(prev_label, ["abnormal_v", "svt"])
            rows.append({
                "lookahead_ms": lookahead_ms,
                "dataset": dataset,
                "record": record,
                "lag_beats": lag,
                "n_after_abnormal": int(prev_abnormal.sum()),
                "healthy_after_abnormal": float(
                    (future_label[prev_abnormal] == "healthy").mean()
                ) if prev_abnormal.any() else np.nan,
                "n_after_full_fail": int(prev_fail.sum()),
                "healthy_after_full_fail": float(
                    (future_label[prev_fail] == "healthy").mean()
                ) if prev_fail.any() else np.nan,
                "abnormal_after_full_fail": float(
                    (future_label[prev_fail] == "abnormal_v").mean()
                ) if prev_fail.any() else np.nan,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby(["lookahead_ms", "dataset", "lag_beats"], as_index=False).agg(
        n_after_abnormal=("n_after_abnormal", "sum"),
        healthy_after_abnormal=("healthy_after_abnormal", "mean"),
        n_after_full_fail=("n_after_full_fail", "sum"),
        healthy_after_full_fail=("healthy_after_full_fail", "mean"),
        abnormal_after_full_fail=("abnormal_after_full_fail", "mean"),
    )


def same_beat_summary(
    per_beat_path: Path,
    lookahead_ms: int,
    nstdb_min_snr: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(per_beat_path, dtype={"record": str}, low_memory=False)
    df = _filter_nstdb_snr(df, nstdb_min_snr)
    df = df[
        (df["benchmark_mode"] == "zero_shot")
        & (df["feature_set"] == "all")
        & (df["eval_mode"] == "fast_causal_gated")
        & (df["label"].isin(["healthy", "abnormal_v"]))
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["permit"] = _as_bool(df["permit"])
    rows = []
    for dataset, ds in df.groupby("dataset"):
        rows.append({
            "lookahead_ms": lookahead_ms,
            "dataset": dataset,
            "policy": "same_beat_full_layer2",
            **_metrics(ds, ds["permit"]),
        })
    return pd.DataFrame(rows)


def parse_lookahead(path: Path) -> int:
    name = path.name
    marker = "lookahead_"
    if marker not in name:
        raise ValueError(f"cannot parse lookahead from {path}")
    return int(name.split(marker, 1)[1].split("ms", 1)[0])


def find_per_beat(sweep_dirs: List[Path], lookahead_ms: int) -> Path | None:
    for sweep_dir in sweep_dirs:
        p = sweep_dir / f"lookahead_{lookahead_ms:03d}ms" / "per_beat.csv"
        if p.exists():
            return p
        p = sweep_dir / f"lookahead_{lookahead_ms}ms" / "per_beat.csv"
        if p.exists():
            return p
    return None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path, default=Path("Results/layer2/nextbeat_lookahead_sweep_quick"))
    p.add_argument("--extra-sweep-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--out-dir", type=Path, default=Path("Results/layer2/cadenced_stateful_analysis"))
    p.add_argument("--lookahead-ms", nargs="+", type=int, default=[30, 40, 50, 80, 100, 150, 200])
    p.add_argument("--cadences", nargs="+", type=int, default=[4, 8])
    p.add_argument("--max-history", type=int, default=8)
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument(
        "--nstdb-min-snr",
        type=int,
        default=None,
        help="If set, keep only NSTDB records with SNR >= this value; other datasets are unchanged.",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_frames = []
    transition_frames = []
    same_beat_frames = []

    sweep_dirs = [args.sweep_dir, *args.extra_sweep_dirs]
    for look_ms in args.lookahead_ms:
        per_beat = find_per_beat(sweep_dirs, look_ms)
        if per_beat is None:
            continue
        base = _load_base(per_beat, args.nstdb_min_snr)
        policy_frames.append(
            simulate_history_policies(base, look_ms, args.cadences, args.max_history)
        )
        transition_frames.append(transition_stats(base, look_ms, args.max_lag))
        same_beat_frames.append(same_beat_summary(per_beat, look_ms, args.nstdb_min_snr))

    policies = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    same_beat = pd.concat(same_beat_frames, ignore_index=True) if same_beat_frames else pd.DataFrame()

    policies.to_csv(args.out_dir / "cadenced_policy_sweep.csv", index=False)
    transitions.to_csv(args.out_dir / "carryover_transition_stats.csv", index=False)
    same_beat.to_csv(args.out_dir / "same_beat_lookahead_summary.csv", index=False)

    if not policies.empty:
        candidates = policies[
            policies["phase"].eq("all")
            & policies["policy"].str.startswith("history", na=False)
        ].copy()
        candidates["score"] = (
            candidates["healthy_permit"]
            + candidates["abnormal_inhibit"]
            - 2.0 * candidates["false_permit"]
        )
        best = candidates.sort_values(
            ["lookahead_ms", "dataset", "cadence", "score"],
            ascending=[True, True, True, False],
        ).groupby(["lookahead_ms", "dataset", "cadence"], as_index=False).head(5)
        best.to_csv(args.out_dir / "best_cadenced_policies.csv", index=False)
        print(best[[
            "lookahead_ms", "dataset", "cadence", "policy",
            "healthy_permit", "abnormal_inhibit", "false_permit", "score",
        ]].to_string(index=False))

    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
