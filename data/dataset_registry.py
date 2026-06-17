"""
PhysioNet ECG dataset registry.

The repository keeps this registry in Git, but not the downloaded WFDB files.
Folder names on disk are descriptive. Old PhysioNet short IDs such as "mitdb"
still work as CLI aliases via resolve_dataset().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional


# WFDB symbols that correspond to beat/QRS annotations. Rhythm-change markers
# such as "+", signal-quality markers such as "~", and comments are excluded
# when scoring R-peak detectors.
BEAT_SYMBOLS: FrozenSet[str] = frozenset({
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E",
    "/", "f", "Q", "?",
})


@dataclass(frozen=True)
class DatasetInfo:
    folder: str
    physionet_id: str
    title: str
    group: str
    description: str
    ann_ext: str = "atr"
    channel: int = 0
    normal_beats: FrozenSet[str] = frozenset({"N", "L", "R", "e", "j"})
    abnormal_beats: FrozenSet[str] = frozenset({"V", "F", "E", "/", "f", "!"})
    rpeak_ann_ext: Optional[str] = "atr"
    rpeak_reference: str = "expert_or_reviewed_beat"
    rpeak_benchmark_default: bool = True
    rpeak_notes: str = ""


DATASETS: Dict[str, DatasetInfo] = {
    "mit_bih_arrhythmia": DatasetInfo(
        folder="mit_bih_arrhythmia",
        physionet_id="mitdb",
        title="MIT-BIH Arrhythmia Database",
        group="mixed",
        description="48 ambulatory ECG records with beat and rhythm annotations.",
        rpeak_notes="Use .atr beat annotations; '+' rhythm annotations are ignored for R-peak scoring.",
    ),
    "normal_sinus_rhythm": DatasetInfo(
        folder="normal_sinus_rhythm",
        physionet_id="nsrdb",
        title="MIT-BIH Normal Sinus Rhythm Database",
        group="clean_sinus",
        description="18 long-duration records, predominantly normal sinus rhythm.",
        rpeak_notes="Use .atr beat annotations. Long records are usually truncated for benchmarking.",
    ),
    "supraventricular_arrhythmia": DatasetInfo(
        folder="supraventricular_arrhythmia",
        physionet_id="svdb",
        title="MIT-BIH Supraventricular Arrhythmia Database",
        group="clean_sinus",
        description="78 half-hour records with supraventricular arrhythmias.",
        abnormal_beats=frozenset({"A", "a", "J", "S", "V", "F", "E"}),
        rpeak_notes="Use .atr reference beat and signal-quality annotations.",
    ),
    "atrial_fibrillation": DatasetInfo(
        folder="atrial_fibrillation",
        physionet_id="afdb",
        title="MIT-BIH Atrial Fibrillation Database",
        group="af_dominant",
        description="Long-term recordings with paroxysmal atrial fibrillation.",
        rpeak_ann_ext="qrs",
        rpeak_reference="automated_or_secondary_qrs",
        rpeak_benchmark_default=False,
        rpeak_notes=(
            ".atr files mainly mark rhythm changes. Local .qrs files contain QRS "
            "annotations, but they are treated as secondary/non-default references."
        ),
    ),
    "long_term_atrial_fibrillation": DatasetInfo(
        folder="long_term_atrial_fibrillation",
        physionet_id="ltafdb",
        title="Long-Term Atrial Fibrillation Database",
        group="af_dominant",
        description="Long Holter recordings dominated by atrial fibrillation.",
        rpeak_notes="Use .atr beat annotations; rhythm-change markers are ignored for R-peak scoring.",
    ),
    "malignant_ventricular_arrhythmia": DatasetInfo(
        folder="malignant_ventricular_arrhythmia",
        physionet_id="vfdb",
        title="MIT-BIH Malignant Ventricular Ectopy Database",
        group="vt_vfib",
        description="Half-hour ECG records with sustained VT, flutter, and VF episodes.",
        normal_beats=frozenset({"N"}),
        abnormal_beats=frozenset({"V", "F", "E"}),
        rpeak_ann_ext=None,
        rpeak_reference="rhythm_only",
        rpeak_benchmark_default=False,
        rpeak_notes=".atr files contain rhythm-change annotations, not beat-level R-peak references.",
    ),
    "creighton_vfib": DatasetInfo(
        folder="creighton_vfib",
        physionet_id="cudb",
        title="CU Ventricular Tachyarrhythmia Database",
        group="vt_vfib",
        description="Short records with induced and spontaneous ventricular fibrillation.",
        normal_beats=frozenset({"N"}),
        abnormal_beats=frozenset({"V", "F", "E"}),
        rpeak_reference="nondefinitive_beat",
        rpeak_benchmark_default=False,
        rpeak_notes=(
            "PhysioNet notes that beat annotations aid event location but are not "
            "definitive; use only for exploratory stress tests."
        ),
    ),
    "noise_stress_test": DatasetInfo(
        folder="noise_stress_test",
        physionet_id="nstdb",
        title="MIT-BIH Noise Stress Test Database",
        group="noisy",
        description="MIT-BIH records with calibrated added noise.",
        rpeak_reference="copied_mitdb_beat",
        rpeak_notes="Use .atr annotations copied from the clean source records; report performance by SNR.",
    ),
    "st_petersburg_12lead": DatasetInfo(
        folder="st_petersburg_12lead",
        physionet_id="incartdb",
        title="St Petersburg INCART 12-lead Arrhythmia Database",
        group="mixed",
        description="75 twelve-lead records. Channel 1 is used as lead II for single-lead tests.",
        channel=1,
        rpeak_notes="Use .atr beat annotations; channel 1 is used by default.",
    ),
}


# Old PhysioNet folder names -> canonical registry key.
PHYSIONET_ALIASES: Dict[str, str] = {
    info.physionet_id: key for key, info in DATASETS.items()
}

GROUP_ORDER = ["clean_sinus", "mixed", "af_dominant", "vt_vfib", "noisy"]

# Convenience dicts keyed by canonical folder name.
DATASET_GROUPS: Dict[str, str] = {info.folder: info.group for info in DATASETS.values()}
DATASET_ANN_EXT: Dict[str, str] = {info.folder: info.ann_ext for info in DATASETS.values()}
DATASET_CHANNEL: Dict[str, int] = {info.folder: info.channel for info in DATASETS.values()}
DATASET_NORMAL: Dict[str, FrozenSet[str]] = {
    info.folder: info.normal_beats for info in DATASETS.values()
}
DATASET_ABNORMAL: Dict[str, FrozenSet[str]] = {
    info.folder: info.abnormal_beats for info in DATASETS.values()
}


def resolve_dataset(name: str) -> DatasetInfo:
    """Accept canonical folder name or old PhysioNet ID."""
    key = PHYSIONET_ALIASES.get(name, name)
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {list(DATASETS)} or aliases: {list(PHYSIONET_ALIASES)}"
        )
    return DATASETS[key]


def dataset_dir(data_dir: Path, name: str) -> Path:
    return data_dir / resolve_dataset(name).folder


def list_datasets(names: Optional[Iterable[str]] = None) -> List[DatasetInfo]:
    if names is None:
        return list(DATASETS.values())
    return [resolve_dataset(n) for n in names]


def default_rpeak_datasets(names: Optional[Iterable[str]] = None) -> List[DatasetInfo]:
    """Datasets with default beat-level references suitable for R-peak scoring."""
    infos = list_datasets(names)
    return [
        info for info in infos
        if info.rpeak_benchmark_default and info.rpeak_ann_ext is not None
    ]


def is_beat_symbol(symbol: str) -> bool:
    return symbol in BEAT_SYMBOLS


def record_stems(data_dir: Path, name: str) -> List[str]:
    """Return WFDB record stems (path without extension) for one dataset."""
    d = dataset_dir(data_dir, name)
    return [str(p.with_suffix("")) for p in sorted(d.glob("*.hea"))]
