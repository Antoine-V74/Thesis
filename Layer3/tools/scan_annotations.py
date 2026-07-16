#!/usr/bin/env python3
"""
Read-only annotation scanner for Layer 3 label grouping.

Purpose
-------
Layer 3 danger-grouped evaluation needs an auditable mapping from WFDB
annotation symbols and rhythm `aux_note` tokens to safety groups
(NORMAL / DANGEROUS / BENIGN_ABNORMAL / NOISE / AF_CONTEXT). Rather than
hardcode guessed rhythm tokens, this tool tallies what actually appears in the
downloaded datasets so the grouping table in `label_grouping.py` is derived
from ground truth.

This script does NOT read waveforms, modify anything, or make safety decisions.
It only reads `.atr` (or chosen) annotation files and counts:
    * beat-symbol frequencies per dataset
    * rhythm `aux_note` token frequencies per dataset

Output
------
- <out-dir>/annotation_symbols_by_dataset.csv
- <out-dir>/annotation_aux_tokens_by_dataset.csv
- <out-dir>/annotation_scan_summary.md
and a compact console summary.

Usage
-----
    .venv\\Scripts\\python.exe Layer3\\tools\\scan_annotations.py \\
        --data-dir data \\
        --out-dir Results\\layer3\\annotation_scan
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
LAYER3_ROOT = THIS_DIR.parent
PROJECT_ROOT = LAYER3_ROOT.parent
for path in (PROJECT_ROOT, LAYER3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# Default folder -> annotation extension. We deliberately read the rhythm-level
# `.atr` for every dataset (including afdb/vfdb/cudb), because the danger and AF
# context labels come from rhythm aux_note spans, not beat symbols.
DEFAULT_DATASETS: Dict[str, str] = {
    "mit_bih_arrhythmia": "atr",
    "normal_sinus_rhythm": "atr",
    "supraventricular_arrhythmia": "atr",
    "long_term_atrial_fibrillation": "atr",
    "noise_stress_test": "atr",
    "st_petersburg_12lead": "atr",
    "atrial_fibrillation": "atr",
    "malignant_ventricular_arrhythmia": "atr",
    "creighton_vfib": "atr",
}


def import_wfdb():
    try:
        import wfdb  # type: ignore
        return wfdb
    except Exception as exc:
        raise RuntimeError("scan_annotations needs wfdb. Install with: pip install wfdb") from exc


def find_records(dataset_dir: Path) -> List[str]:
    records: List[str] = []
    for hea in sorted(dataset_dir.rglob("*.hea")):
        records.append(str(hea.relative_to(dataset_dir).with_suffix("")).replace("\\", "/"))
    return records


def scan_dataset(
    data_dir: Path,
    dataset: str,
    ann_ext: str,
    max_records: int | None,
) -> Tuple[Counter, Counter, int, int]:
    """Return (symbol_counter, aux_counter, n_records_with_ann, n_annotations)."""
    wfdb = import_wfdb()
    ds_dir = data_dir / dataset
    symbols: Counter = Counter()
    aux: Counter = Counter()
    n_records = 0
    n_ann = 0
    records = find_records(ds_dir)
    if max_records is not None:
        records = records[: int(max_records)]
    for record in records:
        record_path = ds_dir / record
        try:
            ann = wfdb.rdann(str(record_path), ann_ext)
        except Exception as exc:
            print(f"[WARN] {dataset}/{record}: no '{ann_ext}' annotation ({exc})", file=sys.stderr)
            continue
        n_records += 1
        syms = [str(s) for s in ann.symbol]
        symbols.update(syms)
        n_ann += len(syms)
        aux_notes = getattr(ann, "aux_note", None)
        if aux_notes is not None:
            for a in aux_notes:
                token = str(a).strip().replace("\x00", "")
                if token:
                    aux.update([token])
    return symbols, aux, n_records, n_ann


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only WFDB annotation symbol/aux-token scanner for Layer 3 label grouping.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="Results/layer3/annotation_scan")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Folder names to scan. Default: all nine known datasets.")
    p.add_argument("--ann-ext", default="atr", help="Annotation extension to read for every dataset.")
    p.add_argument("--max-records", type=int, default=None, help="Limit records per dataset (smoke testing).")
    args = p.parse_args()

    import pandas as pd

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = {d: args.ann_ext for d in args.datasets}
    else:
        datasets = {d: args.ann_ext for d in DEFAULT_DATASETS}

    sym_rows: List[Dict[str, object]] = []
    aux_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for dataset, ann_ext in datasets.items():
        ds_dir = data_dir / dataset
        if not ds_dir.exists():
            print(f"[WARN] missing dataset directory: {ds_dir}", file=sys.stderr)
            continue
        symbols, aux, n_records, n_ann = scan_dataset(data_dir, dataset, ann_ext, args.max_records)
        print(f"\n=== {dataset} (ann='{ann_ext}', records={n_records}, annotations={n_ann}) ===")
        print("  symbols:", dict(symbols.most_common()))
        print("  aux_tokens:", dict(aux.most_common()))
        for sym, c in symbols.most_common():
            sym_rows.append({"dataset": dataset, "symbol": sym, "count": int(c)})
        for tok, c in aux.most_common():
            aux_rows.append({"dataset": dataset, "aux_token": tok, "count": int(c)})
        summary_rows.append({
            "dataset": dataset,
            "ann_ext": ann_ext,
            "n_records_with_ann": n_records,
            "n_annotations": n_ann,
            "n_unique_symbols": len(symbols),
            "n_unique_aux_tokens": len(aux),
        })

    sym_df = pd.DataFrame(sym_rows)
    aux_df = pd.DataFrame(aux_rows)
    summary_df = pd.DataFrame(summary_rows)
    sym_df.to_csv(out_dir / "annotation_symbols_by_dataset.csv", index=False)
    aux_df.to_csv(out_dir / "annotation_aux_tokens_by_dataset.csv", index=False)
    summary_df.to_csv(out_dir / "annotation_scan_summary.csv", index=False)

    with (out_dir / "annotation_scan_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Layer 3 annotation scan (read-only)\n\n")
        f.write("Ground-truth beat symbols and rhythm aux_note tokens per dataset, used to build "
                "`Layer3/pipeline/label_grouping.py`.\n\n")
        f.write("## Per-dataset summary\n\n")
        f.write(summary_df.to_markdown(index=False) if not summary_df.empty else "(no datasets found)\n")
        f.write("\n\n## Beat symbols per dataset\n\n")
        for dataset in datasets:
            sub = sym_df[sym_df["dataset"] == dataset] if not sym_df.empty else sym_df
            if sub.empty:
                continue
            f.write(f"### {dataset}\n\n")
            f.write(", ".join(f"`{r.symbol}`={r.count}" for r in sub.itertuples(index=False)))
            f.write("\n\n")
        f.write("## Rhythm aux_note tokens per dataset\n\n")
        for dataset in datasets:
            sub = aux_df[aux_df["dataset"] == dataset] if not aux_df.empty else aux_df
            if sub.empty:
                continue
            f.write(f"### {dataset}\n\n")
            f.write(", ".join(f"`{r.aux_token}`={r.count}" for r in sub.itertuples(index=False)))
            f.write("\n\n")

    print(f"\n[DONE] wrote scan outputs to {out_dir}")


if __name__ == "__main__":
    main()
