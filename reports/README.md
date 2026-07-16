# Curated Reports

Use this folder for small, GitHub-friendly outputs that help a reader understand
the thesis results without downloading multi-GB generated artifacts.

Good candidates:

- final summary CSV files
- selected figures used in the thesis or slides
- short notes explaining how each figure/table was generated
- links to archived full outputs on Zenodo, OSF, institutional storage, or
GitHub Releases

Current reports:

- `Whole_Thesis_Map.md`: **A–O discussion bullets** for supervisor meetings and
chapter planning (full thesis scope, not intro-only)
- `All_Summary.md`: **supervisor-oriented summaries** for the R-peak detector and
Layers 1–3 (single document, same structure throughout)
- `All_Summary.docx`: Word export of the above (`python reports/export_all_summary_docx.py`)
- `ECG_SAFETY_PIPELINE_ARCHITECTURE.md` / `.docx`: **Layers 1–3 architecture**
summary with Mermaid diagrams (start here for the full pipeline)
- `rpeak_detection/`: compact `adaptive_threshold_v2` R-peak benchmark figures,
worst-record tables, and ECG overlay examples.

Do not use this folder for raw PhysioNet data, full per-beat exports, model
checkpoints, or large generated logs. Those belong in local `Results/` or in an
external archive.