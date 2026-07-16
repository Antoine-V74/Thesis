# Pilot allowlists (tracked)

Gold transition-record lists for the MIT-BIH-first Soft GO pilot.

| File | Role |
| --- | --- |
| `MANIFEST.md` | Decision + inventory interpretation |
| `pilot_primary_mitbih_gold.csv` | **Primary** Phase 1 `--records-csv` |
| `pilot_secondary_creighton_gold.csv` | VF robustness (later) |
| `pilot_secondary_ltafdb_gold.csv` | AF secondary (later; not headline) |
| `pilot_dataset_roles.csv` | Role table |
| `transition_summary_by_dataset.csv` | Dataset-level counts |

Regenerate full inventory with `Layer3/tools/count_transition_records.py` (writes under `Results/`, gitignored).
