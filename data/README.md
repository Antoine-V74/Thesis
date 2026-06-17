# ECG Dataset Setup

Raw ECG databases are not stored in this Git repository. They should be
downloaded locally into this `data/` folder. The tracked file
`dataset_registry.py` tells the scripts which folder, annotation extension, and
lead/channel to use for each dataset.

## Why The Data Is Not In Git

The local WFDB files are large and are distributed by PhysioNet under their own
licenses and citation requirements. Keep the repository focused on code,
configuration, documentation, and small curated results.

## Download Pattern

Install the Python dependencies first from the project root:

```powershell
python -m pip install -r requirements.txt
```

Then download a database with `wfdb`:

```powershell
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mit_bih_arrhythmia')"
```

Replace the PhysioNet ID and destination folder using the table below.

## Dataset Registry

| Local folder | PhysioNet ID | R-peak benchmark default | Reference used for R-peak scoring |
|---|---:|---:|---|
| `mit_bih_arrhythmia` | `mitdb` | yes | `.atr` beat annotations |
| `normal_sinus_rhythm` | `nsrdb` | yes | `.atr` beat annotations |
| `supraventricular_arrhythmia` | `svdb` | yes | `.atr` reference beat annotations |
| `long_term_atrial_fibrillation` | `ltafdb` | yes | `.atr` beat annotations |
| `noise_stress_test` | `nstdb` | yes | `.atr` annotations copied from clean MIT-BIH records |
| `st_petersburg_12lead` | `incartdb` | yes | `.atr` beat annotations, channel 1 |
| `atrial_fibrillation` | `afdb` | no | `.atr` is rhythm-level; `.qrs` is treated as secondary/non-default |
| `malignant_ventricular_arrhythmia` | `vfdb` | no | `.atr` is rhythm-level, not beat-level |
| `creighton_vfib` | `cudb` | no | `.atr` exists but PhysioNet describes it as non-definitive |

## Download Commands

```powershell
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mit_bih_arrhythmia')"
python -c "import wfdb; wfdb.dl_database('nsrdb', dl_dir='data/normal_sinus_rhythm')"
python -c "import wfdb; wfdb.dl_database('svdb', dl_dir='data/supraventricular_arrhythmia')"
python -c "import wfdb; wfdb.dl_database('ltafdb', dl_dir='data/long_term_atrial_fibrillation')"
python -c "import wfdb; wfdb.dl_database('nstdb', dl_dir='data/noise_stress_test')"
python -c "import wfdb; wfdb.dl_database('incartdb', dl_dir='data/st_petersburg_12lead')"
```

Optional exploratory datasets:

```powershell
python -c "import wfdb; wfdb.dl_database('afdb', dl_dir='data/atrial_fibrillation')"
python -c "import wfdb; wfdb.dl_database('vfdb', dl_dir='data/malignant_ventricular_arrhythmia')"
python -c "import wfdb; wfdb.dl_database('cudb', dl_dir='data/creighton_vfib')"
```

## How WFDB Annotations Work

A WFDB record usually has:

- `.hea`: header and sampling metadata
- `.dat`: waveform samples
- `.atr`, `.qrs`, or similar: sample-index annotations

For R-peak benchmarking, this project uses beat/QRS annotations as reference
events. Rhythm-change markers such as `+` and signal-quality markers such as
`~` are ignored. The reference is not mathematically perfect ground truth: it is
the best available standard from expert or reviewed annotations. The benchmark
therefore matches detections to reference events within a tolerance window
instead of requiring exact sample equality.

## Sources

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- MIT-BIH Normal Sinus Rhythm Database: https://physionet.org/content/nsrdb/1.0.0/
- MIT-BIH Supraventricular Arrhythmia Database: https://physionet.org/content/svdb/1.0.0/
- MIT-BIH Atrial Fibrillation Database: https://physionet.org/content/afdb/1.0.0/
- Long-Term Atrial Fibrillation Database: https://physionet.org/content/ltafdb/1.0.0/
- MIT-BIH Malignant Ventricular Ectopy Database: https://physionet.org/content/vfdb/1.0.0/
- CU Ventricular Tachyarrhythmia Database: https://physionet.org/content/cudb/1.0.0/
- MIT-BIH Noise Stress Test Database: https://physionet.org/content/nstdb/1.0.0/
- St Petersburg INCART 12-lead Arrhythmia Database: https://physionet.org/content/incartdb/1.0.0/
