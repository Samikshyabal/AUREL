# AUREL — local research interface

From the project root, run `.\.venv\Scripts\python.exe -m streamlit run app.py`.
Use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` to install
the frontend dependencies into the existing inference environment.

Chapters 1–2 describe the experiment; chapter 3 calls the Stage 2 inference
pipeline on uploaded PNG, JPEG or WebP images. Chapters 4–6 separate measured
test results, best validation results and explanations of the training objective.
Comparative test figures use `03_Results_Charts/accuracy_test (1).csv`. The
teacher's original notebook result is reported separately, not averaged.

Stage 5: Performance Lab now includes marker-only training and validation
histories. `03_Results_Charts/recorded_epochs.csv` transcribes the saved
teacher epoch output (cell 20) and Role 3 student outputs (cells 13, 19),
using one-based cell numbering. Blank values mean not logged, never zero.
The baseline best-validation summary is not assigned to an unrecorded epoch.
Latency is attributed to the specific export, and the KD explanation follows
the implemented teacher-to-student KL rather than the reversed notebook prose.

## Stage 6 — local handover, 22 November 2026

**Status:** Ready for local use in the tested environment. The [README](../README.md)
provides installation, launch, test, inference and results interpretation instructions.
No deployment performed or authorized.

| Check | Handover result |
|---|---|
| Required assets | Entry point, inference modules, six-chapter UI, CSS, SVG, theme, result CSVs, both test scripts and all three trained weights present. All checkpoint SHA-256 hashes match the original audit. |
| Environment | Windows x64, Python 3.13.5; torch 2.11.0+cu130, torchvision 0.26.0+cu130, Pillow 13.0.0, Streamlit 1.67.0, Plotly 6.9.0. `pip check`: no broken requirements. Optional browser checker: Playwright 1.65.2 and installed Edge. |
| Inference | 7/7 tests passed, no skips. Each real checkpoint tested on CPU and CUDA (RTX 3050 Laptop GPU), including probabilities, class and confidence. Missing/incompatible checkpoint tests passed. |
| Launch and frontend | Started from PowerShell on 127.0.0.1:18521 (8501 already occupied). Edge checks passed: all six chapters; empty, corrupt and valid upload states; three predictions and 30 chart probabilities versus direct inference; performance controls, findings and mobile overview. Audit server stopped. |
| Research verification | Stage 5 verified all five comparisons and four history views against source records. Handover did not repeat training, the full CIFAR-10 evaluation, or a hardware latency benchmark. |

**Folder audit and retention**

- No byte-identical standalone project files found outside `.venv` in the
  pre-handover SHA-256 scan. Similar names or `(1)` suffixes do not prove duplication.
- Root and UI `__pycache__` folders contain regenerable bytecode. `.venv` accounted
  for about 3.55 GB at inspection; keep it for this installation but omit it and
  caches from portable source copies. The environment is not a missing dependency.
- `Student_Distillation.ipynb` is an empty placeholder. `Knowledge folder_Main.ipynb`,
  `Results_Analysis.ipynb` and `Main_Demo.ipynb` are setup/smoke notebooks, not app
  entry points. Historical archives, charts and screenshots are not required at
  runtime. Retain originals for provenance; do not use placeholder data as findings.
- The missing handover README has been added. No essential runtime file was found
  missing. Fresh-machine installation was documented, not repeated from scratch
  during this handover. The original CIFAR-10 files and intermediate
  `best_teacher.pth` are absent but not required for inference.
- No project files were deleted. No notebooks, checkpoints, inference modules,
  application behavior or visual design were changed in Stage 6.