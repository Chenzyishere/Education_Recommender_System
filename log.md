# Optimization Log

## 2026-03-24

### Round 1: Stabilize Training and Selection
- File: `train_and_eval.py`
- Changes:
- Added train/validation/test split by user.
- Added early stopping with validation-based best checkpoint restore.
- Added best-epoch reporting in the final table.
- Reason:
- Previous experiments showed overfitting after early epochs (especially DKT), so fixed-epoch reporting was unstable.

### Round 2: Time Feature as Optional, Not Default
- Files: `preprocess/dataset_loader.py`, `models/kg_sakt.py`, `train_and_eval.py`, `preprocess/clean_data.py`
- Changes:
- Implemented `time_gap -> log1p -> bucket` pipeline.
- Added model support for time-bucket embedding.
- Added fallback behavior when `time_gap` is unavailable.
- Reason:
- Time feature caused unstable gains. Made it optional so KG-focused experiments stay clean and reproducible.

### Round 3: KG-Centric Optimization (Current)
- Files: `models/kg_sakt.py`, `train_and_eval.py`
- Changes:
- Disabled time feature by default (`USE_TIME_GAP = False`) for main experiments.
- Upgraded logic loss from simple prerequisite penalty to margin-based consistency:
  - Enforce `P(prereq) >= P(target) - margin` through hinge-style penalty.
- Added two-hop graph diffusion path in KG encoder:
  - Mix one-hop and two-hop graph contexts with a learnable gate.
- Reason:
- Research focus is KG-enhanced SAKT recommendation quality.
- Margin logic aligns better with educational prerequisite ordering.
- Two-hop propagation improves use of indirect prerequisite signals.

### Documentation Sync
- Files: `README.md`, `Algorithm_Overview.md`
- Changes:
- Updated project description to reflect KG-first optimization path.
- Clarified default experiment config and optional Time Gap mode.
- Added explanation of margin logic loss and graph diffusion.

### Round 3 Validation Run Snapshot
- Command: `.\\.venv\\Scripts\\python.exe train_and_eval.py`
- Test metrics:
- `Pure-CF`: AUC 0.5039, RMSE 0.6520
- `DKT`: AUC 0.8307, RMSE 0.3982, Path 82.53
- `SAKT`: AUC 0.8004, RMSE 0.4175
- `KG-SAKT`: AUC 0.8087, RMSE 0.4126, Path 74.47
- Observation:
- Compared with the previous config, KG-SAKT improved path consistency while AUC still needs further tuning.

### Round 4: AUC-oriented KG Fine-tuning
- Files: `train_and_eval.py`, `models/kg_sakt.py`, `readme_zhCN.md`
- Changes:
- Reduced logic strength for KG (`LOGIC_LAMBDA_MAX` and `LOGIC_MARGIN` to 0.02).
- Added KG-specific optimizer setup (`AdamW`, higher LR).
- Changed model selection to AUC-first scoring and weaker Path bonus for KG.
- Reduced KG-SAKT dropout from 0.2 to 0.1 to improve fit capacity.
- Added Chinese documentation file `readme_zhCN.md`.
- Reason:
- Current objective is to improve KG-SAKT AUC while keeping logic constraints as auxiliary guidance.

### Round 4 Validation Run Snapshot
- Command: `.\\.venv\\Scripts\\python.exe train_and_eval.py`
- Test metrics:
- `Pure-CF`: AUC 0.5039, RMSE 0.6520
- `DKT`: AUC 0.8377, RMSE 0.3933, Path 75.22
- `SAKT`: AUC 0.7981, RMSE 0.4186
- `KG-SAKT`: AUC 0.8140, RMSE 0.4112, Path 68.90
- Observation:
- KG-SAKT test AUC improved from 0.8087 to 0.8140 after AUC-oriented tuning.

### Round 5: Add Logic-focused Evaluation Metrics
- File: `train_and_eval.py`
- Changes:
- Added `PVR` (Prerequisite Violation Rate).
- Added `APC` (Average Prerequisite Coverage).
- Added `VS` (Violation Severity).
- Added `RDC` (Recommendation Depth Consistency).
- Added skill-depth computation from KG for RDC.
- Expanded epoch logs and final result table to include the new metrics.
- Added inline comments to improve readability of training/evaluation logic.
- Reason:
- Path Compliance alone is too coarse; these metrics provide stronger evidence for recommendation logic quality.

### Round 5 Validation Run Snapshot
- Command: `.\\.venv\\Scripts\\python.exe train_and_eval.py`
- Test metrics:
- `Pure-CF`: AUC 0.5039, RMSE 0.6520
- `DKT`: AUC 0.8377, RMSE 0.3933, Path 75.22, PVR 24.81, APC 75.19, VS 0.3840, RDC 38.41
- `SAKT`: AUC 0.7981, RMSE 0.4186
- `KG-SAKT`: AUC 0.8140, RMSE 0.4112, Path 68.90, PVR 31.30, APC 68.70, VS 0.4124, RDC 58.98

### Round 6: Auto-export CSV and Charts
- File: `train_and_eval.py`
- Changes:
- Added `save_metrics_and_plots(...)` to export final metrics as CSV.
- Added automatic bar chart export for logic metrics.
- Added automatic radar chart export with direction-aware normalization.
- Switched matplotlib backend to `Agg` for headless environments.
- Output files:
- `data/logic_metrics_comparison.csv`
- `data/logic_metrics_bar.png`
- `data/logic_metrics_radar.png`
- Reason:
- Support direct experiment reporting and thesis-ready visualization without manual plotting scripts.
