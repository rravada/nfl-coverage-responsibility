Explainable Coverage Responsibility (WIP)
========================================

This project analyzes NFL player-tracking data to infer **defensive coverage responsibility** – which defender is responsible for which receiver – with an emphasis on explainability over raw predictive accuracy.

Current status (Phase 2)
------------------------

- Load and validate Big Data Bowl Week 1 tracking data (`week1.csv`, `plays.csv`, `games.csv`).
- Normalize all plays so the offense always moves left-to-right for consistent spatial logic.
- Implement a deterministic coverage engine in `src/coverage_assignment.py`:
  - **Zone coverage**: Voronoi-based assignment from defender positions at the snap.
  - **Man coverage**: Motion-correlation + distance–based assignment between snap and pass.
- Engineer pre-snap features in `src/coverage_features.py` (cushion distance, safety depth, CB alignment, defenders in box, etc.).
- Validate spatial and temporal correctness via notebooks:
  - `notebooks/01_data_validation.ipynb`: data sanity checks, static plots, animations.
  - `notebooks/02_coverage_assignment.ipynb`: coverage assignment validation and feature matrix preview.

Planned work
------------

- Train an ML model to predict Man vs Zone (`pff_passCoverageType`) from pre-snap features.
- Add SHAP-based explanations for coverage predictions.
- Build a small dashboard to visualize assignments over time.

Project structure
-----------------

- `src/` – reusable Python modules for data processing, coverage assignment, features, and visualization.
- `notebooks/` – Jupyter notebooks for validation and exploration.
- `data/raw/` – local tracking CSVs (ignored by git; download from Big Data Bowl).
- `requirements.txt` – pinned dependencies for Python 3.10+.

