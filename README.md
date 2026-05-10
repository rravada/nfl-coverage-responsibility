# Explainable Coverage Responsibility Engine

A deterministic, geometry-first engine that infers which defender is
responsible for which receiver on every passing play in the NFL Big Data
Bowl 2023 tracking data, with a focus on explainability over black-box
prediction.

## How it works

For zone coverage, the engine builds a Voronoi diagram over defender
positions at the snap; each defender's cell defines their area of
responsibility, and every receiver is assigned to the cell that contains
them, with receivers outside the convex hull falling back to the nearest
defender by Euclidean distance.

For man coverage, the engine compares each receiver's motion to every
defender's motion across the frames between the snap and the pass.
Velocities are computed as per-frame deltas, and a Pearson correlation in
each axis is averaged into a single shadowing score. That motion score is
combined with average separation into a composite (60% motion correlation,
40% inverse distance), and each receiver is assigned to the defender with
the highest composite.

Pre-snap feature extraction runs on the snap frame and produces the inputs
that drive man-vs-zone reasoning: cushion distance from each receiver to
the nearest defender, deepest safety depth, cornerback alignment relative
to receivers, and the count of defenders in the box. All field coordinates
are normalized so the offense always moves left-to-right, which keeps the
geometry consistent across plays and games.

## What's built

- Data loading, normalization, and validation pipeline (`src/data_processing.py`)
- Deterministic coverage assignment engine for zone and man (`src/coverage_assignment.py`)
- Pre-snap feature extraction (`src/coverage_features.py`)
- Field plots and play animations (`src/visualization.py`)
- Validation notebooks for the data pipeline and the assignment engine (`notebooks/`)

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_data_validation.ipynb` first, then
`notebooks/02_coverage_assignment.ipynb`.

Data files are not included — download from
[NFL Big Data Bowl 2023](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data)
and place CSVs in `data/raw/`.

![Voronoi Assignment Example](notebooks/outputs/voronoi_assignment_example.png)
