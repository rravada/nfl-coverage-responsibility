# Explainable Coverage Responsibility Engine

An end-to-end data pipeline and machine learning engine that processes multi-agent NFL spatiotemporal tracking data. The system transforms raw player coordinates into real-time metrics to calculate live matchup probabilities, isolate pre-snap coverage disguises, and predict defensive breakdowns.

## How it works

For zone coverage, the engine builds a Voronoi diagram over defender positions at the snap; each defender's cell defines their area of responsibility, and every receiver is assigned to the cell that contains them, with receivers outside the convex hull falling back to the nearest defender by Euclidean distance.

For man coverage, the engine compares each receiver's motion to every defender's motion across the frames between the snap and the pass. Velocities are computed as per-frame deltas, and a Pearson correlation in each axis is averaged into a single shadowing score. That motion score is combined with average separation into a composite (60% motion correlation, 40% inverse distance), and each receiver is assigned to the defender with the highest composite.

Pre-snap feature extraction runs on the snap frame and produces the inputs that drive man-vs-zone reasoning: cushion distance from each receiver to the nearest defender, deepest safety depth, cornerback alignment relative to receivers, and the count of defenders in the box. All field coordinates are normalized so the offense always moves left-to-right, which keeps the geometry consistent across plays and games.

## Core Features
* **Dynamic Matchup Matrix:** Uses an XGBoost classifier to compute a live frame-by-frame probability stream mapping every defender's responsibility as passing plays unfold.
* **Spatial Disguise Index:** Combines Kinematic Voronoi space-ownership boundaries at the snap with post-snap tracking vectors to mathematically score defensive deception.
* **Real-Time Breakdown Predictor:** Leverages optimized `cKDTree` geometries to monitor cushion decay rates and automatically flag the exact frame a coverage assignment fails.

## Project Structure
* `data/raw/` - Raw Next Gen Stats spatiotemporal coordinate tracking files.
* `src/` - Core engineering modules (data normalization, feature extraction, Voronoi space mapping, and state-driven canvas animations).
* `pipelines/` - Terminal-driven automation scripts for training the XGBoost model (`train.py`) and exporting interpretability metrics (`evaluate.py`).
* `app/` - Lightweight FastAPI backend web service exposing high-performance REST endpoints to query situational tracking data.

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline directly from your terminal to process tracking data and train the model
python pipelines/train.py

# 3. Launch the local query engine service
uvicorn app.main:app --reload