# Explainable Coverage Responsibility Engine

An end-to-end full-stack software application and spatiotemporal engineering framework that translates raw NFL Next Gen Stats player tracking data into an explainable, real-time diagnostic tool for football coaches and analysts.

Traditional football metrics are retrospective—they tell you what happened, but fail to capture fluid, split-second structural failures. This engine transforms raw player coordinates into localized matchup probabilities, isolates pre-snap coverage disguises, and flags real-time coverage breakdowns with human-readable SHAP translations.

---

## The 3-Tier Application Architecture

The system is built as a fully decoupled, full-stack software architecture:

```
[ 1. Data & ML Brain ]   ──>   [ 2. FastAPI Backend ]   ──>   [ 3. Interactive UI ]
Vectorized Kinematic          Asynchronous REST API           2D Vector Animation Dashboard
Math + XGBoost Model          Data-Serving Layers             with Live Metric Tickers
```

1. **The Analytical Backbone (`src/` & `pipelines/`):** Rather than using raw coordinates, the data layer applies computational geometry (Kinematic Voronoi Tessellation and Delaunay Triangulation) to calculate field ownership and leverage. A multi-class XGBoost model trained via play-disjoint GroupKFold cross-validation ingests these matrices to output fluid, frame-by-frame assignment probabilities for all 11 defenders.
2. **The Asynchronous Bridge (`app/`):** A high-performance FastAPI server that caches pre-computed tracking matrices and model outputs, exposing optimized REST endpoints to serve data payloads quickly without local runtime lag.
3. **The 2D Animation Frontend:** A localized web interface that visualizes tracking dots moving across a digital football field canvas in real time.

---

## The 3 Core Visual Features (Analytics Deliverables)

As a user selects a play and watches the 2D animation frontend unfold, the interface dynamically renders three core diagnostic tools:

- **The Dynamic Matchup Matrix:** A live, updating UI grid displaying the exact probability stream of every defender's coverage responsibility at that specific millisecond. It captures coverage adjustments in real time, showing a safety fluidly transition from a deep zone landmark to a tight man-to-man assignment as a receiver crosses his threshold.
- **The Spatial Disguise Index (SDI):** A proprietary metric that quantifies defensive deception on a 0–100 scale. By extracting the multi-class TreeSHAP value slice for the zone-drop class index, the system isolates marginal attributions of field-ownership geometry to highlight high-IQ secondary players who successfully masked their coverage intentions to bait quarterbacks.
- **The Real-Time Breakdown Predictor ("Burn Risk"):** A fully vectorized engine driven by Pandas `groupby + shift` operations that continuously monitors coverage integrity. The moment a defender drifts outside a zone landmark or a matchup cushion collapses, the UI flags a localized "Burn Risk" failure frame. Simultaneously, the SHAP engine extracts the model weights for that exact frame, translating them into human-readable coaching notes (e.g., *"Safety bit on play-action, sacrificing 4.2 yards of vertical leverage"*).

---

## Project Structure

```
├── data/raw/        # Ingested Next Gen Stats tracking datasets, play metadata, and PFF scouting files
├── src/             # Modular production engineering library (spatial normalizations, feature matrix assembly, geometric boundary calculations)
├── pipelines/
│   ├── train.py     # Matrix assembly, class imbalance mitigation, data type downcasting, and multi-class model training
│   └── evaluate.py  # Native C++ DMatrix inference, TreeSHAP tensor extraction, and tabular deliverable persistence
├── app/             # FastAPI backend exposing optimized REST endpoints for situational tracking data
└── outputs/         # Serialized analytics deliverables (matchup_matrix.parquet, spatial_disguise_index.parquet, burn_risk_log.parquet)
```

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Production Training
```bash
python pipelines/train.py
```

### 3. Run Pipeline Evaluation & Metric Export
```bash
python pipelines/evaluate.py --game-id 2021090900 --smoke-test
```

### 4. Launch the Local Query Engine Service
```bash
uvicorn app.main:app --reload
```

---

## License

[MIT](LICENSE)