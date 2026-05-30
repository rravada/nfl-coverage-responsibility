from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
    "x",
    "y",
    "s",
    "a",
    "o_rad",
    "dir_rad",
    "receiver_cushion",
    "leverage_x",
    "leverage_y",
    "territory_area_sq_yd",
    "territory_grid_points",
    "safety_depth_mean",
    "safety_depth_std",
    "velocity_angular_divergence",
    "territory_ratio",
]

TARGET_COLUMN: str = "coverage_label"
TRAIN_HASH_THRESHOLD: float = 0.80
NAN_FILL_VALUE: float = -999.0
SNAP_EVENT: str = "ball_snap"
PASS_FORWARD_EVENT: str = "pass_forward"
PASS_ARRIVAL_EVENT: str = "pass_arrived"
PRE_SNAP_WINDOW: int = 30
DATA_RAW_DIR: str = "data/raw"
FEATURE_STORE_DIR: str = "data/feature_store"
MODELS_DIR: str = "models"
MAX_AGENTS: int = 22
MAX_FRAMES: int = 200
SEASON: int = 2022
