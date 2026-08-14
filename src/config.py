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

# Transformer feature store uses raw kinematics only (paper §3.3 — the attention
# mechanism learns spatial relationships from coordinates directly; no derived features).
TRANSFORMER_FEATURE_COLUMNS: list[str] = ["x", "y", "o_rad", "dir_rad"]

# Time-since-snap channel, appended to TRANSFORMER_FEATURE_COLUMNS at the tensor
# level only (not appended to the list itself — ETL sentinel-fill iterates over
# TRANSFORMER_FEATURE_COLUMNS, and this column has its own transform: seconds
# since snap = frames_since_snap / 10).
TIME_FEATURE_COLUMN: str = "frames_since_snap"
TRANSFORMER_INPUT_DIM: int = len(TRANSFORMER_FEATURE_COLUMNS) + 1  # + time channel

TARGET_COLUMN: str = "coverage_label"
TRAIN_HASH_THRESHOLD: float = 0.80
NAN_FILL_VALUE: float = -999.0
SNAP_EVENT: str = "ball_snap"
PASS_FORWARD_EVENT: str = "pass_forward"
PASS_ARRIVAL_EVENT: str = "pass_arrived"
PRE_SNAP_WINDOW: int = 30
# Truncation scheme (paper Appendix A.1): fixed offsets from snap (frames) and
# the probability of drawing a fixed strategy vs. the truly-random strategy.
PRE_SNAP_OFFSETS: tuple[int, ...] = (-30, -20, -10)
TRUNCATION_FIXED_PROB: float = 0.6
DATA_RAW_DIR: str = "data/raw"
FEATURE_STORE_DIR: str = "data/feature_store"
MODELS_DIR: str = "models"
MAX_AGENTS: int = 22
MAX_FRAMES: int = 200
MAX_RECEIVERS: int = 5           # eligible receiver slots per play
NO_MATCHUP_SLOT: int = 5         # last index — zone / no man matchup
NUM_MATCHUP_CLASSES: int = 6     # MAX_RECEIVERS + 1
MATCHUP_TARGET_COLUMN: str = "matchup_slot"
SEASON: int = 2022

# slot_ids channel: index 0..MAX_RECEIVERS-1 identifies "this agent IS the
# play's i-th eligible receiver"; NO_SLOT_ID marks every other agent
# (defenders, QB, and padded slots).
NO_SLOT_ID: int = MAX_RECEIVERS

NUM_POSITION_CLASSES: int = 11
NUM_TEAM_CLASSES: int = 2
POSITION_UNKNOWN_ID: int = 10
POSITION_MAP: dict[str, int] = {
    "CB": 0, "S": 1, "FS": 1, "SS": 1, "LB": 2, "ILB": 2, "OLB": 2, "MLB": 2,
    "DB": 3, "DL": 4, "OL": 5, "WR": 6, "TE": 7, "RB": 8, "QB": 9, "FB": 8,
}
