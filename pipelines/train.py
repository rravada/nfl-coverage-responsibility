"""
pipelines/train.py
==================
Production training pipeline for the NFL Coverage Responsibility multi-class model.

Entry point
-----------
    python pipelines/train.py --data-dir data/raw --output-dir models/

Architecture
------------
The feature matrix is **defender-centric**: one row per (gameId, playId, frameId,
defender_nflId).  Feature families assembled here:

  * Kinematic state       – x, y, s, a, o_rad, dir_rad  (from normalizer)
  * Receiver cushion      – Euclidean distance to assigned/nearest receiver
  * Spatial leverage      – signed x/y offset to receiver
  * Voronoi territory     – kinematic territory area and grid-point count
  * Pre-snap safety depth – play-level mean/std of safety depth behind LOS

Leakage-prevention contract
----------------------------
Entire plays (all frames) are atomically assigned to train or test via a
deterministic MD5 hash of gameId.  No randomised row-level shuffle ever
occurs.  GroupKFold keyed on composite gameId_playId is provided for
cross-validation sweeps.

Target label
------------
For each (defender, frame):
  - ``No_Matchup_Zone``  if no Voronoi receiver assignment exists **or** PFF
    data classifies the defender as playing zone coverage.
  - ``<receiver_nflId>`` (str) for man-coverage assignments.

Class imbalance
---------------
``No_Matchup_Zone`` will dominate.  Balanced sample weights are computed via
sklearn and injected into XGBoost's ``sample_weight`` parameter — zone frames
are never down-sampled, preserving spatiotemporal continuity.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from src.coverage_assignment import (
    build_coverage_assignments,
    build_field_grid,
    compute_defender_territories,
)
from src.data.normalizer import load_and_build
from src.features.coverage_features import (
    compute_presnap_safety_depth,
    compute_receiver_cushion,
    compute_spatial_leverage,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LABEL_ZONE: str = "No_Matchup_Zone"

# Explicit sentinel for unresolvable spatial features.
# XGBoost can handle true NaN natively, but an out-of-range sentinel lets
# SHAP and downstream tools reason about missingness as a first-class signal.
NAN_FILL_VALUE: float = -999.0

# Deterministic 80/20 boundary: games whose MD5 digest maps to [0, threshold)
# go to train; [threshold, 1.0) go to test.
TRAIN_HASH_THRESHOLD: float = 0.80

RANDOM_SEED: int = 42

# Ordered feature column list — must stay consistent between train and serve.
# Territory column is ``territory_area_sq_yd`` (matches compute_defender_territories schema).
FEATURE_COLUMNS: list[str] = [
    # Defender kinematic state
    "x",
    "y",
    "s",
    "a",
    "o_rad",
    "dir_rad",
    # Coverage geometry
    "receiver_cushion",
    "leverage_x",
    "leverage_y",
    # Voronoi territory
    "territory_area_sq_yd",
    "territory_grid_points",
    # Pre-snap context (play-level, broadcast to all frames of the play)
    "safety_depth_mean",
    "safety_depth_std",
]

TARGET_COLUMN: str = "coverage_label"

# Composite key used by GroupKFold — ensures whole plays stay in one fold.
GROUP_COLUMN: str = "play_group_key"

# Positions treated as coverage defenders for kinematic base extraction.
# Kept in sync with src/coverage_assignment.py::COVERAGE_POSITIONS.
_COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)

_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_DEFENDER_KEY: list[str] = _FRAME_KEY + ["defender_nflId"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the NFL Coverage Responsibility multi-class XGBoost model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to write trained model artifacts.",
    )
    p.add_argument(
        "--tracking-week",
        type=int,
        default=1,
        help="Tracking file week number (e.g. 1 → week1.csv).",
    )
    p.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Restrict ingestion to a single gameId (useful for smoke-tests).",
    )
    p.add_argument(
        "--grid-resolution",
        type=float,
        default=1.0,
        help="Voronoi field-grid resolution in yards.",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold folds for cross-validation.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Short-circuit Optuna tuning and boosting rounds for rapid pipeline "
            "verification: 2 Optuna trials, 2 CV folds, max 50 boosting rounds "
            "throughout. Combine with --game-id for sub-30-second iteration."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# 3. Data ingestion
# ---------------------------------------------------------------------------


def ingest_tracking(
    data_dir: Path,
    week: int,
    game_id: Optional[int],
) -> pd.DataFrame:
    """Load, merge, and spatially normalize all raw tracking + metadata CSVs."""
    tracking_path = data_dir / f"week{week}.csv"
    players_path = data_dir / "players.csv"
    pff_path = data_dir / "pffScoutingData.csv"
    plays_path = data_dir / "plays.csv"
    games_path = data_dir / "games.csv"

    for p in (tracking_path, players_path, pff_path, plays_path, games_path):
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    log.info(
        "Ingesting tracking data — week=%d, game_id_filter=%s", week, game_id
    )
    df = load_and_build(
        tracking_path=str(tracking_path),
        players_path=str(players_path),
        pff_path=str(pff_path),
        plays_path=str(plays_path),
        games_path=str(games_path),
        game_id=game_id,
        play_id=None,
    )
    log.info("Normalized tracking shape: %s | games: %d | plays: %d",
             df.shape,
             df["gameId"].nunique(),
             df["playId"].nunique())
    return df


# ---------------------------------------------------------------------------
# 4. Feature matrix assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(
    tracking_df: pd.DataFrame,
    grid_resolution: float,
) -> pd.DataFrame:
    """
    Assemble the defender-centric feature matrix.

    Row identity: (gameId, playId, frameId, defender_nflId).

    Assembly order
    --------------
    1. Kinematic base  – all coverage-position rows from normalized tracking.
    2. Voronoi assignments – left-joined to populate assigned_receiver_nflId;
       when a defender covers multiple receivers (rare), keep the first
       assignment per (frame, defender) to maintain row uniqueness.
    3. Voronoi territory metrics – territory_area_sq_yd, territory_grid_points.
    4. Receiver cushion   – Euclidean distance to assigned/nearest receiver.
    5. Spatial leverage   – signed (x, y) offset to assigned/nearest receiver.
    6. Pre-snap safety depth – play-level stats broadcast to every frame.
    """
    log.info("Building field grid at %.1f yd resolution...", grid_resolution)
    grid_xy = build_field_grid(resolution=grid_resolution)

    # ── 4a. Kinematic base (one row per defender per frame) ─────────────────
    kine_cols = _FRAME_KEY + ["nflId", "x", "y", "s", "a", "o_rad", "dir_rad"]
    kine_df = (
        tracking_df
        .loc[tracking_df["officialPosition"].isin(_COVERAGE_POSITIONS), kine_cols]
        .rename(columns={"nflId": "defender_nflId"})
        .reset_index(drop=True)
    )
    log.info("Kinematic base: %d defender-frame rows.", len(kine_df))

    # ── 4b. Voronoi coverage assignments ─────────────────────────────────────
    log.info("Computing frame-by-frame Voronoi coverage assignments...")
    assignments_df = build_coverage_assignments(
        tracking_df, grid_resolution=grid_resolution, grid_xy=grid_xy
    )
    # build_coverage_assignments is receiver-centric (one row per receiver per
    # frame).  Deduplicate on (frame, defender) to keep the model row-unique;
    # when a defender is assigned to multiple receivers the first alphabetically
    # by receiver nflId is retained — this edge case is flagged in the log.
    dup_mask = assignments_df.duplicated(subset=_DEFENDER_KEY, keep=False)
    n_multi = dup_mask.sum()
    if n_multi > 0:
        log.debug(
            "%d rows have a defender assigned to >1 receiver in the same frame "
            "(keeping first). This may indicate zone-like spread or a data artifact.",
            n_multi,
        )
    assignments_dedup = (
        assignments_df
        .sort_values("assigned_receiver_nflId")
        .drop_duplicates(subset=_DEFENDER_KEY, keep="first")
    )

    # ── 4c. Voronoi territory metrics ─────────────────────────────────────────
    log.info("Computing Voronoi territory metrics...")
    territory_df = compute_defender_territories(
        tracking_df, grid_xy=grid_xy, grid_resolution=grid_resolution
    )

    # ── 4d. Receiver cushion ──────────────────────────────────────────────────
    log.info("Computing receiver cushion per defender per frame...")
    cushion_df = compute_receiver_cushion(tracking_df, fallback_to_nearest=True)
    cushion_df = cushion_df.rename(columns={"nflId": "defender_nflId"})

    # ── 4e. Spatial leverage ──────────────────────────────────────────────────
    log.info("Computing spatial leverage vectors...")
    leverage_df = compute_spatial_leverage(tracking_df, fallback_to_nearest=True)
    leverage_df = leverage_df.rename(columns={"nflId": "defender_nflId"})

    # ── 4f. Pre-snap safety depth (play-level) ────────────────────────────────
    log.info("Computing pre-snap safety depth...")
    safety_df = compute_presnap_safety_depth(tracking_df)

    # ── 4g. Sequential left-join onto kinematic base ──────────────────────────
    log.info("Assembling feature matrix via sequential left-joins...")
    feature_df = (
        kine_df
        .merge(
            assignments_dedup[_DEFENDER_KEY + ["assigned_receiver_nflId"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            territory_df[
                _DEFENDER_KEY + ["territory_grid_points", "territory_area_sq_yd"]
            ],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            cushion_df[_DEFENDER_KEY + ["receiver_cushion"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            leverage_df[_DEFENDER_KEY + ["leverage_x", "leverage_y"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            safety_df[["gameId", "playId", "safety_depth_mean", "safety_depth_std"]],
            on=["gameId", "playId"],
            how="left",
        )
    )

    log.info(
        "Feature matrix assembled: %s | NaN counts:\n%s",
        feature_df.shape,
        feature_df[FEATURE_COLUMNS].isna().sum().to_string(),
    )
    return feature_df


# ---------------------------------------------------------------------------
# 5. Target label construction
# ---------------------------------------------------------------------------


def build_target_labels(
    feature_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive the multi-class coverage-responsibility label for every row.

    Labelling priority
    ------------------
    1. **Zone flag from PFF**: if ``pff_coverage`` is present and does NOT
       contain the substring ``"man"`` (case-insensitive), mark as
       ``LABEL_ZONE``.
    2. **No Voronoi assignment**: if ``assigned_receiver_nflId`` is NaN
       (defender never won a Voronoi cell containing a receiver), mark as
       ``LABEL_ZONE``.
    3. **Man assignment**: use ``str(int(assigned_receiver_nflId))`` as the
       label, uniquely identifying the specific offensive player.

    Rows where PFF data is absent fall through to criterion 2.

    Returns
    -------
    feature_df with a new ``TARGET_COLUMN`` column appended.
    """
    feature_df = feature_df.copy()

    # Resolve PFF coverage type once per (gameId, playId, nflId)
    if "pff_coverage" in tracking_df.columns:
        pff_cols = ["gameId", "playId", "nflId", "pff_coverage"]
        pff_snap = (
            tracking_df[pff_cols]
            .drop_duplicates(subset=["gameId", "playId", "nflId"])
            .rename(columns={"nflId": "defender_nflId"})
        )
        feature_df = feature_df.merge(
            pff_snap,
            on=["gameId", "playId", "defender_nflId"],
            how="left",
        )
        pff_is_zone: pd.Series = (
            feature_df["pff_coverage"].isna()
            | ~feature_df["pff_coverage"].str.lower().str.contains("man", na=False)
        )
    else:
        log.warning(
            "pff_coverage column absent — zone/man classification relies "
            "entirely on Voronoi assignment presence."
        )
        pff_is_zone = pd.Series(False, index=feature_df.index)

    no_voronoi_assignment: pd.Series = feature_df["assigned_receiver_nflId"].isna()

    is_zone: pd.Series = pff_is_zone | no_voronoi_assignment

    feature_df[TARGET_COLUMN] = np.where(
        is_zone,
        LABEL_ZONE,
        feature_df["assigned_receiver_nflId"]
        .dropna()
        .astype(np.int64)
        .astype(str)
        .reindex(feature_df.index, fill_value=LABEL_ZONE),
    )

    label_counts = feature_df[TARGET_COLUMN].value_counts()
    zone_pct = 100.0 * label_counts.get(LABEL_ZONE, 0) / len(feature_df)
    log.info(
        "Target label distribution (%d total rows, %.1f%% zone):\n%s",
        len(feature_df),
        zone_pct,
        label_counts.head(25).to_string(),
    )
    return feature_df


# ---------------------------------------------------------------------------
# 6. NaN sentinel fill
# ---------------------------------------------------------------------------


def fill_feature_nans(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all NaN values in FEATURE_COLUMNS with NAN_FILL_VALUE (-999.0).

    Any expected feature column absent from the DataFrame is synthesised
    as a constant column of NAN_FILL_VALUE and logged as a warning so
    downstream CI surfaces the gap immediately.
    """
    feature_df = feature_df.copy()

    missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
    if missing_cols:
        log.warning(
            "Expected feature columns not found in DataFrame — "
            "filling with NAN_FILL_VALUE sentinel: %s",
            missing_cols,
        )
        for col in missing_cols:
            feature_df[col] = NAN_FILL_VALUE

    existing_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    nan_before = int(feature_df[existing_cols].isna().sum().sum())
    feature_df[existing_cols] = feature_df[existing_cols].fillna(NAN_FILL_VALUE)
    log.info(
        "NaN fill complete: %d values replaced with sentinel %.1f across %d columns.",
        nan_before,
        NAN_FILL_VALUE,
        len(existing_cols),
    )
    return feature_df


# ---------------------------------------------------------------------------
# 7. Label encoding
# ---------------------------------------------------------------------------


def fit_label_encoder(
    labels: pd.Series,
) -> tuple[LabelEncoder, np.ndarray]:
    """
    Fit a LabelEncoder over all observed coverage labels.

    Guarantees
    ----------
    * ``LABEL_ZONE`` ("No_Matchup_Zone") is present in ``le.classes_`` and its
      integer index is logged explicitly — downstream SHAP analyses reference
      this index when explaining zone-vs-man boundaries.
    * The encoder is deterministic: given the same label set, the same integer
      mapping is always produced (sklearn sorts classes_ alphabetically).

    Returns
    -------
    le       : fitted LabelEncoder
    classes_ : ordered class array (le.classes_)
    """
    le = LabelEncoder()
    le.fit(labels)

    if LABEL_ZONE not in le.classes_:
        raise ValueError(
            f"'{LABEL_ZONE}' was not found in target labels. "
            "Check PFF coverage ingestion — every play must produce at least "
            "one zone-labelled defender row."
        )

    zone_index = int(np.where(le.classes_ == LABEL_ZONE)[0][0])
    log.info(
        "LabelEncoder fitted: %d classes | '%s' → encoded index %d",
        len(le.classes_),
        LABEL_ZONE,
        zone_index,
    )
    return le, le.classes_


# ---------------------------------------------------------------------------
# 8. Leakage-free group split
# ---------------------------------------------------------------------------


def _game_hash_fraction(game_id: int) -> float:
    """
    Map a gameId to a deterministic float in [0, 1) via MD5.

    The first 8 hex digits of MD5(str(gameId)) are interpreted as a 32-bit
    unsigned integer and divided by 0xFFFFFFFF, yielding a stable, uniform
    pseudo-random value that is identical across Python versions and platforms.
    """
    digest = hashlib.md5(str(game_id).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFF_FFFF


def split_by_game(
    feature_df: pd.DataFrame,
    train_threshold: float = TRAIN_HASH_THRESHOLD,
) -> tuple[pd.Index, pd.Index]:
    """
    Deterministic 80/20 split at the **game** level.

    Every frame of every play within a gameId is atomically routed to either
    the training set or the test set.  No individual row shuffle ever occurs.

    The hash-based approach (vs. sklearn train_test_split) guarantees:
      * Reproducibility without a random seed argument.
      * Zero possibility of frame-level correlation bleed between train/test.
      * Identical splits when re-run on subsets (e.g. a single week vs. all
        weeks), since each game's assignment depends only on its own gameId.

    Returns
    -------
    train_idx : pd.Index  positional (integer) index into feature_df for train rows
    test_idx  : pd.Index  positional (integer) index into feature_df for test rows
    """
    game_ids: np.ndarray = feature_df["gameId"].unique()
    train_games = {g for g in game_ids if _game_hash_fraction(int(g)) < train_threshold}
    test_games = {g for g in game_ids if g not in train_games}

    train_mask = feature_df["gameId"].isin(train_games)
    test_mask = feature_df["gameId"].isin(test_games)

    train_idx = feature_df.index[train_mask]
    test_idx = feature_df.index[test_mask]

    actual_train_pct = 100.0 * train_mask.sum() / len(feature_df)
    log.info(
        "Game-level split: %d train games (%d rows, %.1f%%) | "
        "%d test games (%d rows, %.1f%%)",
        len(train_games), train_mask.sum(), actual_train_pct,
        len(test_games), test_mask.sum(), 100.0 - actual_train_pct,
    )
    return train_idx, test_idx


def build_group_kfold_splits(
    feature_df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build GroupKFold cross-validation splits keyed on composite
    ``gameId_playId``.

    Each fold guarantees zero play overlap between its inner train and
    validation subsets — suitable for hyperparameter search with Optuna or
    GridSearchCV without risk of intra-play correlation.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Must contain TARGET_COLUMN and all FEATURE_COLUMNS.
        Typically the training-set slice (feature_df.loc[train_idx]).
    n_splits : int
        Number of folds.

    Returns
    -------
    List of (train_positions, val_positions) index arrays (positional into
    the passed feature_df).
    """
    df = feature_df.copy()
    df[GROUP_COLUMN] = (
        df["gameId"].astype(str) + "_" + df["playId"].astype(str)
    )

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].values
    y = df[TARGET_COLUMN].values
    groups = df[GROUP_COLUMN].values

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups=groups))

    n_unique_groups = len(np.unique(groups))
    log.info(
        "GroupKFold: %d folds over %d unique plays (gameId_playId groups).",
        n_splits,
        n_unique_groups,
    )
    return folds


# ---------------------------------------------------------------------------
# 9. Data integrity validation
# ---------------------------------------------------------------------------


def validate_split(
    feature_df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    le: LabelEncoder,
) -> None:
    """
    Assert all data-integrity contracts before model training begins.

    Checks (all must pass — any failure raises AssertionError/ValueError)
    -----------------------------------------------------------------------
    1. **Disjoint play sets**: the intersection of composite
       ``gameId_playId`` keys between train and test is exactly empty.
       A non-empty intersection would be direct evidence of data leakage.

    2. **No residual NaNs**: all FEATURE_COLUMNS present in the DataFrame
       must be NaN-free after the sentinel fill step.

    3. **Label encoder completeness**: every label observed in
       TARGET_COLUMN is covered by the fitted LabelEncoder, and
       ``LABEL_ZONE`` is guaranteed to be present.
    """
    log.info("Running data integrity validation...")

    train_df = feature_df.loc[train_idx]
    test_df = feature_df.loc[test_idx]

    # ── Check 1: Strictly disjoint play sets ─────────────────────────────────
    def _play_keys(df: pd.DataFrame) -> set[str]:
        return set(df["gameId"].astype(str) + "_" + df["playId"].astype(str))

    train_plays = _play_keys(train_df)
    test_plays = _play_keys(test_df)
    leaked_plays = train_plays & test_plays

    assert len(leaked_plays) == 0, (
        f"DATA LEAKAGE DETECTED: {len(leaked_plays)} play(s) appear in both "
        f"train and test sets.\nExample leaks: {sorted(leaked_plays)[:10]}"
    )
    log.info(
        "[PASS] Play disjointness: 0 leaked plays "
        "(%d train plays | %d test plays).",
        len(train_plays),
        len(test_plays),
    )

    # ── Check 2: NaN-free feature columns ────────────────────────────────────
    present_feature_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    nan_counts = feature_df[present_feature_cols].isna().sum()
    dirty_cols = nan_counts[nan_counts > 0]

    if not dirty_cols.empty:
        raise ValueError(
            f"Residual NaNs detected in feature columns after sentinel fill.\n"
            f"Columns with NaN counts:\n{dirty_cols.to_string()}\n"
            "Ensure fill_feature_nans() ran before validate_split()."
        )
    log.info(
        "[PASS] NaN check: all %d feature columns are clean.",
        len(present_feature_cols),
    )

    # ── Check 3: Label encoder covers all observed labels ────────────────────
    live_labels = set(feature_df[TARGET_COLUMN].unique())
    known_labels = set(le.classes_)
    unseen_labels = live_labels - known_labels

    assert not unseen_labels, (
        f"LabelEncoder does not cover all observed labels.\n"
        f"Unseen labels: {unseen_labels}\n"
        "Re-fit the encoder on the full dataset, not a subset."
    )
    assert LABEL_ZONE in known_labels, (
        f"'{LABEL_ZONE}' is missing from LabelEncoder — zone coverage "
        "frames will be mis-classified as an unknown class."
    )
    log.info(
        "[PASS] Label encoder: %d observed classes all known | '%s' present.",
        len(live_labels),
        LABEL_ZONE,
    )

    log.info("All integrity checks passed. Safe to proceed to training.")


# ---------------------------------------------------------------------------
# 10. Class imbalance — balanced sample weights
# ---------------------------------------------------------------------------


def compute_sample_weights(
    y_encoded: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    """
    Derive per-sample weights that correct for ``No_Matchup_Zone`` dominance.

    Strategy
    --------
    sklearn's ``'balanced'`` mode computes::

        w_c = n_samples / (n_classes × n_samples_in_class_c)

    These weights are mapped from class-level to sample-level and passed
    directly to XGBoost's ``sample_weight`` parameter at fit time.

    **Down-sampling is deliberately avoided.**  Removing zone frames would
    destroy the temporal continuity of the spatial timeseries, losing the
    slow evolution of defender positioning that encodes zone-coverage intent.

    Returns
    -------
    sample_weights : np.ndarray of shape (n_samples,)
        One weight per training row.
    """
    class_weights: np.ndarray = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(le.classes_)),
        y=y_encoded,
    )

    zone_idx = int(np.where(le.classes_ == LABEL_ZONE)[0][0])
    log.info(
        "Balanced class weights computed: %d classes | "
        "'%s' (idx=%d) weight=%.6f | "
        "min weight=%.6f | max weight=%.6f",
        len(le.classes_),
        LABEL_ZONE,
        zone_idx,
        class_weights[zone_idx],
        class_weights.min(),
        class_weights.max(),
    )

    sample_weights = class_weights[y_encoded]
    return sample_weights.astype(np.float32)


# ---------------------------------------------------------------------------
# 11. XGBoost model compilation
# ---------------------------------------------------------------------------


def compile_xgboost_model(
    num_classes: int,
    params: Optional[dict] = None,
):  # type: ignore[return]
    """
    Initialise the XGBoost multi-class classifier.

    Parameters
    ----------
    num_classes : int
        Number of coverage-responsibility classes (output neurons).
    params : dict, optional
        Hyperparameter overrides — typically the ``best_params`` dict
        returned by :func:`optimize_hyperparameters`.  Any key present
        here overrides the corresponding conservative default.  When
        ``None``, the baseline football-analytics defaults below are used.

    Notes
    -----
    * ``tree_method="hist"`` enables histogram-based split finding (fast on
      CPU).  Switch to ``device="cuda"`` for GPU acceleration.
    * ``sample_weight`` is injected at ``model.fit()`` time, not here, so
      the DMatrix-level weight array is fully controlled by the caller.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed.  Add it to requirements.txt "
            "and run: pip install xgboost"
        ) from exc

    base_params: dict = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    if params:
        base_params.update(params)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **base_params,
    )
    source = "Optuna-tuned" if params else "baseline defaults"
    log.info(
        "XGBoost model compiled (%s): objective=multi:softprob | "
        "num_class=%d | n_estimators=%d | max_depth=%d | lr=%.4f",
        source,
        num_classes,
        model.n_estimators,
        model.max_depth,
        model.learning_rate,
    )
    return model


# ---------------------------------------------------------------------------
# 11.5. Optuna hyperparameter optimization
# ---------------------------------------------------------------------------


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    sample_weights: np.ndarray,
    num_classes: int,
    n_trials: int = 20,
    smoke_test: bool = False,
) -> dict:
    """
    Run an Optuna TPE sweep over GroupKFold splits to find the best XGBoost
    hyperparameters for the multi-class coverage-responsibility problem.

    For each trial the objective function:

    1. Iterates over all ``cv_folds`` (positional index arrays into ``X_train``).
    2. Slices fold-train / fold-validation arrays and the matching
       ``sample_weights`` slice.
    3. Fits a native XGBoost booster via ``xgb.DMatrix`` + ``xgb.train()`` so
       that ``num_class`` in the params dict is the sole authority on class
       count — preventing the scikit-learn wrapper from inferring classes from
       ``np.unique(y_fold_train)`` and crashing when a fold slice is missing a
       minority class.
    4. Collects the best out-of-fold ``mlogloss`` across all evaluated rounds.

    The average OOF ``mlogloss`` across all folds is the minimisation metric.

    Search spaces
    -------------
    ``n_estimators``     : int  [200, 1500, step=100]
    ``max_depth``        : int  [4, 8]   — shallow enough to avoid memorising
                                           Voronoi tracking anomalies.
    ``learning_rate``    : float [0.01, 0.1, log=True]
    ``subsample``        : float [0.6, 0.9] — forces broad positional coverage.
    ``colsample_bytree`` : float [0.6, 0.9] — explores all kinematic / Voronoi
                                              feature families each tree.
    ``min_child_weight`` : int  [5, 30]  — prevents individual noisy frames
                                           from creating deep leaves.
    ``reg_alpha``        : float [1e-3, 10.0, log=True] — L1 sparsity.
    ``reg_lambda``       : float [1e-3, 10.0, log=True] — L2 shrinkage.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train_samples, n_features)
    y_train : np.ndarray of shape (n_train_samples,)  — integer-encoded labels.
    cv_folds : list of (fold_train_pos, fold_val_pos) positional index arrays,
        as returned by :func:`build_group_kfold_splits`.
    sample_weights : np.ndarray of shape (n_train_samples,)
        Full training-set balanced weights; sliced per fold inside objective.
    num_classes : int
        Total number of coverage-responsibility classes.
    n_trials : int
        Number of Optuna trials to execute (default 20).
    smoke_test : bool
        When ``True`` clamps the search to 2 folds and caps ``n_boost_rounds``
        at 50 per fold, enabling sub-30-second pipeline verification runs.

    Returns
    -------
    dict
        ``study.best_params`` — ready to pass into :func:`compile_xgboost_model`.
    """
    import xgboost as xgb  # already guarded at module level; re-import for clarity

    # Smoke-test guards: restrict fold count and trial budget before entering
    # the study so every hot path inside objective() runs at minimal cost.
    active_folds = cv_folds[:2] if smoke_test else cv_folds
    _SMOKE_MAX_ROUNDS = 50

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        # Extract n_estimators separately — it becomes num_boost_round in the
        # native xgb.train() API rather than a booster param key.
        n_boost_rounds = trial.suggest_int("n_estimators", 200, 1500, step=100)
        if smoke_test:
            n_boost_rounds = min(n_boost_rounds, _SMOKE_MAX_ROUNDS)

        # Use the native booster params dict rather than XGBClassifier so that
        # num_class is the sole authority on class count.  XGBClassifier's
        # sklearn wrapper infers classes from np.unique(y_fold_train), which
        # crashes with a class-mismatch ValueError whenever a fold slice is
        # missing one or more minority classes — common in single-game smoke
        # tests.  xgb.DMatrix accepts integer labels verbatim; missing classes
        # in a fold simply contribute zero training examples for that class.
        booster_params: dict = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": RANDOM_SEED,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        fold_scores: list[float] = []
        for fold_train_pos, fold_val_pos in active_folds:
            dtrain = xgb.DMatrix(
                X_train[fold_train_pos],
                label=y_train[fold_train_pos],
                weight=sample_weights[fold_train_pos],
            )
            dval = xgb.DMatrix(
                X_train[fold_val_pos],
                label=y_train[fold_val_pos],
            )

            evals_result: dict = {}
            xgb.train(
                booster_params,
                dtrain,
                num_boost_round=n_boost_rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                evals_result=evals_result,
                verbose_eval=False,
            )
            fold_scores.append(float(min(evals_result["val"]["mlogloss"])))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    log.info(
        "Optuna TPE study started: %d trials × %d GroupKFold folds%s...",
        n_trials,
        len(active_folds),
        " [smoke-test mode: capped at 50 rounds/fold]" if smoke_test else "",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info(
        "Optuna study complete: %d trials finished | best OOF mlogloss=%.6f",
        len(study.trials),
        study.best_value,
    )
    return study.best_params


# ---------------------------------------------------------------------------
# 12. SHAP explainer initialisation  [PLACEHOLDER — run after fit]
# ---------------------------------------------------------------------------


def init_shap_explainer(model, X_train: np.ndarray):  # type: ignore[return]
    """
    Initialise a SHAP TreeExplainer for post-training interpretability.

    Call this immediately after ``model.fit()`` and persist the explainer
    alongside the model artifact.  At evaluation time, use::

        shap_values = explainer.shap_values(X_test)   # shape: (n_samples, n_features, n_classes)
        # shap_values[zone_class_idx] → per-feature attribution for No_Matchup_Zone

    The SHAP values over Voronoi territory and leverage features feed
    directly into the Spatial Disguise Index decomposition described in the
    README.

    Returns None if shap is not installed (non-fatal; logs a warning).
    """
    try:
        import shap
    except ImportError:
        log.warning(
            "shap is not installed — SHAP explainer skipped. "
            "Install with: pip install shap"
        )
        return None

    explainer = shap.TreeExplainer(model)
    log.info(
        "SHAP TreeExplainer initialised over %d training samples, "
        "%d features.",
        X_train.shape[0],
        X_train.shape[1],
    )
    return explainer


# ---------------------------------------------------------------------------
# 13. Orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:  # noqa: PLR0915
    args = parse_args(argv)
    setup_logging(args.log_level)

    log.info("=" * 72)
    log.info("NFL Coverage Responsibility — Training Pipeline")
    log.info("=" * 72)
    log.info("Run config: %s", vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ingest & normalize ───────────────────────────────────────────
    tracking_df = ingest_tracking(args.data_dir, args.tracking_week, args.game_id)

    # ── Step 2: Assemble defender-centric feature matrix ─────────────────────
    feature_df = build_feature_matrix(tracking_df, args.grid_resolution)

    # ── Step 3: Derive coverage-responsibility labels ─────────────────────────
    feature_df = build_target_labels(feature_df, tracking_df)

    # ── Step 4: Fill NaN features with explicit sentinel ─────────────────────
    feature_df = fill_feature_nans(feature_df)

    # ── Step 5: Encode labels (LabelEncoder preserves No_Matchup_Zone mapping) ─
    le, classes = fit_label_encoder(feature_df[TARGET_COLUMN])
    y_encoded = le.transform(feature_df[TARGET_COLUMN].values).astype(np.int32)

    # ── Step 6: Deterministic leakage-free train/test split ──────────────────
    train_idx, test_idx = split_by_game(feature_df)

    # ── Step 7: Data integrity validation ────────────────────────────────────
    validate_split(feature_df, train_idx, test_idx, le)

    # ── Step 8: Extract typed NumPy arrays ───────────────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    X = feature_df[available_features].values.astype(np.float32)

    # Convert pd.Index to positional for NumPy indexing
    train_pos = feature_df.index.get_indexer(train_idx)
    test_pos = feature_df.index.get_indexer(test_idx)

    X_train, y_train = X[train_pos], y_encoded[train_pos]
    X_test, y_test = X[test_pos], y_encoded[test_pos]

    log.info(
        "Array shapes → X_train=%s  X_test=%s  y_train=%s  y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    # ── Step 9: Balanced sample weights for No_Matchup_Zone correction ───────
    sample_weights_train = compute_sample_weights(y_train, le)

    # ── Step 10: GroupKFold splits for cross-validation / Optuna sweeps ──────
    cv_folds = build_group_kfold_splits(
        feature_df.loc[train_idx].reset_index(drop=True),
        n_splits=args.n_splits,
    )
    log.info("%d GroupKFold folds ready for hyperparameter search.", len(cv_folds))

    # ── Step 11: Optuna hyperparameter optimization ───────────────────────────
    n_optuna_trials = 2 if args.smoke_test else 20
    if _OPTUNA_AVAILABLE:
        log.info(
            "Launching Optuna hyperparameter sweep: %d trials × %d GroupKFold folds%s...",
            n_optuna_trials,
            min(len(cv_folds), 2) if args.smoke_test else len(cv_folds),
            " [smoke-test mode]" if args.smoke_test else "",
        )
        best_params = optimize_hyperparameters(
            X_train,
            y_train,
            cv_folds,
            sample_weights_train,
            num_classes=len(classes),
            n_trials=n_optuna_trials,
            smoke_test=args.smoke_test,
        )
        log.info("Best hyperparameters discovered by Optuna:")
        for k, v in sorted(best_params.items()):
            log.info("  %-22s = %s", k, v)
    else:
        log.warning(
            "optuna is not installed — hyperparameter sweep skipped.  "
            "Install with: pip install optuna  "
            "Proceeding with baseline XGBoost hyperparameters."
        )
        best_params = None

    # ── Step 12: Compile final production model with optimized parameters ─────
    # In smoke-test mode cap n_estimators at 50 so the production fit completes
    # in seconds regardless of what Optuna sampled.
    if args.smoke_test:
        smoke_params = dict(best_params or {})
        smoke_params["n_estimators"] = min(smoke_params.get("n_estimators", 500), 50)
        model = compile_xgboost_model(num_classes=len(classes), params=smoke_params)
        log.warning(
            "smoke-test mode: final fit capped at n_estimators=%d.",
            smoke_params["n_estimators"],
        )
    else:
        model = compile_xgboost_model(num_classes=len(classes), params=best_params)

    # ── Step 13: Final production fit on full train split ─────────────────────
    # When --game-id restricts ingestion to a single game the deterministic
    # hash split can route 100 % of rows to X_train, leaving X_test empty.
    # Passing an empty matrix to XGBoost's eval_set triggers a
    # '-nan(ind)' ValueError from the metric parser.  Fall back to the first
    # GroupKFold validation slice so smoke tests always have a valid eval set.
    if len(X_test) > 0:
        fit_eval_set = [(X_test, y_test)]
        eval_label = "locked-out test set"
    else:
        val_fold_pos = cv_folds[0][1]
        X_val_smoke = X_train[val_fold_pos]
        y_val_smoke = y_train[val_fold_pos]
        fit_eval_set = [(X_val_smoke, y_val_smoke)]
        eval_label = "GroupKFold fold-0 validation (smoke-test fallback — X_test is empty)"
        log.warning(
            "X_test is empty (single-game smoke test?). "
            "Eval set for final fit replaced with GroupKFold fold-0 "
            "validation slice (%d rows).",
            len(val_fold_pos),
        )

    log.info(
        "Final production fit | %d train rows | %d classes | "
        "sample_weight injected | eval against %s...",
        len(X_train),
        len(classes),
        eval_label,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        eval_set=fit_eval_set,
        verbose=50,
    )

    # ── Step 14: Persist artifacts ───────────────────────────────────────────
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is not installed.  Add it to requirements.txt "
            "and run: pip install joblib"
        ) from exc

    le_path = args.output_dir / "label_encoder.pkl"
    model_path = args.output_dir / "coverage_model.pkl"
    feat_path = args.output_dir / "feature_columns.txt"

    joblib.dump(le, le_path)
    joblib.dump(model, model_path)
    feat_path.write_text("\n".join(available_features), encoding="utf-8")

    log.info("Artifacts saved:")
    log.info("  Label encoder  → %s", le_path)
    log.info("  XGBoost model  → %s", model_path)
    log.info("  Feature list   → %s", feat_path)

    # ── Step 15: SHAP TreeExplainer ──────────────────────────────────────────
    explainer = init_shap_explainer(model, X_train)
    if explainer is not None:
        shap_path = args.output_dir / "shap_explainer.pkl"
        joblib.dump(explainer, shap_path)
        log.info("  SHAP explainer → %s", shap_path)

    log.info("=" * 72)
    log.info("Pipeline complete.  All artifacts written to: %s", args.output_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
