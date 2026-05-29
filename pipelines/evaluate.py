"""
pipelines/evaluate.py
=====================
Production evaluation and translation pipeline for the NFL Coverage
Responsibility multi-class XGBoost model.

Entry point
-----------
    # Production run (locked-out test games, full dataset):
    python pipelines/evaluate.py \\
        --data-dir  data/raw \\
        --model-dir models \\
        --output-dir outputs/eval

    # Single-week production evaluation (week 1):
    python pipelines/evaluate.py \\
        --data-dir  data/raw \\
        --model-dir models \\
        --output-dir outputs/eval \\
        --tracking-weeks 1

    # Local smoke-test (bypasses MD5 hash, forces all rows as test set):
    python pipelines/evaluate.py \\
        --data-dir  data/raw \\
        --model-dir models \\
        --output-dir outputs/smoke \\
        --game-ids 2021090900 \\
        --smoke-test

Architecture
------------
This pipeline is intentionally self-contained: it imports only from the
``src/`` library modules and the standard library.  No cross-dependency on
``pipelines/train.py`` exists so that the evaluation artifact can be run
independently in a CI/CD context without the training environment.

The deterministic MD5 hash split logic (``TRAIN_HASH_THRESHOLD = 0.80``) is
re-implemented here, bit-for-bit identical to the training pipeline, guaranteeing
that the exact same game-level partition is recovered without needing to ship
the training data split index as a separate artifact.

Feature column ordering is loaded at runtime from ``feature_columns.txt`` in
``--model-dir`` — the canonical artifact written by ``pipelines/train.py``.

Outputs written to ``--output-dir``
------------------------------------
classification_report.txt    — sklearn precision/recall/F1 per class on test set.
matchup_matrix.[parquet|csv] — Frame-aggregated defender–receiver assignment
                               share matrix per play (long format, dashboard-ready).
spatial_disguise_index.[parquet|csv]
                             — Per (game, play, defender) SDI scores [0–100].
burn_risk_log.[parquet|csv]  — Per (game, play, defender) failure-frame and
                               normalized burn-risk score.

No interactive matplotlib windows are ever opened.
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

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.coverage_assignment import (
    build_coverage_assignments,
    build_field_grid,
    compute_defender_territories,
)
from src.data.normalizer import (
    build_normalized_tracking,
    load_games,
    load_players,
    load_plays,
    load_pff_scouting,
)
from src.features.coverage_features import (
    compute_presnap_safety_depth,
    compute_receiver_cushion,
    compute_spatial_leverage,
    compute_velocity_angular_divergence,
)

# ---------------------------------------------------------------------------
# Module-level constants
# These values MUST remain bit-for-bit identical to pipelines/train.py.
# Any change here must be mirrored there, and vice versa.
# ---------------------------------------------------------------------------

#: String label used for zone-coverage / no-man-assignment frames.
LABEL_ZONE: str = "No_Matchup_Zone"

#: Sentinel fill value for missing features — matches XGBoost's native NaN handling.
NAN_FILL_VALUE: float = -999.0

#: Games whose MD5 digest maps to [0, threshold) → train; rest → test.
TRAIN_HASH_THRESHOLD: float = 0.80

#: Event string marking the ball snap in the NGS tracking stream.
SNAP_EVENT: str = "ball_snap"

#: Column name for the derived multi-class coverage label.
TARGET_COLUMN: str = "coverage_label"

#: Defensive positions included in the coverage model.
_COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)

_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_DEFENDER_KEY: list[str] = _FRAME_KEY + ["defender_nflId"]

# ---------------------------------------------------------------------------
# SDI — Spatial Disguise Index configuration
# ---------------------------------------------------------------------------

#: Feature names whose SHAP attributions drive the disguise signal.
#: Must be a subset of the serialized feature_columns.txt.
_SDI_FEATURE_NAMES: tuple[str, ...] = (
    "territory_area_sq_yd",
    "territory_grid_points",
    "leverage_x",
    "leverage_y",
)

# ---------------------------------------------------------------------------
# Burn-risk operational thresholds
# ---------------------------------------------------------------------------

#: Cushion delta (yards/frame) that constitutes a rapid-press collapse event.
_CUSHION_DECAY_THRESHOLD: float = -1.5

#: Minimum man-assignment probability for a surge event to be recorded.
_SURGE_PROB_THRESHOLD: float = 0.35

#: Fractional Voronoi territory drop (≥ 30 %) that triggers a landmark
#: discipline failure.
_TERRITORY_COLLAPSE_RATIO: float = 0.30

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the NFL Coverage Responsibility multi-class XGBoost model "
            "and generate coaching-staff analytics: Matchup Matrix, "
            "Spatial Disguise Index, and Burn Risk Log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files (tracking, players, plays, etc.).",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model artifacts.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory to write evaluation outputs.",
    )
    p.add_argument(
        "--tracking-weeks",
        nargs="+",
        type=int,
        default=[1],
        metavar="WEEK",
        help=(
            "One or more tracking week numbers to load "
            "(e.g. --tracking-weeks 1 → week1.csv). "
            "Duplicate values are silently deduplicated."
        ),
    )
    p.add_argument(
        "--game-ids",
        nargs="*",
        type=int,
        default=None,
        metavar="GAME_ID",
        help=(
            "Restrict evaluation to specific game IDs across the loaded weeks. "
            "Omit entirely to evaluate all games from the specified weeks. "
            "Example: --game-ids 2021090900 2021091300"
        ),
    )
    p.add_argument(
        "--grid-resolution",
        type=float,
        default=1.0,
        help="Voronoi field-grid resolution in yards.",
    )
    p.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output serialisation format for analytics tables.",
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
            "Force the entire ingested matrix to act as the evaluation test set, "
            "bypassing the MD5 hash partition.  Use with --game-ids for rapid local "
            "pipeline validation when the target game routes entirely to the training "
            "split under the deterministic hash."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 2.  Logging
# ---------------------------------------------------------------------------


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# 3.  Artifact loading
# ---------------------------------------------------------------------------


def load_artifacts(
    model_dir: Path,
) -> tuple:
    """
    Deserialise all model artifacts from *model_dir*.

    Returns
    -------
    model        : fitted XGBClassifier
    le           : fitted LabelEncoder
    explainer    : SHAP TreeExplainer, or ``None`` if unavailable
    feature_cols : ordered ``list[str]`` loaded from ``feature_columns.txt``
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load model artifacts.  "
            "Install with: pip install joblib"
        ) from exc

    required = {
        "coverage_model.pkl": "XGBoost model",
        "label_encoder.pkl": "LabelEncoder",
        "feature_columns.txt": "feature column list",
    }
    for filename, desc in required.items():
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact ({desc}) not found: {path}"
            )

    log.info("Loading model artifacts from: %s", model_dir)
    model = joblib.load(model_dir / "coverage_model.pkl")
    le = joblib.load(model_dir / "label_encoder.pkl")
    feature_cols: list[str] = [
        c.strip()
        for c in (model_dir / "feature_columns.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if c.strip()
    ]

    if LABEL_ZONE not in le.classes_:
        raise ValueError(
            f"'{LABEL_ZONE}' is absent from the loaded LabelEncoder — "
            "the artifact may have been trained with a different labelling scheme."
        )

    zone_idx = int(np.where(le.classes_ == LABEL_ZONE)[0][0])
    log.info(
        "Artifacts loaded: %d classes | '%s' → encoded index %d | %d features",
        len(le.classes_),
        LABEL_ZONE,
        zone_idx,
        len(feature_cols),
    )

    shap_path = model_dir / "shap_explainer.pkl"
    explainer = None
    if shap_path.exists():
        try:
            import shap as _shap_check  # noqa: F401

            explainer = joblib.load(shap_path)
            log.info("SHAP TreeExplainer loaded from: %s", shap_path)
        except ImportError:
            log.warning(
                "shap package not installed — SDI feature will be skipped.  "
                "Install with: pip install shap"
            )
    else:
        log.warning(
            "shap_explainer.pkl not found in %s — Spatial Disguise Index "
            "will be unavailable for this run.",
            model_dir,
        )

    return model, le, explainer, feature_cols


# ---------------------------------------------------------------------------
# 4.  Deterministic MD5 hash split
#     (Re-implemented for self-containment; algorithm is identical to
#      pipelines/train.py::_game_hash_fraction / split_by_game)
# ---------------------------------------------------------------------------


def _game_hash_fraction(game_id: int) -> float:
    """
    Map a gameId to a deterministic float in [0, 1) via MD5.

    The first 8 hex digits of MD5(str(gameId)) are interpreted as a 32-bit
    unsigned integer and divided by 0xFFFFFFFF.  This is identical to the
    implementation in ``pipelines/train.py``.
    """
    digest = hashlib.md5(str(game_id).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFF_FFFF


def split_by_game(
    feature_df: pd.DataFrame,
    train_threshold: float = TRAIN_HASH_THRESHOLD,
    smoke_test: bool = False,
) -> tuple[pd.Index, pd.Index]:
    """
    Recover the exact game-level train/test partition used during training.

    Every frame of every play within a gameId is atomically routed to one
    split.  The MD5 hash ensures this function returns an identical partition
    regardless of call order or platform — no random seed is required.

    Parameters
    ----------
    feature_df      : DataFrame containing a ``gameId`` column.
    train_threshold : Hash boundary; games below this map to train.
    smoke_test      : When ``True``, bypasses the MD5 partition entirely and
                      routes **all** ingested rows to the test index.  Use
                      with ``--game-id`` to force local pipeline validation
                      on a game that the hash would otherwise assign to the
                      training split.

    Returns
    -------
    train_idx : pd.Index  positional index into *feature_df* for train rows
    test_idx  : pd.Index  positional index into *feature_df* for test rows
    """
    if smoke_test:
        log.warning(
            "[WARNING] Smoke-test mode active: Bypassing MD5 hash splits to "
            "force local evaluation.  Results are NOT representative of "
            "held-out model performance."
        )
        empty_idx: pd.Index = feature_df.index[:0]
        return empty_idx, feature_df.index

    game_ids: np.ndarray = feature_df["gameId"].unique()
    train_games = {g for g in game_ids if _game_hash_fraction(int(g)) < train_threshold}
    test_games = {g for g in game_ids if g not in train_games}

    train_mask = feature_df["gameId"].isin(train_games)
    test_mask = feature_df["gameId"].isin(test_games)

    log.info(
        "Game-level split: %d train games | %d test games "
        "(%d test rows, %.1f%% of total)",
        len(train_games),
        len(test_games),
        int(test_mask.sum()),
        100.0 * test_mask.sum() / max(len(feature_df), 1),
    )
    return feature_df.index[train_mask], feature_df.index[test_mask]


# ---------------------------------------------------------------------------
# 4.5.  Memory optimization (mirrors pipelines/train.py::_downcast_memory)
# ---------------------------------------------------------------------------


def _downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory footprint for multi-week production datasets.

    Float64 spatial / kinematic columns are downcast to float32 while
    ``nflId`` and PFF assignment ID columns are preserved at float64 (they
    carry NaN for the ball entity and serve as merge keys).

    Integer ID columns are downcast to low-overhead types:
      * ``gameId``  → int32  (max 2,147,483,647 > any foreseeable game ID)
      * ``playId``  → int32  (play IDs are small positive integers)
      * ``frameId`` → int16  (per-play frames typically < 300; int16 max = 32,767)

    NaN-safety: integer downcasting is skipped for any column that contains
    NaN values, since plain NumPy integer dtypes cannot represent NaN.
    """
    df = df.copy()
    _PRESERVE_FLOAT64: frozenset[str] = frozenset({
        "nflId",
        "pff_primaryDefensiveCoveredReceiverId",
    })
    for col in df.select_dtypes(include="float64").columns:
        if col not in _PRESERVE_FLOAT64:
            df[col] = df[col].astype(np.float32)

    _INT_CASTS: dict[str, type] = {
        "gameId":  np.int32,
        "playId":  np.int32,
        "frameId": np.int16,
    }
    for col, dtype in _INT_CASTS.items():
        if col in df.columns and not df[col].isna().any():
            df[col] = df[col].astype(dtype)

    return df


# ---------------------------------------------------------------------------
# 5.  Data ingestion (mirrors pipelines/train.py::ingest_tracking)
# ---------------------------------------------------------------------------


def _ingest_tracking(
    data_dir: Path,
    weeks: list[int],
    game_ids: Optional[list[int]],
) -> pd.DataFrame:
    """
    Load, merge, and spatially normalise tracking data across multiple weeks.

    Metadata files are loaded once and reused across all week files.  Each
    week's tracking CSV is early-filtered on ``gameId`` before normalization
    when *game_ids* is specified, minimizing peak memory overhead.

    Parameters
    ----------
    data_dir : Path
        Directory containing all raw CSV files.
    weeks : list[int]
        Tracking week numbers to load (``week{w}.csv`` per entry).
    game_ids : list[int] | None
        When provided, only rows whose ``gameId`` appears in this list are
        retained after loading each week CSV.  ``None`` loads all games.

    Returns
    -------
    pd.DataFrame
        Concatenated, normalized, float32-downcast tracking DataFrame.
    """
    static_files = {
        "players": data_dir / "players.csv",
        "pff":     data_dir / "pffScoutingData.csv",
        "plays":   data_dir / "plays.csv",
        "games":   data_dir / "games.csv",
    }
    for label, p in static_files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Required data file ({label}) not found: {p}"
            )

    log.info("Loading shared metadata files from: %s", data_dir)
    players_df = load_players(str(static_files["players"]))
    pff_df     = load_pff_scouting(str(static_files["pff"]))
    plays_df   = load_plays(str(static_files["plays"]))
    games_df   = load_games(str(static_files["games"]))

    frames: list[pd.DataFrame] = []
    for week in sorted(set(weeks)):
        tracking_path = data_dir / f"week{week}.csv"
        if not tracking_path.exists():
            raise FileNotFoundError(
                f"Tracking file not found: {tracking_path}"
            )

        log.info("Reading tracking CSV — week=%d ...", week)
        tracking_raw: pd.DataFrame = pd.read_csv(
            str(tracking_path), low_memory=False
        )

        if game_ids:
            gid_dtype = type(tracking_raw["gameId"].iloc[0])
            typed_ids = [gid_dtype(g) for g in game_ids]
            tracking_raw = tracking_raw.loc[
                tracking_raw["gameId"].isin(typed_ids)
            ].copy()
            if tracking_raw.empty:
                log.warning(
                    "Week %d: no rows match game_ids=%s — skipping.",
                    week, game_ids,
                )
                continue

        week_game_ids = tracking_raw["gameId"].unique()
        pff_week = (
            pff_df.loc[pff_df["gameId"].isin(week_game_ids)].copy()
            if "gameId" in pff_df.columns else pff_df
        )
        plays_week = (
            plays_df.loc[plays_df["gameId"].isin(week_game_ids)].copy()
            if "gameId" in plays_df.columns else plays_df
        )
        games_week = (
            games_df.loc[games_df["gameId"].isin(week_game_ids)].copy()
            if "gameId" in games_df.columns else games_df
        )

        df = build_normalized_tracking(
            tracking_df=tracking_raw,
            players_df=players_df,
            pff_df=pff_week,
            plays_df=plays_week,
            games_df=games_week,
        )
        log.info(
            "Week %d normalised: shape=%s | games=%d | plays=%d",
            week, df.shape, df["gameId"].nunique(), df["playId"].nunique(),
        )
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No tracking data loaded for weeks={weeks} game_ids={game_ids}. "
            "Verify that the specified weeks and game IDs exist in the data directory."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = _downcast_memory(combined)
    log.info(
        "Multi-week ingest complete: shape=%s | games=%d | plays=%d",
        combined.shape,
        combined["gameId"].nunique(),
        combined["playId"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# 6.  Feature matrix assembly (mirrors pipelines/train.py::build_feature_matrix)
# ---------------------------------------------------------------------------


def _build_feature_matrix(
    tracking_df: pd.DataFrame,
    grid_resolution: float,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Assemble the defender-centric feature matrix.

    Row identity: (gameId, playId, frameId, defender_nflId).

    Uses the same sequential left-join pattern as ``pipelines/train.py`` and
    validates that all columns listed in *feature_cols* are present after
    assembly.  Includes ``territory_ratio`` (frame-normalized Voronoi share)
    and ``velocity_angular_divergence`` (cosine similarity of defender vs
    receiver velocity vectors) to match the expanded training feature set.
    """
    log.info("Building field grid at %.1f yd resolution...", grid_resolution)
    grid_xy = build_field_grid(resolution=grid_resolution)

    # ── Kinematic base ────────────────────────────────────────────────────────
    kine_cols = _FRAME_KEY + ["nflId", "x", "y", "s", "a", "o_rad", "dir_rad"]
    kine_df = (
        tracking_df
        .loc[tracking_df["officialPosition"].isin(_COVERAGE_POSITIONS), kine_cols]
        .rename(columns={"nflId": "defender_nflId"})
        .reset_index(drop=True)
    )
    log.info("Kinematic base: %d defender-frame rows.", len(kine_df))

    # ── Voronoi coverage assignments ──────────────────────────────────────────
    log.info("Computing frame-by-frame Voronoi coverage assignments...")
    assignments_df = build_coverage_assignments(
        tracking_df, grid_resolution=grid_resolution, grid_xy=grid_xy
    )
    n_multi = int(assignments_df.duplicated(subset=_DEFENDER_KEY, keep=False).sum())
    if n_multi > 0:
        log.debug(
            "%d rows have a defender assigned to >1 receiver in the same frame "
            "(keeping first).",
            n_multi,
        )
    assignments_dedup = (
        assignments_df
        .sort_values("assigned_receiver_nflId")
        .drop_duplicates(subset=_DEFENDER_KEY, keep="first")
    )

    # ── Voronoi territory metrics ─────────────────────────────────────────────
    log.info("Computing Voronoi territory metrics...")
    territory_df = compute_defender_territories(
        tracking_df, grid_xy=grid_xy, grid_resolution=grid_resolution
    )

    # ── Receiver cushion ──────────────────────────────────────────────────────
    log.info("Computing receiver cushion...")
    cushion_df = compute_receiver_cushion(tracking_df, fallback_to_nearest=True)
    cushion_df = cushion_df.rename(columns={"nflId": "defender_nflId"})

    # ── Spatial leverage ──────────────────────────────────────────────────────
    log.info("Computing spatial leverage vectors...")
    leverage_df = compute_spatial_leverage(tracking_df, fallback_to_nearest=True)
    leverage_df = leverage_df.rename(columns={"nflId": "defender_nflId"})

    # ── Pre-snap safety depth ─────────────────────────────────────────────────
    log.info("Computing pre-snap safety depth...")
    safety_df = compute_presnap_safety_depth(tracking_df)

    # ── Velocity angular divergence ───────────────────────────────────────────
    log.info("Computing velocity angular divergence...")
    vad_df = compute_velocity_angular_divergence(tracking_df, fallback_to_nearest=True)
    vad_df = vad_df.rename(columns={"nflId": "defender_nflId"})

    # ── Sequential left-join ──────────────────────────────────────────────────
    log.info("Assembling feature matrix via sequential left-joins...")

    # Build up to territory first so territory_ratio can be derived in-place.
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
    )

    # Territory ratio: vectorized groupby transform — no loop required.
    _frame_total = (
        feature_df
        .groupby(_FRAME_KEY)["territory_area_sq_yd"]
        .transform("sum")
    )
    feature_df["territory_ratio"] = np.where(
        _frame_total > 0,
        feature_df["territory_area_sq_yd"] / _frame_total,
        np.nan,
    )

    feature_df = (
        feature_df
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
            vad_df[_DEFENDER_KEY + ["velocity_angular_divergence"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            safety_df[["gameId", "playId", "safety_depth_mean", "safety_depth_std"]],
            on=["gameId", "playId"],
            how="left",
        )
    )
    log.info("Feature matrix assembled: %s", feature_df.shape)
    return feature_df


# ---------------------------------------------------------------------------
# 7.  Target label construction (mirrors pipelines/train.py::build_target_labels)
# ---------------------------------------------------------------------------


def _build_target_labels(
    feature_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive the multi-class coverage-responsibility label for every row.

    Priority: (1) PFF zone flag → (2) no Voronoi assignment → (3) man label.
    Identical logic to ``pipelines/train.py::build_target_labels``.
    """
    feature_df = feature_df.copy()

    if "pff_coverage" in tracking_df.columns:
        pff_snap = (
            tracking_df[["gameId", "playId", "nflId", "pff_coverage"]]
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
            "pff_coverage column absent — zone/man classification uses "
            "Voronoi assignment presence only."
        )
        pff_is_zone = pd.Series(False, index=feature_df.index)

    no_voronoi: pd.Series = feature_df["assigned_receiver_nflId"].isna()
    is_zone: pd.Series = pff_is_zone | no_voronoi

    feature_df[TARGET_COLUMN] = np.where(
        is_zone,
        LABEL_ZONE,
        feature_df["assigned_receiver_nflId"]
        .dropna()
        .astype(np.int64)
        .astype(str)
        .reindex(feature_df.index, fill_value=LABEL_ZONE),
    )
    return feature_df


# ---------------------------------------------------------------------------
# 8.  NaN sentinel fill (mirrors pipelines/train.py::fill_feature_nans)
# ---------------------------------------------------------------------------


def _fill_feature_nans(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Replace all NaN values in *feature_cols* with ``NAN_FILL_VALUE``."""
    feature_df = feature_df.copy()
    missing = [c for c in feature_cols if c not in feature_df.columns]
    if missing:
        log.warning(
            "Synthesising %d missing feature columns as NAN_FILL_VALUE: %s",
            len(missing),
            missing,
        )
        for col in missing:
            feature_df[col] = NAN_FILL_VALUE
    existing = [c for c in feature_cols if c in feature_df.columns]
    feature_df[existing] = feature_df[existing].fillna(NAN_FILL_VALUE)
    return feature_df


# ---------------------------------------------------------------------------
# 9.  Test-set construction
# ---------------------------------------------------------------------------


def build_test_arrays(
    data_dir: Path,
    tracking_weeks: list[int],
    game_ids: Optional[list[int]],
    grid_resolution: float,
    feature_cols: list[str],
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Rebuild the full feature matrix and isolate the locked-out test split.

    The deterministic MD5 hash split is applied here so the caller receives
    exactly the same row subset that was held out during training, without
    needing any persisted split-index artifact.

    Parameters
    ----------
    tracking_weeks : list[int]
        Week numbers to load (e.g. ``[1]``).
    game_ids : list[int] | None
        Restrict to specific game IDs; ``None`` loads all games.
    smoke_test : When ``True``, the hash split is bypassed and every ingested
                 row is treated as the test set.  See ``split_by_game`` for the
                 full contract.

    Returns
    -------
    test_df     : pd.DataFrame  full test-set rows (features + labels + identity)
    tracking_df : pd.DataFrame  normalised tracking data (for snap-frame lookup)
    X_test      : np.ndarray    shape ``(n_test, n_features)`` float32 array
                                aligned row-by-row with *test_df*
    """
    tracking_df = _ingest_tracking(data_dir, tracking_weeks, game_ids)
    feature_df = _build_feature_matrix(tracking_df, grid_resolution, feature_cols)
    feature_df = _build_target_labels(feature_df, tracking_df)
    feature_df = _fill_feature_nans(feature_df, feature_cols)

    _, test_idx = split_by_game(feature_df, smoke_test=smoke_test)
    test_df = feature_df.loc[test_idx].reset_index(drop=True)

    avail_cols = [c for c in feature_cols if c in test_df.columns]
    X_test = test_df[avail_cols].values.astype(np.float32)

    log.info(
        "Test set: %d rows | %d games | %d plays | %d features",
        len(test_df),
        test_df["gameId"].nunique(),
        test_df["playId"].nunique(),
        X_test.shape[1],
    )
    return test_df, tracking_df, X_test


# ---------------------------------------------------------------------------
# 10.  SHAP value extraction
# ---------------------------------------------------------------------------


def compute_shap_arrays(
    explainer,
    X_test: np.ndarray,
    le,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-class SHAP values and extract the ``No_Matchup_Zone`` slice.

    Handles both the legacy shap API (returns a ``list`` of per-class 2-D
    arrays) and the modern API (returns a single 3-D array).

    Returns
    -------
    shap_3d   : np.ndarray  shape ``(n_test, n_features, n_classes)``
    zone_shap : np.ndarray  shape ``(n_test, n_features)`` — zone class only
    """
    log.info(
        "Computing SHAP values for %d test rows × %d features...",
        X_test.shape[0],
        X_test.shape[1],
    )
    raw = explainer.shap_values(X_test)

    if isinstance(raw, list):
        # Legacy shap: list of (n_samples, n_features) per class
        shap_3d: np.ndarray = np.stack(raw, axis=2)
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_3d = raw
    else:
        log.warning(
            "Unexpected SHAP output type=%s shape=%s — attempting coercion.",
            type(raw).__name__,
            getattr(raw, "shape", "unknown"),
        )
        shap_3d = np.array(raw)
        if shap_3d.ndim == 2:
            shap_3d = shap_3d[:, :, np.newaxis]

    zone_idx = int(np.where(le.classes_ == LABEL_ZONE)[0][0])
    zone_shap: np.ndarray = shap_3d[:, :, zone_idx]

    log.info(
        "SHAP values ready: shap_3d=%s | zone class index=%d",
        shap_3d.shape,
        zone_idx,
    )
    return shap_3d, zone_shap


# ---------------------------------------------------------------------------
# 11.  Feature 1 — Dynamic Matchup Matrix
# ---------------------------------------------------------------------------


def build_matchup_matrix(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    le,
) -> pd.DataFrame:
    """
    Aggregate frame-level class probability streams to the play level,
    producing a defender × assignment-class matchup share matrix.

    For each ``(gameId, playId, defender_nflId)`` triple the mean frame-level
    probability for every coverage class is computed.  The ``No_Matchup_Zone``
    column isolates zone-vs-man at the play level, while high probabilities
    for a specific receiver ``nflId`` class identify precise man assignments.

    Returns
    -------
    pd.DataFrame — long format, dashboard-ready:
        gameId, playId, defender_nflId, assignment_class, matchup_share,
        is_zone_class
    """
    class_labels: list[str] = le.classes_.tolist()

    # Attach per-frame probabilities to identity columns
    id_cols = ["gameId", "playId", "frameId", "defender_nflId"]
    proba_df = pd.DataFrame(proba, columns=class_labels)
    frame_proba = pd.concat(
        [test_df[id_cols].reset_index(drop=True), proba_df],
        axis=1,
    )

    # Aggregate to play level: mean probability per (game, play, defender, class)
    group_keys = ["gameId", "playId", "defender_nflId"]
    matchup_agg = (
        frame_proba
        .groupby(group_keys)[class_labels]
        .mean()
        .reset_index()
    )

    # Melt to long format for dashboard ingestion
    matchup_long = matchup_agg.melt(
        id_vars=group_keys,
        value_vars=class_labels,
        var_name="assignment_class",
        value_name="matchup_share",
    )

    matchup_long["is_zone_class"] = (
        matchup_long["assignment_class"] == LABEL_ZONE
    )

    # Drop trivially zero rows to keep the output compact
    matchup_long = matchup_long[matchup_long["matchup_share"] > 0.01].copy()
    matchup_long = matchup_long.sort_values(
        ["gameId", "playId", "defender_nflId", "matchup_share"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    log.info(
        "Matchup matrix: %d (play, defender, class) rows | %d unique plays",
        len(matchup_long),
        matchup_long["playId"].nunique(),
    )
    return matchup_long


# ---------------------------------------------------------------------------
# 12.  Feature 2 — Spatial Disguise Index (SDI)
# ---------------------------------------------------------------------------


def compute_spatial_disguise_index(
    test_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    zone_shap: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute the Spatial Disguise Index for every (gameId, playId, defender).

    Algorithm
    ---------
    1. For each test row, compute the *Voronoi territory SHAP signal* — the
       sum of zone-class SHAP attributions for the territory and leverage
       features (``territory_area_sq_yd``, ``territory_grid_points``,
       ``leverage_x``, ``leverage_y``).  A high positive value means the
       defender's current field-ownership geometry is strongly driving a zone
       prediction.

    2. Identify the snap frame for each play from the raw tracking data.

    3. Use the snap-frame Voronoi SHAP signal as the **pre-snap baseline**
       for each (play, defender) pair.

    4. Compute the *cumulative divergence*: for every post-snap frame, sum
       ``|signal_t − snap_baseline|``.  A large cumulative divergence means
       the defender's post-snap territory evolved significantly away from
       their pre-snap alignment posture — the hallmark of disguise or
       post-snap rotation.

    5. Scale globally to [0, 100]:
       ``0`` = completely transparent (pre and post-snap behaviour identical);
       ``100`` = elite disguise / rotation.

    Returns
    -------
    pd.DataFrame:
        gameId, playId, defender_nflId,
        snap_territory_shap,       — baseline signal at the snap frame
        presnap_territory_shap,    — mean signal across all pre-snap frames
        postsnap_territory_shap,   — mean signal across all post-snap frames
        cumulative_divergence,     — sum of |post_t − snap_baseline|
        sdi_score                  — normalised [0, 100]
    """
    # Resolve SHAP feature indices for SDI components
    sdi_feat_indices: list[int] = [
        feature_cols.index(f) for f in _SDI_FEATURE_NAMES if f in feature_cols
    ]
    if not sdi_feat_indices:
        log.warning(
            "None of the SDI features %s are present in feature_cols — "
            "SDI will be zero for all defenders.",
            _SDI_FEATURE_NAMES,
        )

    # Per-row: sum SHAP contributions for Voronoi + leverage features
    if sdi_feat_indices:
        voronoi_shap_signal: np.ndarray = (
            zone_shap[:, sdi_feat_indices].sum(axis=1)
        )
    else:
        voronoi_shap_signal = np.zeros(len(test_df), dtype=np.float64)

    # Build snap-frame lookup: (gameId, playId) → snap frameId (vectorized merge,
    # avoids per-row dict lookup that is fragile with int32/int16 downcast key types).
    snap_rows = tracking_df[tracking_df["event"] == SNAP_EVENT]
    snap_fids: pd.DataFrame = (
        snap_rows
        .groupby(["gameId", "playId"])["frameId"]
        .first()
        .rename("snap_frameId")
        .reset_index()
    )

    # Construct working DataFrame aligned to test_df
    work = test_df[["gameId", "playId", "frameId", "defender_nflId"]].copy()
    work = work.reset_index(drop=True)
    work["voronoi_shap_signal"] = voronoi_shap_signal
    work = work.merge(snap_fids, on=["gameId", "playId"], how="left")

    # Snap-frame SHAP baseline per (gameId, playId, defender_nflId)
    snap_mask = work["frameId"] == work["snap_frameId"]
    snap_baseline = (
        work[snap_mask]
        .groupby(["gameId", "playId", "defender_nflId"])["voronoi_shap_signal"]
        .first()
        .rename("snap_territory_shap")
        .reset_index()
    )

    work = work.merge(
        snap_baseline,
        on=["gameId", "playId", "defender_nflId"],
        how="left",
    )

    # Pre-snap and post-snap masks
    snap_fid = work["snap_frameId"].fillna(-1)
    work["is_presnap"] = work["frameId"] <= snap_fid
    work["is_postsnap"] = work["frameId"] > snap_fid

    # Per-frame divergence from the snap baseline (post-snap frames only)
    work["frame_divergence"] = np.where(
        work["is_postsnap"] & work["snap_territory_shap"].notna(),
        (work["voronoi_shap_signal"] - work["snap_territory_shap"]).abs(),
        np.nan,
    )

    group_keys = ["gameId", "playId", "defender_nflId"]

    presnap_mean = (
        work[work["is_presnap"]]
        .groupby(group_keys)["voronoi_shap_signal"]
        .mean()
        .rename("presnap_territory_shap")
    )
    postsnap_mean = (
        work[work["is_postsnap"]]
        .groupby(group_keys)["voronoi_shap_signal"]
        .mean()
        .rename("postsnap_territory_shap")
    )
    cumulative_div = (
        work[work["is_postsnap"]]
        .groupby(group_keys)["frame_divergence"]
        .sum()
        .rename("cumulative_divergence")
    )
    snap_base_per_def = (
        snap_baseline.set_index(group_keys)["snap_territory_shap"]
    )

    sdi_df = (
        presnap_mean.to_frame()
        .join(postsnap_mean, how="outer")
        .join(cumulative_div, how="outer")
        .join(snap_base_per_def, how="left")
        .reset_index()
    )

    for col in ("presnap_territory_shap", "postsnap_territory_shap",
                "cumulative_divergence", "snap_territory_shap"):
        sdi_df[col] = sdi_df[col].fillna(0.0)

    # Scale cumulative_divergence to [0, 100]
    raw_min = sdi_df["cumulative_divergence"].min()
    raw_max = sdi_df["cumulative_divergence"].max()
    if raw_max > raw_min:
        sdi_df["sdi_score"] = (
            100.0 * (sdi_df["cumulative_divergence"] - raw_min)
            / (raw_max - raw_min)
        ).round(2)
    else:
        sdi_df["sdi_score"] = 0.0

    sdi_df = sdi_df.sort_values("sdi_score", ascending=False).reset_index(drop=True)

    log.info(
        "SDI computed: %d (play, defender) pairs | "
        "mean SDI=%.1f | max SDI=%.1f",
        len(sdi_df),
        float(sdi_df["sdi_score"].mean()),
        float(sdi_df["sdi_score"].max()),
    )
    return sdi_df


# ---------------------------------------------------------------------------
# 13.  Feature 3 — Real-Time Breakdown Predictor ("Burn Risk")
# ---------------------------------------------------------------------------


def compute_burn_risk(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    le,
) -> pd.DataFrame:
    """
    Detect structural assignment failures and score their severity.

    Failure conditions (either triggers a flagged frame)
    -----------------------------------------------------
    1. **Cushion collapse + surge**: ``receiver_cushion`` delta drops below
       ``_CUSHION_DECAY_THRESHOLD`` (rapid press) AND the model's
       man-assignment probability for that frame exceeds
       ``_SURGE_PROB_THRESHOLD``.

    2. **Voronoi discipline failure**: ``territory_area_sq_yd`` drops by
       more than ``_TERRITORY_COLLAPSE_RATIO × previous_value`` (landmark
       collapse) AND man-assignment probability exceeds the surge threshold.

    ``failure_frame`` is the earliest ``frameId`` satisfying either condition
    per (gameId, playId, defender_nflId) triple.

    ``burn_risk_score`` is a normalised [0, 100] composite of three
    peak-signal magnitudes across all flagged frames for that triple:
        - 40 % cushion decay rate
        - 40 % man-assignment probability surge
        - 20 % Voronoi territory collapse ratio

    The composite is linearly rescaled so the highest-risk event in the test
    set maps to 100.

    Returns
    -------
    pd.DataFrame:
        gameId, playId, defender_nflId, failure_frame, failure_type,
        max_cushion_decay, max_prob_surge, max_territory_collapse,
        burn_risk_score

    Only rows with at least one flagged failure frame are returned.
    """
    zone_idx = int(np.where(le.classes_ == LABEL_ZONE)[0][0])

    # Man-assignment probability = 1 − P(No_Matchup_Zone)
    man_proba: np.ndarray = 1.0 - proba[:, zone_idx]

    # Build working DataFrame
    work = test_df[
        ["gameId", "playId", "frameId", "defender_nflId",
         "receiver_cushion", "territory_area_sq_yd"]
    ].copy().reset_index(drop=True)
    work["man_proba"] = man_proba

    # Restore NaN sentinel → proper NaN for delta computations
    for col in ("receiver_cushion", "territory_area_sq_yd"):
        work[col] = work[col].replace(NAN_FILL_VALUE, np.nan)

    # Sort for correct lag alignment
    group_keys = ["gameId", "playId", "defender_nflId"]
    work = work.sort_values(group_keys + ["frameId"]).reset_index(drop=True)

    # Vectorised within-group lag using groupby + shift
    grp = work.groupby(group_keys)
    work["cushion_prev"] = grp["receiver_cushion"].shift(1)
    work["territory_prev"] = grp["territory_area_sq_yd"].shift(1)

    work["cushion_delta"] = work["receiver_cushion"] - work["cushion_prev"]
    work["territory_ratio"] = np.where(
        work["territory_prev"] > 0,
        (work["territory_area_sq_yd"] - work["territory_prev"]) / work["territory_prev"],
        np.nan,
    )

    # Failure flags
    work["is_cushion_failure"] = (
        work["cushion_delta"].notna()
        & (work["cushion_delta"] < _CUSHION_DECAY_THRESHOLD)
        & (work["man_proba"] > _SURGE_PROB_THRESHOLD)
    )
    work["is_territory_failure"] = (
        work["territory_ratio"].notna()
        & (work["territory_ratio"] < -_TERRITORY_COLLAPSE_RATIO)
        & (work["man_proba"] > _SURGE_PROB_THRESHOLD)
    )
    work["is_failure"] = work["is_cushion_failure"] | work["is_territory_failure"]

    failures = work[work["is_failure"]].copy()

    if failures.empty:
        log.info("No burn-risk failures detected in the test set.")
        return pd.DataFrame(
            columns=[
                "gameId", "playId", "defender_nflId",
                "failure_frame", "failure_type",
                "max_cushion_decay", "max_prob_surge",
                "max_territory_collapse", "burn_risk_score",
            ]
        )

    # Earliest failure frame per triple
    first_frame = (
        failures
        .groupby(group_keys)["frameId"]
        .first()
        .rename("failure_frame")
        .reset_index()
    )

    # Dominant failure type at the first-failure frame.
    # `failures` still carries the original column name "frameId"; `first_frame`
    # has renamed it to "failure_frame".  Explicit left/right key mapping bridges
    # the mismatch without mutating either DataFrame.
    first_frame_details = failures.merge(
        first_frame,
        left_on=["gameId", "playId", "defender_nflId", "frameId"],
        right_on=["gameId", "playId", "defender_nflId", "failure_frame"],
        how="inner",
    )

    first_frame_details = first_frame_details.copy()
    first_frame_details["failure_type"] = np.select(
        [
            first_frame_details["is_cushion_failure"] & first_frame_details["is_territory_failure"],
            first_frame_details["is_cushion_failure"],
        ],
        ["cushion_and_territory", "cushion_collapse"],
        default="territory_discipline",
    )

    # Peak signal magnitudes across all flagged frames per triple
    peak_signals = (
        failures
        .groupby(group_keys)
        .agg(
            max_cushion_decay=(
                "cushion_delta",
                lambda x: float(x.dropna().abs().max()) if x.notna().any() else 0.0,
            ),
            max_prob_surge=("man_proba", "max"),
            max_territory_collapse=(
                "territory_ratio",
                lambda x: float(x.dropna().abs().max()) if x.notna().any() else 0.0,
            ),
        )
        .reset_index()
    )

    # Merge first-frame metadata with peak signals
    burn_df = first_frame_details[group_keys + ["failure_frame", "failure_type"]].merge(
        peak_signals,
        on=group_keys,
        how="left",
    )

    # Composite raw score [0, 1] — weighted blend of three signal magnitudes
    burn_df["_raw_score"] = (
        0.4 * (burn_df["max_cushion_decay"] / abs(_CUSHION_DECAY_THRESHOLD)).clip(0.0, 1.0)
        + 0.4 * (
            (burn_df["max_prob_surge"] - _SURGE_PROB_THRESHOLD)
            / max(1.0 - _SURGE_PROB_THRESHOLD, 1e-6)
        ).clip(0.0, 1.0)
        + 0.2 * (burn_df["max_territory_collapse"] / _TERRITORY_COLLAPSE_RATIO).clip(0.0, 1.0)
    )

    # Normalise to [0, 100] globally
    raw_min = burn_df["_raw_score"].min()
    raw_max = burn_df["_raw_score"].max()
    if raw_max > raw_min:
        burn_df["burn_risk_score"] = (
            100.0 * (burn_df["_raw_score"] - raw_min) / (raw_max - raw_min)
        ).round(2)
    else:
        burn_df["burn_risk_score"] = 50.0  # all events equally severe

    burn_df = (
        burn_df
        .drop(columns=["_raw_score"])
        .sort_values("burn_risk_score", ascending=False)
        .reset_index(drop=True)
    )

    log.info(
        "Burn risk: %d failure events | %d plays affected | "
        "mean score=%.1f | max score=%.1f",
        len(burn_df),
        burn_df["playId"].nunique(),
        float(burn_df["burn_risk_score"].mean()),
        float(burn_df["burn_risk_score"].max()),
    )
    return burn_df


# ---------------------------------------------------------------------------
# 14.  Classification performance report
# ---------------------------------------------------------------------------


def generate_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    le,
    output_dir: Path,
    smoke_test: bool = False,
) -> None:
    """
    Compute and persist sklearn classification metrics on the held-out test set.

    Writes ``classification_report.txt`` into *output_dir*.  No interactive
    windows are opened.

    Degenerate-label handling
    -------------------------
    In smoke-test mode a single game may not contain every class the
    LabelEncoder knows about.  ``classification_report`` is called with
    ``labels`` restricted to the integer-encoded classes that appear in
    *either* ``y_test`` or ``y_pred`` so that sklearn never receives a
    ``labels`` argument referencing a class absent from both arrays — which
    would otherwise raise a ``UndefinedMetricWarning`` or, in some sklearn
    versions, an assertion error.  Classes absent from the test slice are
    noted explicitly in the saved report.
    """
    from sklearn.metrics import accuracy_score, classification_report  # noqa: PLC0415

    all_class_indices: np.ndarray = np.arange(len(le.classes_))

    if smoke_test:
        # Restrict labels to those observed in at least one of the two arrays
        # to prevent assertion failures on degenerate single-game slices.
        observed = np.union1d(np.unique(y_test), np.unique(y_pred))
        active_labels: np.ndarray = np.intersect1d(all_class_indices, observed)
        active_names: np.ndarray = le.classes_[active_labels]

        absent = sorted(
            set(le.classes_.tolist()) - set(active_names.tolist())
        )
        absent_note = (
            f"\nNote (smoke-test): {len(absent)} class(es) absent from this "
            f"game slice and excluded from the report:\n  {absent}\n"
            if absent
            else ""
        )
    else:
        active_labels = all_class_indices
        active_names = le.classes_
        absent_note = ""

    accuracy = float(accuracy_score(y_test, y_pred))
    report_str = classification_report(
        y_test,
        y_pred,
        labels=active_labels,
        target_names=active_names,
        zero_division=0,
    )

    log.info("Test-set classification report:\n%s", report_str)
    if absent_note:
        log.info(absent_note.strip())
    log.info("Overall test-set accuracy: %.4f", accuracy)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(
        f"NFL Coverage Responsibility — Test-Set Classification Report\n"
        f"{'=' * 64}\n"
        f"Overall Accuracy: {accuracy:.6f}\n"
        f"{absent_note}\n"
        f"{report_str}",
        encoding="utf-8",
    )
    log.info("Classification report saved → %s", report_path)


# ---------------------------------------------------------------------------
# 15.  Output persistence
# ---------------------------------------------------------------------------


def save_outputs(
    output_dir: Path,
    matchup_matrix: pd.DataFrame,
    sdi_df: Optional[pd.DataFrame],
    burn_risk_df: pd.DataFrame,
    output_format: str,
) -> None:
    """
    Serialise all analytics DataFrames to *output_dir*.

    Supports ``"parquet"`` (default, columnar, typed) and ``"csv"`` formats.
    The directory is created if it does not already exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = output_format

    def _write(df: pd.DataFrame, name: str) -> None:
        path = output_dir / f"{name}.{ext}"
        if ext == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        log.info("Saved %-30s → %s  (%d rows)", name, path, len(df))

    _write(matchup_matrix, "matchup_matrix")
    if sdi_df is not None and not sdi_df.empty:
        _write(sdi_df, "spatial_disguise_index")
    else:
        log.info("SDI table empty or unavailable — skipping spatial_disguise_index.")
    _write(burn_risk_df, "burn_risk_log")


# ---------------------------------------------------------------------------
# 16.  Orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:  # noqa: PLR0915
    args = parse_args(argv)
    setup_logging(args.log_level)

    log.info("=" * 72)
    log.info("NFL Coverage Responsibility — Evaluation Pipeline")
    log.info("=" * 72)
    log.info("Run config: %s", vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load serialised model artifacts ───────────────────────────────
    model, le, explainer, feature_cols = load_artifacts(args.model_dir)

    # ── Step 2: Rebuild feature matrix and isolate the locked-out test split ──
    test_df, tracking_df, X_test = build_test_arrays(
        data_dir=args.data_dir,
        tracking_weeks=args.tracking_weeks,
        game_ids=args.game_ids,
        grid_resolution=args.grid_resolution,
        feature_cols=feature_cols,
        smoke_test=args.smoke_test,
    )

    if len(X_test) == 0:
        log.error(
            "The test set is empty — every game in the dataset was routed to "
            "the training split by the MD5 hash.  To force local validation on "
            "a specific game, re-run with --smoke-test (e.g. "
            "--game-ids 2021090900 --smoke-test).  To use production held-out "
            "data, remove --game-ids so all weeks are ingested.",
        )
        return

    # ── Step 3: Frame-level probability inference via native XGBoost DMatrix ──
    try:
        import xgboost as xgb  # noqa: PLC0415

        dtest = xgb.DMatrix(X_test)
        # model.get_booster().predict returns raw softprob: (n_test, n_classes)
        proba: np.ndarray = model.get_booster().predict(dtest)
        if proba.ndim == 1:
            # Single-class degenerate case (smoke-test with one game)
            proba = proba.reshape(-1, 1)
        log.info(
            "Native DMatrix inference complete: proba shape=%s  "
            "[min=%.4f, max=%.4f]",
            proba.shape,
            float(proba.min()),
            float(proba.max()),
        )
    except Exception as xgb_exc:
        log.warning(
            "Native DMatrix predict raised (%s) — falling back to "
            "model.predict_proba().",
            xgb_exc,
        )
        proba = model.predict_proba(X_test)

    # Argmax frame-level prediction for classification metrics
    y_pred_encoded: np.ndarray = proba.argmax(axis=1).astype(np.int32)

    # ── Step 4: Encode ground-truth labels with the loaded LabelEncoder ────────
    has_labels = TARGET_COLUMN in test_df.columns
    y_test: Optional[np.ndarray] = None

    if has_labels:
        live_labels = set(test_df[TARGET_COLUMN].unique())
        unseen = live_labels - set(le.classes_)
        if unseen:
            log.warning(
                "Test set contains %d label(s) unseen by the LabelEncoder: %s.  "
                "Classification report will be skipped.",
                len(unseen),
                unseen,
            )
            has_labels = False
        else:
            y_test = le.transform(test_df[TARGET_COLUMN]).astype(np.int32)

    # ── Step 5: Classification report ─────────────────────────────────────────
    if has_labels and y_test is not None:
        generate_classification_report(
            y_test, y_pred_encoded, le, args.output_dir,
            smoke_test=args.smoke_test,
        )
    else:
        log.info(
            "Ground-truth labels unavailable or incomplete — "
            "classification report skipped."
        )

    # ── Step 6: Feature 1 — Dynamic Matchup Matrix ────────────────────────────
    log.info("-" * 60)
    log.info("Feature 1: Building Matchup Matrix...")
    matchup_matrix = build_matchup_matrix(test_df, proba, le)

    # ── Step 7: SHAP extraction for SDI ───────────────────────────────────────
    sdi_df: Optional[pd.DataFrame] = None

    if explainer is not None:
        log.info("-" * 60)
        log.info("Computing SHAP values for Spatial Disguise Index...")
        try:
            _, zone_shap = compute_shap_arrays(explainer, X_test, le, feature_cols)

            # ── Step 8: Feature 2 — Spatial Disguise Index ────────────────────
            log.info("Feature 2: Computing Spatial Disguise Index...")
            sdi_df = compute_spatial_disguise_index(
                test_df, tracking_df, zone_shap, feature_cols
            )
        except Exception as shap_exc:
            log.warning(
                "SHAP computation failed (%s) — SDI skipped.  "
                "This may indicate a version mismatch between shap and the "
                "serialised explainer.",
                shap_exc,
            )
    else:
        log.info(
            "SHAP explainer not available — "
            "Spatial Disguise Index (Feature 2) skipped."
        )

    # ── Step 9: Feature 3 — Burn Risk ─────────────────────────────────────────
    log.info("-" * 60)
    log.info("Feature 3: Computing Burn Risk scores...")
    burn_risk_df = compute_burn_risk(test_df, proba, le)

    # ── Step 10: Persist all analytics outputs ────────────────────────────────
    log.info("-" * 60)
    log.info("Persisting outputs to: %s", args.output_dir)
    save_outputs(
        output_dir=args.output_dir,
        matchup_matrix=matchup_matrix,
        sdi_df=sdi_df,
        burn_risk_df=burn_risk_df,
        output_format=args.output_format,
    )

    log.info("=" * 72)
    log.info("Evaluation pipeline complete.  Outputs: %s", args.output_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
