"""
pipelines/etl_job.py
====================
Feature store ETL job. Reads week1.csv once, computes the full defender-centric
feature matrix per game, and writes partitioned Parquet output to data/feature_store/.

Usage
-----
    python pipelines/etl_job.py --data-dir data/raw --output-dir data/feature_store
    python pipelines/etl_job.py --data-dir data/raw --game-ids 2021090900 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import FEATURE_COLUMNS, NAN_FILL_VALUE, SEASON
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
from src.etl.schema import validate_schema
from src.etl.transforms import (
    add_frames_since_snap,
    add_play_group_key,
    add_snap_frame_marker,
    apply_nan_sentinel,
    derive_coverage_labels,
    derive_territory_ratio,
    downcast_memory,
)
from src.features.coverage_features import (
    compute_presnap_safety_depth,
    compute_receiver_cushion,
    compute_spatial_leverage,
    compute_velocity_angular_divergence,
)

_COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)
_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_DEFENDER_KEY: list[str] = _FRAME_KEY + ["defender_nflId"]

log = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature store ETL: raw CSVs → partitioned Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/feature_store"))
    p.add_argument("--game-ids", nargs="*", type=int, default=None, metavar="GAME_ID")
    p.add_argument("--grid-resolution", type=float, default=1.0)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _update_manifest(
    manifest_path: Path,
    new_rows: pd.DataFrame,
) -> None:
    if manifest_path.exists():
        existing = pd.read_parquet(manifest_path)
        manifest = (
            pd.concat([existing, new_rows], ignore_index=True)
            .drop_duplicates(subset=["gameId"], keep="last")
            .reset_index(drop=True)
        )
    else:
        manifest = new_rows
    manifest.to_parquet(manifest_path, index=False)
    log.info("Manifest updated: %d total game(s) → %s", len(manifest), manifest_path)


def process_game(
    game_id: int,
    tracking_full: pd.DataFrame,
    players_df: pd.DataFrame,
    plays_df: pd.DataFrame,
    games_df: pd.DataFrame,
    pff_df: pd.DataFrame,
    field_grid: np.ndarray,
    output_dir: Path,
) -> dict:
    t0 = time.perf_counter()

    slice_raw = tracking_full[tracking_full["gameId"] == game_id].copy()

    game_pff = (
        pff_df[pff_df["gameId"] == game_id].copy()
        if "gameId" in pff_df.columns else pff_df
    )
    game_plays = (
        plays_df[plays_df["gameId"] == game_id].copy()
        if "gameId" in plays_df.columns else plays_df
    )
    game_games = (
        games_df[games_df["gameId"] == game_id].copy()
        if "gameId" in games_df.columns else games_df
    )

    normalized = build_normalized_tracking(
        tracking_df=slice_raw,
        players_df=players_df,
        pff_df=game_pff,
        plays_df=game_plays,
        games_df=game_games,
    )
    log.debug(
        "game=%s normalized: shape=%s plays=%d",
        game_id, normalized.shape, normalized["playId"].nunique(),
    )

    assignments = build_coverage_assignments(normalized, grid_xy=field_grid)
    territory = compute_defender_territories(normalized, grid_xy=field_grid)
    cushion = compute_receiver_cushion(normalized, fallback_to_nearest=True)
    leverage = compute_spatial_leverage(normalized, fallback_to_nearest=True)
    safety = compute_presnap_safety_depth(normalized)
    vad = compute_velocity_angular_divergence(normalized, fallback_to_nearest=True)

    kine_cols = _FRAME_KEY + ["nflId", "x", "y", "s", "a", "o_rad", "dir_rad"]
    kine_df = (
        normalized
        .loc[normalized["officialPosition"].isin(_COVERAGE_POSITIONS), kine_cols]
        .rename(columns={"nflId": "defender_nflId"})
        .reset_index(drop=True)
    )

    assignments_dedup = (
        assignments
        .sort_values("assigned_receiver_nflId")
        .drop_duplicates(subset=_DEFENDER_KEY, keep="first")
    )

    cushion = cushion.rename(columns={"nflId": "defender_nflId"})
    leverage = leverage.rename(columns={"nflId": "defender_nflId"})
    vad = vad.rename(columns={"nflId": "defender_nflId"})

    feature_df = (
        kine_df
        .merge(
            assignments_dedup[_DEFENDER_KEY + ["assigned_receiver_nflId"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            territory[_DEFENDER_KEY + ["territory_grid_points", "territory_area_sq_yd"]],
            on=_DEFENDER_KEY,
            how="left",
        )
    )

    feature_df = derive_territory_ratio(feature_df)

    feature_df = (
        feature_df
        .merge(
            cushion[_DEFENDER_KEY + ["receiver_cushion"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            leverage[_DEFENDER_KEY + ["leverage_x", "leverage_y"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            vad[_DEFENDER_KEY + ["velocity_angular_divergence"]],
            on=_DEFENDER_KEY,
            how="left",
        )
        .merge(
            safety[["gameId", "playId", "safety_depth_mean", "safety_depth_std"]],
            on=["gameId", "playId"],
            how="left",
        )
    )

    event_lookup = (
        normalized
        .groupby(_FRAME_KEY)["event"]
        .first()
        .reset_index()
    )
    feature_df = feature_df.merge(event_lookup, on=_FRAME_KEY, how="left")

    feature_df = derive_coverage_labels(feature_df, pff_df)
    feature_df = add_snap_frame_marker(feature_df)
    feature_df = add_frames_since_snap(feature_df)
    feature_df = add_play_group_key(feature_df)
    feature_df = apply_nan_sentinel(feature_df, FEATURE_COLUMNS, NAN_FILL_VALUE)
    feature_df = downcast_memory(feature_df)

    drop_cols = [c for c in ["event", "assigned_receiver_nflId", "pff_coverage"] if c in feature_df.columns]
    feature_df = feature_df.drop(columns=drop_cols)

    validate_schema(feature_df)

    feature_df = feature_df.sort_values(["playId", "frameId", "defender_nflId"]).reset_index(drop=True)

    out_path = output_dir / f"season={SEASON}" / "week=01" / f"game={game_id}" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    elapsed = time.perf_counter() - t0
    n_plays = int(feature_df["playId"].nunique())
    row_count = len(feature_df)
    log.info(
        "game=%s | rows=%d | plays=%d | elapsed=%.1fs → %s",
        game_id, row_count, n_plays, elapsed, out_path,
    )

    return {
        "season": SEASON,
        "week": 1,
        "gameId": int(game_id),
        "row_count": row_count,
        "n_plays": n_plays,
        "etl_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0.0",
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    log.info("=" * 72)
    log.info("NFL Coverage Responsibility — Feature Store ETL")
    log.info("=" * 72)
    log.info("config: data_dir=%s output_dir=%s grid_res=%.1f",
             args.data_dir, args.output_dir, args.grid_resolution)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading shared metadata...")
    players_df = load_players(str(args.data_dir / "players.csv"))
    plays_df = load_plays(str(args.data_dir / "plays.csv"))
    games_df = load_games(str(args.data_dir / "games.csv"))
    pff_df = load_pff_scouting(str(args.data_dir / "pffScoutingData.csv"))

    log.info("Reading week1.csv...")
    tracking_full = pd.read_csv(str(args.data_dir / "week1.csv"), low_memory=False)

    log.info("Building field grid at %.1f yd resolution...", args.grid_resolution)
    field_grid = build_field_grid(resolution=args.grid_resolution)

    game_ids_in_data = sorted(tracking_full["gameId"].unique())
    game_ids_to_run = (
        [g for g in game_ids_in_data if g in args.game_ids]
        if args.game_ids else game_ids_in_data
    )
    log.info("Games to process: %d", len(game_ids_to_run))

    manifest_rows: list[dict] = []

    for game_id in game_ids_to_run:
        log.info("-" * 60)
        log.info("Processing game=%s ...", game_id)
        try:
            row = process_game(
                game_id=game_id,
                tracking_full=tracking_full,
                players_df=players_df,
                plays_df=plays_df,
                games_df=games_df,
                pff_df=pff_df,
                field_grid=field_grid,
                output_dir=args.output_dir,
            )
            manifest_rows.append(row)
        except Exception as exc:
            log.error("game=%s FAILED: %s", game_id, exc, exc_info=True)
            raise

    if manifest_rows:
        _update_manifest(
            manifest_path=args.output_dir / "manifest.parquet",
            new_rows=pd.DataFrame(manifest_rows),
        )

    log.info("=" * 72)
    log.info("ETL complete. %d game(s) written to: %s", len(manifest_rows), args.output_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
