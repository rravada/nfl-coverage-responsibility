"""
pipelines/etl_job.py
====================
Feature store ETL job. Reads week*.csv files, computes the raw-kinematic
defender-centric feature matrix per game (paper §3.3: x, y, o_rad, dir_rad),
and writes partitioned Parquet output to data/feature_store/.

Receiver-defender matchup labels are derived from:
  1. pff_primaryDefensiveCoveredReceiverId (guarded; absent in BDB 2023 data)
  2. Nearest eligible receiver by Euclidean distance (drives all matchups here)
  Man/zone determination uses plays.csv → pff_passCoverageType.

Week partition is derived from games.csv (games_df["week"]) — not hardcoded.

Usage
-----
    # All weeks
    python pipelines/etl_job.py --data-dir data/raw --output-dir data/feature_store

    # Specific weeks
    python pipelines/etl_job.py --data-dir data/raw --weeks 1 2 3

    # Specific games within specific weeks
    python pipelines/etl_job.py --data-dir data/raw --weeks 1 --game-ids 2021090900
"""

from __future__ import annotations

import argparse
import logging
import re
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

from src.config import NAN_FILL_VALUE, SEASON, TRANSFORMER_FEATURE_COLUMNS
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
    add_position_and_team_ids,
    add_snap_frame_marker,
    apply_nan_sentinel,
    derive_assigned_receiver,
    derive_coverage_labels,
    derive_matchup_slots,
    downcast_memory,
)

_COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)
_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_DEFENDER_KEY: list[str] = _FRAME_KEY + ["defender_nflId"]
_WEEK_RE: re.Pattern[str] = re.compile(r"week(\d+)\.csv", re.IGNORECASE)

log = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature store ETL: raw CSVs → partitioned Parquet (4 raw kinematic features).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/feature_store"))
    p.add_argument("--game-ids", nargs="*", type=int, default=None, metavar="GAME_ID")
    p.add_argument(
        "--weeks",
        nargs="*",
        type=int,
        default=None,
        metavar="WEEK",
        help="Week numbers to process (e.g. --weeks 1 2 3). Defaults to all week*.csv files found.",
    )
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

    # Derive the true week number from games_df (authoritative; not hardcoded).
    if len(game_games) and "week" in game_games.columns:
        week = int(game_games["week"].iloc[0])
    else:
        log.warning("game=%s: week not found in games_df; defaulting to 1", game_id)
        week = 1

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

    # --- Defender kinematic base (4 raw features only; paper §3.3) ---
    kine_cols = _FRAME_KEY + ["nflId", "x", "y", "o_rad", "dir_rad"]
    kine_df = (
        normalized
        .loc[normalized["officialPosition"].isin(_COVERAGE_POSITIONS), kine_cols]
        .rename(columns={"nflId": "defender_nflId"})
        .reset_index(drop=True)
    )

    # --- Nearest-receiver assignment (no Voronoi; pure Euclidean distance) ---
    receiver_assignments = derive_assigned_receiver(normalized)

    feature_df = kine_df.merge(
        receiver_assignments[_DEFENDER_KEY + ["assigned_receiver_nflId"]],
        on=_DEFENDER_KEY,
        how="left",
    )

    # --- Event column needed for snap marker ---
    event_lookup = (
        normalized
        .groupby(_FRAME_KEY)["event"]
        .first()
        .reset_index()
    )
    feature_df = feature_df.merge(event_lookup, on=_FRAME_KEY, how="left")

    # --- Labels ---
    feature_df = derive_coverage_labels(feature_df, game_plays)
    feature_df = derive_matchup_slots(feature_df, normalized, game_plays)

    # --- Metadata ---
    feature_df = add_snap_frame_marker(feature_df)
    feature_df = add_frames_since_snap(feature_df)
    feature_df = add_play_group_key(feature_df)

    # Sentinel-fill transformer feature columns only (not the full 15-col XGBoost set).
    feature_df = apply_nan_sentinel(feature_df, TRANSFORMER_FEATURE_COLUMNS, NAN_FILL_VALUE)
    feature_df = downcast_memory(feature_df)

    # Drop working columns that were needed only for label derivation.
    drop_cols = [c for c in ["event", "assigned_receiver_nflId", "pff_coverage"] if c in feature_df.columns]
    feature_df = feature_df.drop(columns=drop_cols)

    # --- position_id / team_id (needs team column from raw slice) ---
    team_lookup = (
        slice_raw[["nflId", "team"]]
        .drop_duplicates("nflId")
        .rename(columns={"nflId": "defender_nflId"})
    )
    feature_df = feature_df.merge(team_lookup, on="defender_nflId", how="left")
    feature_df = add_position_and_team_ids(feature_df, players_df, game_plays)

    validate_schema(feature_df)

    feature_df = feature_df.sort_values(["playId", "frameId", "defender_nflId"]).reset_index(drop=True)

    out_path = (
        output_dir
        / f"season={SEASON}"
        / f"week={week:02d}"
        / f"game={game_id}"
        / "features.parquet"
    )
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
        "week": week,
        "gameId": int(game_id),
        "row_count": row_count,
        "n_plays": n_plays,
        "etl_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "2.0.0",
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    log.info("=" * 72)
    log.info("NFL Coverage Responsibility — Feature Store ETL")
    log.info("=" * 72)
    log.info("config: data_dir=%s output_dir=%s", args.data_dir, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading shared metadata...")
    players_df = load_players(str(args.data_dir / "players.csv"))
    plays_df = load_plays(str(args.data_dir / "plays.csv"))
    games_df = load_games(str(args.data_dir / "games.csv"))
    pff_df = load_pff_scouting(str(args.data_dir / "pffScoutingData.csv"))

    # --- Discover week*.csv files ---
    all_week_files = sorted(args.data_dir.glob("week*.csv"))
    week_file_map: dict[int, Path] = {}
    for wf in all_week_files:
        m = _WEEK_RE.match(wf.name)
        if m:
            week_file_map[int(m.group(1))] = wf

    if args.weeks:
        requested = set(args.weeks)
        week_file_map = {w: p for w, p in week_file_map.items() if w in requested}
        missing = requested - set(week_file_map)
        if missing:
            log.warning("Requested weeks not found: %s", sorted(missing))

    if not week_file_map:
        log.warning("No week*.csv files matched. Nothing to process.")
        return

    log.info(
        "Processing weeks: %s",
        ", ".join(str(w) for w in sorted(week_file_map)),
    )

    manifest_rows: list[dict] = []

    for week_num, week_path in sorted(week_file_map.items()):
        log.info("=" * 60)
        log.info("Reading %s ...", week_path.name)
        tracking_full = pd.read_csv(str(week_path), low_memory=False)

        game_ids_in_data = sorted(tracking_full["gameId"].unique())
        game_ids_to_run = (
            [g for g in game_ids_in_data if g in args.game_ids]
            if args.game_ids else game_ids_in_data
        )
        log.info(
            "Week %d: %d game(s) found in CSV, %d to process",
            week_num, len(game_ids_in_data), len(game_ids_to_run),
        )

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
