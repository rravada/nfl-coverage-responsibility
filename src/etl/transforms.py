from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MATCHUP_TARGET_COLUMN,
    MAX_RECEIVERS,
    NAN_FILL_VALUE,
    NO_MATCHUP_SLOT,
    POSITION_MAP,
    POSITION_UNKNOWN_ID,
    SNAP_EVENT,
)

_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_PRESERVE_FLOAT64: frozenset[str] = frozenset(
    {"defender_nflId", "nflId", "pff_primaryDefensiveCoveredReceiverId"}
)

log = logging.getLogger(__name__)


def apply_nan_sentinel(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_value: float,
) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        log.warning(
            "Synthesising %d missing feature columns as fill_value=%.1f: %s",
            len(missing),
            fill_value,
            missing,
        )
        for col in missing:
            df[col] = fill_value
    existing = [c for c in feature_columns if c in df.columns]
    nan_before = int(df[existing].isna().sum().sum())
    df[existing] = df[existing].fillna(fill_value)
    log.debug(
        "NaN sentinel: %d values replaced with %.1f across %d columns.",
        nan_before,
        fill_value,
        len(existing),
    )
    return df


def add_snap_frame_marker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_snap_frame"] = df["event"] == SNAP_EVENT
    return df


def add_frames_since_snap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    snap_frameId = (
        df[df["is_snap_frame"]]
        .groupby(["gameId", "playId"])["frameId"]
        .first()
        .rename("snap_frameId")
        .reset_index()
    )

    df = df.merge(snap_frameId, on=["gameId", "playId"], how="left")
    df["snap_frameId"] = df["snap_frameId"].fillna(df["frameId"])

    raw = df["frameId"].astype(np.int32) - df["snap_frameId"].astype(np.int32)
    df["frames_since_snap"] = raw.astype(np.int16)
    df = df.drop(columns=["snap_frameId"])
    return df


def add_play_group_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["play_group_key"] = df["gameId"].astype(str) + "_" + df["playId"].astype(str)
    return df


def downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="float64").columns:
        if col not in _PRESERVE_FLOAT64:
            df[col] = df[col].astype(np.float32)
    # Feature columns that happen to be int64 (no NaN introduced by left-join)
    # must also become float32 to match the schema.
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].dtype == np.int64:
            df[col] = df[col].astype(np.float32)
    _INT_CASTS: dict[str, type] = {
        "gameId": np.int32,
        "playId": np.int32,
        "frameId": np.int16,
    }
    for col, dtype in _INT_CASTS.items():
        if col in df.columns and not df[col].isna().any():
            df[col] = df[col].astype(dtype)
    return df


def derive_territory_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _frame_total = (
        df.groupby(_FRAME_KEY)["territory_area_sq_yd"].transform("sum")
    )
    df["territory_ratio"] = np.where(
        _frame_total > 0,
        df["territory_area_sq_yd"] / _frame_total,
        np.nan,
    )
    return df


def add_position_and_team_ids(
    df: pd.DataFrame,
    players_df: pd.DataFrame,
    plays_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    pos_lookup = (
        players_df[["nflId", "officialPosition"]]
        .drop_duplicates("nflId")
        .rename(columns={"nflId": "defender_nflId"})
    )
    df = df.merge(pos_lookup, on="defender_nflId", how="left")
    df["position_id"] = (
        df["officialPosition"].map(POSITION_MAP).fillna(POSITION_UNKNOWN_ID).astype(np.int8)
    )
    df = df.drop(columns=["officialPosition"])

    poss_lookup = (
        plays_df[["gameId", "playId", "possessionTeam"]]
        .drop_duplicates(["gameId", "playId"])
    )
    df = df.merge(poss_lookup, on=["gameId", "playId"], how="left")
    df["team_id"] = np.where(df["team"] == df["possessionTeam"], 0, 1).astype(np.int8)
    df = df.drop(columns=["possessionTeam", "team"])

    return df


_ELIGIBLE_POSITIONS: frozenset[str] = frozenset({"WR", "TE", "RB", "FB"})


def derive_matchup_slots(
    feature_df: pd.DataFrame,
    normalized: pd.DataFrame,
    plays_df: pd.DataFrame,  # retained for API compatibility; not used directly
) -> pd.DataFrame:
    """Attach per-play relative receiver slot labels to the defender feature rows.

    Adds two columns to *feature_df* (which must already contain
    ``assigned_receiver_nflId`` and ``coverage_label``):

    ``matchup_slot`` (int8)
        0‥MAX_RECEIVERS-1 — the left-to-right slot of the covered receiver;
        NO_MATCHUP_SLOT — zone coverage or receiver absent from roster.

    ``slot_receiver_nflIds`` (object — list[int])
        The ordered roster of eligible receiver nflIds for that play (length
        ≤ MAX_RECEIVERS).  Broadcast to every defender-frame row of the play
        so that offline inference can re-expand slot probabilities back to
        actual receiver IDs.

    Eligible receivers are skill-position offensive players (WR / TE / RB / FB
    on the possessionTeam) present at the snap frame (or the play's first frame
    when no snap row exists), ordered left-to-right by *y*-coordinate with
    nflId as a tie-break.  Only the first MAX_RECEIVERS are kept.

    ``possessionTeam`` is read directly from *normalized* (build_normalized_tracking
    already merges plays metadata in before this function is called).

    This function is vectorised (groupby / merge) and does not iterate over
    individual rows or frames.
    """
    df = feature_df.copy()

    # ------------------------------------------------------------------ #
    # 1.  Build per-play eligible-receiver rosters from the full tracking  #
    #     frame (normalized contains all 22 players, not just defenders).  #
    # ------------------------------------------------------------------ #
    # Identify each player's snap frame; fall back to the play's minimum frameId.
    # normalized already carries possessionTeam (merged in by build_normalized_tracking).
    snap_flag = normalized["event"].eq(SNAP_EVENT)
    snap_frames = (
        normalized[snap_flag]
        .groupby(["gameId", "playId"])["frameId"]
        .first()
        .rename("snap_frameId")
        .reset_index()
    )
    first_frames = (
        normalized.groupby(["gameId", "playId"])["frameId"]
        .min()
        .rename("snap_frameId")
        .reset_index()
    )
    # Prefer actual snap; fall back to first frame when absent.
    ref_frames = (
        first_frames
        .merge(snap_frames, on=["gameId", "playId"], how="left", suffixes=("_first", "_snap"))
    )
    ref_frames["snap_frameId"] = (
        ref_frames["snap_frameId_snap"].combine_first(ref_frames["snap_frameId_first"])
    )
    ref_frames = ref_frames[["gameId", "playId", "snap_frameId"]]

    # Filter normalized to only the reference frame rows.
    snap_tracking = normalized.merge(ref_frames, on=["gameId", "playId"], how="inner")
    snap_tracking = snap_tracking[snap_tracking["frameId"] == snap_tracking["snap_frameId"]]

    # Keep only eligible offensive skill players (possessionTeam is already present
    # in normalized / snap_tracking from build_normalized_tracking).
    eligible = snap_tracking[
        snap_tracking["officialPosition"].isin(_ELIGIBLE_POSITIONS)
        & (snap_tracking["team"] == snap_tracking["possessionTeam"])
    ].copy()

    # Sort left-to-right (ascending y) with nflId as tie-break; keep ≤ MAX_RECEIVERS.
    eligible = eligible.sort_values(["gameId", "playId", "y", "nflId"])
    eligible["_slot_rank"] = (
        eligible.groupby(["gameId", "playId"]).cumcount()
    )
    eligible = eligible[eligible["_slot_rank"] < MAX_RECEIVERS]

    # Build roster map: (gameId, playId, nflId) → slot index.
    roster_slot = eligible[["gameId", "playId", "nflId", "_slot_rank"]].rename(
        columns={"nflId": "assigned_receiver_nflId", "_slot_rank": "_recv_slot"}
    )

    # Build per-play roster list (ordered list[int] of nflIds).
    roster_list = (
        eligible.sort_values(["gameId", "playId", "_slot_rank"])
        .groupby(["gameId", "playId"])["nflId"]
        .apply(lambda s: [int(x) for x in s])
        .rename("slot_receiver_nflIds")
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 2.  Map each defender-frame row to its receiver's slot index.        #
    # ------------------------------------------------------------------ #
    # assigned_receiver_nflId is float64 in feature_df; copy to a safe key.
    df["_assigned_int"] = df["assigned_receiver_nflId"]

    # Left join: each df row matches at most one roster_slot row.
    # reset_index on both sides guarantees positional alignment.
    df_reset = df.reset_index(drop=True)
    roster_for_join = roster_slot.rename(columns={"assigned_receiver_nflId": "_assigned_int"})
    merged = df_reset.merge(roster_for_join, on=["gameId", "playId", "_assigned_int"], how="left")

    # Default is NO_MATCHUP_SLOT; man-coverage rows get their receiver slot rank.
    is_man = df_reset["coverage_label"].ne("No_Matchup_Zone")
    recv_slot = merged["_recv_slot"].reset_index(drop=True)
    matchup_slot = np.where(
        is_man.values & recv_slot.notna().values,
        recv_slot.values,
        NO_MATCHUP_SLOT,
    ).astype(np.int8)

    df_reset[MATCHUP_TARGET_COLUMN] = matchup_slot
    df = df_reset.drop(columns=["_assigned_int"])

    # ------------------------------------------------------------------ #
    # 3.  Broadcast the roster list to every row of the play.             #
    # ------------------------------------------------------------------ #
    df = df.merge(roster_list, on=["gameId", "playId"], how="left")
    # Plays with no eligible receivers (rare): fill NaN cells with empty list.
    # Using apply on the whole column avoids pandas .loc/list-assignment gotchas.
    df["slot_receiver_nflIds"] = df["slot_receiver_nflIds"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    return df


def derive_coverage_labels(df: pd.DataFrame, pff_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pff_coverage" in pff_df.columns:
        pff_snap = (
            pff_df[["gameId", "playId", "nflId", "pff_coverage"]]
            .drop_duplicates(subset=["gameId", "playId", "nflId"])
            .rename(columns={"nflId": "defender_nflId"})
        )
        df = df.merge(pff_snap, on=["gameId", "playId", "defender_nflId"], how="left")
        pff_is_zone: pd.Series = (
            df["pff_coverage"].isna()
            | ~df["pff_coverage"].str.lower().str.contains("man", na=False)
        )
    else:
        pff_is_zone = pd.Series(False, index=df.index)

    no_voronoi: pd.Series = df["assigned_receiver_nflId"].isna()
    is_zone: pd.Series = pff_is_zone | no_voronoi

    df["coverage_label"] = np.where(
        is_zone,
        "No_Matchup_Zone",
        df["assigned_receiver_nflId"]
        .dropna()
        .astype(np.int64)
        .astype(str)
        .reindex(df.index, fill_value="No_Matchup_Zone"),
    )
    return df
