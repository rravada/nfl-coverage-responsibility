from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    MATCHUP_TARGET_COLUMN,
    MAX_RECEIVERS,
    NAN_FILL_VALUE,
    NO_MATCHUP_SLOT,
    PASS_ARRIVAL_EVENT,
    PASS_FORWARD_EVENT,
    POSITION_MAP,
    POSITION_UNKNOWN_ID,
    SNAP_EVENT,
    TRANSFORMER_FEATURE_COLUMNS,
)

_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]
_DEFENDER_KEY: list[str] = _FRAME_KEY + ["defender_nflId"]
_PRESERVE_FLOAT64: frozenset[str] = frozenset(
    {"defender_nflId", "nflId", "pff_primaryDefensiveCoveredReceiverId"}
)

# Coverage defender positions (same set used in etl_job.py).
_COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)

# Eligible offensive skill positions for receiver-slot and nearest-receiver logic.
_ELIGIBLE_POSITIONS: frozenset[str] = frozenset({"WR", "TE", "RB", "FB"})

# PFF per-defender covered-receiver column (absent in BDB 2023 data; guarded).
_PFF_RECEIVER_COL: str = "pff_primaryDefensiveCoveredReceiverId"

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


def add_pass_event_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Mark pass_forward and pass_arrival frames (paper §3.3 event landmarks).

    Adds two boolean columns derived from the ``event`` column:
      ``is_pass_forward_frame`` — frame where the QB releases the ball.
      ``is_pass_arrival_frame`` — frame where the ball reaches the receiver.

    Uses literal event-string matching only (no auto-event fallback) to stay
    faithful to the paper's event-landmark definitions.  The ``event`` column
    must be present in *df* before this is called (merge it in first).
    """
    df = df.copy()
    df["is_pass_forward_frame"] = df["event"] == PASS_FORWARD_EVENT
    df["is_pass_arrival_frame"] = df["event"] == PASS_ARRIVAL_EVENT
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
    # Transformer feature columns that happen to be int64 (no NaN from left-join)
    # must also become float32 to match the schema.
    for col in TRANSFORMER_FEATURE_COLUMNS:
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
    """Compute territory_ratio from territory_area_sq_yd (XGBoost path only).

    No longer called by the transformer ETL; retained for the XGBoost pipeline
    (train.py / evaluate.py call their own feature functions inline) and for
    backward compatibility.
    """
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


def derive_assigned_receiver(normalized: pd.DataFrame) -> pd.DataFrame:
    """Assign each defender-frame the nearest eligible receiver by Euclidean distance.

    Replaces the Voronoi-based ``build_coverage_assignments`` call in the transformer
    ETL path, computing receiver identity purely from raw tracking coordinates.

    Priority
    --------
    1. ``pff_primaryDefensiveCoveredReceiverId`` (play-level PFF annotation) — used
       when the column is present and non-NaN; broadcast to every frame for that
       defender on that play.  **No-op on BDB 2023 data** (column absent).
    2. Nearest eligible receiver (WR / TE / RB / FB on the possessionTeam) at the
       same frame by Euclidean distance.

    Parameters
    ----------
    normalized:
        Full per-frame tracking DataFrame for one game, produced by
        ``build_normalized_tracking``.  Must carry ``officialPosition``,
        ``team``, ``possessionTeam``, ``x``, ``y``.

    Returns
    -------
    pd.DataFrame
        One row per (gameId, playId, frameId, defender_nflId).  Columns:
        ``gameId, playId, frameId, defender_nflId, assigned_receiver_nflId``.
        ``assigned_receiver_nflId`` is NaN where no eligible receiver exists
        on the play.
    """
    # --- Defenders (coverage positions only) ---
    def_df = (
        normalized
        .loc[
            normalized["officialPosition"].isin(_COVERAGE_POSITIONS),
            _FRAME_KEY + ["nflId", "x", "y"],
        ]
        .rename(columns={"nflId": "defender_nflId", "x": "_def_x", "y": "_def_y"})
        .reset_index(drop=True)
    )

    # --- Eligible receivers on the possession team ---
    rec_df = (
        normalized
        .loc[
            normalized["officialPosition"].isin(_ELIGIBLE_POSITIONS)
            & (normalized["team"] == normalized["possessionTeam"]),
            _FRAME_KEY + ["nflId", "x", "y"],
        ]
        .rename(columns={"nflId": "receiver_nflId", "x": "_rec_x", "y": "_rec_y"})
        .reset_index(drop=True)
    )

    # --- Cross-join per frame: one (defender, receiver) row for each pairing ---
    cross = def_df.merge(rec_df, on=_FRAME_KEY, how="left")
    cross["_dist2"] = (cross["_def_x"] - cross["_rec_x"]) ** 2 + (
        cross["_def_y"] - cross["_rec_y"]
    ) ** 2

    # Keep the closest receiver per defender-frame.
    nearest = (
        cross.sort_values("_dist2")
        .drop_duplicates(subset=_DEFENDER_KEY, keep="first")
        [_DEFENDER_KEY + ["receiver_nflId"]]
        .rename(columns={"receiver_nflId": "assigned_receiver_nflId"})
        .reset_index(drop=True)
    )

    # --- Priority-1 PFF override (guarded — no-op when column absent) ---
    if _PFF_RECEIVER_COL in normalized.columns:
        pff_recv = (
            normalized
            .loc[
                normalized["officialPosition"].isin(_COVERAGE_POSITIONS)
                & normalized[_PFF_RECEIVER_COL].notna(),
                ["gameId", "playId", "nflId", _PFF_RECEIVER_COL],
            ]
            .rename(columns={"nflId": "defender_nflId", _PFF_RECEIVER_COL: "_pff_recv"})
            .drop_duplicates(subset=["gameId", "playId", "defender_nflId"])
        )
        # Broadcast the play-level PFF assignment across all frames for that defender.
        nearest = nearest.merge(
            pff_recv[["gameId", "playId", "defender_nflId", "_pff_recv"]],
            on=["gameId", "playId", "defender_nflId"],
            how="left",
        )
        nearest["assigned_receiver_nflId"] = nearest["_pff_recv"].combine_first(
            nearest["assigned_receiver_nflId"]
        )
        nearest = nearest.drop(columns=["_pff_recv"])

    return nearest


def derive_coverage_labels(df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``coverage_label`` to each defender-frame row.

    Zone / man determination uses the play-level ``pff_passCoverageType`` column from
    *plays_df* (``Man`` → man; ``Zone``, ``Other``, or NaN → ``No_Matchup_Zone``).
    Additionally, any defender-frame with no assigned receiver (``assigned_receiver_nflId``
    is NaN) is forced to ``No_Matchup_Zone`` regardless of the coverage type.

    ``coverage_label`` values:
        ``"No_Matchup_Zone"`` — zone coverage, missing annotation, or no receiver.
        ``str(int(nflId))``   — man coverage: the covered receiver's nflId as string.

    Parameters
    ----------
    df:
        Defender-frame feature DataFrame.  Must contain ``gameId``, ``playId``,
        and ``assigned_receiver_nflId``.
    plays_df:
        Play-context DataFrame (from ``plays.csv``).  Must contain ``gameId``,
        ``playId``, and ``pff_passCoverageType``.
    """
    df = df.copy()

    cov_type = (
        plays_df[["gameId", "playId", "pff_passCoverageType"]]
        .drop_duplicates(subset=["gameId", "playId"])
    )
    df = df.merge(cov_type, on=["gameId", "playId"], how="left")

    # Non-"man" coverage type (Zone, Other, NaN) → zone.
    pff_is_zone: pd.Series = (
        df["pff_passCoverageType"].isna()
        | ~df["pff_passCoverageType"].str.lower().str.contains("man", na=False)
    )
    no_receiver: pd.Series = df["assigned_receiver_nflId"].isna()
    is_zone: pd.Series = pff_is_zone | no_receiver

    df["coverage_label"] = np.where(
        is_zone,
        "No_Matchup_Zone",
        df["assigned_receiver_nflId"]
        .dropna()
        .astype(np.int64)
        .astype(str)
        .reindex(df.index, fill_value="No_Matchup_Zone"),
    )
    df = df.drop(columns=["pff_passCoverageType"])
    return df


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
