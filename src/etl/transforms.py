from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, NAN_FILL_VALUE, SNAP_EVENT

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
