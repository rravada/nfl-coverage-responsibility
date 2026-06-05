"""
NFLCoverageDataset — PyTorch Dataset for the factorized attention transformer
(AWS/NGS architecture, arxiv:2603.25901).

Each sample is one play. __getitem__ returns a dict with six keys:

    features     float32  (MAX_FRAMES, MAX_AGENTS, 15)
                          Padded with NAN_FILL_VALUE where no defender is present.
    labels       int64    (MAX_FRAMES, MAX_AGENTS)
                          -1 where padded, unknown label, or no defender.
    agent_mask   bool     (MAX_FRAMES, MAX_AGENTS)
                          True only where a real defender row exists.
    position_ids int64    (MAX_AGENTS,)  one position_id per agent slot;
                          padded slots → POSITION_UNKNOWN_ID.
    team_ids     int64    (MAX_AGENTS,)  one team_id per agent slot (0=offense,
                          1=defense); padded slots → 0.
    meta         dict     gameId (int), playId (int),
                          frameIds (np.int32, real frames only),
                          defenderIds (np.float64, real agents only)

Agent slot assignment: the union of all defender_nflIds across every frame in the
play is sorted ascending and assigned slot indices 0…A-1. The same agent occupies
the same slot in every frame; absent-agent slots for a given frame remain padded.

Truncation strategies (selected randomly in training mode, always 0 in inference):

    0   Full play
    1   Pre-snap: frames 0 through snap (inclusive)
    2   Snap frame only (single frame)
    3   Post-snap, keep 10 % of post-snap frames (ceil, min 1)
    4   Post-snap, keep 20 %
    5   Post-snap, keep 30 %
    6   Post-snap, keep 40 %
    7   Post-snap, keep 50 %
    8   Post-snap, keep 60 %
    9   Post-snap, keep 70 %
    10  Post-snap through pass arrival: slice [snap+1, pass_arrival+1)
        Pass arrival = frame where frames_since_snap is maximum and > 0.
        Falls back to full play when snap or pass-arrival index is absent.

Strategies 1, 2, 3–9 fall back to (0, T) when snap_local_idx == -1.
Strategies 3–9 fall back to (snap, snap+1) when n_post == 0.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    FEATURE_COLUMNS,
    MAX_AGENTS,
    MAX_FRAMES,
    NAN_FILL_VALUE,
    POSITION_UNKNOWN_ID,
    TARGET_COLUMN,
)

_N_FEATURES = len(FEATURE_COLUMNS)
_ZONE_LABEL = "No_Matchup_Zone"


@dataclass(frozen=True)
class _PlayData:
    feature_array: np.ndarray        # (T, A, _N_FEATURES) float32
    label_array: np.ndarray          # (T, A) int64
    presence_mask: np.ndarray        # (T, A) bool
    n_frames: int
    n_agents: int
    snap_local_idx: int              # -1 if absent
    pass_arrival_local_idx: int      # -1 if absent
    frame_ids: np.ndarray            # (T,) int32
    agent_ids: np.ndarray            # (A,) float64
    position_ids: np.ndarray         # (A,) int64
    team_ids: np.ndarray             # (A,) int64
    game_id: int
    play_id: int


class NFLCoverageDataset(Dataset):

    def __init__(
        self,
        parquet_paths: list[Path],
        label_map: dict[str, int] | None,
        truncate: bool,
    ) -> None:
        self.truncate = truncate

        if not parquet_paths:
            self.label_map: dict[str, int] = label_map or {}
            self.plays: list[_PlayData] = []
            return

        all_df = pd.concat(
            [pd.read_parquet(p) for p in parquet_paths], ignore_index=True
        )

        if label_map is None:
            all_labels = set(all_df[TARGET_COLUMN].dropna().unique())
            self.label_map = self._build_label_map(all_labels)
        else:
            self.label_map = label_map

        self.plays = []
        for (game_id, play_id), play_df in all_df.groupby(["gameId", "playId"]):
            self.plays.append(
                self._build_play_arrays(play_df, int(game_id), int(play_id))
            )

    # ------------------------------------------------------------------
    # Label map
    # ------------------------------------------------------------------

    def _build_label_map(self, all_labels: set[str]) -> dict[str, int]:
        numeric = sorted(
            (l for l in all_labels if l != _ZONE_LABEL),
            key=lambda l: int(l),
        )
        ordered = numeric + [_ZONE_LABEL]
        return {label: idx for idx, label in enumerate(ordered)}

    # ------------------------------------------------------------------
    # Per-play pre-computation
    # ------------------------------------------------------------------

    def _build_play_arrays(
        self, play_df: pd.DataFrame, game_id: int, play_id: int
    ) -> _PlayData:
        # --- frame index ---
        frame_ids = np.sort(play_df["frameId"].unique()).astype(np.int32)
        frame_id_map: dict[Any, int] = {fid: i for i, fid in enumerate(frame_ids)}
        T = len(frame_ids)

        # --- agent slot assignment ---
        agent_ids = np.sort(play_df["defender_nflId"].dropna().unique())
        if len(agent_ids) > MAX_AGENTS:
            agent_ids = agent_ids[:MAX_AGENTS]
        A = len(agent_ids)
        agent_slot_map: dict[float, int] = {nid: i for i, nid in enumerate(agent_ids)}

        # --- array initialisation ---
        feature_array = np.full((T, A, _N_FEATURES), NAN_FILL_VALUE, dtype=np.float32)
        label_array = np.full((T, A), -1, dtype=np.int64)
        presence_mask = np.zeros((T, A), dtype=bool)

        # --- vectorised fill ---
        fi = play_df["frameId"].map(frame_id_map).values.astype(np.int32)
        ai_raw = play_df["defender_nflId"].map(agent_slot_map)
        valid = ai_raw.notna().values
        fi_v = fi[valid]
        ai_v = ai_raw[valid].values.astype(np.int32)
        feature_vals = play_df.loc[valid, FEATURE_COLUMNS].values.astype(np.float32)
        label_v = (
            play_df.loc[valid, TARGET_COLUMN]
            .map(lambda l: self.label_map.get(l, -1))
            .values.astype(np.int64)
        )
        feature_array[fi_v, ai_v, :] = feature_vals
        label_array[fi_v, ai_v] = label_v
        presence_mask[fi_v, ai_v] = True

        # --- snap frame detection ---
        snap_rows = play_df[play_df["is_snap_frame"]]
        if len(snap_rows):
            snap_local_idx = frame_id_map[snap_rows["frameId"].iloc[0]]
        else:
            snap_local_idx = -1

        # --- pass arrival detection ---
        fss = play_df.groupby("frameId")["frames_since_snap"].first()
        candidates = fss[fss > 0]
        if candidates.empty:
            pass_arrival_local_idx = -1
        else:
            pass_arrival_local_idx = frame_id_map[candidates.idxmax()]

        # --- per-agent position and team ids (first valid frame per agent) ---
        agent_meta = (
            play_df[["defender_nflId", "position_id", "team_id"]]
            .dropna(subset=["defender_nflId"])
            .groupby("defender_nflId")[["position_id", "team_id"]]
            .first()
        )
        reindexed = agent_meta.reindex(agent_ids)
        position_ids_arr = reindexed["position_id"].fillna(POSITION_UNKNOWN_ID).values.astype(np.int64)
        team_ids_arr = reindexed["team_id"].fillna(0).values.astype(np.int64)

        return _PlayData(
            feature_array=feature_array,
            label_array=label_array,
            presence_mask=presence_mask,
            n_frames=T,
            n_agents=A,
            snap_local_idx=snap_local_idx,
            pass_arrival_local_idx=pass_arrival_local_idx,
            frame_ids=frame_ids,
            agent_ids=agent_ids,
            position_ids=position_ids_arr,
            team_ids=team_ids_arr,
            game_id=game_id,
            play_id=play_id,
        )

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def _get_frame_slice(self, play: _PlayData, strategy: int) -> tuple[int, int]:
        T = play.n_frames
        snap = play.snap_local_idx
        pass_arr = play.pass_arrival_local_idx

        if strategy == 0:
            return 0, T

        if strategy == 1:
            if snap == -1:
                return 0, T
            return 0, snap + 1

        if strategy == 2:
            if snap == -1:
                return 0, T
            return snap, snap + 1

        if 3 <= strategy <= 9:
            if snap == -1:
                return 0, T
            post_start = snap + 1
            n_post = T - post_start
            if n_post == 0:
                return snap, snap + 1
            pct = (strategy - 2) * 0.1
            n_keep = max(1, math.ceil(n_post * pct))
            n_keep = min(n_keep, n_post)
            return post_start, post_start + n_keep

        if strategy == 10:
            if snap == -1 or pass_arr == -1:
                return 0, T
            start = snap + 1
            end = pass_arr + 1
            if start >= end:
                return 0, T
            return start, end

        return 0, T

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plays)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        play = self.plays[idx]
        strategy = random.randint(0, 10) if self.truncate else 0
        start, end = self._get_frame_slice(play, strategy)
        T_s = end - start
        A = play.n_agents

        features = np.full(
            (MAX_FRAMES, MAX_AGENTS, _N_FEATURES), NAN_FILL_VALUE, dtype=np.float32
        )
        labels = np.full((MAX_FRAMES, MAX_AGENTS), -1, dtype=np.int64)
        agent_mask = np.zeros((MAX_FRAMES, MAX_AGENTS), dtype=bool)

        features[:T_s, :A, :] = play.feature_array[start:end]
        labels[:T_s, :A] = play.label_array[start:end]
        agent_mask[:T_s, :A] = play.presence_mask[start:end]

        position_ids_out = np.full(MAX_AGENTS, POSITION_UNKNOWN_ID, dtype=np.int64)
        team_ids_out = np.zeros(MAX_AGENTS, dtype=np.int64)
        position_ids_out[:A] = play.position_ids
        team_ids_out[:A] = play.team_ids

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "agent_mask": torch.from_numpy(agent_mask),
            "position_ids": torch.from_numpy(position_ids_out),
            "team_ids": torch.from_numpy(team_ids_out),
            "meta": {
                "gameId": play.game_id,
                "playId": play.play_id,
                "frameIds": play.frame_ids[start:end],
                "defenderIds": play.agent_ids,
            },
        }
