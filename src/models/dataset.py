"""
NFLCoverageDataset — PyTorch Dataset for the factorized attention transformer
(AWS/NGS architecture, arxiv:2603.25901).

Each sample is one play. __getitem__ returns a dict with seven keys:

    features     float32  (MAX_FRAMES, MAX_AGENTS, 5)
                          5 raw kinematic features: x, y, o_rad, dir_rad, and
                          frames_since_snap/10 (seconds since snap; negative
                          pre-snap is fine). Padded with NAN_FILL_VALUE where
                          no agent is present.
    labels       int64    (MAX_FRAMES, MAX_AGENTS)
                          -1 where padded, unknown label, or a non-defender
                          (offensive) agent — offensive agents are visible to
                          attention but are never a coverage target.
    agent_mask   bool     (MAX_FRAMES, MAX_AGENTS)
                          True only where a real agent row exists.
    position_ids int64    (MAX_AGENTS,)  one position_id per agent slot;
                          padded slots → POSITION_UNKNOWN_ID.
    team_ids     int64    (MAX_AGENTS,)  one team_id per agent slot (0=offense,
                          1=defense); padded slots → 0.
    slot_ids     int64    (MAX_AGENTS,)  index 0..MAX_RECEIVERS-1 if this agent
                          IS that play's i-th eligible receiver (left-to-right
                          at snap); NO_SLOT_ID for every other agent
                          (defenders, QB, and padded slots).
    meta         dict     gameId (int), playId (int),
                          frameIds (np.int32, real frames only),
                          defenderIds (np.float64, real agents only)

Agent slot assignment: the union of all defender_nflIds across every frame in the
play is sorted ascending and assigned slot indices 0…A-1. The same agent occupies
the same slot in every frame; absent-agent slots for a given frame remain padded.

Truncation strategy (paper Appendix A.1, Song et al. 2025):

  Training uses a 60/40 split between fixed windows and a truly-random window:

  60 % FIXED — one of 11 fixed (start, end) windows, each defined by a
  (pre_snap_offset, end_landmark) pair:
      start ∈ {snap−30, snap−20, snap−10, snap}   (frames at 10 Hz)
      end   ∈ {snap, pass_forward, pass_arrival}
      minus the degenerate (snap, snap) case → 11 windows total.

  40 % RANDOM — start and end both sampled uniformly in [snap−30, pass_arrival].

  "pass_arrival" = pass_arrived event when present, else pass_forward (Option A).
  This fallback is justified by ≤2pt accuracy difference in the paper's own tables
  and the ~0.5s gap between events in BDB-2023 data (only 4.3% have pass_arrived).

  Play filter (require_events=True, used for train/val only):
      Drop plays lacking a snap or pass_forward event.  These are sacks and
      scrambles where arrival/forward windows are undefined (paper §3.3).
      Inference keeps all plays (require_events=False, full sequence).

Inference / val: always uses the full available sequence (start=0, end=T).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from src.config import (
    MATCHUP_TARGET_COLUMN,
    MAX_AGENTS,
    MAX_FRAMES,
    MAX_RECEIVERS,
    NAN_FILL_VALUE,
    NO_MATCHUP_SLOT,
    NO_SLOT_ID,
    NUM_MATCHUP_CLASSES,
    POSITION_UNKNOWN_ID,
    PRE_SNAP_OFFSETS,
    PRE_SNAP_WINDOW,
    TARGET_COLUMN,
    TIME_FEATURE_COLUMN,
    TRANSFORMER_FEATURE_COLUMNS,
    TRANSFORMER_INPUT_DIM,
    TRUNCATION_FIXED_PROB,
)

_N_FEATURES = TRANSFORMER_INPUT_DIM  # 5: x, y, o_rad, dir_rad, frames_since_snap/10
_ZONE_LABEL = "No_Matchup_Zone"

# Fixed label map: slot_0..slot_4 map to indices 0-4; No_Matchup_Zone → 5.
# This is play-relative: slot index i corresponds to that play's i-th eligible
# receiver (sorted left-to-right at snap).  The map is constant across plays
# and across the full corpus.
_FIXED_LABEL_MAP: dict[str, int] = {
    f"slot_{i}": i for i in range(MAX_RECEIVERS)
}
_FIXED_LABEL_MAP[_ZONE_LABEL] = NO_MATCHUP_SLOT

# 11 fixed (start_offset, end_marker) strategies from paper Appendix A.1.
# start_offset is relative to snap (frames); end_marker is one of:
#   "snap"     → snap frame (inclusive end)
#   "forward"  → pass_forward frame
#   "arrival"  → pass_arrival frame (or pass_forward via Option A fallback)
# The degenerate (offset=0, end="snap") case is intentionally excluded.
_FIXED_STRATEGIES: list[tuple[int, str]] = [
    # start = 30 frames before snap
    (-30, "snap"), (-30, "forward"), (-30, "arrival"),
    # start = 20 frames before snap
    (-20, "snap"), (-20, "forward"), (-20, "arrival"),
    # start = 10 frames before snap
    (-10, "snap"), (-10, "forward"), (-10, "arrival"),
    # start = snap (degenerate snap→snap excluded → only 2 remain)
    (0, "forward"), (0, "arrival"),
]
assert len(_FIXED_STRATEGIES) == 11, "Paper Appendix A.1 requires exactly 11 fixed strategies"


@dataclass(frozen=True)
class _PlayData:
    feature_array: np.ndarray        # (T, A, _N_FEATURES) float32
    label_array: np.ndarray          # (T, A) int64  — slot indices 0‥NO_MATCHUP_SLOT
    presence_mask: np.ndarray        # (T, A) bool
    n_frames: int
    n_agents: int
    snap_local_idx: int              # local frame index of ball_snap; -1 if absent
    pass_forward_local_idx: int      # local frame index of pass_forward; -1 if absent
    pass_arrival_local_idx: int      # local frame index of pass_arrival (Option A); -1 if absent
    frame_ids: np.ndarray            # (T,) int32
    agent_ids: np.ndarray            # (A,) float64
    position_ids: np.ndarray         # (A,) int64
    team_ids: np.ndarray             # (A,) int64
    slot_ids: np.ndarray             # (A,) int64 — 0..MAX_RECEIVERS-1 or NO_SLOT_ID
    game_id: int
    play_id: int
    slot_receiver_ids: list          # ordered list[int] of eligible receiver nflIds
    num_eligible: int                # number of filled receiver slots (≤ MAX_RECEIVERS)


class NFLCoverageDataset(Dataset):

    def __init__(
        self,
        parquet_paths: list[Path],
        label_map: dict[str, int] | None,
        truncate: bool,
        require_events: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parquet_paths:
            Feature-store Parquet files to load (one per game).
        label_map:
            Accepted for API compatibility; the canonical fixed map is always used.
        truncate:
            True → apply paper's 60/40 fixed/random truncation (training).
            False → full sequence for every play (val / inference).
        require_events:
            True → drop plays where snap or pass_forward event is missing.
            Use True for train/val; False for inference (paper §3.3 filter).
        """
        self.truncate = truncate
        # label_map is fixed and play-relative; any caller-supplied map is
        # accepted for API compatibility but the canonical map is always used.
        self.label_map: dict[str, int] = _FIXED_LABEL_MAP

        if not parquet_paths:
            self.plays: list[_PlayData] = []
            return

        all_df = pd.concat(
            [pd.read_parquet(p) for p in parquet_paths], ignore_index=True
        )

        self.plays = []
        for (game_id, play_id), play_df in all_df.groupby(["gameId", "playId"]):
            self.plays.append(
                self._build_play_arrays(play_df, int(game_id), int(play_id))
            )

        if require_events:
            n_before = len(self.plays)
            self.plays = [
                p for p in self.plays
                if p.snap_local_idx != -1 and p.pass_forward_local_idx != -1
            ]
            n_dropped = n_before - len(self.plays)
            if n_dropped:
                import logging
                logging.getLogger(__name__).info(
                    "require_events: dropped %d/%d plays missing snap or pass_forward",
                    n_dropped, n_before,
                )

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
        feature_vals = play_df.loc[valid, TRANSFORMER_FEATURE_COLUMNS].values.astype(np.float32)
        # Time-since-snap channel: seconds since snap (negative pre-snap is fine).
        time_raw = pd.to_numeric(
            play_df.loc[valid, TIME_FEATURE_COLUMN], errors="coerce"
        ).to_numpy(dtype=np.float32)
        time_vals = (np.nan_to_num(time_raw, nan=0.0) / 10.0).astype(np.float32).reshape(-1, 1)
        feature_vals = np.concatenate([feature_vals, time_vals], axis=1)
        # Use the per-play relative slot column (int8, values 0‥NO_MATCHUP_SLOT).
        # Rows without a valid slot (e.g. NaN from a missing column) fall back to -1.
        raw_slots = play_df.loc[valid, MATCHUP_TARGET_COLUMN]
        label_v = pd.to_numeric(raw_slots, errors="coerce").fillna(-1).values.astype(np.int64)
        feature_array[fi_v, ai_v, :] = feature_vals
        label_array[fi_v, ai_v] = label_v
        presence_mask[fi_v, ai_v] = True

        # --- snap frame detection ---
        snap_rows = play_df[play_df["is_snap_frame"]]
        if len(snap_rows):
            snap_local_idx = frame_id_map[snap_rows["frameId"].iloc[0]]
        else:
            snap_local_idx = -1

        # --- pass_forward detection ---
        if "is_pass_forward_frame" in play_df.columns:
            fwd_rows = play_df[play_df["is_pass_forward_frame"]]
            if len(fwd_rows):
                pass_forward_local_idx = frame_id_map[fwd_rows["frameId"].iloc[0]]
            else:
                pass_forward_local_idx = -1
        else:
            pass_forward_local_idx = -1

        # --- pass_arrival detection (Option A: fall back to pass_forward) ---
        if "is_pass_arrival_frame" in play_df.columns:
            arr_rows = play_df[play_df["is_pass_arrival_frame"]]
            if len(arr_rows):
                pass_arrival_local_idx = frame_id_map[arr_rows["frameId"].iloc[0]]
            elif pass_forward_local_idx != -1:
                # Option A: pass_arrived absent → use pass_forward as proxy.
                # Justified by ≤2pt accuracy gap and ~0.5s event spacing.
                pass_arrival_local_idx = pass_forward_local_idx
            else:
                pass_arrival_local_idx = -1
        else:
            # Column absent (old feature store): Option A unconditionally.
            pass_arrival_local_idx = pass_forward_local_idx

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

        # --- per-play receiver roster (for inference slot → nflId re-expansion) ---
        # slot_receiver_nflIds is broadcast to every row; grab the first non-null value.
        # Normalise to list[int] regardless of how pyarrow reconstructs the column
        # (Python list, numpy array, or pyarrow ListScalar are all iterable).
        slot_receiver_ids: list = []
        if "slot_receiver_nflIds" in play_df.columns:
            roster_series = play_df["slot_receiver_nflIds"].dropna()
            if len(roster_series):
                raw = roster_series.iloc[0]
                try:
                    slot_receiver_ids = [int(x) for x in raw]
                except (TypeError, ValueError):
                    slot_receiver_ids = []
        num_eligible = len(slot_receiver_ids)

        # --- slot_ids: which agents ARE this play's eligible receivers ---
        # ≤ MAX_RECEIVERS entries, so a small Python loop is fine (no per-frame
        # or per-defender loop; this runs once per play over ≤ MAX_RECEIVERS ids).
        slot_id_lookup = {nid: i for i, nid in enumerate(slot_receiver_ids)}
        slot_ids_arr = np.fromiter(
            (slot_id_lookup.get(int(nid), NO_SLOT_ID) for nid in agent_ids),
            dtype=np.int64,
            count=A,
        )

        return _PlayData(
            feature_array=feature_array,
            label_array=label_array,
            presence_mask=presence_mask,
            n_frames=T,
            n_agents=A,
            snap_local_idx=snap_local_idx,
            pass_forward_local_idx=pass_forward_local_idx,
            pass_arrival_local_idx=pass_arrival_local_idx,
            frame_ids=frame_ids,
            agent_ids=agent_ids,
            position_ids=position_ids_arr,
            team_ids=team_ids_arr,
            slot_ids=slot_ids_arr,
            game_id=game_id,
            play_id=play_id,
            slot_receiver_ids=slot_receiver_ids,
            num_eligible=num_eligible,
        )

    # ------------------------------------------------------------------
    # Truncation (paper Appendix A.1)
    # ------------------------------------------------------------------

    def _fixed_slice(self, play: _PlayData, strategy_idx: int) -> tuple[int, int]:
        """Return (start, end) for one of the 11 fixed landmark windows."""
        snap = play.snap_local_idx
        fwd = play.pass_forward_local_idx
        arr = play.pass_arrival_local_idx  # already Option A fallback

        if snap == -1 or fwd == -1:
            # Fallback: shouldn't occur after require_events filter.
            return 0, play.n_frames

        offset, end_marker = _FIXED_STRATEGIES[strategy_idx]

        # Start: snap + offset (offset ≤ 0), clamped to [0, T-1].
        start = max(0, snap + offset)

        # End: inclusive landmark → exclusive slice index.
        if end_marker == "snap":
            end = snap + 1
        elif end_marker == "forward":
            end = fwd + 1
        else:  # "arrival"
            end = (arr if arr != -1 else fwd) + 1

        end = min(end, play.n_frames)
        # Guarantee at least a 1-frame window.
        if start >= end:
            start = max(0, end - 1)

        return start, end

    def _random_slice(self, play: _PlayData) -> tuple[int, int]:
        """Return (start, end) sampled uniformly within [snap−30, pass_arrival]."""
        snap = play.snap_local_idx
        arr = play.pass_arrival_local_idx  # already Option A fallback

        if snap == -1 or arr == -1:
            return 0, play.n_frames

        lo = max(0, snap - PRE_SNAP_WINDOW)
        hi = min(arr, play.n_frames - 1)

        if lo > hi:
            return 0, play.n_frames

        a = random.randint(lo, hi)
        b = random.randint(a, hi)
        return a, b + 1  # exclusive end

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.plays)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        play = self.plays[idx]

        if self.truncate:
            # 60% fixed (uniform over 11 strategies), 40% truly random.
            if random.random() < TRUNCATION_FIXED_PROB:
                i = random.randint(0, len(_FIXED_STRATEGIES) - 1)
                start, end = self._fixed_slice(play, i)
            else:
                start, end = self._random_slice(play)
        else:
            # Val / inference: full available sequence.
            start, end = 0, play.n_frames

        # Cap to MAX_FRAMES: plays longer than the capacity limit are truncated
        # from the start of the requested window.
        T_s = min(end - start, MAX_FRAMES)
        A = play.n_agents

        features = np.full(
            (MAX_FRAMES, MAX_AGENTS, _N_FEATURES), NAN_FILL_VALUE, dtype=np.float32
        )
        labels = np.full((MAX_FRAMES, MAX_AGENTS), -1, dtype=np.int64)
        agent_mask = np.zeros((MAX_FRAMES, MAX_AGENTS), dtype=bool)

        features[:T_s, :A, :] = play.feature_array[start:start + T_s]
        labels[:T_s, :A] = play.label_array[start:start + T_s]
        agent_mask[:T_s, :A] = play.presence_mask[start:start + T_s]

        position_ids_out = np.full(MAX_AGENTS, POSITION_UNKNOWN_ID, dtype=np.int64)
        team_ids_out = np.zeros(MAX_AGENTS, dtype=np.int64)
        slot_ids_out = np.full(MAX_AGENTS, NO_SLOT_ID, dtype=np.int64)
        position_ids_out[:A] = play.position_ids
        team_ids_out[:A] = play.team_ids
        slot_ids_out[:A] = play.slot_ids

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "agent_mask": torch.from_numpy(agent_mask),
            "position_ids": torch.from_numpy(position_ids_out),
            "team_ids": torch.from_numpy(team_ids_out),
            "slot_ids": torch.from_numpy(slot_ids_out),
            "meta": {
                "gameId": play.game_id,
                "playId": play.play_id,
                "frameIds": play.frame_ids[start:start + T_s],
                "defenderIds": play.agent_ids,
                "slotReceiverIds": play.slot_receiver_ids,
                "numEligible": play.num_eligible,
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    # meta.frameIds is variable-length across plays and cannot be stacked by the
    # default collator; keep meta as a list of dicts and collate everything else.
    meta_list = [item["meta"] for item in batch]
    tensor_batch = [{k: v for k, v in item.items() if k != "meta"} for item in batch]
    result = default_collate(tensor_batch)
    result["meta"] = meta_list
    return result
