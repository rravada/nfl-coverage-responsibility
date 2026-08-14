from __future__ import annotations

import pandas as pd

from src.config import MATCHUP_TARGET_COLUMN, NO_MATCHUP_SLOT, TRANSFORMER_FEATURE_COLUMNS

FEATURE_STORE_SCHEMA: dict[str, str] = {
    "gameId": "int32",
    "playId": "int32",
    "frameId": "int16",
    # defender_nflId: historical name, now means "agent nflId" — the union of
    # coverage-position defenders AND possession-team offensive skill players
    # (WR/TE/RB/FB/QB).  Kept unrenamed to avoid rippling through the codebase.
    "defender_nflId": "float64",
    **{col: "float32" for col in TRANSFORMER_FEATURE_COLUMNS},
    "coverage_label": "object",
    "matchup_slot": "int8",
    "slot_receiver_nflIds": "object",
    "is_snap_frame": "bool",
    "is_pass_forward_frame": "bool",
    "is_pass_arrival_frame": "bool",
    "frames_since_snap": "int16",
    "play_group_key": "object",
    "position_id": "int8",
    "team_id": "int8",
}

# matchup_slot valid value range:
#   -1                    — non-defender agent row (offensive player); no training label.
#   0 .. NO_MATCHUP_SLOT-1 — man coverage on receiver slot i.
#   NO_MATCHUP_SLOT        — zone / no matchup (defender row).
_MATCHUP_SLOT_VALID_VALUES: frozenset[int] = frozenset(range(-1, NO_MATCHUP_SLOT + 1))


def validate_schema(df: pd.DataFrame) -> None:
    errors: list[str] = []

    for col, expected_dtype in FEATURE_STORE_SCHEMA.items():
        if col not in df.columns:
            errors.append(f"missing column: {col!r}")
            continue
        actual = str(df[col].dtype)
        if actual != expected_dtype:
            errors.append(
                f"wrong dtype for {col!r}: expected {expected_dtype!r}, got {actual!r}"
            )

    if MATCHUP_TARGET_COLUMN in df.columns:
        bad = ~df[MATCHUP_TARGET_COLUMN].isin(_MATCHUP_SLOT_VALID_VALUES)
        if bad.any():
            errors.append(
                f"{MATCHUP_TARGET_COLUMN!r} has {int(bad.sum())} value(s) outside "
                f"{sorted(_MATCHUP_SLOT_VALID_VALUES)}"
            )

    if errors:
        raise ValueError(
            f"Schema validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
