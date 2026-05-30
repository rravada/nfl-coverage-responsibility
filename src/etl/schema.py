from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS

FEATURE_STORE_SCHEMA: dict[str, str] = {
    "gameId": "int32",
    "playId": "int32",
    "frameId": "int16",
    "defender_nflId": "float64",
    **{col: "float32" for col in FEATURE_COLUMNS},
    "coverage_label": "object",
    "is_snap_frame": "bool",
    "frames_since_snap": "int16",
    "play_group_key": "object",
}


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

    if errors:
        raise ValueError(
            f"Schema validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
