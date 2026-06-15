"""Data-access layer for pre-computed per-play coverage predictions.

Reads the Parquet files written by ``pipelines/run_inference.py`` and shapes them
into JSON-serializable per-defender records. Stateless: every function takes its
inputs and returns new objects.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import PREDICTIONS_ROOT

# Column that identifies the defender; every other column is a probability for
# one coverage class (a receiver nflId as a string, or "No_Matchup_Zone").
DEFENDER_ID_COLUMN = "nflId"


def prediction_path(game_id: int, play_id: int) -> Path:
    """Return the Parquet path for a given game/play under the predictions root."""
    return (
        PREDICTIONS_ROOT
        / f"game={game_id}"
        / f"play={play_id}"
        / "predictions.parquet"
    )


def load_play_predictions(game_id: int, play_id: int) -> list[dict] | None:
    """Load one play's predictions as a list of per-defender records.

    Returns ``None`` when no prediction file exists for the game/play, allowing
    the caller to translate that into a 404. Otherwise returns:

        [{"nflId": int, "probabilities": {class_name: float, ...}}, ...]

    one record per defender, with the full probability distribution (which sums
    to 1.0).
    """
    path = prediction_path(game_id, play_id)
    if not path.exists():
        return None

    # S3 SWAP POINT (see app/config.py): when predictions live on S3, ``path``
    # becomes an "s3://..." URI and pandas reads it directly via s3fs. The rest
    # of this function is storage-agnostic.
    frame = pd.read_parquet(path)

    class_columns = [c for c in frame.columns if c != DEFENDER_ID_COLUMN]

    defenders: list[dict] = []
    for record in frame.to_dict(orient="records"):
        defenders.append(
            {
                # Cast away numpy scalar types so the response is plain JSON.
                "nflId": int(record[DEFENDER_ID_COLUMN]),
                "probabilities": {
                    str(cls): float(record[cls]) for cls in class_columns
                },
            }
        )
    return defenders
