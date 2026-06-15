"""FastAPI serving layer for transformer coverage predictions.

Serves pre-computed per-play predictions as JSON. Runs locally for development
with::

    uvicorn app.main:app --reload

and deploys to AWS Lambda unchanged via the module-level ``handler`` (Mangum).
"""
from __future__ import annotations

import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel

from app.predictions import load_play_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str


class DefenderPrediction(BaseModel):
    nflId: int
    probabilities: dict[str, float]


class PlayPredictionResponse(BaseModel):
    gameId: int
    playId: int
    defenders: list[DefenderPrediction]


app = FastAPI(
    title="NFL Coverage Responsibility API",
    description="Serves pre-computed transformer coverage-responsibility predictions.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/predictions/{game_id}/{play_id}", response_model=PlayPredictionResponse)
def get_play_predictions(game_id: int, play_id: int) -> PlayPredictionResponse:
    defenders = load_play_predictions(game_id, play_id)
    if defenders is None:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for game {game_id}, play {play_id}.",
        )
    return PlayPredictionResponse(
        gameId=game_id,
        playId=play_id,
        defenders=defenders,
    )


# AWS Lambda entry point. The same ``app`` object above is served locally by
# uvicorn, so no code changes are needed between local and Lambda deployment.
handler = Mangum(app)
