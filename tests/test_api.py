"""Integration tests for the FastAPI serving layer.

Runs against the real prediction Parquet files already present under
``data/predictions/`` (game 2021090900, play 137).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# A game/play that exists on disk under data/predictions/.
EXISTING_GAME_ID = 2021090900
EXISTING_PLAY_ID = 137


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_play_returns_defenders_with_normalized_probabilities() -> None:
    response = client.get(f"/predictions/{EXISTING_GAME_ID}/{EXISTING_PLAY_ID}")
    assert response.status_code == 200

    body = response.json()
    assert body["gameId"] == EXISTING_GAME_ID
    assert body["playId"] == EXISTING_PLAY_ID

    defenders = body["defenders"]
    assert len(defenders) > 0

    for defender in defenders:
        assert isinstance(defender["nflId"], int)
        probabilities = defender["probabilities"]
        assert len(probabilities) > 0
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_missing_play_returns_404() -> None:
    response = client.get(f"/predictions/{EXISTING_GAME_ID}/999999")
    assert response.status_code == 404


def test_lambda_handler_imports_cleanly() -> None:
    from app.main import handler

    assert callable(handler)
