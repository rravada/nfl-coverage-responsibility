"""Configuration for the coverage-prediction serving layer.

Resolves the root location of the pre-computed per-play prediction Parquet files
written by ``pipelines/run_inference.py``:

    data/predictions/game={gameId}/play={playId}/predictions.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# S3 SWAP POINT
# ---------------------------------------------------------------------------
# Today predictions are read from a local directory. When they are migrated to
# S3, this constant and the read call in ``app/predictions.py`` are the ONLY two
# places that change:
#   * Set PREDICTIONS_DIR (or this default) to an S3 URI, e.g.
#         s3://my-bucket/predictions
#   * In app/predictions.py, point the parquet read at that URI (pandas reads
#     "s3://..." directly via s3fs, or use pyarrow.fs.S3FileSystem).
# The partition layout (game={id}/play={id}/predictions.parquet) is identical on
# S3, so nothing else in the serving layer needs to change.
# ---------------------------------------------------------------------------
PREDICTIONS_ROOT: Path = Path(
    os.environ.get("PREDICTIONS_DIR", str(_project_root / "data" / "predictions"))
)
