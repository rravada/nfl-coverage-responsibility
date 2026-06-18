"""
Offline inference job for the CoverageMatchupTransformer.

Loads the trained checkpoint (models/transformer_best.pt), runs the model over
every play in the feature store, and writes per-play prediction Parquet files to:

    data/predictions/game={gameId}/play={playId}/predictions.parquet

The partition layout mirrors the S3 structure that will replace the local prefix
in a later task; the inference logic here is intended to remain unchanged.

The transformer mean-pools over time, producing one probability distribution per
agent slot per play. Each output file therefore has one row per agent slot that
is present in at least one frame — columns are nflId plus one float32 column per
class in the checkpoint's label_map. No frameId column, no ground-truth column.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import FEATURE_STORE_DIR, NO_MATCHUP_SLOT, NUM_MATCHUP_CLASSES
from src.models.dataset import NFLCoverageDataset
from src.models.transformer import CoverageMatchupTransformer

_ZONE_COL = "No_Matchup_Zone"

OUTPUT_DIR = Path("data/predictions")
CHECKPOINT_PATH = Path("models/transformer_best.pt")

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _discover_paths(feature_store_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in feature_store_dir.rglob("features.parquet"):
        for part in path.parts:
            if part.startswith("game="):
                result[int(part.split("=")[1])] = path
                break
    return result


def _load_checkpoint(device: torch.device) -> dict:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run the training pipeline (pipelines/train_transformer.py) first."
        )
    return torch.load(CHECKPOINT_PATH, map_location=device)


def _predict_play(
    model: CoverageMatchupTransformer,
    item: dict,
    device: torch.device,
) -> pd.DataFrame | None:
    """Run model for one play and re-expand slot probabilities to receiver nflIds.

    Steps:
    1. Mask invalid receiver slot logits to -1e9 (slots >= numEligible, except
       NO_MATCHUP_SLOT which is always valid).
    2. Softmax over the masked 6-class logit vector.
    3. Re-expand: slot i → str(slotReceiverIds[i]); NO_MATCHUP_SLOT → "No_Matchup_Zone".
    4. Renormalise the kept columns to sum to exactly 1.0.
    5. Output one row per present defender, columns = receiver nflId strings + no-matchup.
    """
    num_eligible: int = item["meta"]["numEligible"]
    slot_receiver_ids: list = item["meta"]["slotReceiverIds"]

    features = item["features"].unsqueeze(0).to(device)
    agent_mask_t = item["agent_mask"].unsqueeze(0).to(device)
    position_ids = item["position_ids"].unsqueeze(0).to(device)
    team_ids = item["team_ids"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features, agent_mask_t, position_ids, team_ids)  # (1, A, C)

        # Mask empty receiver slots — same masking applied during training.
        C = logits.size(-1)
        eligible_mask = torch.zeros(1, C, dtype=torch.bool, device=device)
        if num_eligible > 0:
            eligible_mask[0, :num_eligible] = True
        eligible_mask[0, NO_MATCHUP_SLOT] = True
        logits = logits.masked_fill(~eligible_mask.unsqueeze(1), -1e9)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()       # (A, C)

    agent_mask = item["agent_mask"].numpy()                              # (T, A) bool
    a_idx = np.where(agent_mask.any(axis=0))[0]
    if a_idx.size == 0:
        return None

    defender_ids = np.asarray(item["meta"]["defenderIds"])
    nfl_ids = defender_ids[a_idx].astype(np.int64)
    prob_rows = probs[a_idx].astype(np.float32)                          # (present, C)

    # Build output columns: eligible receiver nflId strings + no-matchup.
    col_names = [str(rid) for rid in slot_receiver_ids] + [_ZONE_COL]
    valid_slot_indices = list(range(num_eligible)) + [NO_MATCHUP_SLOT]
    kept_probs = prob_rows[:, valid_slot_indices]                         # (present, ne+1)

    # Renormalise so columns sum to exactly 1.0 per defender.
    row_sums = kept_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    kept_probs = (kept_probs / row_sums).astype(np.float32)

    out = pd.DataFrame(kept_probs, columns=col_names)
    out.insert(0, "nflId", nfl_ids)
    return out


def _select_indices(
    dataset: NFLCoverageDataset, smoke_test: bool
) -> dict[int, list[int]]:
    game_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, play in enumerate(dataset.plays):
        game_to_indices[play.game_id].append(idx)

    games = sorted(game_to_indices)
    if smoke_test:
        games = games[:2]

    selected: dict[int, list[int]] = {}
    for game_id in games:
        indices = game_to_indices[game_id]
        if smoke_test:
            indices = indices[:3]
        selected[game_id] = indices
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline coverage inference")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    _setup_logging()
    device = _select_device()
    log.info("Device: %s", device)

    game_id_to_path = _discover_paths(Path(FEATURE_STORE_DIR))
    if not game_id_to_path:
        raise RuntimeError(f"No features.parquet files found under {FEATURE_STORE_DIR}")

    checkpoint = _load_checkpoint(device)
    num_classes: int = checkpoint.get("num_classes", NUM_MATCHUP_CLASSES)

    selected_game_ids = sorted(game_id_to_path)
    if args.smoke_test:
        selected_game_ids = selected_game_ids[:2]
    paths = [game_id_to_path[g] for g in selected_game_ids]

    dataset = NFLCoverageDataset(paths, label_map=None, truncate=False)

    model = CoverageMatchupTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    game_to_indices = _select_indices(dataset, args.smoke_test)
    total_plays = sum(len(v) for v in game_to_indices.values())
    log.info("Total plays to process: %d", total_plays)

    plays_written = 0
    files_written = 0
    for game_id in sorted(game_to_indices):
        indices = game_to_indices[game_id]
        log.info("Processing game %d — %d plays", game_id, len(indices))
        for idx in indices:
            item = dataset[idx]
            play_id = item["meta"]["playId"]
            log.debug("Predicting game %d play %d", game_id, play_id)

            predictions = _predict_play(model, item, device)
            if predictions is None:
                continue

            out_dir = OUTPUT_DIR / f"game={game_id}" / f"play={play_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(out_dir / "predictions.parquet", index=False)
            plays_written += 1
            files_written += 1

    log.info(
        "Inference complete. Plays written: %d | Files written: %d | Output: %s",
        plays_written,
        files_written,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
