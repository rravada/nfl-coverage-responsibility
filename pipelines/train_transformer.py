from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    FEATURE_STORE_DIR,
    MAX_RECEIVERS,
    MODELS_DIR,
    NO_MATCHUP_SLOT,
    NUM_MATCHUP_CLASSES,
    TRAIN_HASH_THRESHOLD,
)
from src.models.dataset import NFLCoverageDataset, collate_fn
from src.models.transformer import CoverageMatchupTransformer

NUM_EPOCHS = 200
BATCH_SIZE = 16


def _game_hash(game_id: int) -> float:
    return int(hashlib.md5(str(game_id).encode()).hexdigest(), 16) / (2**128)


def _setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _discover_paths(feature_store_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in feature_store_dir.rglob("features.parquet"):
        for part in path.parts:
            if part.startswith("game="):
                result[int(part.split("=")[1])] = path
                break
    return result


def _split_games(
    game_id_to_path: dict[int, Path], smoke_test: bool
) -> tuple[list[Path], list[Path]]:
    train_ids: list[int] = []
    val_ids: list[int] = []
    for gid in game_id_to_path:
        if _game_hash(gid) < TRAIN_HASH_THRESHOLD:
            train_ids.append(gid)
        else:
            val_ids.append(gid)
    if smoke_test:
        train_ids = sorted(train_ids)[:3]
        val_ids = sorted(val_ids)[:1]
    return [game_id_to_path[g] for g in train_ids], [game_id_to_path[g] for g in val_ids]


def _aggregate_labels(labels: torch.Tensor) -> torch.Tensor:
    # labels: (B, T, A) int64 — frame-level, -1 where padded
    B, T, A = labels.shape
    valid = labels != -1                                      # (B, T, A)
    has_valid = valid.any(dim=1)                              # (B, A)
    flipped_idx = valid.flip(1).long().argmax(dim=1)          # (B, A)
    last_idx = (T - 1) - flipped_idx                         # (B, A)
    play_labels = labels.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B, A)
    play_labels[~has_valid] = -1
    return play_labels


def _compute_class_weights(
    train_ds: NFLCoverageDataset, num_classes: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for play in train_ds.plays:
        labels = torch.from_numpy(play.label_array).unsqueeze(0)  # (1, T, A)
        play_labels = _aggregate_labels(labels).squeeze(0)         # (A,)
        valid = play_labels[play_labels != -1]
        if valid.numel() > 0:
            counts.scatter_add_(0, valid, torch.ones_like(valid))
    total_valid = counts.sum().item()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = total_valid / (num_classes * counts[nonzero].float())
    return weights.to(device)


def _run_epoch(
    model: CoverageMatchupTransformer,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: AdamW | None,
    scheduler: OneCycleLR | None,
    device: torch.device,
    num_classes: int,
    train: bool,
) -> tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total_valid = 0
    last_lr = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            agent_mask = batch["agent_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            team_ids = batch["team_ids"].to(device)

            logits = model(features, agent_mask, position_ids, team_ids)  # (B, A, C)

            # Mask receiver slots that are empty for this play to -1e9 so the
            # head cannot put probability mass on non-existent receivers.
            # NO_MATCHUP_SLOT is always kept.  (Paper §A.2 masking approach.)
            B_size = logits.size(0)
            eligible_mask = torch.zeros(B_size, num_classes, dtype=torch.bool, device=device)
            for b, m in enumerate(batch["meta"]):
                ne = int(m["numEligible"])
                if ne > 0:
                    eligible_mask[b, :ne] = True
            eligible_mask[:, NO_MATCHUP_SLOT] = True
            logits = logits.masked_fill(~eligible_mask.unsqueeze(1), -1e9)

            play_labels = _aggregate_labels(labels)                        # (B, A)

            loss = criterion(logits.reshape(-1, num_classes), play_labels.reshape(-1))

            if train:
                loss.backward()
                # Fully-padded agent slots produce softmax(all -inf) → NaN gradients.
                # Replace NaN grads with 0 before clipping and stepping.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                last_lr = scheduler.get_last_lr()[0]

            total_loss += loss.item()
            valid_mask = play_labels != -1
            correct += (logits.argmax(dim=-1)[valid_mask] == play_labels[valid_mask]).sum().item()
            total_valid += valid_mask.sum().item()

    n = max(len(loader), 1)
    return total_loss / n, correct / max(total_valid, 1), last_lr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoverageMatchupTransformer")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(models_dir / "transformer_training.log")
    log = logging.getLogger(__name__)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    game_id_to_path = _discover_paths(Path(FEATURE_STORE_DIR))
    if not game_id_to_path:
        raise RuntimeError(f"No features.parquet files found under {FEATURE_STORE_DIR}")

    train_paths, val_paths = _split_games(game_id_to_path, args.smoke_test)
    log.info("Train games: %d | Val games: %d", len(train_paths), len(val_paths))

    train_ds = NFLCoverageDataset(train_paths, label_map=None, truncate=True)
    val_ds = NFLCoverageDataset(val_paths, label_map=train_ds.label_map, truncate=False)
    log.info("Train plays: %d | Val plays: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=collate_fn,
    )

    num_classes = NUM_MATCHUP_CLASSES  # fixed 6-class per-play relative slot head
    model = CoverageMatchupTransformer(num_classes=num_classes).to(device)
    log.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    class_weights = _compute_class_weights(train_ds, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else NUM_EPOCHS)

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-4,
        total_steps=len(train_loader) * epochs,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_val_acc = float("-inf")
    checkpoint_path = models_dir / "transformer_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss, _, lr = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, num_classes, train=True
        )
        val_loss, val_acc, _ = _run_epoch(
            model, val_loader, criterion, None, None, device, num_classes, train=False
        )
        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
            epoch, epochs, train_loss, val_loss, val_acc, lr,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "label_map": train_ds.label_map,    # fixed slot→name map
                    "num_classes": num_classes,          # always NUM_MATCHUP_CLASSES
                    "max_receivers": MAX_RECEIVERS,
                    "no_matchup_slot": NO_MATCHUP_SLOT,
                },
                checkpoint_path,
            )
            log.info("Checkpoint saved (val_acc=%.4f)", val_acc)

    log.info("Training complete. Best val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    main()
