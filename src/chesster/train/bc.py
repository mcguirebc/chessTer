"""Behavior cloning (supervised learning) trainer for SmallNetPolicy."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chesster.policies.net import (
    SmallChessNet,
    board_to_tensor,
    get_move_vocab,
    legal_move_mask,
)
from chesster.selfplay.record import GameRecord, load_games

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for behavior cloning training."""

    # Data
    data_paths: list[str] = field(default_factory=list)
    val_split: float = 0.1

    # Model
    num_channels: int = 128
    num_blocks: int = 4

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    device: str = "cpu"

    # Checkpointing
    output_dir: str = "runs/bc"
    save_every_epoch: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_paths": self.data_paths,
            "val_split": self.val_split,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "device": self.device,
            "output_dir": self.output_dir,
            "save_every_epoch": self.save_every_epoch,
        }


@dataclass
class TrainingSample:
    """A single training sample: position + target move."""

    fen: str
    target_uci: str


class BCDataset(Dataset):
    """Dataset for behavior cloning from game records."""

    def __init__(self, samples: list[TrainingSample]) -> None:
        self.samples = samples
        self._uci_to_idx, _ = get_move_vocab()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        # Parse board and get features
        board = chess.Board(sample.fen)
        features = board_to_tensor(board)
        mask = legal_move_mask(board)

        # Get target index
        target_idx = self._uci_to_idx.get(sample.target_uci, -1)
        if target_idx == -1:
            # Target move not in vocabulary (shouldn't happen for legal moves)
            # Use first legal move as fallback
            for move in board.legal_moves:
                uci = move.uci()
                if uci in self._uci_to_idx:
                    target_idx = self._uci_to_idx[uci]
                    break

        return features, mask, target_idx


def extract_samples_from_games(
    games: Iterator[GameRecord],
    require_teacher: bool = True,
) -> list[TrainingSample]:
    """
    Extract training samples from game records.

    Args:
        games: Iterator of GameRecord objects.
        require_teacher: If True, only use positions with teacher_uci annotation.

    Returns:
        List of TrainingSample objects.
    """
    samples: list[TrainingSample] = []

    for game in games:
        for move in game.moves:
            if require_teacher:
                if move.teacher_uci is None:
                    continue
                target_uci = move.teacher_uci
            else:
                target_uci = move.uci

            samples.append(TrainingSample(fen=move.fen, target_uci=target_uci))

    return samples


def train_bc(config: BCConfig) -> Path:
    """
    Train a SmallNetPolicy via behavior cloning.

    Args:
        config: Training configuration.

    Returns:
        Path to the saved model checkpoint.
    """
    logger.info("Starting behavior cloning training")
    logger.info(f"Config: {config.to_dict()}")

    # Load data
    logger.info(f"Loading data from {len(config.data_paths)} files")
    all_samples: list[TrainingSample] = []
    for path in config.data_paths:
        games = load_games(path)
        samples = extract_samples_from_games(games, require_teacher=True)
        all_samples.extend(samples)
        logger.info(f"  {path}: {len(samples)} samples")

    if not all_samples:
        raise ValueError("No training samples found. Ensure games have teacher_uci annotations.")

    logger.info(f"Total samples: {len(all_samples)}")

    # Split train/val
    n_val = int(len(all_samples) * config.val_split)
    n_train = len(all_samples) - n_val

    # Shuffle deterministically
    import random

    rng = random.Random(42)
    rng.shuffle(all_samples)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets and loaders
    train_dataset = BCDataset(train_samples)
    val_dataset = BCDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    device = torch.device(config.device)
    model = SmallChessNet(
        num_channels=config.num_channels,
        num_blocks=config.num_blocks,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Training loop
    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, masks, targets in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits, _ = model(features, masks)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += features.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, masks, targets in val_loader:
                features = features.to(device)
                masks = masks.to(device)
                targets = targets.to(device)

                logits, _ = model(features, masks)
                loss = F.cross_entropy(logits, targets, ignore_index=-1)

                val_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += features.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )

        # Save checkpoint
        if config.save_every_epoch:
            epoch_path = output_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), epoch_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model saved (val_acc={val_acc:.2%})")

    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)

    # Save training info
    info_path = output_dir / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "best_val_acc": best_val_acc,
                "total_samples": len(all_samples),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
            f,
            indent=2,
        )

    logger.info(f"Training complete. Best model: {best_model_path}")
    return best_model_path


def main() -> None:
    """CLI entrypoint for behavior cloning training."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train SmallNetPolicy via behavior cloning")
    parser.add_argument("--data", type=str, required=True, nargs="+", help="Path(s) to JSONL game files")
    parser.add_argument("--out", type=str, default="runs/bc", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    config = BCConfig(
        data_paths=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    train_bc(config)


if __name__ == "__main__":
    main()

