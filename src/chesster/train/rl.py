"""REINFORCE-style RL trainer for SmallNetPolicy."""
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

from chesster.policies.net import (
    SmallChessNet,
    SmallNetPolicy,
    board_to_tensor,
    get_move_vocab,
    legal_move_mask,
)
from chesster.selfplay.record import GameRecord, load_games
from chesster.train.rewards import (
    RewardType,
    compute_discounted_returns,
    compute_game_rewards,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training."""

    # Data
    data_paths: list[str] = field(default_factory=list)

    # Rewards
    reward_type: RewardType = RewardType.MATCH_BESTMOVE
    gamma: float = 0.99  # Discount factor
    normalize_returns: bool = True

    # Model
    init_model_path: str | None = None
    num_channels: int = 128
    num_blocks: int = 4

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    grad_clip: float = 1.0
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    value_coef: float = 0.5  # Value loss coefficient
    device: str = "cpu"

    # Checkpointing
    output_dir: str = "runs/rl"
    save_every_epoch: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_paths": self.data_paths,
            "reward_type": self.reward_type.value,
            "gamma": self.gamma,
            "normalize_returns": self.normalize_returns,
            "init_model_path": self.init_model_path,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "grad_clip": self.grad_clip,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "device": self.device,
            "output_dir": self.output_dir,
            "save_every_epoch": self.save_every_epoch,
        }


@dataclass
class RLSample:
    """A single RL training sample."""

    fen: str
    action_uci: str
    reward: float
    return_: float  # Discounted return (reward-to-go)


def extract_rl_samples(
    games: Iterator[GameRecord],
    reward_type: RewardType,
    gamma: float = 0.99,
    policy_id_filter: str | None = None,
) -> list[RLSample]:
    """
    Extract RL training samples from games.

    Args:
        games: Iterator of game records.
        reward_type: Which reward function to use.
        gamma: Discount factor for returns.
        policy_id_filter: If set, only include moves from this policy.

    Returns:
        List of RLSample objects.
    """
    samples: list[RLSample] = []

    for game in games:
        game_rewards = compute_game_rewards(game, reward_type)
        returns = compute_discounted_returns(game_rewards.move_rewards, gamma)

        for move, mr, ret in zip(game.moves, game_rewards.move_rewards, returns):
            if policy_id_filter is not None and move.policy_id != policy_id_filter:
                continue

            samples.append(
                RLSample(
                    fen=move.fen,
                    action_uci=move.uci,
                    reward=mr.reward,
                    return_=ret,
                )
            )

    return samples


class RLDataset(torch.utils.data.Dataset):
    """Dataset for RL training."""

    def __init__(self, samples: list[RLSample]) -> None:
        self.samples = samples
        self._uci_to_idx, _ = get_move_vocab()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        sample = self.samples[idx]

        # Parse board and get features
        board = chess.Board(sample.fen)
        features = board_to_tensor(board)
        mask = legal_move_mask(board)

        # Get action index
        action_idx = self._uci_to_idx.get(sample.action_uci, -1)
        if action_idx == -1:
            # Fallback to first legal move (shouldn't happen)
            for move in board.legal_moves:
                uci = move.uci()
                if uci in self._uci_to_idx:
                    action_idx = self._uci_to_idx[uci]
                    break

        return features, mask, action_idx, sample.return_


def train_rl(config: RLConfig) -> Path:
    """
    Train a SmallNetPolicy via REINFORCE.

    Args:
        config: Training configuration.

    Returns:
        Path to the saved model checkpoint.
    """
    logger.info("Starting RL training (REINFORCE)")
    logger.info(f"Config: {config.to_dict()}")

    # Load data
    logger.info(f"Loading data from {len(config.data_paths)} files")
    all_samples: list[RLSample] = []
    for path in config.data_paths:
        games = load_games(path)
        samples = extract_rl_samples(
            games,
            reward_type=config.reward_type,
            gamma=config.gamma,
        )
        all_samples.extend(samples)
        logger.info(f"  {path}: {len(samples)} samples")

    if not all_samples:
        raise ValueError("No training samples found.")

    logger.info(f"Total samples: {len(all_samples)}")

    # Normalize returns if requested
    if config.normalize_returns and len(all_samples) > 1:
        returns = [s.return_ for s in all_samples]
        mean_ret = sum(returns) / len(returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8
        all_samples = [
            RLSample(
                fen=s.fen,
                action_uci=s.action_uci,
                reward=s.reward,
                return_=(s.return_ - mean_ret) / std_ret,
            )
            for s in all_samples
        ]
        logger.info(f"Normalized returns: mean={mean_ret:.4f}, std={std_ret:.4f}")

    # Create dataset and loader
    dataset = RLDataset(all_samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create or load model
    device = torch.device(config.device)
    model = SmallChessNet(
        num_channels=config.num_channels,
        num_blocks=config.num_blocks,
    ).to(device)

    if config.init_model_path is not None:
        logger.info(f"Loading initial weights from {config.init_model_path}")
        model.load_state_dict(
            torch.load(config.init_model_path, map_location=device, weights_only=True)
        )

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
    best_policy_loss = float("inf")
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(config.epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for features, masks, actions, returns in loader:
            features = features.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            returns = returns.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            logits, values = model(features, masks)

            # Policy loss (REINFORCE): -log_prob(action) * return
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            policy_loss = -(action_log_probs * returns).mean()

            # Value loss (optional baseline)
            value_loss = F.mse_loss(values.squeeze(-1), returns)

            # Entropy bonus (encourage exploration)
            probs = F.softmax(logits, dim=-1)
            # Mask out illegal moves for entropy computation
            probs = probs * masks.float()
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

            # Total loss
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            loss.backward()

            # Gradient clipping
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_batches += 1

        avg_policy_loss = total_policy_loss / max(n_batches, 1)
        avg_value_loss = total_value_loss / max(n_batches, 1)
        avg_entropy = total_entropy / max(n_batches, 1)

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, "
            f"entropy={avg_entropy:.4f}"
        )

        # Save checkpoint
        if config.save_every_epoch:
            epoch_path = output_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), epoch_path)

        # Save best model (by policy loss)
        if avg_policy_loss < best_policy_loss:
            best_policy_loss = avg_policy_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model saved (policy_loss={avg_policy_loss:.4f})")

    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)

    # Save training info
    info_path = output_dir / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "best_policy_loss": best_policy_loss,
                "total_samples": len(all_samples),
                "reward_type": config.reward_type.value,
            },
            f,
            indent=2,
        )

    logger.info(f"Training complete. Best model: {best_model_path}")
    return best_model_path


def main() -> None:
    """CLI entrypoint for RL training."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train SmallNetPolicy via REINFORCE")
    parser.add_argument("--data", type=str, required=True, nargs="+", help="Path(s) to JSONL game files")
    parser.add_argument("--init", type=str, default=None, help="Path to initial model weights")
    parser.add_argument("--out", type=str, default="runs/rl", help="Output directory")
    parser.add_argument("--reward", type=str, default="match_bestmove", choices=["match_bestmove", "cp_delta", "outcome"], help="Reward type")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    config = RLConfig(
        data_paths=args.data,
        init_model_path=args.init,
        output_dir=args.out,
        reward_type=RewardType(args.reward),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )

    train_rl(config)


if __name__ == "__main__":
    main()

