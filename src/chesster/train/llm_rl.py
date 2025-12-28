"""REINFORCE trainer for LLM chess policies with LoRA."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess

logger = logging.getLogger(__name__)


@dataclass
class LLMRLConfig:
    """Configuration for LLM RL training."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str | None = None  # Auto-detect
    load_in_4bit: bool = False

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training settings
    games_per_batch: int = 10
    epochs: int = 1
    lr: float = 1e-4
    gamma: float = 0.99  # Discount factor for returns
    max_moves: int = 100  # Max moves per game

    # Opponent settings
    opponent: str = "random"  # "random" or "self"

    # Output settings
    output_dir: str = "runs/llm/v1"
    save_every: int = 1  # Save every N epochs


@dataclass
class GameTrajectory:
    """A single game trajectory for training."""

    moves: list[str]  # UCI moves
    log_probs: list[Any]  # torch.Tensor log probs
    outcome: float  # +1 win, 0 draw, -1 loss (from white's perspective)
    fallback_count: int = 0  # Number of moves that were fallbacks
    num_moves: int = field(init=False)

    def __post_init__(self):
        self.num_moves = len(self.moves)


@dataclass
class TrainingStats:
    """Statistics from a training run."""

    epoch: int
    games_played: int
    avg_loss: float
    avg_return: float
    win_rate: float
    draw_rate: float
    avg_game_length: float
    valid_move_rate: float  # Fraction of moves that were valid (not fallback)
    elapsed_seconds: float


def compute_discounted_returns(
    outcome: float,
    num_moves: int,
    gamma: float,
    white_played: bool,
) -> list[float]:
    """
    Compute discounted returns for each move in a game.

    The outcome is from White's perspective (+1 if white wins).
    For Black's moves, we flip the sign.

    Args:
        outcome: Game outcome from White's perspective (+1, 0, -1).
        num_moves: Total number of moves in the game.
        gamma: Discount factor.
        white_played: Whether we played as White.

    Returns:
        List of discounted returns for each of our moves.
    """
    # Flip outcome if we played as Black
    our_outcome = outcome if white_played else -outcome

    # Compute returns working backwards from game end
    # Move at index i gets return = outcome * gamma^(num_moves - i - 1)
    returns = []
    for i in range(num_moves):
        discount = gamma ** (num_moves - i - 1)
        returns.append(our_outcome * discount)

    return returns


def play_training_game(
    policy,
    opponent_policy,
    config: LLMRLConfig,
    we_play_white: bool,
    seed: int | None = None,
) -> GameTrajectory | None:
    """
    Play a single training game, collecting trajectories.

    Args:
        policy: Our LoRA policy (trainable).
        opponent_policy: Opponent policy (random or self).
        config: Training configuration.
        we_play_white: Whether we play as White.
        seed: Random seed for opponent.

    Returns:
        GameTrajectory with moves, log_probs, and outcome.
    """
    from chesster.policies.base import ChooseMoveParams

    board = chess.Board()
    our_moves = []
    our_log_probs = []
    fallback_count = 0

    move_count = 0
    rng = random.Random(seed)

    while not board.is_game_over() and move_count < config.max_moves:
        is_our_turn = (board.turn == chess.WHITE) == we_play_white

        if is_our_turn:
            # Our turn - use LoRA policy
            params = ChooseMoveParams(temperature=0.3)
            result = policy.forward(board, params)

            our_moves.append(result.uci)
            our_log_probs.append(result.log_prob)
            if result.info.get("fallback"):
                fallback_count += 1

            board.push_uci(result.uci)
        else:
            # Opponent's turn
            if opponent_policy is None:
                # Random move
                legal = list(board.legal_moves)
                move = rng.choice(legal)
                board.push(move)
            else:
                # Self-play or other policy
                params = ChooseMoveParams(seed=rng.randint(0, 2**31))
                result = opponent_policy.choose_move(board, params)
                board.push_uci(result.uci)

        move_count += 1

    # Determine outcome
    if board.is_game_over():
        result = board.outcome()
        if result.winner is None:
            outcome = 0.0  # Draw
        elif result.winner:  # White wins
            outcome = 1.0
        else:  # Black wins
            outcome = -1.0
    else:
        # Game didn't finish - treat as draw
        outcome = 0.0

    if len(our_moves) == 0:
        return None

    return GameTrajectory(
        moves=our_moves,
        log_probs=our_log_probs,
        outcome=outcome,
        fallback_count=fallback_count,
    )


def train_epoch(
    policy,
    optimizer,
    config: LLMRLConfig,
    epoch: int,
) -> TrainingStats:
    """
    Train for one epoch (batch of games).

    Args:
        policy: LoRA policy to train.
        optimizer: PyTorch optimizer.
        config: Training configuration.
        epoch: Current epoch number.

    Returns:
        TrainingStats for this epoch.
    """
    import torch
    from chesster.policies.random import RandomPolicy

    start_time = time.time()
    policy.train_mode()

    # Opponent
    if config.opponent == "random":
        opponent = None  # Use inline random
    else:
        opponent = policy  # Self-play

    # Collect trajectories
    trajectories = []
    wins, draws, losses = 0, 0, 0

    for game_idx in range(config.games_per_batch):
        # Alternate colors
        we_play_white = game_idx % 2 == 0

        traj = play_training_game(
            policy=policy,
            opponent_policy=opponent if config.opponent == "self" else None,
            config=config,
            we_play_white=we_play_white,
            seed=epoch * 10000 + game_idx,
        )

        if traj is not None:
            trajectories.append((traj, we_play_white))

            # Track outcomes from our perspective
            our_outcome = traj.outcome if we_play_white else -traj.outcome
            if our_outcome > 0:
                wins += 1
            elif our_outcome < 0:
                losses += 1
            else:
                draws += 1

    if not trajectories:
        logger.warning("No valid trajectories collected")
        return TrainingStats(
            epoch=epoch,
            games_played=0,
            avg_loss=0.0,
            avg_return=0.0,
            win_rate=0.0,
            draw_rate=0.0,
            avg_game_length=0.0,
            elapsed_seconds=time.time() - start_time,
        )

    # Compute stats (no actual training on M4 - needs more memory for backprop)
    total_return = 0.0
    total_moves = 0

    for traj, we_play_white in trajectories:
        returns = compute_discounted_returns(
            outcome=traj.outcome,
            num_moves=traj.num_moves,
            gamma=config.gamma,
            white_played=we_play_white,
        )

        for ret in returns:
            total_return += ret
            total_moves += 1

    # Note: Full REINFORCE training disabled for MVP (needs more GPU memory)
    # On M4 MacBook, we validate the pipeline works; actual training on GCP
    avg_loss = 0.0  # Placeholder - no actual backprop in MVP

    elapsed = time.time() - start_time
    games_played = len(trajectories)

    # Calculate valid move rate
    total_fallbacks = sum(t.fallback_count for t, _ in trajectories)
    valid_move_rate = 1.0 - (total_fallbacks / max(total_moves, 1))

    return TrainingStats(
        epoch=epoch,
        games_played=games_played,
        avg_loss=avg_loss,  # Already a float in MVP mode
        avg_return=total_return / max(total_moves, 1),
        win_rate=wins / max(games_played, 1),
        draw_rate=draws / max(games_played, 1),
        avg_game_length=sum(t.num_moves for t, _ in trajectories) / max(games_played, 1),
        valid_move_rate=valid_move_rate,
        elapsed_seconds=elapsed,
    )


def train_llm_rl(config: LLMRLConfig) -> dict[str, Any]:
    """
    Main training function for LLM RL.

    Args:
        config: Training configuration.

    Returns:
        Training results dictionary.
    """
    import torch
    from chesster.policies.llm_lora import LoRAConfig, LoRALLMPolicy

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"Starting LLM RL training with config: {config}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize policy with LoRA
    lora_config = LoRAConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    policy = LoRALLMPolicy(
        model_name=config.model_name,
        device=config.device,
        lora_config=lora_config,
        load_in_4bit=config.load_in_4bit,
    )

    # Force load to get parameters
    policy._ensure_loaded()

    logger.info(f"Trainable parameters: {policy.trainable_parameters():,}")
    logger.info(f"Total parameters: {policy.total_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

    # Training loop
    all_stats = []
    best_win_rate = 0.0

    for epoch in range(config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")

        stats = train_epoch(policy, optimizer, config, epoch)
        all_stats.append(asdict(stats))

        logger.info(
            f"Epoch {epoch + 1}: loss={stats.avg_loss:.4f}, "
            f"return={stats.avg_return:.4f}, win_rate={stats.win_rate:.2%}, "
            f"valid_moves={stats.valid_move_rate:.1%}, "
            f"games={stats.games_played}, time={stats.elapsed_seconds:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_dir = output_dir / f"epoch_{epoch + 1}"
            policy.save_adapter(checkpoint_dir)

        # Track best
        if stats.win_rate > best_win_rate:
            best_win_rate = stats.win_rate
            policy.save_adapter(output_dir / "best_adapter")

    # Save final adapter
    policy.save_adapter(output_dir / "final_adapter")

    # Save training info
    training_info = {
        "config": asdict(config),
        "stats": all_stats,
        "best_win_rate": best_win_rate,
        "trainable_params": policy.trainable_parameters(),
        "total_params": policy.total_parameters(),
    }

    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    logger.info(f"Training complete. Best win rate: {best_win_rate:.2%}")
    logger.info(f"Results saved to {output_dir}")

    policy.unload()

    return training_info


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train LLM chess policy with REINFORCE")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu, or auto)",
    )
    parser.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")

    # Training
    parser.add_argument("--games", type=int, default=100, help="Games per epoch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Opponent
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "self"],
        help="Opponent type",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="runs/llm/v1",
        help="Output directory",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = LLMRLConfig(
        model_name=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        games_per_batch=args.games,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        opponent=args.opponent,
        output_dir=args.output,
    )

    train_llm_rl(config)


if __name__ == "__main__":
    main()
