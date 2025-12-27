"""End-to-end RL training loop with self-play and gated checkpointing."""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chesster.league.gating import GatingConfig, should_promote
from chesster.league.registry import ModelRegistry
from chesster.league.sampler import SamplerFactory
from chesster.policies.net import SmallChessNet, SmallNetPolicy
from chesster.selfplay.generator import GeneratorConfig, generate_selfplay_games
from chesster.selfplay.record import save_games
from chesster.train.rewards import RewardType
from chesster.train.rl import RLConfig, train_rl

if TYPE_CHECKING:
    from chesster.policies.base import Policy

logger = logging.getLogger(__name__)


@dataclass
class TrainingLoopConfig:
    """Configuration for the full RL training loop."""

    # Initialization
    init_model_path: str | None = None
    num_channels: int = 128
    num_blocks: int = 4

    # Loop
    iterations: int = 100
    games_per_iteration: int = 200
    max_moves: int = 500

    # Self-play
    n_opponents: int = 3
    sampler_type: str = "recent"  # uniform, recent, latest, mixed
    include_random: bool = True
    include_stockfish: bool = False

    # Reward
    reward_type: RewardType = RewardType.MATCH_BESTMOVE
    annotate_with_teacher: bool = True
    teacher_depth: int = 10

    # RL Training
    rl_epochs: int = 3
    rl_batch_size: int = 64
    rl_learning_rate: float = 1e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01

    # Gating
    gating_games: int = 50
    gating_threshold: float = 0.55

    # Infrastructure
    registry_path: str = "runs/registry"
    output_dir: str = "runs/loop"
    device: str = "cpu"
    base_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "init_model_path": self.init_model_path,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "iterations": self.iterations,
            "games_per_iteration": self.games_per_iteration,
            "max_moves": self.max_moves,
            "n_opponents": self.n_opponents,
            "sampler_type": self.sampler_type,
            "include_random": self.include_random,
            "include_stockfish": self.include_stockfish,
            "reward_type": self.reward_type.value,
            "annotate_with_teacher": self.annotate_with_teacher,
            "teacher_depth": self.teacher_depth,
            "rl_epochs": self.rl_epochs,
            "rl_batch_size": self.rl_batch_size,
            "rl_learning_rate": self.rl_learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "gating_games": self.gating_games,
            "gating_threshold": self.gating_threshold,
            "registry_path": self.registry_path,
            "output_dir": self.output_dir,
            "device": self.device,
            "base_seed": self.base_seed,
        }


@dataclass
class IterationResult:
    """Result of a single training iteration."""

    iteration: int
    games_generated: int
    rl_loss: float
    gating_passed: bool
    gating_score: float
    promoted: bool
    snapshot_name: str | None


def run_training_loop(config: TrainingLoopConfig) -> list[IterationResult]:
    """
    Run the full RL training loop.

    Loop:
        1. Sample opponents from registry + builtins
        2. Generate self-play games with optional teacher annotation
        3. Train via RL
        4. Gate: if better than baseline, register snapshot

    Args:
        config: Training loop configuration.

    Returns:
        List of IterationResult for each iteration.
    """
    logger.info("Starting RL training loop")
    logger.info(f"Config: {config.to_dict()}")

    # Setup directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Initialize registry
    registry = ModelRegistry(config.registry_path)

    # Initialize or load model
    device = torch.device(config.device)
    model = SmallChessNet(
        num_channels=config.num_channels,
        num_blocks=config.num_blocks,
    ).to(device)

    if config.init_model_path is not None:
        logger.info(f"Loading initial model from {config.init_model_path}")
        model.load_state_dict(
            torch.load(config.init_model_path, map_location=device, weights_only=True)
        )

    # Create policy wrapper
    policy = SmallNetPolicy(model=model, device=config.device)

    # Register initial model as baseline if registry is empty
    baseline_name: str | None = None
    if not registry.list_snapshots():
        init_path = models_dir / "init.pt"
        torch.save(model.state_dict(), init_path)
        registry.register("init", init_path, {"type": "initial"})
        baseline_name = "init"
        logger.info("Registered initial model as 'init'")
    else:
        baseline_name = registry.list_snapshots()[0].name
        logger.info(f"Using existing baseline: {baseline_name}")

    # Build built-in opponents
    builtin_opponents: list[Policy] = []
    if config.include_random:
        from chesster.policies.random import RandomPolicy
        builtin_opponents.append(RandomPolicy())
    if config.include_stockfish:
        from chesster.policies.stockfish import StockfishPolicy
        builtin_opponents.append(StockfishPolicy())

    # Teacher for annotation
    teacher: Policy | None = None
    if config.annotate_with_teacher:
        from chesster.policies.stockfish import StockfishPolicy
        teacher = StockfishPolicy(default_depth=config.teacher_depth)

    # Training loop
    results: list[IterationResult] = []

    for iteration in range(config.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{config.iterations}")
        logger.info(f"{'='*60}")

        iter_seed = (config.base_seed + iteration * 100000) if config.base_seed else None

        # 1. Sample opponents
        opponents: list[Policy] = list(builtin_opponents)

        if registry.list_snapshots():
            sampler = SamplerFactory.create(
                config.sampler_type,
                registry,
                seed=iter_seed,
            )
            registry_opponents = sampler.sample_policies(config.n_opponents, device=config.device)
            opponents.extend(registry_opponents)

        if not opponents:
            logger.warning("No opponents available, using self-play only")
            opponents = [policy]  # Self-play against current policy

        logger.info(f"Opponents: {[getattr(o, 'policy_id', 'unknown') for o in opponents]}")

        # 2. Generate self-play games
        gen_config = GeneratorConfig(
            n_games=config.games_per_iteration,
            max_moves=config.max_moves,
            base_seed=iter_seed,
            annotate_with_teacher=config.annotate_with_teacher,
            teacher_depth=config.teacher_depth,
        )

        games = generate_selfplay_games(policy, opponents, gen_config, teacher)
        logger.info(f"Generated {len(games)} games")

        # Save games
        games_path = games_dir / f"iter_{iteration:04d}.jsonl"
        save_games(games_path, games)

        # 3. Train via RL
        rl_output_dir = models_dir / f"iter_{iteration:04d}"
        rl_config = RLConfig(
            data_paths=[str(games_path)],
            init_model_path=None,  # We'll pass model directly
            reward_type=config.reward_type,
            gamma=config.gamma,
            epochs=config.rl_epochs,
            batch_size=config.rl_batch_size,
            learning_rate=config.rl_learning_rate,
            entropy_coef=config.entropy_coef,
            device=config.device,
            output_dir=str(rl_output_dir),
            save_every_epoch=False,
        )

        # Save current model for RL to load
        temp_model_path = rl_output_dir / "temp_init.pt"
        rl_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), temp_model_path)
        rl_config.init_model_path = str(temp_model_path)

        best_model_path = train_rl(rl_config)

        # Load trained model
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
        policy = SmallNetPolicy(model=model, device=config.device)

        # Read training info
        info_path = rl_output_dir / "training_info.json"
        with open(info_path) as f:
            train_info = json.load(f)
        rl_loss = train_info.get("best_policy_loss", 0.0)

        # 4. Gate: evaluate against baseline
        baseline_policy = registry.load_policy(baseline_name, device=config.device)

        gating_config = GatingConfig(
            score_threshold=config.gating_threshold,
            n_games=config.gating_games,
            max_moves=config.max_moves,
            seed=iter_seed,
        )
        gating_result = should_promote(policy, baseline_policy, gating_config)

        # 5. Register if passed
        snapshot_name: str | None = None
        promoted = False

        if gating_result.passed:
            snapshot_name = f"v{iteration + 1:04d}"
            final_model_path = models_dir / f"{snapshot_name}.pt"
            torch.save(model.state_dict(), final_model_path)

            registry.register(
                snapshot_name,
                final_model_path,
                metadata={
                    "iteration": iteration,
                    "gating_score": gating_result.score_rate,
                    "gating_games": gating_result.games_played,
                    "rl_loss": rl_loss,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            baseline_name = snapshot_name
            promoted = True
            logger.info(f"Promoted new baseline: {snapshot_name}")
        else:
            # Revert to baseline
            model.load_state_dict(
                torch.load(
                    registry.get_artifact_path(baseline_name),
                    map_location=device,
                    weights_only=True,
                )
            )
            policy = SmallNetPolicy(model=model, device=config.device)
            logger.info("Reverted to baseline (gating failed)")

        # Record result
        result = IterationResult(
            iteration=iteration,
            games_generated=len(games),
            rl_loss=rl_loss,
            gating_passed=gating_result.passed,
            gating_score=gating_result.score_rate,
            promoted=promoted,
            snapshot_name=snapshot_name,
        )
        results.append(result)

        # Log progress
        n_promoted = sum(1 for r in results if r.promoted)
        logger.info(
            f"Progress: {n_promoted}/{iteration + 1} iterations promoted, "
            f"current baseline: {baseline_name}"
        )

    # Save final results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "total_iterations": len(results),
                "total_promoted": sum(1 for r in results if r.promoted),
                "final_baseline": baseline_name,
                "iterations": [
                    {
                        "iteration": r.iteration,
                        "games_generated": r.games_generated,
                        "rl_loss": r.rl_loss,
                        "gating_passed": r.gating_passed,
                        "gating_score": r.gating_score,
                        "promoted": r.promoted,
                        "snapshot_name": r.snapshot_name,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    logger.info(f"\nTraining loop complete. Results saved to {results_path}")
    logger.info(f"Final baseline: {baseline_name}")

    return results


def main() -> None:
    """CLI entrypoint for the training loop."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run RL training loop with self-play and gating")
    parser.add_argument("--init", type=str, default=None, help="Path to initial model weights")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--games-per-iter", type=int, default=200, help="Games per iteration")
    parser.add_argument("--n-opponents", type=int, default=3, help="Registry opponents to sample")
    parser.add_argument("--sampler", type=str, default="recent", choices=["uniform", "recent", "latest"], help="Opponent sampler type")
    parser.add_argument("--reward", type=str, default="match_bestmove", choices=["match_bestmove", "cp_delta", "outcome"], help="Reward type")
    parser.add_argument("--no-annotate", action="store_true", help="Disable Stockfish annotation")
    parser.add_argument("--teacher-depth", type=int, default=10, help="Stockfish depth for annotation")
    parser.add_argument("--rl-epochs", type=int, default=3, help="RL epochs per iteration")
    parser.add_argument("--gating-games", type=int, default=50, help="Games for gating evaluation")
    parser.add_argument("--gating-threshold", type=float, default=0.55, help="Gating score threshold")
    parser.add_argument("--registry", type=str, default="runs/registry", help="Model registry path")
    parser.add_argument("--out", type=str, default="runs/loop", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--include-random", action="store_true", default=True, help="Include random opponent")
    parser.add_argument("--include-stockfish", action="store_true", help="Include Stockfish opponent")

    args = parser.parse_args()

    config = TrainingLoopConfig(
        init_model_path=args.init,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        n_opponents=args.n_opponents,
        sampler_type=args.sampler,
        include_random=args.include_random,
        include_stockfish=args.include_stockfish,
        reward_type=RewardType(args.reward),
        annotate_with_teacher=not args.no_annotate,
        teacher_depth=args.teacher_depth,
        rl_epochs=args.rl_epochs,
        gating_games=args.gating_games,
        gating_threshold=args.gating_threshold,
        registry_path=args.registry,
        output_dir=args.out,
        device=args.device,
        base_seed=args.seed,
    )

    run_training_loop(config)


if __name__ == "__main__":
    main()

