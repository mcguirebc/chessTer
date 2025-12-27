"""Self-play game generator with opponent sampling and Stockfish annotation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chesster.policies.base import ChooseMoveParams
from chesster.selfplay.record import GameRecord, save_games
from chesster.selfplay.runner import play_game

if TYPE_CHECKING:
    from chesster.policies.base import Policy

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for self-play generation."""

    n_games: int = 100
    max_moves: int = 500
    base_seed: int | None = None
    alternate_colors: bool = True

    # Teacher annotation
    annotate_with_teacher: bool = False
    teacher_depth: int = 10

    # Output
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_games": self.n_games,
            "max_moves": self.max_moves,
            "base_seed": self.base_seed,
            "alternate_colors": self.alternate_colors,
            "annotate_with_teacher": self.annotate_with_teacher,
            "teacher_depth": self.teacher_depth,
            "output_path": self.output_path,
        }


def generate_selfplay_games(
    policy: Policy,
    opponents: list[Policy],
    config: GeneratorConfig,
    teacher: Policy | None = None,
) -> list[GameRecord]:
    """
    Generate self-play games between a policy and opponents.

    Args:
        policy: The main policy to train.
        opponents: List of opponent policies.
        config: Generation configuration.
        teacher: Optional teacher policy for move annotation.

    Returns:
        List of GameRecord objects.
    """
    if not opponents:
        raise ValueError("At least one opponent is required")

    games: list[GameRecord] = []
    games_per_opponent = config.n_games // len(opponents)
    remainder = config.n_games % len(opponents)

    seed_offset = 0

    for i, opponent in enumerate(opponents):
        n = games_per_opponent + (1 if i < remainder else 0)
        opponent_id = getattr(opponent, "policy_id", f"opponent_{i}")

        logger.info(f"Generating {n} games vs {opponent_id}")

        for j in range(n):
            game_seed = (config.base_seed + seed_offset) if config.base_seed is not None else None
            seed_offset += 1000

            # Alternate colors
            if config.alternate_colors and j % 2 == 1:
                white, black = opponent, policy
            else:
                white, black = policy, opponent

            # Determine teacher for annotation
            annotation_teacher = teacher if config.annotate_with_teacher else None

            game = play_game(
                white,
                black,
                max_moves=config.max_moves,
                teacher=annotation_teacher,
                teacher_depth=config.teacher_depth,
                seed=game_seed,
            )
            games.append(game)

    # Save if output path specified
    if config.output_path is not None:
        path = Path(config.output_path)
        save_games(path, games)
        logger.info(f"Saved {len(games)} games to {path}")

    return games


def generate_with_registry_opponents(
    policy: Policy,
    registry_path: str,
    opponent_names: list[str] | None = None,
    config: GeneratorConfig | None = None,
    teacher: Policy | None = None,
) -> list[GameRecord]:
    """
    Generate games using opponents from a model registry.

    Args:
        policy: The main policy.
        registry_path: Path to the model registry.
        opponent_names: Specific opponent names, or None for all.
        config: Generation configuration.
        teacher: Optional teacher for annotation.

    Returns:
        List of GameRecord objects.
    """
    from chesster.league.registry import ModelRegistry

    if config is None:
        config = GeneratorConfig()

    registry = ModelRegistry(registry_path)
    snapshots = registry.list_snapshots()

    if opponent_names is not None:
        # Load specific opponents
        opponents = []
        for name in opponent_names:
            try:
                opp = registry.load_policy(name)
                opponents.append(opp)
            except KeyError:
                logger.warning(f"Opponent {name!r} not found in registry, skipping")
    else:
        # Load all opponents
        opponents = [registry.load_policy(s.name) for s in snapshots]

    if not opponents:
        raise ValueError("No opponents available from registry")

    return generate_selfplay_games(policy, opponents, config, teacher)


def main() -> None:
    """CLI entrypoint for self-play generation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate self-play games")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy model weights")
    parser.add_argument("--opponents", type=str, required=True, help="Comma-separated opponent types (random,stockfish) or registry:<path>")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves per game")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    parser.add_argument("--annotate", action="store_true", help="Annotate moves with Stockfish")
    parser.add_argument("--teacher-depth", type=int, default=10, help="Stockfish depth for annotation")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for policy inference")

    args = parser.parse_args()

    # Load the policy
    from chesster.policies.net import SmallNetPolicy

    policy = SmallNetPolicy(device=args.device)
    policy.load(args.policy)
    logger.info(f"Loaded policy from {args.policy}")

    # Parse opponents
    opponents: list[Policy] = []

    if args.opponents.startswith("registry:"):
        # Load from registry
        registry_path = args.opponents[len("registry:"):]
        from chesster.league.registry import ModelRegistry

        registry = ModelRegistry(registry_path)
        for snapshot in registry.list_snapshots():
            opp = registry.load_policy(snapshot.name, device=args.device)
            opponents.append(opp)
        logger.info(f"Loaded {len(opponents)} opponents from registry")
    else:
        # Parse comma-separated opponent types
        for opp_type in args.opponents.split(","):
            opp_type = opp_type.strip().lower()
            if opp_type == "random":
                from chesster.policies.random import RandomPolicy
                opponents.append(RandomPolicy())
            elif opp_type == "stockfish":
                from chesster.policies.stockfish import StockfishPolicy
                opponents.append(StockfishPolicy())
            else:
                logger.warning(f"Unknown opponent type: {opp_type}")

    if not opponents:
        raise ValueError("No valid opponents specified")

    # Teacher for annotation
    teacher: Policy | None = None
    if args.annotate:
        from chesster.policies.stockfish import StockfishPolicy
        teacher = StockfishPolicy(default_depth=args.teacher_depth)
        logger.info(f"Using Stockfish teacher for annotation (depth={args.teacher_depth})")

    # Generate games
    config = GeneratorConfig(
        n_games=args.games,
        max_moves=args.max_moves,
        base_seed=args.seed,
        annotate_with_teacher=args.annotate,
        teacher_depth=args.teacher_depth,
        output_path=args.out,
    )

    games = generate_selfplay_games(policy, opponents, config, teacher)
    logger.info(f"Generated {len(games)} games")


if __name__ == "__main__":
    main()

