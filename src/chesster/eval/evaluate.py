"""Main evaluation engine with CLI.

Evaluates any policy against configurable baselines and reports results.

CLI Usage:
    python -m chesster.eval.evaluate --model hf:Qwen/Qwen2.5-1.5B-Instruct --suite standard
    python -m chesster.eval.evaluate --model pt:runs/bc/v1/best_model.pt --suite quick
    python -m chesster.eval.evaluate --model reg:init --opponents stockfish:1,stockfish:5 --games 20

Example:
    >>> from chesster.eval import evaluate_model
    >>> result = evaluate_model("hf:Qwen/Qwen2.5-1.5B-Instruct", suite="quick")
    >>> print(f"ELO: {result.estimated_elo}")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chesster.eval.baselines import get_baseline_suite, get_reference_elo, list_suites
from chesster.eval.loader import PolicyLoadError, load_policy, parse_opponent_specs
from chesster.league.arena import MatchResult, run_arena
from chesster.league.elo import DEFAULT_ELO, calculate_elo_change

if TYPE_CHECKING:
    from chesster.policies.base import Policy

logger = logging.getLogger(__name__)


@dataclass
class OpponentResult:
    """Result against a single opponent."""

    opponent_id: str
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float  # (wins + 0.5*draws) / total
    win_rate: float  # wins / total
    reference_elo: float  # Known ELO of opponent

    @classmethod
    def from_match_result(cls, match: MatchResult) -> OpponentResult:
        """Create from arena MatchResult."""
        total = match.total
        score_rate = match.score_rate if total > 0 else 0.0
        win_rate = match.win_rate if total > 0 else 0.0
        ref_elo = get_reference_elo(match.opponent_id)

        return cls(
            opponent_id=match.opponent_id,
            games=total,
            wins=match.wins,
            draws=match.draws,
            losses=match.losses,
            score_rate=score_rate,
            win_rate=win_rate,
            reference_elo=ref_elo,
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    model_id: str
    suite_name: str | None
    timestamp: str

    # Per-opponent results
    opponents: dict[str, OpponentResult] = field(default_factory=dict)

    # Aggregate metrics
    total_games: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0

    # Key metrics
    score_rate: float = 0.0  # Overall score rate
    win_rate: float = 0.0  # Overall win rate
    estimated_elo: float = DEFAULT_ELO

    # Specific opponent scores (for easy access)
    vs_random_score: float | None = None
    vs_stockfish_1_score: float | None = None
    vs_stockfish_3_score: float | None = None
    vs_stockfish_5_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert OpponentResult to dicts
        result["opponents"] = {k: asdict(v) for k, v in self.opponents.items()}
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationResult:
        """Create from dictionary."""
        d = d.copy()
        opponents = {k: OpponentResult(**v) for k, v in d.pop("opponents", {}).items()}
        return cls(opponents=opponents, **d)


def evaluate_model(
    model: str | Policy,
    *,
    suite: str | None = None,
    opponents: list[str | Policy] | None = None,
    games_per_opponent: int | None = None,
    registry_path: str | Path | None = None,
    device: str | None = None,
    base_seed: int | None = 42,
    register_as: str | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate a model against baselines.

    Args:
        model: Policy spec string or Policy instance.
        suite: Baseline suite name ("quick", "standard", "full").
        opponents: Custom list of opponent specs/policies (overrides suite).
        games_per_opponent: Games per opponent (overrides suite default).
        registry_path: Path to model registry.
        device: Device for model loading.
        base_seed: Base seed for reproducibility.
        register_as: If provided, register model in registry with this name.
        verbose: Print progress during evaluation.

    Returns:
        EvaluationResult with all metrics.

    Example:
        >>> result = evaluate_model("hf:Qwen/Qwen2.5-1.5B-Instruct", suite="quick")
        >>> print(f"ELO: {result.estimated_elo:.0f}")
    """
    # Load model policy
    if isinstance(model, str):
        if verbose:
            print(f"Loading model: {model}")
        policy = load_policy(model, registry_path=registry_path, device=device)
    else:
        policy = model

    model_id = getattr(policy, "policy_id", "unknown")

    # Determine opponents
    opponent_policies: list[Policy] = []
    suite_name: str | None = None

    if opponents is not None:
        # Custom opponents
        for opp in opponents:
            if isinstance(opp, str):
                opponent_policies.append(load_policy(opp, registry_path=registry_path))
            else:
                opponent_policies.append(opp)
        games = games_per_opponent or 10
    elif suite is not None:
        # Use baseline suite
        baseline_suite = get_baseline_suite(suite)
        suite_name = baseline_suite.name
        opponent_policies = baseline_suite.load_opponents()
        games = games_per_opponent or baseline_suite.games_per_opponent
        if verbose:
            print(f"Using suite: {suite_name} ({len(opponent_policies)} opponents)")
    else:
        # Default to standard suite
        baseline_suite = get_baseline_suite("standard")
        suite_name = baseline_suite.name
        opponent_policies = baseline_suite.load_opponents()
        games = games_per_opponent or baseline_suite.games_per_opponent
        if verbose:
            print(f"Using default suite: {suite_name}")

    total_games = len(opponent_policies) * games
    if verbose:
        print(f"Running {total_games} games ({games} per opponent)...")

    # Run arena
    try:
        arena_result = run_arena(
            challenger=policy,
            opponents=opponent_policies,
            games_per_opponent=games,
            base_seed=base_seed,
            alternate_colors=True,
        )
    finally:
        # Clean up policies (especially Stockfish)
        for p in [policy] + opponent_policies:
            if hasattr(p, "close") and callable(p.close):
                try:
                    p.close()
                except Exception as e:
                    logger.warning(f"Failed to close policy {getattr(p, 'policy_id', 'unknown')}: {e}")

    # Build evaluation result
    timestamp = datetime.now(timezone.utc).isoformat()
    result = EvaluationResult(
        model_id=model_id,
        suite_name=suite_name,
        timestamp=timestamp,
    )

    # Process opponent results
    for opp_id, match in arena_result.match_results.items():
        opp_result = OpponentResult.from_match_result(match)
        result.opponents[opp_id] = opp_result

        # Store specific opponent scores
        if opp_id == "random":
            result.vs_random_score = opp_result.score_rate
        elif opp_id == "stockfish:1":
            result.vs_stockfish_1_score = opp_result.score_rate
        elif opp_id == "stockfish:3":
            result.vs_stockfish_3_score = opp_result.score_rate
        elif opp_id == "stockfish:5":
            result.vs_stockfish_5_score = opp_result.score_rate

    # Aggregate metrics
    result.total_games = arena_result.total_games
    result.total_wins = arena_result.total_wins
    result.total_draws = arena_result.total_draws
    result.total_losses = arena_result.total_losses
    result.score_rate = arena_result.overall_score_rate
    result.win_rate = arena_result.overall_win_rate

    # Estimate ELO from results
    result.estimated_elo = _estimate_elo(result)

    # Optionally register in model registry
    if register_as and registry_path:
        _register_result(result, register_as, registry_path)
        if verbose:
            print(f"Registered as '{register_as}' in model registry")

    return result


def _estimate_elo(result: EvaluationResult) -> float:
    """
    Estimate model ELO from evaluation results.

    Uses performance rating formula based on opponents' reference ELOs.
    """
    if not result.opponents:
        return DEFAULT_ELO

    # Weighted average based on games and expected vs actual performance
    total_weight = 0.0
    elo_sum = 0.0

    for opp_id, opp_result in result.opponents.items():
        if opp_result.games == 0:
            continue

        ref_elo = opp_result.reference_elo
        score = opp_result.score_rate

        # Performance rating: opponent_elo + 400 * (score - 0.5) / 0.5
        # Simplified: if score=1.0 -> +400, score=0.5 -> +0, score=0 -> -400
        if score >= 1.0:
            perf_elo = ref_elo + 400
        elif score <= 0.0:
            perf_elo = ref_elo - 400
        else:
            # Linear interpolation
            perf_elo = ref_elo + 800 * (score - 0.5)

        weight = opp_result.games
        elo_sum += perf_elo * weight
        total_weight += weight

    if total_weight == 0:
        return DEFAULT_ELO

    return elo_sum / total_weight


def _register_result(
    result: EvaluationResult,
    name: str,
    registry_path: str | Path,
) -> None:
    """Register evaluation result in model registry."""
    from chesster.league.registry import ModelRegistry

    registry = ModelRegistry(registry_path)

    # Update ELO for existing snapshot or create metadata
    if name in registry.list_snapshots():
        # Update existing snapshot's ELO
        registry.update_elo(
            name=name,
            new_elo=result.estimated_elo,
            opponent="evaluation",
            games_played=result.total_games,
            result=f"{result.total_wins}W-{result.total_draws}D-{result.total_losses}L",
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a chess policy against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate HuggingFace model with standard suite
  python -m chesster.eval.evaluate --model hf:Qwen/Qwen2.5-1.5B-Instruct --suite standard

  # Quick evaluation of PyTorch checkpoint
  python -m chesster.eval.evaluate --model pt:runs/bc/v1/best_model.pt --suite quick

  # Custom opponents
  python -m chesster.eval.evaluate --model reg:init --opponents stockfish:1,stockfish:5 --games 20

  # Full evaluation with JSON output
  python -m chesster.eval.evaluate --model hf:Qwen/Qwen2.5-1.5B-Instruct --suite full --output results.json
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model spec: hf:name, pt:path, reg:name, stockfish:depth, or random",
    )
    parser.add_argument(
        "--suite",
        choices=list_suites(),
        default="standard",
        help="Baseline suite (default: standard)",
    )
    parser.add_argument(
        "--opponents",
        help="Custom opponents (comma-separated specs), overrides --suite",
    )
    parser.add_argument(
        "--games",
        type=int,
        help="Games per opponent (overrides suite default)",
    )
    parser.add_argument(
        "--registry",
        default="runs/registry",
        help="Path to model registry (default: runs/registry)",
    )
    parser.add_argument(
        "--device",
        help="Device for model (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--register",
        metavar="NAME",
        help="Register model in registry with this name",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write JSON results to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    try:
        # Parse custom opponents if provided
        opponents = None
        suite = args.suite
        if args.opponents:
            opponents = parse_opponent_specs(args.opponents)
            suite = None  # Custom opponents override suite

        # Run evaluation
        result = evaluate_model(
            model=args.model,
            suite=suite,
            opponents=opponents,
            games_per_opponent=args.games,
            registry_path=args.registry,
            device=args.device,
            base_seed=args.seed,
            register_as=args.register,
            verbose=not args.quiet,
        )

        # Print report
        if not args.quiet:
            from chesster.eval.report import format_report

            print(format_report(result))

        # Write JSON output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            if not args.quiet:
                print(f"\nResults written to: {output_path}")

        return 0

    except PolicyLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
