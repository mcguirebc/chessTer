"""Checkpoint gating: decide whether to promote a new model to the registry."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chesster.league.arena import run_arena

if TYPE_CHECKING:
    from chesster.policies.base import Policy

logger = logging.getLogger(__name__)


@dataclass
class GatingResult:
    """Result of a gating evaluation."""

    passed: bool
    score_rate: float
    win_rate: float
    games_played: int
    threshold: float
    details: str


@dataclass
class GatingConfig:
    """Configuration for checkpoint gating."""

    # Minimum score rate to pass (0.5 = break even, 0.55 = clearly better)
    score_threshold: float = 0.55

    # Number of games to play for evaluation
    n_games: int = 50

    # Maximum moves per game
    max_moves: int = 500

    # Random seed for reproducibility
    seed: int | None = None


def should_promote(
    challenger: Policy,
    baseline: Policy,
    config: GatingConfig | None = None,
) -> GatingResult:
    """
    Decide whether a challenger policy should be promoted over the baseline.

    Args:
        challenger: The new policy being evaluated.
        baseline: The current best policy.
        config: Gating configuration.

    Returns:
        GatingResult indicating whether the challenger passed.
    """
    if config is None:
        config = GatingConfig()

    challenger_id = getattr(challenger, "policy_id", "challenger")
    baseline_id = getattr(baseline, "policy_id", "baseline")

    logger.info(
        f"Gating evaluation: {challenger_id} vs {baseline_id} "
        f"({config.n_games} games, threshold={config.score_threshold:.1%})"
    )

    # Run arena matches
    arena_result = run_arena(
        challenger=challenger,
        opponents=[baseline],
        games_per_opponent=config.n_games,
        max_moves=config.max_moves,
        base_seed=config.seed,
        alternate_colors=True,
    )

    score_rate = arena_result.overall_score_rate
    win_rate = arena_result.overall_win_rate
    games_played = arena_result.total_games

    passed = score_rate >= config.score_threshold

    details = (
        f"{arena_result.total_wins}W / {arena_result.total_draws}D / {arena_result.total_losses}L "
        f"(score={score_rate:.1%}, threshold={config.score_threshold:.1%})"
    )

    if passed:
        logger.info(f"PASSED: {details}")
    else:
        logger.info(f"FAILED: {details}")

    return GatingResult(
        passed=passed,
        score_rate=score_rate,
        win_rate=win_rate,
        games_played=games_played,
        threshold=config.score_threshold,
        details=details,
    )


def gate_against_multiple(
    challenger: Policy,
    baselines: list[Policy],
    config: GatingConfig | None = None,
    require_all: bool = False,
) -> tuple[bool, list[GatingResult]]:
    """
    Gate a challenger against multiple baselines.

    Args:
        challenger: The new policy.
        baselines: List of baseline policies.
        config: Gating configuration (games split across baselines).
        require_all: If True, must beat all baselines. If False, must beat average.

    Returns:
        (passed, list of GatingResult per baseline)
    """
    if config is None:
        config = GatingConfig()

    if not baselines:
        return True, []

    # Split games across baselines
    games_per_baseline = max(1, config.n_games // len(baselines))
    per_baseline_config = GatingConfig(
        score_threshold=config.score_threshold,
        n_games=games_per_baseline,
        max_moves=config.max_moves,
        seed=config.seed,
    )

    results: list[GatingResult] = []
    for baseline in baselines:
        result = should_promote(challenger, baseline, per_baseline_config)
        results.append(result)

    if require_all:
        passed = all(r.passed for r in results)
    else:
        # Pass if average score rate exceeds threshold
        avg_score = sum(r.score_rate for r in results) / len(results)
        passed = avg_score >= config.score_threshold

    return passed, results


class AdaptiveGating:
    """
    Adaptive gating with increasing confidence.

    Runs quick initial check, then longer evaluation if promising.
    """

    def __init__(
        self,
        quick_games: int = 20,
        quick_threshold: float = 0.45,  # Lower threshold for quick check
        full_games: int = 100,
        full_threshold: float = 0.55,
        max_moves: int = 500,
        seed: int | None = None,
    ) -> None:
        self.quick_games = quick_games
        self.quick_threshold = quick_threshold
        self.full_games = full_games
        self.full_threshold = full_threshold
        self.max_moves = max_moves
        self.seed = seed

    def evaluate(self, challenger: Policy, baseline: Policy) -> GatingResult:
        """
        Evaluate challenger with adaptive game count.

        1. Quick check with fewer games and lower threshold.
        2. If quick check passes, run full evaluation.
        """
        # Quick check
        quick_config = GatingConfig(
            score_threshold=self.quick_threshold,
            n_games=self.quick_games,
            max_moves=self.max_moves,
            seed=self.seed,
        )
        quick_result = should_promote(challenger, baseline, quick_config)

        if not quick_result.passed:
            # Failed quick check - definitely not promoting
            logger.info("Failed quick check, skipping full evaluation")
            return GatingResult(
                passed=False,
                score_rate=quick_result.score_rate,
                win_rate=quick_result.win_rate,
                games_played=quick_result.games_played,
                threshold=self.full_threshold,
                details=f"Quick check failed: {quick_result.details}",
            )

        # Quick check passed - run full evaluation
        logger.info("Quick check passed, running full evaluation")
        full_config = GatingConfig(
            score_threshold=self.full_threshold,
            n_games=self.full_games,
            max_moves=self.max_moves,
            seed=(self.seed + 10000) if self.seed is not None else None,
        )
        full_result = should_promote(challenger, baseline, full_config)

        return GatingResult(
            passed=full_result.passed,
            score_rate=full_result.score_rate,
            win_rate=full_result.win_rate,
            games_played=quick_result.games_played + full_result.games_played,
            threshold=self.full_threshold,
            details=f"Full evaluation: {full_result.details}",
        )

