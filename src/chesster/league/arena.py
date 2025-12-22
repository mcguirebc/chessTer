"""Arena: run matches between policies and report statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chesster.selfplay.runner import play_game

if TYPE_CHECKING:
    from chesster.policies.base import Policy
    from chesster.selfplay.record import GameRecord


@dataclass
class MatchResult:
    """Result of matches against a single opponent."""

    opponent_id: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: list[GameRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins / self.total

    @property
    def score(self) -> float:
        """Score as wins + 0.5 * draws."""
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        """Score per game."""
        if self.total == 0:
            return 0.0
        return self.score / self.total


@dataclass
class ArenaResult:
    """Result of an arena evaluation."""

    challenger_id: str
    match_results: dict[str, MatchResult] = field(default_factory=dict)

    @property
    def total_wins(self) -> int:
        return sum(m.wins for m in self.match_results.values())

    @property
    def total_draws(self) -> int:
        return sum(m.draws for m in self.match_results.values())

    @property
    def total_losses(self) -> int:
        return sum(m.losses for m in self.match_results.values())

    @property
    def total_games(self) -> int:
        return sum(m.total for m in self.match_results.values())

    @property
    def overall_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games

    @property
    def overall_score_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.total_wins + 0.5 * self.total_draws) / self.total_games

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Arena results for {self.challenger_id}:",
            f"  Total: {self.total_wins}W / {self.total_draws}D / {self.total_losses}L "
            f"({self.total_games} games, {self.overall_score_rate:.1%} score)",
            "",
        ]
        for opp_id, match in sorted(self.match_results.items()):
            lines.append(
                f"  vs {opp_id}: {match.wins}W / {match.draws}D / {match.losses}L "
                f"({match.score_rate:.1%} score)"
            )
        return "\n".join(lines)


def run_arena(
    challenger: Policy,
    opponents: list[Policy],
    games_per_opponent: int,
    *,
    max_moves: int = 500,
    base_seed: int | None = None,
    alternate_colors: bool = True,
) -> ArenaResult:
    """
    Run arena matches between a challenger and multiple opponents.

    Args:
        challenger: The policy being evaluated.
        opponents: List of opponent policies.
        games_per_opponent: Number of games to play against each opponent.
        max_moves: Maximum ply per game.
        base_seed: Base seed for reproducibility.
        alternate_colors: If True, alternate colors between games.

    Returns:
        ArenaResult with per-opponent and aggregate statistics.
    """
    challenger_id = getattr(challenger, "policy_id", "challenger")
    result = ArenaResult(challenger_id=challenger_id)

    seed_offset = 0

    for opponent in opponents:
        opponent_id = getattr(opponent, "policy_id", "opponent")
        match = MatchResult(opponent_id=opponent_id)

        for i in range(games_per_opponent):
            game_seed = (base_seed + seed_offset) if base_seed is not None else None
            seed_offset += 1000

            # Alternate colors
            if alternate_colors and i % 2 == 1:
                white, black = opponent, challenger
                challenger_is_white = False
            else:
                white, black = challenger, opponent
                challenger_is_white = True

            game = play_game(
                white,
                black,
                max_moves=max_moves,
                seed=game_seed,
            )
            match.games.append(game)

            # Determine result from challenger's perspective
            if game.result == "1-0":
                if challenger_is_white:
                    match.wins += 1
                else:
                    match.losses += 1
            elif game.result == "0-1":
                if challenger_is_white:
                    match.losses += 1
                else:
                    match.wins += 1
            else:
                # "1/2-1/2" or "*" (unfinished) count as draws
                match.draws += 1

        result.match_results[opponent_id] = match

    return result


def run_match(
    policy_a: Policy,
    policy_b: Policy,
    n_games: int,
    *,
    max_moves: int = 500,
    base_seed: int | None = None,
    alternate_colors: bool = True,
) -> tuple[MatchResult, MatchResult]:
    """
    Run a head-to-head match between two policies.

    Returns results from both perspectives: (a_vs_b, b_vs_a).
    """
    a_id = getattr(policy_a, "policy_id", "policy_a")
    b_id = getattr(policy_b, "policy_id", "policy_b")

    a_result = MatchResult(opponent_id=b_id)
    b_result = MatchResult(opponent_id=a_id)

    for i in range(n_games):
        game_seed = (base_seed + i * 1000) if base_seed is not None else None

        # Alternate colors
        if alternate_colors and i % 2 == 1:
            white, black = policy_b, policy_a
            a_is_white = False
        else:
            white, black = policy_a, policy_b
            a_is_white = True

        game = play_game(white, black, max_moves=max_moves, seed=game_seed)
        a_result.games.append(game)
        b_result.games.append(game)

        # Update results
        if game.result == "1-0":
            if a_is_white:
                a_result.wins += 1
                b_result.losses += 1
            else:
                a_result.losses += 1
                b_result.wins += 1
        elif game.result == "0-1":
            if a_is_white:
                a_result.losses += 1
                b_result.wins += 1
            else:
                a_result.wins += 1
                b_result.losses += 1
        else:
            # "1/2-1/2" or "*" (unfinished) count as draws
            a_result.draws += 1
            b_result.draws += 1

    return a_result, b_result

