"""ELO rating calculations for chess model evaluation."""
from __future__ import annotations

from dataclasses import dataclass

# Standard ELO constants
DEFAULT_K = 32  # K-factor for rating changes
DEFAULT_ELO = 1200  # Starting rating for new models

# Fixed ratings for known policies
RANDOM_ELO = 800
STOCKFISH_ELO = 2800


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Uses the standard ELO formula:
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))

    Args:
        rating_a: ELO rating of player A.
        rating_b: ELO rating of player B.

    Returns:
        Expected score between 0 and 1.
        0.5 means equal chance, >0.5 means A is favored.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating: float,
    expected: float,
    actual: float,
    k: float = DEFAULT_K,
) -> float:
    """
    Calculate new ELO rating after a game or series.

    Uses the standard update formula:
        R_new = R_old + K * (S - E)

    Args:
        rating: Current ELO rating.
        expected: Expected score (from expected_score function).
        actual: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss).
        k: K-factor controlling rating volatility.

    Returns:
        New ELO rating.
    """
    return rating + k * (actual - expected)


@dataclass
class EloUpdate:
    """Result of an ELO calculation for a match."""

    rating_before: float
    rating_after: float
    expected_score: float
    actual_score: float
    games_played: int

    @property
    def rating_change(self) -> float:
        """Change in rating."""
        return self.rating_after - self.rating_before


def calculate_elo_change(
    rating_a: float,
    rating_b: float,
    wins_a: int,
    draws: int,
    losses_a: int,
    k: float = DEFAULT_K,
) -> tuple[EloUpdate, EloUpdate]:
    """
    Calculate new ratings for both players after a match series.

    Args:
        rating_a: Current ELO of player A.
        rating_b: Current ELO of player B.
        wins_a: Number of games won by player A.
        draws: Number of drawn games.
        losses_a: Number of games lost by player A (won by B).
        k: K-factor for rating volatility.

    Returns:
        Tuple of (EloUpdate for A, EloUpdate for B).
    """
    total_games = wins_a + draws + losses_a
    if total_games == 0:
        # No games played, no change
        return (
            EloUpdate(
                rating_before=rating_a,
                rating_after=rating_a,
                expected_score=0.5,
                actual_score=0.5,
                games_played=0,
            ),
            EloUpdate(
                rating_before=rating_b,
                rating_after=rating_b,
                expected_score=0.5,
                actual_score=0.5,
                games_played=0,
            ),
        )

    # Calculate actual scores (1 for win, 0.5 for draw, 0 for loss)
    actual_a = (wins_a + 0.5 * draws) / total_games
    actual_b = (losses_a + 0.5 * draws) / total_games

    # Calculate expected scores
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)

    # Scale K by number of games (more games = more confident update)
    scaled_k = k * total_games

    # Calculate new ratings
    new_rating_a = update_elo(rating_a, expected_a, actual_a, scaled_k)
    new_rating_b = update_elo(rating_b, expected_b, actual_b, scaled_k)

    return (
        EloUpdate(
            rating_before=rating_a,
            rating_after=new_rating_a,
            expected_score=expected_a,
            actual_score=actual_a,
            games_played=total_games,
        ),
        EloUpdate(
            rating_before=rating_b,
            rating_after=new_rating_b,
            expected_score=expected_b,
            actual_score=actual_b,
            games_played=total_games,
        ),
    )


def get_initial_elo(policy_id: str) -> float:
    """
    Get the initial ELO rating for a policy.

    Args:
        policy_id: The policy identifier.

    Returns:
        Initial ELO rating.
    """
    policy_id_lower = policy_id.lower()
    if policy_id_lower == "random":
        return RANDOM_ELO
    elif policy_id_lower == "stockfish":
        return STOCKFISH_ELO
    else:
        return DEFAULT_ELO
