"""Baseline suite definitions for evaluation.

Provides pre-defined sets of opponents for consistent evaluation:
- quick: Fast sanity check (2 opponents, ~2 min)
- standard: Default evaluation (4 opponents, ~10 min)
- full: Comprehensive (5+ opponents, ~30 min)

Example:
    >>> from chesster.eval.baselines import get_baseline_suite, SUITES
    >>> opponents = get_baseline_suite("standard")
    >>> print([o.policy_id for o in opponents])
    ['random', 'stockfish:1', 'stockfish:3', 'stockfish:5']
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chesster.eval.loader import load_policy

if TYPE_CHECKING:
    from chesster.policies.base import Policy


@dataclass
class BaselineSuite:
    """Definition of a baseline evaluation suite."""

    name: str
    description: str
    opponent_specs: list[str]
    games_per_opponent: int = 10
    estimated_time_min: int = 10

    def load_opponents(self) -> list[Policy]:
        """Load all opponent policies for this suite."""
        return [load_policy(spec) for spec in self.opponent_specs]


# Pre-defined suites
QUICK_SUITE = BaselineSuite(
    name="quick",
    description="Fast sanity check - Random and Stockfish depth 1",
    opponent_specs=["random", "stockfish:1"],
    games_per_opponent=5,
    estimated_time_min=2,
)

STANDARD_SUITE = BaselineSuite(
    name="standard",
    description="Standard evaluation - Random and Stockfish depths 1, 3, 5",
    opponent_specs=["random", "stockfish:1", "stockfish:3", "stockfish:5"],
    games_per_opponent=10,
    estimated_time_min=10,
)

FULL_SUITE = BaselineSuite(
    name="full",
    description="Comprehensive - Random and Stockfish depths 1, 3, 5, 10",
    opponent_specs=["random", "stockfish:1", "stockfish:3", "stockfish:5", "stockfish:10"],
    games_per_opponent=20,
    estimated_time_min=30,
)


# Registry of all suites
SUITES: dict[str, BaselineSuite] = {
    "quick": QUICK_SUITE,
    "standard": STANDARD_SUITE,
    "full": FULL_SUITE,
}


def get_baseline_suite(name: str) -> BaselineSuite:
    """
    Get a baseline suite by name.

    Args:
        name: Suite name ("quick", "standard", or "full").

    Returns:
        BaselineSuite instance.

    Raises:
        ValueError: If suite name is not recognized.
    """
    if name not in SUITES:
        available = ", ".join(SUITES.keys())
        raise ValueError(f"Unknown suite: {name!r}. Available: {available}")
    return SUITES[name]


def list_suites() -> list[str]:
    """Return list of available suite names."""
    return list(SUITES.keys())


@dataclass
class EloReference:
    """Reference ELO ratings for calibration."""

    random: float = 800.0
    stockfish_1: float = 1200.0
    stockfish_3: float = 1600.0
    stockfish_5: float = 1900.0
    stockfish_10: float = 2200.0
    stockfish_20: float = 2800.0


# Default ELO reference values
DEFAULT_ELO_REFERENCE = EloReference()


def get_reference_elo(opponent_id: str) -> float:
    """
    Get reference ELO for a known opponent.

    Args:
        opponent_id: Policy ID like "random" or "stockfish:5".

    Returns:
        Reference ELO rating, or 1200 if unknown.
    """
    ref = DEFAULT_ELO_REFERENCE

    if opponent_id == "random":
        return ref.random
    elif opponent_id.startswith("stockfish:"):
        try:
            depth = int(opponent_id.split(":")[1])
            if depth <= 1:
                return ref.stockfish_1
            elif depth <= 3:
                return ref.stockfish_3
            elif depth <= 5:
                return ref.stockfish_5
            elif depth <= 10:
                return ref.stockfish_10
            else:
                return ref.stockfish_20
        except (ValueError, IndexError):
            return 1500.0  # Default Stockfish
    elif opponent_id == "stockfish":
        return ref.stockfish_10  # Default depth is 10

    # Unknown opponent
    return 1200.0
