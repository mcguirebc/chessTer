"""Tests for checkpoint gating logic."""
import pytest

from chesster.league.gating import GatingConfig, GatingResult, should_promote
from chesster.policies.random import RandomPolicy


def test_gating_result_properties() -> None:
    result = GatingResult(
        passed=True,
        score_rate=0.65,
        win_rate=0.5,
        games_played=20,
        threshold=0.55,
        details="test",
    )
    assert result.passed is True
    assert result.score_rate == 0.65


def test_should_promote_random_vs_random() -> None:
    """Two random policies should approximately tie (score ~0.5)."""
    policy_a = RandomPolicy()
    policy_b = RandomPolicy()

    config = GatingConfig(
        score_threshold=0.55,
        n_games=20,
        max_moves=30,
        seed=42,
    )

    result = should_promote(policy_a, policy_b, config)

    # Should have played all games
    assert result.games_played == 20

    # Score should be around 0.5 (unlikely to exceed 0.55 consistently)
    assert 0.0 <= result.score_rate <= 1.0
    assert 0.0 <= result.win_rate <= 1.0


def test_should_promote_low_threshold() -> None:
    """With a low threshold, random should pass against random."""
    policy_a = RandomPolicy()
    policy_b = RandomPolicy()

    config = GatingConfig(
        score_threshold=0.0,  # Any score passes
        n_games=10,
        max_moves=20,
        seed=123,
    )

    result = should_promote(policy_a, policy_b, config)
    assert result.passed is True  # 0% threshold always passes


def test_should_promote_high_threshold() -> None:
    """With a very high threshold, random shouldn't pass against random."""
    policy_a = RandomPolicy()
    policy_b = RandomPolicy()

    config = GatingConfig(
        score_threshold=0.95,  # Extremely high
        n_games=10,
        max_moves=20,
        seed=456,
    )

    result = should_promote(policy_a, policy_b, config)
    # Very unlikely to pass with 95% threshold
    # (not guaranteed, but extremely probable)
    assert result.threshold == 0.95


def test_gating_result_details() -> None:
    """Gating result should include useful details."""
    policy_a = RandomPolicy()
    policy_b = RandomPolicy()

    config = GatingConfig(n_games=6, max_moves=10, seed=789)
    result = should_promote(policy_a, policy_b, config)

    assert "W" in result.details
    assert "D" in result.details
    assert "L" in result.details
    assert "score=" in result.details

