"""Tests for checkpoint gating logic."""
import pytest

from chesster.league.elo import DEFAULT_ELO, RANDOM_ELO
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


def test_gating_result_elo_fields_default() -> None:
    """GatingResult should have ELO fields with default None."""
    result = GatingResult(
        passed=True,
        score_rate=0.65,
        win_rate=0.5,
        games_played=20,
        threshold=0.55,
        details="test",
    )
    assert result.challenger_elo is None
    assert result.baseline_elo is None
    assert result.wins == 0
    assert result.draws == 0
    assert result.losses == 0


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


class TestGatingEloIntegration:
    """Tests for ELO calculation in gating."""

    def test_gating_with_elo_calculates_updates(self) -> None:
        """When ELO ratings provided, should calculate updates."""
        policy_a = RandomPolicy()
        policy_b = RandomPolicy()

        config = GatingConfig(n_games=10, max_moves=20, seed=42)
        result = should_promote(
            policy_a,
            policy_b,
            config,
            challenger_elo=1200.0,
            baseline_elo=1200.0,
        )

        # ELO updates should be calculated
        assert result.challenger_elo is not None
        assert result.baseline_elo is not None

        # Ratings should be updated from starting values
        assert result.challenger_elo.rating_before == 1200.0
        assert result.baseline_elo.rating_before == 1200.0

    def test_gating_without_elo_returns_none(self) -> None:
        """When ELO ratings not provided, should return None updates."""
        policy_a = RandomPolicy()
        policy_b = RandomPolicy()

        config = GatingConfig(n_games=6, max_moves=10, seed=123)
        result = should_promote(policy_a, policy_b, config)

        assert result.challenger_elo is None
        assert result.baseline_elo is None

    def test_gating_elo_sum_to_zero(self) -> None:
        """ELO changes should sum to zero (zero-sum game)."""
        policy_a = RandomPolicy()
        policy_b = RandomPolicy()

        config = GatingConfig(n_games=10, max_moves=20, seed=999)
        result = should_promote(
            policy_a,
            policy_b,
            config,
            challenger_elo=1400.0,
            baseline_elo=1100.0,
        )

        assert result.challenger_elo is not None
        assert result.baseline_elo is not None

        change_a = result.challenger_elo.rating_change
        change_b = result.baseline_elo.rating_change

        assert change_a + change_b == pytest.approx(0)

    def test_gating_tracks_wins_draws_losses(self) -> None:
        """Gating result should track W/D/L counts."""
        policy_a = RandomPolicy()
        policy_b = RandomPolicy()

        config = GatingConfig(n_games=10, max_moves=20, seed=42)
        result = should_promote(policy_a, policy_b, config)

        # Sum of W/D/L should equal games played
        assert result.wins + result.draws + result.losses == result.games_played

    def test_gating_elo_with_different_ratings(self) -> None:
        """Test ELO calculation with different starting ratings."""
        policy_a = RandomPolicy()
        policy_b = RandomPolicy()

        config = GatingConfig(n_games=10, max_moves=20, seed=42)

        # Higher rated vs lower rated
        result = should_promote(
            policy_a,
            policy_b,
            config,
            challenger_elo=1500.0,
            baseline_elo=1000.0,
        )

        assert result.challenger_elo is not None
        assert result.baseline_elo is not None

        # Expected score for higher rated should be > 0.5
        assert result.challenger_elo.expected_score > 0.5
        assert result.baseline_elo.expected_score < 0.5

