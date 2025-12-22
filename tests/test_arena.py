"""Tests for Arena evaluation."""
import pytest

from chesster.league.arena import ArenaResult, MatchResult, run_arena, run_match
from chesster.policies.random import RandomPolicy


def test_match_result_properties() -> None:
    result = MatchResult(opponent_id="test", wins=3, draws=2, losses=1)

    assert result.total == 6
    assert result.win_rate == 0.5  # 3/6
    assert result.score == 4.0  # 3 + 0.5*2
    assert result.score_rate == pytest.approx(4.0 / 6)


def test_arena_result_aggregation() -> None:
    arena = ArenaResult(challenger_id="challenger")
    arena.match_results["opp1"] = MatchResult(opponent_id="opp1", wins=2, draws=1, losses=0)
    arena.match_results["opp2"] = MatchResult(opponent_id="opp2", wins=1, draws=0, losses=2)

    assert arena.total_wins == 3
    assert arena.total_draws == 1
    assert arena.total_losses == 2
    assert arena.total_games == 6


def test_run_match_basic() -> None:
    policy_a = RandomPolicy()
    policy_b = RandomPolicy()

    a_result, b_result = run_match(
        policy_a,
        policy_b,
        n_games=4,
        max_moves=30,
        base_seed=42,
    )

    # Results should be complementary
    assert a_result.wins + a_result.draws + a_result.losses == 4
    assert b_result.wins + b_result.draws + b_result.losses == 4
    assert a_result.wins == b_result.losses
    assert a_result.losses == b_result.wins
    assert a_result.draws == b_result.draws


def test_run_arena_basic() -> None:
    challenger = RandomPolicy()
    opponents = [RandomPolicy(), RandomPolicy()]

    # Temporarily rename policies for testing
    opponents[0].policy_id = "opponent_1"
    opponents[1].policy_id = "opponent_2"

    result = run_arena(
        challenger,
        opponents,
        games_per_opponent=2,
        max_moves=20,
        base_seed=42,
    )

    assert result.challenger_id == "random"
    assert len(result.match_results) == 2
    assert "opponent_1" in result.match_results
    assert "opponent_2" in result.match_results
    assert result.total_games == 4


def test_run_arena_deterministic() -> None:
    challenger = RandomPolicy()
    opponent = RandomPolicy()
    opponent.policy_id = "opponent"

    result1 = run_arena(challenger, [opponent], games_per_opponent=4, max_moves=20, base_seed=123)
    result2 = run_arena(challenger, [opponent], games_per_opponent=4, max_moves=20, base_seed=123)

    # Same seed should produce same results
    assert result1.total_wins == result2.total_wins
    assert result1.total_draws == result2.total_draws
    assert result1.total_losses == result2.total_losses


def test_arena_result_summary() -> None:
    arena = ArenaResult(challenger_id="test_challenger")
    arena.match_results["opp"] = MatchResult(opponent_id="opp", wins=5, draws=3, losses=2)

    summary = arena.summary()

    assert "test_challenger" in summary
    assert "5W" in summary
    assert "3D" in summary
    assert "2L" in summary
    assert "opp" in summary

