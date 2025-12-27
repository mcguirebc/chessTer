"""Tests for ELO rating calculations."""
import pytest

from chesster.league.elo import (
    DEFAULT_ELO,
    DEFAULT_K,
    RANDOM_ELO,
    STOCKFISH_ELO,
    EloUpdate,
    calculate_elo_change,
    expected_score,
    get_initial_elo,
    update_elo,
)


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings_gives_half(self):
        """Two players with equal ratings should have 0.5 expected score."""
        result = expected_score(1200, 1200)
        assert result == pytest.approx(0.5)

    def test_higher_rated_favored(self):
        """Higher rated player should have >0.5 expected score."""
        result = expected_score(1400, 1200)
        assert result > 0.5
        assert result < 1.0

    def test_lower_rated_unfavored(self):
        """Lower rated player should have <0.5 expected score."""
        result = expected_score(1200, 1400)
        assert result < 0.5
        assert result > 0.0

    def test_symmetric(self):
        """Expected scores should sum to 1."""
        e_a = expected_score(1200, 1400)
        e_b = expected_score(1400, 1200)
        assert e_a + e_b == pytest.approx(1.0)

    def test_400_point_difference(self):
        """400 point difference should give ~0.91 expected score for higher rated."""
        # Standard ELO: 400 points = 10x expected score ratio
        result = expected_score(1600, 1200)
        assert result == pytest.approx(0.909, abs=0.01)

    def test_large_difference(self):
        """Very large rating difference should approach 1.0."""
        result = expected_score(2800, 800)
        assert result > 0.99


class TestUpdateElo:
    """Tests for update_elo function."""

    def test_win_increases_rating(self):
        """Winning should increase rating."""
        expected = expected_score(1200, 1200)  # 0.5
        new_rating = update_elo(1200, expected, 1.0)  # actual = 1.0 (win)
        assert new_rating > 1200

    def test_loss_decreases_rating(self):
        """Losing should decrease rating."""
        expected = expected_score(1200, 1200)  # 0.5
        new_rating = update_elo(1200, expected, 0.0)  # actual = 0.0 (loss)
        assert new_rating < 1200

    def test_draw_against_equal_no_change(self):
        """Draw against equal opponent should not change rating."""
        expected = expected_score(1200, 1200)  # 0.5
        new_rating = update_elo(1200, expected, 0.5)  # actual = 0.5 (draw)
        assert new_rating == pytest.approx(1200)

    def test_draw_against_higher_rated_increases(self):
        """Draw against higher rated opponent should increase rating."""
        expected = expected_score(1200, 1400)  # < 0.5
        new_rating = update_elo(1200, expected, 0.5)
        assert new_rating > 1200

    def test_draw_against_lower_rated_decreases(self):
        """Draw against lower rated opponent should decrease rating."""
        expected = expected_score(1400, 1200)  # > 0.5
        new_rating = update_elo(1400, expected, 0.5)
        assert new_rating < 1400

    def test_k_factor_scales_change(self):
        """Higher K factor should produce larger rating changes."""
        expected = expected_score(1200, 1200)
        new_k16 = update_elo(1200, expected, 1.0, k=16)
        new_k32 = update_elo(1200, expected, 1.0, k=32)
        new_k64 = update_elo(1200, expected, 1.0, k=64)

        assert new_k16 < new_k32 < new_k64

    def test_expected_win_gives_k_points(self):
        """Win against equal opponent gives K/2 points (since expected=0.5)."""
        new_rating = update_elo(1200, 0.5, 1.0, k=DEFAULT_K)
        change = new_rating - 1200
        assert change == pytest.approx(DEFAULT_K * 0.5)  # K * (1.0 - 0.5)


class TestCalculateEloChange:
    """Tests for calculate_elo_change function."""

    def test_no_games_no_change(self):
        """No games played should result in no rating change."""
        update_a, update_b = calculate_elo_change(1200, 1200, wins_a=0, draws=0, losses_a=0)

        assert update_a.rating_after == 1200
        assert update_b.rating_after == 1200
        assert update_a.games_played == 0
        assert update_b.games_played == 0

    def test_single_win(self):
        """Single win by A should increase A's rating and decrease B's."""
        update_a, update_b = calculate_elo_change(1200, 1200, wins_a=1, draws=0, losses_a=0)

        assert update_a.rating_after > 1200
        assert update_b.rating_after < 1200
        assert update_a.games_played == 1
        assert update_b.games_played == 1

    def test_single_loss(self):
        """Single loss by A should decrease A's rating and increase B's."""
        update_a, update_b = calculate_elo_change(1200, 1200, wins_a=0, draws=0, losses_a=1)

        assert update_a.rating_after < 1200
        assert update_b.rating_after > 1200

    def test_all_draws_equal_ratings(self):
        """All draws between equal players should not change ratings."""
        update_a, update_b = calculate_elo_change(1200, 1200, wins_a=0, draws=10, losses_a=0)

        assert update_a.rating_after == pytest.approx(1200)
        assert update_b.rating_after == pytest.approx(1200)

    def test_rating_changes_sum_to_zero(self):
        """Total rating points should be conserved (zero-sum)."""
        update_a, update_b = calculate_elo_change(1200, 1400, wins_a=3, draws=2, losses_a=5)

        change_a = update_a.rating_after - update_a.rating_before
        change_b = update_b.rating_after - update_b.rating_before

        assert change_a + change_b == pytest.approx(0)

    def test_more_games_larger_change(self):
        """More games should result in larger rating changes."""
        update_1, _ = calculate_elo_change(1200, 1200, wins_a=1, draws=0, losses_a=0)
        update_5, _ = calculate_elo_change(1200, 1200, wins_a=5, draws=0, losses_a=0)

        change_1 = abs(update_1.rating_after - 1200)
        change_5 = abs(update_5.rating_after - 1200)

        assert change_5 > change_1

    def test_expected_score_calculated(self):
        """Expected score should be properly calculated."""
        update_a, update_b = calculate_elo_change(1200, 1400, wins_a=1, draws=0, losses_a=0)

        # Lower rated player has < 0.5 expected
        assert update_a.expected_score < 0.5
        # Higher rated player has > 0.5 expected
        assert update_b.expected_score > 0.5
        # Sum to 1
        assert update_a.expected_score + update_b.expected_score == pytest.approx(1.0)

    def test_elo_update_dataclass(self):
        """EloUpdate dataclass should have correct properties."""
        update_a, _ = calculate_elo_change(1200, 1200, wins_a=1, draws=0, losses_a=0)

        assert update_a.rating_before == 1200
        assert update_a.rating_after > 1200
        assert update_a.rating_change == update_a.rating_after - update_a.rating_before
        assert update_a.actual_score == 1.0  # All wins


class TestGetInitialElo:
    """Tests for get_initial_elo function."""

    def test_random_policy(self):
        """Random policy should get low ELO."""
        assert get_initial_elo("random") == RANDOM_ELO
        assert get_initial_elo("Random") == RANDOM_ELO
        assert get_initial_elo("RANDOM") == RANDOM_ELO

    def test_stockfish_policy(self):
        """Stockfish policy should get high ELO."""
        assert get_initial_elo("stockfish") == STOCKFISH_ELO
        assert get_initial_elo("Stockfish") == STOCKFISH_ELO

    def test_unknown_policy(self):
        """Unknown policies should get default ELO."""
        assert get_initial_elo("smallnet") == DEFAULT_ELO
        assert get_initial_elo("llm") == DEFAULT_ELO
        assert get_initial_elo("my_custom_policy") == DEFAULT_ELO

    def test_constants(self):
        """ELO constants should have expected values."""
        assert DEFAULT_ELO == 1200
        assert RANDOM_ELO == 800
        assert STOCKFISH_ELO == 2800
        assert DEFAULT_K == 32
