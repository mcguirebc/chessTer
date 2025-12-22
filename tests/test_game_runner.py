"""Tests for GameRunner."""
import chess
import pytest

from chesster.policies.random import RandomPolicy
from chesster.selfplay.runner import play_game


def test_play_game_returns_game_record() -> None:
    white = RandomPolicy()
    black = RandomPolicy()

    game = play_game(white, black, max_moves=20, seed=42)

    assert game.id is not None
    assert game.white_policy == "random"
    assert game.black_policy == "random"
    assert game.result in ("1-0", "0-1", "1/2-1/2", "*")
    assert len(game.moves) > 0


def test_play_game_moves_are_legal() -> None:
    white = RandomPolicy()
    black = RandomPolicy()

    game = play_game(white, black, max_moves=50, seed=123)

    # Replay moves and verify legality
    board = chess.Board()
    for move_record in game.moves:
        assert move_record.fen == board.fen()
        move = chess.Move.from_uci(move_record.uci)
        assert move in board.legal_moves, f"Illegal move: {move_record.uci}"
        board.push(move)


def test_play_game_deterministic_with_seed() -> None:
    white = RandomPolicy()
    black = RandomPolicy()

    game1 = play_game(white, black, max_moves=30, seed=999)
    game2 = play_game(white, black, max_moves=30, seed=999)

    # Same seed should produce same moves
    assert len(game1.moves) == len(game2.moves)
    for m1, m2 in zip(game1.moves, game2.moves):
        assert m1.uci == m2.uci


def test_play_game_max_moves_limit() -> None:
    white = RandomPolicy()
    black = RandomPolicy()

    game = play_game(white, black, max_moves=10, seed=42)

    # Should stop at max_moves if game doesn't end naturally
    assert len(game.moves) <= 10


def test_play_game_checkmate_detection() -> None:
    """Test that checkmate is properly detected."""
    # Use a very short game limit to ensure we hit max_moves first in most cases
    # This test just verifies the runner doesn't crash and returns valid termination
    white = RandomPolicy()
    black = RandomPolicy()

    game = play_game(white, black, max_moves=500, seed=12345)

    assert game.termination in (
        "checkmate",
        "stalemate",
        "50-move",
        "repetition",
        "insufficient",
        "max_moves",
        "in_progress",
    )

    if game.termination == "checkmate":
        assert game.result in ("1-0", "0-1")
    elif game.termination in ("stalemate", "50-move", "repetition", "insufficient"):
        assert game.result == "1/2-1/2"


def test_play_game_metadata() -> None:
    white = RandomPolicy()
    black = RandomPolicy()

    game = play_game(white, black, max_moves=10, seed=42)

    assert "started_at" in game.metadata
    assert "finished_at" in game.metadata
    assert "final_fen" in game.metadata
    assert game.metadata["max_moves"] == 10
    assert game.metadata["seed"] == 42

