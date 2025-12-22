"""Tests for GameRecord serialization."""
import json
import tempfile
from pathlib import Path

import pytest

from chesster.selfplay.record import GameRecord, MoveRecord, load_games, save_games


def test_move_record_creation() -> None:
    move = MoveRecord(
        ply=0,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        uci="e2e4",
        san="e4",
        policy_id="random",
    )
    assert move.ply == 0
    assert move.uci == "e2e4"
    assert move.teacher_uci is None


def test_move_record_with_teacher() -> None:
    move = MoveRecord(
        ply=0,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        uci="e2e4",
        san="e4",
        policy_id="random",
        teacher_uci="d2d4",
        teacher_eval_cp=30,
    )
    assert move.teacher_uci == "d2d4"
    assert move.teacher_eval_cp == 30


def test_game_record_to_dict() -> None:
    game = GameRecord(
        id="test-123",
        white_policy="random",
        black_policy="stockfish",
        result="1-0",
        termination="checkmate",
        moves=[
            MoveRecord(
                ply=0,
                fen="startpos",
                uci="e2e4",
                san="e4",
                policy_id="random",
            )
        ],
        metadata={"seed": 42},
    )

    d = game.to_dict()
    assert d["id"] == "test-123"
    assert d["result"] == "1-0"
    assert len(d["moves"]) == 1
    assert d["metadata"]["seed"] == 42


def test_game_record_from_dict() -> None:
    d = {
        "id": "test-456",
        "white_policy": "a",
        "black_policy": "b",
        "result": "0-1",
        "termination": "stalemate",
        "moves": [
            {
                "ply": 0,
                "fen": "test-fen",
                "uci": "e2e4",
                "san": "e4",
                "policy_id": "a",
                "teacher_uci": None,
                "teacher_eval_cp": None,
            }
        ],
        "metadata": {},
    }

    game = GameRecord.from_dict(d)
    assert game.id == "test-456"
    assert game.result == "0-1"
    assert len(game.moves) == 1
    assert game.moves[0].uci == "e2e4"


def test_game_record_jsonl_roundtrip() -> None:
    game = GameRecord(
        id="roundtrip-test",
        white_policy="policy_a",
        black_policy="policy_b",
        result="1/2-1/2",
        termination="repetition",
        moves=[
            MoveRecord(
                ply=0,
                fen="fen1",
                uci="e2e4",
                san="e4",
                policy_id="policy_a",
                teacher_uci="d2d4",
                teacher_eval_cp=15,
            ),
            MoveRecord(
                ply=1,
                fen="fen2",
                uci="e7e5",
                san="e5",
                policy_id="policy_b",
            ),
        ],
        metadata={"test_key": "test_value"},
    )

    # Serialize
    jsonl = game.to_jsonl()

    # Deserialize
    restored = GameRecord.from_jsonl(jsonl)

    assert restored.id == game.id
    assert restored.result == game.result
    assert restored.termination == game.termination
    assert len(restored.moves) == 2
    assert restored.moves[0].teacher_uci == "d2d4"
    assert restored.moves[1].teacher_uci is None
    assert restored.metadata["test_key"] == "test_value"


def test_save_and_load_games() -> None:
    games = [
        GameRecord(
            id=f"game-{i}",
            white_policy="w",
            black_policy="b",
            result="*",
            termination="max_moves",
            moves=[],
            metadata={},
        )
        for i in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "games.jsonl"

        # Save
        count = save_games(path, games)
        assert count == 3
        assert path.exists()

        # Load
        loaded = list(load_games(path))
        assert len(loaded) == 3
        assert loaded[0].id == "game-0"
        assert loaded[2].id == "game-2"


def test_save_games_append() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "games.jsonl"

        # First batch
        games1 = [
            GameRecord(id="a", white_policy="w", black_policy="b", result="*", termination="t", moves=[])
        ]
        save_games(path, games1)

        # Append second batch
        games2 = [
            GameRecord(id="b", white_policy="w", black_policy="b", result="*", termination="t", moves=[])
        ]
        save_games(path, games2, append=True)

        # Load all
        loaded = list(load_games(path))
        assert len(loaded) == 2
        assert loaded[0].id == "a"
        assert loaded[1].id == "b"

