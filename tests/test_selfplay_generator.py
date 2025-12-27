"""Tests for self-play game generator."""
import tempfile
from pathlib import Path

import pytest

from chesster.policies.random import RandomPolicy
from chesster.selfplay.generator import GeneratorConfig, generate_selfplay_games
from chesster.selfplay.record import load_games


def test_generate_selfplay_games_basic() -> None:
    policy = RandomPolicy()
    opponents = [RandomPolicy()]

    config = GeneratorConfig(
        n_games=4,
        max_moves=20,
        base_seed=42,
    )

    games = generate_selfplay_games(policy, opponents, config)

    assert len(games) == 4
    for game in games:
        assert len(game.moves) > 0
        assert game.white_policy == "random"
        assert game.black_policy == "random"


def test_generate_selfplay_games_multiple_opponents() -> None:
    policy = RandomPolicy()

    opp1 = RandomPolicy()
    opp1.policy_id = "opponent_1"
    opp2 = RandomPolicy()
    opp2.policy_id = "opponent_2"

    config = GeneratorConfig(
        n_games=6,
        max_moves=15,
        base_seed=123,
    )

    games = generate_selfplay_games(policy, [opp1, opp2], config)

    assert len(games) == 6

    # Should have games against both opponents
    opponent_ids = set()
    for game in games:
        if game.white_policy != "random":
            opponent_ids.add(game.white_policy)
        if game.black_policy != "random":
            opponent_ids.add(game.black_policy)

    assert "opponent_1" in opponent_ids
    assert "opponent_2" in opponent_ids


def test_generate_selfplay_games_alternates_colors() -> None:
    policy = RandomPolicy()
    policy.policy_id = "main_policy"

    opponent = RandomPolicy()
    opponent.policy_id = "opponent"

    config = GeneratorConfig(
        n_games=4,
        max_moves=10,
        base_seed=42,
        alternate_colors=True,
    )

    games = generate_selfplay_games(policy, [opponent], config)

    # With alternating colors, main_policy should be white in some and black in others
    main_white = sum(1 for g in games if g.white_policy == "main_policy")
    main_black = sum(1 for g in games if g.black_policy == "main_policy")

    assert main_white == 2
    assert main_black == 2


def test_generate_selfplay_games_saves_to_file() -> None:
    policy = RandomPolicy()
    opponent = RandomPolicy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "games.jsonl"

        config = GeneratorConfig(
            n_games=3,
            max_moves=10,
            base_seed=42,
            output_path=str(output_path),
        )

        games = generate_selfplay_games(policy, [opponent], config)

        assert output_path.exists()
        loaded = list(load_games(output_path))
        assert len(loaded) == 3


def test_generate_selfplay_games_deterministic() -> None:
    policy = RandomPolicy()
    opponent = RandomPolicy()

    config = GeneratorConfig(
        n_games=3,
        max_moves=20,
        base_seed=999,
    )

    games1 = generate_selfplay_games(policy, [opponent], config)
    games2 = generate_selfplay_games(policy, [opponent], config)

    # Same seed should produce same games
    assert len(games1) == len(games2)
    for g1, g2 in zip(games1, games2):
        assert len(g1.moves) == len(g2.moves)
        for m1, m2 in zip(g1.moves, g2.moves):
            assert m1.uci == m2.uci

