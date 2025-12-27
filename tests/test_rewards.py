"""Tests for reward functions."""
import pytest

from chesster.selfplay.record import GameRecord, MoveRecord
from chesster.train.rewards import (
    RewardType,
    compute_discounted_returns,
    compute_game_rewards,
    compute_move_rewards,
    cp_delta_reward,
    match_bestmove_reward,
    outcome_reward,
)


def test_match_bestmove_reward_match() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
        teacher_uci="e2e4",
    )
    assert match_bestmove_reward(move) == 1.0


def test_match_bestmove_reward_no_match() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
        teacher_uci="d2d4",
    )
    assert match_bestmove_reward(move) == 0.0


def test_match_bestmove_reward_no_teacher() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
    )
    assert match_bestmove_reward(move) == 0.0


def test_cp_delta_reward_positive() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
        teacher_eval_cp=50,
    )
    # Position improved by 50 cp from 0
    reward = cp_delta_reward(move, prev_eval_cp=0, scale=100.0)
    assert reward == pytest.approx(0.5)


def test_cp_delta_reward_negative() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
        teacher_eval_cp=-100,
    )
    # Position worsened by 100 cp from 0
    reward = cp_delta_reward(move, prev_eval_cp=0, scale=100.0)
    assert reward == pytest.approx(-1.0)


def test_cp_delta_reward_clipped() -> None:
    move = MoveRecord(
        ply=0,
        fen="startpos",
        uci="e2e4",
        san="e4",
        policy_id="test",
        teacher_eval_cp=500,
    )
    # Large improvement clipped to 2.0
    reward = cp_delta_reward(move, prev_eval_cp=0, scale=100.0, clip=2.0)
    assert reward == 2.0


def test_outcome_reward_white_wins() -> None:
    assert outcome_reward("1-0", "white") == 1.0
    assert outcome_reward("1-0", "black") == -1.0


def test_outcome_reward_black_wins() -> None:
    assert outcome_reward("0-1", "white") == -1.0
    assert outcome_reward("0-1", "black") == 1.0


def test_outcome_reward_draw() -> None:
    assert outcome_reward("1/2-1/2", "white") == 0.0
    assert outcome_reward("1/2-1/2", "black") == 0.0


def test_compute_move_rewards_match_bestmove() -> None:
    game = GameRecord(
        id="test",
        white_policy="w",
        black_policy="b",
        result="1-0",
        termination="checkmate",
        moves=[
            MoveRecord(ply=0, fen="f1", uci="e2e4", san="e4", policy_id="w", teacher_uci="e2e4"),
            MoveRecord(ply=1, fen="f2", uci="e7e5", san="e5", policy_id="b", teacher_uci="d7d5"),
        ],
    )

    rewards = compute_move_rewards(game, RewardType.MATCH_BESTMOVE)
    assert len(rewards) == 2
    assert rewards[0].reward == 1.0  # Matched
    assert rewards[1].reward == 0.0  # Didn't match


def test_compute_game_rewards() -> None:
    game = GameRecord(
        id="test",
        white_policy="w",
        black_policy="b",
        result="1-0",
        termination="checkmate",
        moves=[
            MoveRecord(ply=0, fen="f1", uci="e2e4", san="e4", policy_id="w", teacher_uci="e2e4"),
            MoveRecord(ply=1, fen="f2", uci="e7e5", san="e5", policy_id="b", teacher_uci="e7e5"),
        ],
    )

    result = compute_game_rewards(game, RewardType.MATCH_BESTMOVE)
    assert result.game_id == "test"
    assert result.outcome_reward == 1.0  # White won
    assert result.total_reward == 2.0  # Both matched
    assert result.mean_reward == 1.0


def test_compute_discounted_returns() -> None:
    from chesster.train.rewards import MoveReward

    move_rewards = [
        MoveReward(ply=0, reward=1.0, reward_type=RewardType.MATCH_BESTMOVE),
        MoveReward(ply=1, reward=0.0, reward_type=RewardType.MATCH_BESTMOVE),
        MoveReward(ply=2, reward=1.0, reward_type=RewardType.MATCH_BESTMOVE),
    ]

    returns = compute_discounted_returns(move_rewards, gamma=0.9)
    assert len(returns) == 3

    # G_2 = 1.0
    assert returns[2] == pytest.approx(1.0)
    # G_1 = 0.0 + 0.9 * 1.0 = 0.9
    assert returns[1] == pytest.approx(0.9)
    # G_0 = 1.0 + 0.9 * 0.9 = 1.81
    assert returns[0] == pytest.approx(1.81)

