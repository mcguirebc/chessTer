"""Reward functions for RL training."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesster.selfplay.record import GameRecord, MoveRecord


class RewardType(str, Enum):
    """Available reward types."""

    MATCH_BESTMOVE = "match_bestmove"
    CP_DELTA = "cp_delta"
    OUTCOME = "outcome"


@dataclass(frozen=True, slots=True)
class MoveReward:
    """Reward for a single move."""

    ply: int
    reward: float
    reward_type: RewardType


@dataclass(frozen=True, slots=True)
class GameRewards:
    """Rewards for all moves in a game."""

    game_id: str
    move_rewards: list[MoveReward]
    outcome_reward: float  # +1 win, 0 draw, -1 loss (from white's perspective)

    @property
    def total_reward(self) -> float:
        """Sum of all move rewards."""
        return sum(mr.reward for mr in self.move_rewards)

    @property
    def mean_reward(self) -> float:
        """Mean reward per move."""
        if not self.move_rewards:
            return 0.0
        return self.total_reward / len(self.move_rewards)


def match_bestmove_reward(move: MoveRecord) -> float:
    """
    Compute match_bestmove reward for a single move.

    Returns 1.0 if the move matches the teacher's best move, else 0.0.
    Returns 0.0 if teacher_uci is not available.
    """
    if move.teacher_uci is None:
        return 0.0
    return 1.0 if move.uci == move.teacher_uci else 0.0


def cp_delta_reward(
    move: MoveRecord,
    prev_eval_cp: int | None,
    scale: float = 100.0,
    clip: float = 2.0,
) -> float:
    """
    Compute centipawn delta reward for a single move.

    Reward = (current_eval - prev_eval) / scale, clipped to [-clip, +clip].
    Positive reward means the position improved for the side that just moved.

    Args:
        move: The move record with teacher_eval_cp annotation.
        prev_eval_cp: The eval before the move (from the moving side's perspective).
        scale: Divide centipawn delta by this for normalization.
        clip: Clip reward to this range.

    Returns:
        Normalized and clipped reward.
    """
    if move.teacher_eval_cp is None or prev_eval_cp is None:
        return 0.0

    # teacher_eval_cp is from side-to-move perspective after the move
    # prev_eval_cp is from the moving side's perspective before the move
    # So delta = current - prev (both from same perspective)
    delta = move.teacher_eval_cp - prev_eval_cp
    reward = delta / scale
    return max(-clip, min(clip, reward))


def outcome_reward(result: str, perspective: str = "white") -> float:
    """
    Compute game outcome reward.

    Args:
        result: Game result string ("1-0", "0-1", "1/2-1/2", "*").
        perspective: "white" or "black".

    Returns:
        +1.0 for win, 0.0 for draw, -1.0 for loss.
    """
    if result == "1-0":
        return 1.0 if perspective == "white" else -1.0
    elif result == "0-1":
        return -1.0 if perspective == "white" else 1.0
    else:
        # Draw or unfinished
        return 0.0


def compute_move_rewards(
    game: GameRecord,
    reward_type: RewardType,
    cp_scale: float = 100.0,
    cp_clip: float = 2.0,
) -> list[MoveReward]:
    """
    Compute rewards for all moves in a game.

    Args:
        game: The game record.
        reward_type: Which reward function to use.
        cp_scale: Scale factor for cp_delta reward.
        cp_clip: Clip value for cp_delta reward.

    Returns:
        List of MoveReward objects.
    """
    rewards: list[MoveReward] = []

    prev_eval_cp: int | None = None

    for move in game.moves:
        if reward_type == RewardType.MATCH_BESTMOVE:
            r = match_bestmove_reward(move)
        elif reward_type == RewardType.CP_DELTA:
            r = cp_delta_reward(move, prev_eval_cp, scale=cp_scale, clip=cp_clip)
            # Update prev_eval for next iteration (flip sign for opponent's perspective)
            if move.teacher_eval_cp is not None:
                prev_eval_cp = -move.teacher_eval_cp
        elif reward_type == RewardType.OUTCOME:
            # For outcome reward, assign the game result to each move
            # Perspective depends on who made the move
            is_white = move.ply % 2 == 0
            perspective = "white" if is_white else "black"
            r = outcome_reward(game.result, perspective)
        else:
            r = 0.0

        rewards.append(MoveReward(ply=move.ply, reward=r, reward_type=reward_type))

    return rewards


def compute_game_rewards(
    game: GameRecord,
    reward_type: RewardType,
    cp_scale: float = 100.0,
    cp_clip: float = 2.0,
) -> GameRewards:
    """
    Compute all rewards for a game.

    Args:
        game: The game record.
        reward_type: Which reward function to use.
        cp_scale: Scale factor for cp_delta reward.
        cp_clip: Clip value for cp_delta reward.

    Returns:
        GameRewards object with move-level and game-level rewards.
    """
    move_rewards = compute_move_rewards(game, reward_type, cp_scale, cp_clip)

    # Outcome reward from white's perspective
    game_outcome = outcome_reward(game.result, "white")

    return GameRewards(
        game_id=game.id,
        move_rewards=move_rewards,
        outcome_reward=game_outcome,
    )


def compute_discounted_returns(
    move_rewards: list[MoveReward],
    gamma: float = 0.99,
) -> list[float]:
    """
    Compute discounted returns (reward-to-go) for each move.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        move_rewards: List of move rewards.
        gamma: Discount factor.

    Returns:
        List of discounted returns, one per move.
    """
    if not move_rewards:
        return []

    returns: list[float] = []
    g = 0.0

    # Compute returns backwards
    for mr in reversed(move_rewards):
        g = mr.reward + gamma * g
        returns.append(g)

    returns.reverse()
    return returns

