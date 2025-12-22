"""GameRunner: plays full games between two policies."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import chess

from chesster.chess.board import move_to_san, parse_uci_move
from chesster.policies.base import ChooseMoveParams

from .record import GameRecord, MoveRecord

if TYPE_CHECKING:
    from chesster.policies.base import Policy


def _get_termination(board: chess.Board, max_moves_reached: bool) -> tuple[str, str]:
    """
    Determine game result and termination reason.

    Returns:
        (result, termination) tuple.
    """
    if max_moves_reached:
        return ("*", "max_moves")

    if board.is_checkmate():
        # The side to move is checkmated, so the other side wins
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        return (result, "checkmate")

    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")

    if board.is_fifty_moves():
        return ("1/2-1/2", "50-move")

    if board.is_repetition(3):
        return ("1/2-1/2", "repetition")

    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient")

    # Game still in progress
    return ("*", "in_progress")


def play_game(
    white: Policy,
    black: Policy,
    *,
    max_moves: int = 500,
    teacher: Policy | None = None,
    teacher_depth: int | None = None,
    seed: int | None = None,
) -> GameRecord:
    """
    Play a complete game between two policies.

    Args:
        white: Policy for White.
        black: Policy for Black.
        max_moves: Maximum number of half-moves (ply) before stopping.
        teacher: Optional teacher policy to annotate each position with best move/eval.
        teacher_depth: Depth for teacher analysis (passed as params.depth).
        seed: Base seed for deterministic play (each ply uses seed+ply).

    Returns:
        A GameRecord with all moves and optional teacher annotations.
    """
    board = chess.Board()
    moves: list[MoveRecord] = []

    game_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    ply = 0
    while ply < max_moves:
        if board.is_game_over():
            break

        fen = board.fen()
        policy = white if board.turn == chess.WHITE else black
        policy_id = getattr(policy, "policy_id", "unknown")

        # Compute seed for this ply
        ply_seed = (seed + ply) if seed is not None else None

        params = ChooseMoveParams(seed=ply_seed)
        result = policy.choose_move(board, params)

        move = parse_uci_move(board, result.uci)
        san = move_to_san(board, move)

        # Optional teacher annotation
        teacher_uci: str | None = None
        teacher_eval_cp: int | None = None
        if teacher is not None:
            teacher_params = ChooseMoveParams(depth=teacher_depth)
            teacher_result = teacher.choose_move(board, teacher_params)
            teacher_uci = teacher_result.uci
            # Extract eval if available in info
            if "score_cp" in teacher_result.info:
                teacher_eval_cp = teacher_result.info["score_cp"]

        move_record = MoveRecord(
            ply=ply,
            fen=fen,
            uci=move.uci(),
            san=san,
            policy_id=policy_id,
            teacher_uci=teacher_uci,
            teacher_eval_cp=teacher_eval_cp,
        )
        moves.append(move_record)

        board.push(move)
        ply += 1

    # Determine result
    max_moves_reached = ply >= max_moves and not board.is_game_over()
    result_str, termination = _get_termination(board, max_moves_reached)

    return GameRecord(
        id=game_id,
        white_policy=getattr(white, "policy_id", "unknown"),
        black_policy=getattr(black, "policy_id", "unknown"),
        result=result_str,
        termination=termination,
        moves=moves,
        metadata={
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "max_moves": max_moves,
            "seed": seed,
            "final_fen": board.fen(),
        },
    )


def play_games(
    white: Policy,
    black: Policy,
    n_games: int,
    *,
    max_moves: int = 500,
    teacher: Policy | None = None,
    teacher_depth: int | None = None,
    base_seed: int | None = None,
    alternate_colors: bool = True,
) -> list[GameRecord]:
    """
    Play multiple games between two policies.

    Args:
        white: Policy for White (in odd games if alternating).
        black: Policy for Black (in odd games if alternating).
        n_games: Number of games to play.
        max_moves: Maximum ply per game.
        teacher: Optional teacher for annotation.
        teacher_depth: Depth for teacher analysis.
        base_seed: Base seed; each game uses base_seed + game_index.
        alternate_colors: If True, swap colors every other game.

    Returns:
        List of GameRecords.
    """
    records: list[GameRecord] = []

    for i in range(n_games):
        game_seed = (base_seed + i * 1000) if base_seed is not None else None

        if alternate_colors and i % 2 == 1:
            w, b = black, white
        else:
            w, b = white, black

        record = play_game(
            w,
            b,
            max_moves=max_moves,
            teacher=teacher,
            teacher_depth=teacher_depth,
            seed=game_seed,
        )
        records.append(record)

    return records

