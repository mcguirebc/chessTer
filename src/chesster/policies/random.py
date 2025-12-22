from __future__ import annotations

import time

import chess

from chesster.chess.board import choose_fallback_legal_move, move_to_san

from .base import ChooseMoveParams, MoveResult


class RandomPolicy:
    policy_id = "random"
    description = "Choose a random legal move (requires a deterministic seed)."

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        if params.seed is None:
            raise ValueError("RandomPolicy requires params.seed for deterministic behavior.")

        started = time.perf_counter()
        move = choose_fallback_legal_move(board, seed=params.seed)
        uci = move.uci()
        san = move_to_san(board, move)
        took_ms = int((time.perf_counter() - started) * 1000)

        return MoveResult(uci=uci, san=san, info={"took_ms": took_ms})




