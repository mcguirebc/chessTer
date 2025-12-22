from __future__ import annotations

import random
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from collections.abc import Sequence


class FenParseError(ValueError):
    pass


class IllegalMoveError(ValueError):
    pass


def parse_board(fen: str) -> chess.Board:
    """
    Parse a FEN string into a python-chess Board.

    Supports the convenience alias `"startpos"`.
    """
    fen = fen.strip()
    if fen == "startpos":
        return chess.Board()
    try:
        return chess.Board(fen)
    except ValueError as e:
        raise FenParseError(f"Invalid FEN: {fen!r}") from e


def legal_moves_uci(board: chess.Board) -> list[str]:
    """Return legal moves as UCI strings, sorted for stable output/prompting."""
    moves = [m.uci() for m in board.legal_moves]
    moves.sort()
    return moves


def parse_uci_move(board: chess.Board, uci: str) -> chess.Move:
    """
    Parse and validate a UCI move string against the board's legal moves.
    """
    uci = uci.strip().lower()
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as e:
        raise IllegalMoveError(f"Invalid UCI move: {uci!r}") from e
    if move not in board.legal_moves:
        raise IllegalMoveError(f"Illegal move for position: {uci!r}")
    return move


def move_to_san(board: chess.Board, move: chess.Move) -> str:
    """Compute SAN without mutating the caller's board."""
    b = board.copy(stack=False)
    return b.san(move)


def choose_fallback_legal_move(board: chess.Board, seed: int | None) -> chess.Move:
    """
    Choose a legal move deterministically.

    - If seed is provided: pseudo-random choice (stable for a given seed).
    - If seed is None: first legal move in sorted UCI order (fully deterministic).
    """
    legal: Sequence[chess.Move] = tuple(board.legal_moves)
    if not legal:
        raise IllegalMoveError("No legal moves available (game is over).")

    if seed is None:
        uci_sorted = sorted(legal, key=lambda m: m.uci())
        return uci_sorted[0]

    rng = random.Random(seed)
    return rng.choice(list(legal))




