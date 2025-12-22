from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass

import chess
import chess.engine

from chesster.chess.board import move_to_san

from .base import ChooseMoveParams, MoveResult


def _resolve_stockfish_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    env_path = os.getenv("STOCKFISH_PATH")
    if env_path:
        return env_path

    which = shutil.which("stockfish")
    if which:
        return which

    # Common Homebrew locations
    for p in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Stockfish binary not found. Set STOCKFISH_PATH or install stockfish on your PATH."
    )


@dataclass
class _EngineHandle:
    engine: chess.engine.SimpleEngine
    lock: threading.Lock


class StockfishPolicy:
    policy_id = "stockfish"
    description = "Choose Stockfish best move (deterministic depth by default)."

    def __init__(self, stockfish_path: str | None = None, default_depth: int = 10) -> None:
        self._stockfish_path = stockfish_path
        self._default_depth = default_depth
        self._handle: _EngineHandle | None = None

    def _get_handle(self) -> _EngineHandle:
        if self._handle is not None:
            return self._handle

        path = _resolve_stockfish_path(self._stockfish_path)
        engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            engine.configure({"Threads": 1})
        except chess.engine.EngineError:
            # Not all builds support these options; ignore.
            pass

        self._handle = _EngineHandle(engine=engine, lock=threading.Lock())
        return self._handle

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.engine.quit()
        finally:
            self._handle = None

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        started = time.perf_counter()

        depth = params.depth if params.depth is not None else self._default_depth
        limit: chess.engine.Limit
        if params.time_ms is not None:
            limit = chess.engine.Limit(time=max(params.time_ms, 1) / 1000.0)
        else:
            limit = chess.engine.Limit(depth=depth)

        handle = self._get_handle()
        with handle.lock:
            result = handle.engine.play(board, limit)

        if result.move is None:
            raise RuntimeError("Stockfish returned no move.")

        move = result.move
        uci = move.uci()
        san = move_to_san(board, move)
        took_ms = int((time.perf_counter() - started) * 1000)

        info = {"took_ms": took_ms}
        if params.time_ms is not None:
            info["time_ms"] = params.time_ms
        else:
            info["depth"] = depth

        return MoveResult(uci=uci, san=san, info=info)




