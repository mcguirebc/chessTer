from __future__ import annotations

import re
import time
from typing import Any

import chess
import httpx

from chesster.chess.board import (
    choose_fallback_legal_move,
    legal_moves_uci,
    move_to_san,
    parse_uci_move,
)

from .base import ChooseMoveParams, MoveResult


_UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


class OllamaPolicy:
    policy_id = "ollama"
    description = "Choose a move via Ollama (LLM), constrained to legal UCI moves."

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "gpt-oss-20b",
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def _prompt(self, board: chess.Board, legal_uci: list[str]) -> str:
        stm = "white" if board.turn == chess.WHITE else "black"
        legal = " ".join(legal_uci)
        return (
            "You are a chess engine. "
            "Given the position, choose the single best move.\n\n"
            f"Side to move: {stm}\n"
            f"FEN: {board.fen()}\n"
            f"Legal moves (UCI): {legal}\n\n"
            "Return exactly one move in UCI from the legal list. No other text."
        )

    def _extract_first_legal_uci(self, text: str, board: chess.Board) -> str | None:
        for m in _UCI_RE.findall(text.lower()):
            try:
                parse_uci_move(board, m)
            except Exception:
                continue
            return m
        return None

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        started = time.perf_counter()

        legal_uci = legal_moves_uci(board)
        prompt = self._prompt(board, legal_uci)

        model = params.model or self._default_model
        temperature = 0.0 if params.temperature is None else params.temperature
        options: dict[str, Any] = {"temperature": temperature}
        if params.top_p is not None:
            options["top_p"] = params.top_p
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens

        resp = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
        )
        resp.raise_for_status()

        data = resp.json()
        text = data.get("response") or ""

        chosen_uci = self._extract_first_legal_uci(text, board)
        info: dict[str, Any] = {
            "took_ms": int((time.perf_counter() - started) * 1000),
            "model": model,
            "temperature": temperature,
        }

        if chosen_uci is None:
            info["invalid_output"] = True
            fallback = choose_fallback_legal_move(board, seed=params.seed)
            chosen_move = fallback
        else:
            chosen_move = parse_uci_move(board, chosen_uci)

        uci = chosen_move.uci()
        san = move_to_san(board, chosen_move)
        return MoveResult(uci=uci, san=san, info=info)




