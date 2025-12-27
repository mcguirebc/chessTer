from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import chess


@dataclass(frozen=True, slots=True)
class ChooseMoveParams:
    seed: int | None = None
    depth: int | None = None
    time_ms: int | None = None

    # LLM params
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    # ELO context for LLM prompts (enables opponent-aware play)
    opponent_elo: float | None = None
    self_elo: float | None = None
    opponent_is_bot: bool | None = None


@dataclass(frozen=True, slots=True)
class MoveResult:
    uci: str
    san: str
    info: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    policy_id: str
    description: str

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult: ...




