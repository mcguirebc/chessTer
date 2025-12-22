from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MoveRequest(BaseModel):
    fen: str = Field(..., description='FEN string, or the alias "startpos".')
    policy_id: str = Field(..., description="Which policy backend to use.")

    idempotency_key: str | None = Field(
        default=None, description="If provided, enables idempotent replay semantics."
    )

    seed: int | None = None

    # Stockfish params
    depth: int | None = None
    time_ms: int | None = None

    # LLM params
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class MoveResponse(BaseModel):
    uci: str
    san: str
    policy_id: str
    idempotency_key: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class PolicyInfo(BaseModel):
    policy_id: str
    description: str


class PoliciesResponse(BaseModel):
    policies: list[PolicyInfo]




