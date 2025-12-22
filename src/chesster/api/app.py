from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from chesster.api.deps import PolicyRegistry, get_idempotency_store, get_policy_registry
from chesster.api.idempotency import hash_request
from chesster.api.schemas import MoveRequest, MoveResponse, PoliciesResponse, PolicyInfo
from chesster.chess.board import FenParseError, IllegalMoveError, move_to_san, parse_board, parse_uci_move
from chesster.policies.base import ChooseMoveParams, Policy


app = FastAPI(title="chesster")


def _derive_seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/v1/policies", response_model=PoliciesResponse)
def list_policies(registry: PolicyRegistry = Depends(get_policy_registry)) -> PoliciesResponse:
    return PoliciesResponse(
        policies=[PolicyInfo(policy_id=pid, description=desc) for pid, desc in registry.list()]
    )


@app.post("/v1/move", response_model=MoveResponse)
def choose_move(
    req: MoveRequest,
    registry: PolicyRegistry = Depends(get_policy_registry),
    store=Depends(get_idempotency_store),
) -> MoveResponse:
    # Hash excludes idempotency_key (key is used as the storage lookup)
    payload_for_hash = req.model_dump(exclude={"idempotency_key"}, exclude_none=True)
    request_hash = hash_request(payload_for_hash)

    if req.idempotency_key:
        existing = store.get(req.idempotency_key)
        if existing is not None:
            if existing.request_hash != request_hash:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency key reused with different request payload.",
                )
            cached = json.loads(existing.response_json)
            return MoveResponse.model_validate(cached)

    policy_obj = registry.get(req.policy_id)
    if policy_obj is None:
        raise HTTPException(status_code=404, detail=f"Unknown policy_id: {req.policy_id!r}")

    policy: Policy = policy_obj  # typing aid
    try:
        board = parse_board(req.fen)
    except FenParseError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    params = ChooseMoveParams(
        seed=req.seed,
        depth=req.depth,
        time_ms=req.time_ms,
        model=req.model,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    if req.policy_id == "random" and params.seed is None:
        if req.idempotency_key is None:
            raise HTTPException(
                status_code=400,
                detail="random policy requires seed or idempotency_key for deterministic behavior.",
            )
        params = dataclasses.replace(params, seed=_derive_seed_from_key(req.idempotency_key))

    started = time.perf_counter()
    try:
        result = policy.choose_move(board, params)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Server-side legality check (hard guarantee)
    try:
        move = parse_uci_move(board, result.uci)
    except IllegalMoveError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    san = move_to_san(board, move)
    response = MoveResponse(
        uci=move.uci(),
        san=san,
        policy_id=req.policy_id,
        idempotency_key=req.idempotency_key,
        info={
            **(result.info or {}),
            "server_took_ms": int((time.perf_counter() - started) * 1000),
            "legal_count": board.legal_moves.count(),
        },
    )

    if req.idempotency_key:
        inserted = store.put_if_absent(
            key=req.idempotency_key,
            request_hash=request_hash,
            response_json=response.model_dump_json(),
        )
        if not inserted:
            # Another request won the race: return the stored response for this idempotency key.
            existing = store.get(req.idempotency_key)
            if existing is None:
                raise HTTPException(status_code=500, detail="Idempotency store read-after-write failed.")
            if existing.request_hash != request_hash:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency key reused with different request payload.",
                )
            cached = json.loads(existing.response_json)
            return MoveResponse.model_validate(cached)

    return response


