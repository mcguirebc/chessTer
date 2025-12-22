import json
from dataclasses import dataclass

import chess
import pytest
from fastapi.testclient import TestClient

from chesster.api.app import app
from chesster.api.deps import PolicyRegistry, get_idempotency_store, get_policy_registry
from chesster.api.idempotency import IdempotencyStore
from chesster.chess.board import legal_moves_uci
from chesster.policies.base import ChooseMoveParams, MoveResult


@dataclass
class CyclingPolicy:
    policy_id: str = "cycle"
    description: str = "Cycles through legal moves each call (for idempotency tests)."

    _i: int = 0

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        moves = legal_moves_uci(board)
        uci = moves[self._i % len(moves)]
        self._i += 1
        # SAN is computed by server anyway.
        return MoveResult(uci=uci, san="", info={"counter": self._i})


@pytest.fixture()
def client(tmp_path) -> TestClient:
    db_path = tmp_path / "idem.sqlite"
    store = IdempotencyStore(str(db_path))

    registry = PolicyRegistry(policies={"cycle": CyclingPolicy()})

    app.dependency_overrides[get_idempotency_store] = lambda: store
    app.dependency_overrides[get_policy_registry] = lambda: registry
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_idempotency_replay_returns_same_response(client: TestClient) -> None:
    body = {"fen": "startpos", "policy_id": "cycle", "idempotency_key": "k1"}
    r1 = client.post("/v1/move", json=body)
    assert r1.status_code == 200
    r2 = client.post("/v1/move", json=body)
    assert r2.status_code == 200
    assert r1.json() == r2.json()


def test_idempotency_conflict_returns_409(client: TestClient) -> None:
    body1 = {"fen": "startpos", "policy_id": "cycle", "idempotency_key": "k2"}
    body2 = {"fen": chess.Board().mirror().fen(), "policy_id": "cycle", "idempotency_key": "k2"}

    r1 = client.post("/v1/move", json=body1)
    assert r1.status_code == 200

    r2 = client.post("/v1/move", json=body2)
    assert r2.status_code == 409
    assert "Idempotency key reused" in json.dumps(r2.json())




