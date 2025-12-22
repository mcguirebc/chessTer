import chess
from fastapi.testclient import TestClient

from chesster.api.app import app


def test_move_endpoint_random_is_legal() -> None:
    client = TestClient(app)
    r = client.post(
        "/v1/move",
        json={"fen": "startpos", "policy_id": "random", "idempotency_key": "legality-1"},
    )
    assert r.status_code == 200
    data = r.json()
    uci = data["uci"]

    board = chess.Board()
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves




