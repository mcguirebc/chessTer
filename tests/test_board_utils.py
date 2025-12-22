import chess
import pytest

from chesster.chess.board import FenParseError, IllegalMoveError, legal_moves_uci, move_to_san, parse_board, parse_uci_move


def test_parse_board_startpos() -> None:
    b = parse_board("startpos")
    assert isinstance(b, chess.Board)
    assert b.fen() == chess.Board().fen()


def test_parse_board_invalid_fen() -> None:
    with pytest.raises(FenParseError):
        parse_board("not-a-fen")


def test_legal_moves_startpos_len() -> None:
    b = parse_board("startpos")
    moves = legal_moves_uci(b)
    assert len(moves) == 20
    assert moves == sorted(moves)


def test_parse_uci_move_and_san() -> None:
    b = parse_board("startpos")
    m = parse_uci_move(b, "e2e4")
    assert m.uci() == "e2e4"
    assert move_to_san(b, m) == "e4"


def test_parse_uci_move_illegal() -> None:
    b = parse_board("startpos")
    with pytest.raises(IllegalMoveError):
        parse_uci_move(b, "e2e5")




