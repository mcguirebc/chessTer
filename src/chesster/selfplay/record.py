"""GameRecord and MoveRecord dataclasses with JSONL serialization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """A single move within a game."""

    ply: int  # 0-indexed half-move number
    fen: str  # position before the move
    uci: str  # move played in UCI notation
    san: str  # move played in SAN notation
    policy_id: str  # which policy made this move

    # Optional teacher annotations
    teacher_uci: str | None = None  # Stockfish best move
    teacher_eval_cp: int | None = None  # centipawn evaluation (from side-to-move POV)


@dataclass(slots=True)
class GameRecord:
    """A complete game record."""

    id: str  # unique game identifier (uuid)
    white_policy: str
    black_policy: str
    result: str  # "1-0", "0-1", "1/2-1/2", "*"
    termination: str  # "checkmate", "stalemate", "50-move", "repetition", "insufficient", "max_moves", "resignation"
    moves: list[MoveRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "id": self.id,
            "white_policy": self.white_policy,
            "black_policy": self.black_policy,
            "result": self.result,
            "termination": self.termination,
            "moves": [asdict(m) for m in self.moves],
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameRecord:
        """Deserialize from a dict."""
        moves = [MoveRecord(**m) for m in d.get("moves", [])]
        return cls(
            id=d["id"],
            white_policy=d["white_policy"],
            black_policy=d["black_policy"],
            result=d["result"],
            termination=d["termination"],
            moves=moves,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_jsonl(cls, line: str) -> GameRecord:
        """Deserialize from a JSONL line."""
        return cls.from_dict(json.loads(line))


def load_games(path: str | Path) -> Iterator[GameRecord]:
    """
    Load games from a JSONL file.

    Yields GameRecord objects one at a time for memory efficiency.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield GameRecord.from_jsonl(line)


def save_games(path: str | Path, games: Iterable[GameRecord], *, append: bool = False) -> int:
    """
    Save games to a JSONL file.

    Args:
        path: Output file path.
        games: Iterable of GameRecord objects.
        append: If True, append to existing file; otherwise overwrite.

    Returns:
        Number of games written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for game in games:
            f.write(game.to_jsonl())
            f.write("\n")
            count += 1
    return count

