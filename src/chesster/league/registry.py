"""ModelRegistry: save, load, and list model snapshots."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chesster.league.elo import DEFAULT_ELO

if TYPE_CHECKING:
    from chesster.policies.net import SmallNetPolicy


@dataclass
class EloHistoryEntry:
    """A single entry in a model's ELO history."""

    elo: float
    opponent: str
    timestamp: str
    games_played: int
    result: str  # "win", "loss", "draw", or W-D-L summary like "3-2-1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EloHistoryEntry:
        return cls(**d)


@dataclass
class SnapshotInfo:
    """Metadata for a saved model snapshot."""

    name: str
    artifact_path: str  # relative to registry root
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # ELO tracking fields
    elo: float = DEFAULT_ELO
    elo_history: list[dict[str, Any]] = field(default_factory=list)
    games_played: int = 0
    is_bot: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SnapshotInfo:
        # Handle backward compatibility for snapshots without ELO fields
        d = d.copy()
        d.setdefault("elo", DEFAULT_ELO)
        d.setdefault("elo_history", [])
        d.setdefault("games_played", 0)
        d.setdefault("is_bot", True)
        return cls(**d)


class ModelRegistry:
    """
    Registry for model snapshots.

    Storage layout:
        {root}/
            registry.json        # index of all snapshots
            snapshots/
                {name}/
                    model.pt     # model weights
                    info.json    # snapshot metadata
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._index_path = self.root / "registry.json"
        self._snapshots_dir = self.root / "snapshots"

        # Ensure directories exist
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index: dict[str, SnapshotInfo] = {}
        if self._index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load the registry index from disk."""
        with open(self._index_path, encoding="utf-8") as f:
            data = json.load(f)
        self._index = {
            name: SnapshotInfo.from_dict(info) for name, info in data.get("snapshots", {}).items()
        }

    def _save_index(self) -> None:
        """Save the registry index to disk."""
        data = {"snapshots": {name: info.to_dict() for name, info in self._index.items()}}
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        artifact_path: str | Path,
        metadata: dict[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ) -> SnapshotInfo:
        """
        Register a model snapshot.

        Args:
            name: Unique name for this snapshot.
            artifact_path: Path to the model weights file (.pt).
            metadata: Optional metadata dict.
            overwrite: If True, overwrite existing snapshot with same name.

        Returns:
            SnapshotInfo for the registered snapshot.

        Raises:
            ValueError: If name already exists and overwrite=False.
        """
        if name in self._index and not overwrite:
            raise ValueError(f"Snapshot {name!r} already exists. Use overwrite=True to replace.")

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Create snapshot directory
        snapshot_dir = self._snapshots_dir / name
        if snapshot_dir.exists() and overwrite:
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        dest_artifact = snapshot_dir / "model.pt"
        shutil.copy2(artifact_path, dest_artifact)

        # Create snapshot info
        info = SnapshotInfo(
            name=name,
            artifact_path=str(dest_artifact.relative_to(self.root)),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        # Save info to snapshot directory
        info_path = snapshot_dir / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info.to_dict(), f, indent=2)

        # Update index
        self._index[name] = info
        self._save_index()

        return info

    def list_snapshots(self) -> list[SnapshotInfo]:
        """List all registered snapshots, sorted by creation time (newest first)."""
        snapshots = list(self._index.values())
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots

    def get(self, name: str) -> SnapshotInfo | None:
        """Get snapshot info by name."""
        return self._index.get(name)

    def get_artifact_path(self, name: str) -> Path | None:
        """Get the full path to a snapshot's model weights."""
        info = self._index.get(name)
        if info is None:
            return None
        return self.root / info.artifact_path

    def load_policy(self, name: str, device: str = "cpu") -> SmallNetPolicy:
        """
        Load a SmallNetPolicy from a snapshot.

        Args:
            name: Snapshot name.
            device: Device to load the model onto.

        Returns:
            SmallNetPolicy with loaded weights.

        Raises:
            KeyError: If snapshot not found.
        """
        # Import here to avoid circular imports
        from chesster.policies.net import SmallNetPolicy

        artifact_path = self.get_artifact_path(name)
        if artifact_path is None:
            raise KeyError(f"Snapshot not found: {name!r}")

        policy = SmallNetPolicy(device=device)
        policy.load(str(artifact_path))
        return policy

    def delete(self, name: str) -> bool:
        """
        Delete a snapshot.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self._index:
            return False

        # Remove from disk
        snapshot_dir = self._snapshots_dir / name
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        # Update index
        del self._index[name]
        self._save_index()

        return True

    def update_elo(
        self,
        name: str,
        new_elo: float,
        opponent: str,
        result: str,
        games_played: int,
    ) -> SnapshotInfo:
        """
        Update the ELO rating for a snapshot.

        Args:
            name: Snapshot name.
            new_elo: New ELO rating.
            opponent: Name of the opponent.
            result: Result description (e.g., "3-2-1" for 3W-2D-1L).
            games_played: Number of games in this match.

        Returns:
            Updated SnapshotInfo.

        Raises:
            KeyError: If snapshot not found.
        """
        info = self._index.get(name)
        if info is None:
            raise KeyError(f"Snapshot not found: {name!r}")

        # Create history entry
        history_entry = EloHistoryEntry(
            elo=new_elo,
            opponent=opponent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            games_played=games_played,
            result=result,
        )

        # Update the snapshot info
        info.elo_history.append(history_entry.to_dict())
        info.elo = new_elo
        info.games_played += games_played

        # Save to disk
        self._save_index()

        # Also update the individual snapshot info.json
        snapshot_dir = self._snapshots_dir / name
        info_path = snapshot_dir / "info.json"
        if info_path.exists():
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info.to_dict(), f, indent=2)

        return info

    def get_leaderboard(self) -> list[SnapshotInfo]:
        """
        Get all snapshots sorted by ELO rating (highest first).

        Returns:
            List of SnapshotInfo sorted by ELO descending.
        """
        snapshots = list(self._index.values())
        snapshots.sort(key=lambda s: s.elo, reverse=True)
        return snapshots
