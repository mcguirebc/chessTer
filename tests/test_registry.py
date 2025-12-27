"""Tests for ModelRegistry with ELO tracking."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from chesster.league.elo import DEFAULT_ELO
from chesster.league.registry import EloHistoryEntry, ModelRegistry, SnapshotInfo


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ModelRegistry(tmpdir)


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"test": "data"}, f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestSnapshotInfoEloDefaults:
    """Tests for SnapshotInfo ELO default values."""

    def test_default_elo(self):
        """New SnapshotInfo should have default ELO."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
        )
        assert info.elo == DEFAULT_ELO

    def test_default_elo_history(self):
        """New SnapshotInfo should have empty ELO history."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
        )
        assert info.elo_history == []

    def test_default_games_played(self):
        """New SnapshotInfo should have 0 games played."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
        )
        assert info.games_played == 0

    def test_default_is_bot(self):
        """New SnapshotInfo should be marked as bot."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
        )
        assert info.is_bot is True

    def test_custom_elo(self):
        """SnapshotInfo should accept custom ELO."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
            elo=1500.0,
        )
        assert info.elo == 1500.0


class TestSnapshotInfoBackwardCompat:
    """Tests for backward compatibility with old snapshots."""

    def test_from_dict_without_elo_fields(self):
        """Old snapshots without ELO fields should get defaults."""
        old_data = {
            "name": "old_model",
            "artifact_path": "old/model.pt",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": {"version": 1},
        }
        info = SnapshotInfo.from_dict(old_data)

        assert info.name == "old_model"
        assert info.elo == DEFAULT_ELO
        assert info.elo_history == []
        assert info.games_played == 0
        assert info.is_bot is True

    def test_from_dict_with_partial_elo_fields(self):
        """Snapshots with some ELO fields should get defaults for missing ones."""
        partial_data = {
            "name": "partial_model",
            "artifact_path": "partial/model.pt",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": {},
            "elo": 1400.0,
            # missing: elo_history, games_played, is_bot
        }
        info = SnapshotInfo.from_dict(partial_data)

        assert info.elo == 1400.0
        assert info.elo_history == []
        assert info.games_played == 0
        assert info.is_bot is True

    def test_to_dict_includes_elo_fields(self):
        """to_dict should include all ELO fields."""
        info = SnapshotInfo(
            name="test",
            artifact_path="test/model.pt",
            created_at="2024-01-01T00:00:00Z",
            elo=1350.0,
            elo_history=[{"elo": 1350.0, "opponent": "random"}],
            games_played=10,
            is_bot=True,
        )
        d = info.to_dict()

        assert d["elo"] == 1350.0
        assert d["elo_history"] == [{"elo": 1350.0, "opponent": "random"}]
        assert d["games_played"] == 10
        assert d["is_bot"] is True


class TestEloHistoryEntry:
    """Tests for EloHistoryEntry dataclass."""

    def test_to_dict(self):
        """EloHistoryEntry should serialize to dict."""
        entry = EloHistoryEntry(
            elo=1250.0,
            opponent="stockfish",
            timestamp="2024-01-01T00:00:00Z",
            games_played=10,
            result="3-4-3",
        )
        d = entry.to_dict()

        assert d["elo"] == 1250.0
        assert d["opponent"] == "stockfish"
        assert d["games_played"] == 10
        assert d["result"] == "3-4-3"

    def test_from_dict(self):
        """EloHistoryEntry should deserialize from dict."""
        d = {
            "elo": 1300.0,
            "opponent": "random",
            "timestamp": "2024-01-01T00:00:00Z",
            "games_played": 5,
            "result": "5-0-0",
        }
        entry = EloHistoryEntry.from_dict(d)

        assert entry.elo == 1300.0
        assert entry.opponent == "random"
        assert entry.games_played == 5
        assert entry.result == "5-0-0"


class TestRegistryUpdateElo:
    """Tests for ModelRegistry.update_elo method."""

    def test_update_elo_changes_rating(self, temp_registry, temp_model_file):
        """update_elo should change the snapshot's ELO."""
        temp_registry.register("model_v1", temp_model_file)

        updated = temp_registry.update_elo(
            name="model_v1",
            new_elo=1350.0,
            opponent="stockfish",
            result="5-3-2",
            games_played=10,
        )

        assert updated.elo == 1350.0

    def test_update_elo_records_history(self, temp_registry, temp_model_file):
        """update_elo should add entry to elo_history."""
        temp_registry.register("model_v1", temp_model_file)

        temp_registry.update_elo(
            name="model_v1",
            new_elo=1350.0,
            opponent="stockfish",
            result="5-3-2",
            games_played=10,
        )

        info = temp_registry.get("model_v1")
        assert len(info.elo_history) == 1
        assert info.elo_history[0]["elo"] == 1350.0
        assert info.elo_history[0]["opponent"] == "stockfish"
        assert info.elo_history[0]["result"] == "5-3-2"
        assert info.elo_history[0]["games_played"] == 10

    def test_update_elo_accumulates_games(self, temp_registry, temp_model_file):
        """Multiple updates should accumulate games_played."""
        temp_registry.register("model_v1", temp_model_file)

        temp_registry.update_elo("model_v1", 1250.0, "random", "5-0-0", 5)
        temp_registry.update_elo("model_v1", 1300.0, "stockfish", "2-3-5", 10)

        info = temp_registry.get("model_v1")
        assert info.games_played == 15
        assert len(info.elo_history) == 2

    def test_update_elo_persists(self, temp_registry, temp_model_file):
        """ELO updates should persist across registry reloads."""
        temp_registry.register("model_v1", temp_model_file)
        temp_registry.update_elo("model_v1", 1400.0, "stockfish", "6-2-2", 10)

        # Create new registry instance from same root
        reloaded = ModelRegistry(temp_registry.root)
        info = reloaded.get("model_v1")

        assert info.elo == 1400.0
        assert info.games_played == 10
        assert len(info.elo_history) == 1

    def test_update_elo_not_found(self, temp_registry):
        """update_elo should raise KeyError for missing snapshot."""
        with pytest.raises(KeyError, match="not found"):
            temp_registry.update_elo("nonexistent", 1200.0, "random", "0-0-0", 0)


class TestRegistryGetLeaderboard:
    """Tests for ModelRegistry.get_leaderboard method."""

    def test_leaderboard_empty_registry(self, temp_registry):
        """Empty registry should return empty leaderboard."""
        leaderboard = temp_registry.get_leaderboard()
        assert leaderboard == []

    def test_leaderboard_single_model(self, temp_registry, temp_model_file):
        """Single model should be returned in leaderboard."""
        temp_registry.register("model_v1", temp_model_file)

        leaderboard = temp_registry.get_leaderboard()
        assert len(leaderboard) == 1
        assert leaderboard[0].name == "model_v1"

    def test_leaderboard_sorted_by_elo(self, temp_registry, temp_model_file):
        """Leaderboard should be sorted by ELO descending."""
        # Register multiple models
        temp_registry.register("low_elo", temp_model_file, overwrite=True)
        temp_registry.register("high_elo", temp_model_file, overwrite=True)
        temp_registry.register("mid_elo", temp_model_file, overwrite=True)

        # Update their ELOs
        temp_registry.update_elo("low_elo", 1000.0, "test", "0-0-1", 1)
        temp_registry.update_elo("high_elo", 1500.0, "test", "1-0-0", 1)
        temp_registry.update_elo("mid_elo", 1250.0, "test", "0-1-0", 1)

        leaderboard = temp_registry.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0].name == "high_elo"
        assert leaderboard[0].elo == 1500.0
        assert leaderboard[1].name == "mid_elo"
        assert leaderboard[1].elo == 1250.0
        assert leaderboard[2].name == "low_elo"
        assert leaderboard[2].elo == 1000.0

    def test_leaderboard_includes_all_fields(self, temp_registry, temp_model_file):
        """Leaderboard entries should include all ELO fields."""
        temp_registry.register("model_v1", temp_model_file)
        temp_registry.update_elo("model_v1", 1350.0, "stockfish", "5-2-3", 10)

        leaderboard = temp_registry.get_leaderboard()
        entry = leaderboard[0]

        assert entry.elo == 1350.0
        assert entry.games_played == 10
        assert entry.is_bot is True
        assert len(entry.elo_history) == 1


class TestRegistryRegisterWithElo:
    """Tests for ModelRegistry.register preserving default ELO fields."""

    def test_register_sets_default_elo(self, temp_registry, temp_model_file):
        """Newly registered models should have default ELO."""
        info = temp_registry.register("new_model", temp_model_file)

        assert info.elo == DEFAULT_ELO
        assert info.elo_history == []
        assert info.games_played == 0
        assert info.is_bot is True

    def test_register_persists_elo_to_disk(self, temp_registry, temp_model_file):
        """Registered model's ELO should be persisted to info.json."""
        temp_registry.register("new_model", temp_model_file)

        info_path = temp_registry.root / "snapshots" / "new_model" / "info.json"
        with open(info_path) as f:
            data = json.load(f)

        assert data["elo"] == DEFAULT_ELO
        assert data["is_bot"] is True
