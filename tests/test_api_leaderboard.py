"""Tests for the /v1/leaderboard API endpoint."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from chesster.api.app import app
from chesster.league.registry import ModelRegistry


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_registry():
    """Create a temporary registry with some models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(tmpdir)

        # Create a dummy model file
        model_path = Path(tmpdir) / "dummy_model.pt"
        torch.save({"test": "data"}, model_path)

        # Register models with different ELOs
        registry.register("model_low", model_path, overwrite=True)
        registry.register("model_mid", model_path, overwrite=True)
        registry.register("model_high", model_path, overwrite=True)

        # Update their ELOs
        registry.update_elo("model_low", 900.0, "random", "2-0-8", 10)
        registry.update_elo("model_mid", 1200.0, "stockfish", "5-0-5", 10)
        registry.update_elo("model_high", 1500.0, "stockfish", "7-1-2", 10)

        yield tmpdir


class TestLeaderboardEndpoint:
    """Tests for GET /v1/leaderboard."""

    def test_leaderboard_empty_no_registry(self, client):
        """When no registry configured, should return empty list."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = None
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()
        assert data["rankings"] == []

    def test_leaderboard_empty_nonexistent_registry(self, client):
        """When registry path doesn't exist, should return empty list."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = "/nonexistent/path/registry"
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()
        assert data["rankings"] == []

    def test_leaderboard_sorted_by_elo(self, client, temp_registry):
        """Leaderboard should return models sorted by ELO descending."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = temp_registry
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()

        rankings = data["rankings"]
        assert len(rankings) == 3

        # Should be sorted by ELO descending
        assert rankings[0]["name"] == "model_high"
        assert rankings[0]["elo"] == 1500.0
        assert rankings[1]["name"] == "model_mid"
        assert rankings[1]["elo"] == 1200.0
        assert rankings[2]["name"] == "model_low"
        assert rankings[2]["elo"] == 900.0

    def test_leaderboard_includes_all_fields(self, client, temp_registry):
        """Each leaderboard entry should include all required fields."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = temp_registry
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()

        for entry in data["rankings"]:
            assert "name" in entry
            assert "elo" in entry
            assert "games_played" in entry
            assert "is_bot" in entry

    def test_leaderboard_games_played(self, client, temp_registry):
        """Leaderboard should correctly report games played."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = temp_registry
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()

        # Each model played 10 games
        for entry in data["rankings"]:
            assert entry["games_played"] == 10

    def test_leaderboard_is_bot_default_true(self, client, temp_registry):
        """All registered models should be marked as bots by default."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = temp_registry
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()

        for entry in data["rankings"]:
            assert entry["is_bot"] is True


class TestLeaderboardResponseSchema:
    """Tests for LeaderboardResponse schema validation."""

    def test_response_schema_validation(self, client, temp_registry):
        """Response should match the expected schema."""
        with patch("chesster.api.app.get_settings") as mock_settings:
            mock_settings.return_value.registry_path = temp_registry
            response = client.get("/v1/leaderboard")

        assert response.status_code == 200
        data = response.json()

        # Top level should have rankings key
        assert "rankings" in data
        assert isinstance(data["rankings"], list)

        # Each entry should have correct types
        for entry in data["rankings"]:
            assert isinstance(entry["name"], str)
            assert isinstance(entry["elo"], (int, float))
            assert isinstance(entry["games_played"], int)
            assert isinstance(entry["is_bot"], bool)
