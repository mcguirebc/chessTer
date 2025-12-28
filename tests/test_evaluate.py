"""Tests for the evaluation framework."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chesster.eval import (
    FULL_SUITE,
    QUICK_SUITE,
    STANDARD_SUITE,
    BaselineSuite,
    EvaluationResult,
    OpponentResult,
    PolicyLoadError,
    evaluate_model,
    format_compact,
    format_markdown,
    format_report,
    get_baseline_suite,
    get_reference_elo,
    get_supported_types,
    list_suites,
    load_policy,
    parse_opponent_specs,
)


# =============================================================================
# Loader Tests
# =============================================================================


class TestLoadPolicy:
    """Tests for load_policy function."""

    def test_load_random(self):
        """Should load random policy."""
        policy = load_policy("random")
        assert policy.policy_id == "random"

    def test_load_stockfish_default(self):
        """Should load Stockfish with default depth."""
        policy = load_policy("stockfish")
        assert policy.policy_id == "stockfish:10"

    def test_load_stockfish_with_depth(self):
        """Should load Stockfish with specified depth."""
        policy = load_policy("stockfish:5")
        assert policy.policy_id == "stockfish:5"

    def test_load_stockfish_high_depth(self):
        """Should load Stockfish with high depth."""
        policy = load_policy("stockfish:20")
        assert policy.policy_id == "stockfish:20"

    def test_invalid_type(self):
        """Should raise error for unknown policy type."""
        with pytest.raises(PolicyLoadError, match="Unknown policy type"):
            load_policy("invalid:something")

    def test_invalid_stockfish_depth(self):
        """Should raise error for non-integer Stockfish depth."""
        with pytest.raises(PolicyLoadError, match="Invalid Stockfish depth"):
            load_policy("stockfish:abc")

    def test_hf_missing_model_name(self):
        """Should raise error for empty HF model name."""
        with pytest.raises(PolicyLoadError, match="requires model name"):
            load_policy("hf:")

    def test_pt_missing_path(self):
        """Should raise error for empty PT path."""
        with pytest.raises(PolicyLoadError, match="requires path"):
            load_policy("pt:")

    def test_pt_nonexistent_file(self):
        """Should raise error for nonexistent checkpoint."""
        with pytest.raises(PolicyLoadError, match="Checkpoint not found"):
            load_policy("pt:/nonexistent/model.pt")

    def test_reg_missing_name(self):
        """Should raise error for empty registry name."""
        with pytest.raises(PolicyLoadError, match="requires snapshot name"):
            load_policy("reg:")

    def test_reg_nonexistent_registry(self):
        """Should raise error for nonexistent registry."""
        with pytest.raises(PolicyLoadError, match="Registry not found"):
            load_policy("reg:test", registry_path="/nonexistent/registry")


class TestGetSupportedTypes:
    """Tests for get_supported_types."""

    def test_returns_list(self):
        """Should return list of supported types."""
        types = get_supported_types()
        assert isinstance(types, list)
        assert "hf" in types
        assert "pt" in types
        assert "reg" in types
        assert "stockfish" in types
        assert "random" in types


class TestParseOpponentSpecs:
    """Tests for parse_opponent_specs."""

    def test_single_spec(self):
        """Should parse single spec."""
        specs = parse_opponent_specs("stockfish:5")
        assert specs == ["stockfish:5"]

    def test_multiple_specs(self):
        """Should parse comma-separated specs."""
        specs = parse_opponent_specs("stockfish:1,stockfish:5,random")
        assert specs == ["stockfish:1", "stockfish:5", "random"]

    def test_with_whitespace(self):
        """Should handle whitespace in specs."""
        specs = parse_opponent_specs("  stockfish:1 , stockfish:5 , random  ")
        assert specs == ["stockfish:1", "stockfish:5", "random"]

    def test_empty_string(self):
        """Should return empty list for empty string."""
        specs = parse_opponent_specs("")
        assert specs == []


# =============================================================================
# Baseline Tests
# =============================================================================


class TestBaselineSuites:
    """Tests for baseline suite definitions."""

    def test_quick_suite(self):
        """Quick suite should have 2 opponents."""
        assert QUICK_SUITE.name == "quick"
        assert len(QUICK_SUITE.opponent_specs) == 2
        assert "random" in QUICK_SUITE.opponent_specs
        assert "stockfish:1" in QUICK_SUITE.opponent_specs

    def test_standard_suite(self):
        """Standard suite should have 4 opponents."""
        assert STANDARD_SUITE.name == "standard"
        assert len(STANDARD_SUITE.opponent_specs) == 4
        assert "random" in STANDARD_SUITE.opponent_specs

    def test_full_suite(self):
        """Full suite should have 5+ opponents."""
        assert FULL_SUITE.name == "full"
        assert len(FULL_SUITE.opponent_specs) >= 5

    def test_get_baseline_suite(self):
        """Should get suite by name."""
        suite = get_baseline_suite("quick")
        assert suite == QUICK_SUITE

    def test_get_baseline_suite_invalid(self):
        """Should raise error for unknown suite."""
        with pytest.raises(ValueError, match="Unknown suite"):
            get_baseline_suite("invalid")

    def test_list_suites(self):
        """Should list all available suites."""
        suites = list_suites()
        assert "quick" in suites
        assert "standard" in suites
        assert "full" in suites


class TestGetReferenceElo:
    """Tests for get_reference_elo."""

    def test_random_elo(self):
        """Random should have low ELO."""
        assert get_reference_elo("random") == 800.0

    def test_stockfish_depth_1(self):
        """Stockfish depth 1 should have moderate ELO."""
        assert get_reference_elo("stockfish:1") == 1200.0

    def test_stockfish_depth_5(self):
        """Stockfish depth 5 should have higher ELO."""
        assert get_reference_elo("stockfish:5") == 1900.0

    def test_stockfish_depth_10(self):
        """Stockfish depth 10 should have high ELO."""
        assert get_reference_elo("stockfish:10") == 2200.0

    def test_unknown_opponent(self):
        """Unknown opponent should get default ELO."""
        assert get_reference_elo("unknown:policy") == 1200.0


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = EvaluationResult(
            model_id="test",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
        )
        d = result.to_dict()
        assert d["model_id"] == "test"
        assert d["total_games"] == 10
        assert d["estimated_elo"] == 1300.0

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {
            "model_id": "test",
            "suite_name": "quick",
            "timestamp": "2024-01-01T00:00:00",
            "total_games": 10,
            "total_wins": 5,
            "total_draws": 3,
            "total_losses": 2,
            "score_rate": 0.65,
            "win_rate": 0.5,
            "estimated_elo": 1300.0,
            "opponents": {},
        }
        result = EvaluationResult.from_dict(d)
        assert result.model_id == "test"
        assert result.total_games == 10

    def test_roundtrip(self):
        """Should survive to_dict/from_dict roundtrip."""
        original = EvaluationResult(
            model_id="test",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
        )
        restored = EvaluationResult.from_dict(original.to_dict())
        assert restored.model_id == original.model_id
        assert restored.total_games == original.total_games
        assert restored.estimated_elo == original.estimated_elo


class TestOpponentResult:
    """Tests for OpponentResult dataclass."""

    def test_creation(self):
        """Should create opponent result."""
        result = OpponentResult(
            opponent_id="stockfish:5",
            games=10,
            wins=2,
            draws=1,
            losses=7,
            score_rate=0.25,
            win_rate=0.2,
            reference_elo=1900.0,
        )
        assert result.opponent_id == "stockfish:5"
        assert result.games == 10


# =============================================================================
# Report Formatting Tests
# =============================================================================


class TestFormatReport:
    """Tests for format_report function."""

    def test_basic_report(self):
        """Should format basic report."""
        result = EvaluationResult(
            model_id="test:model",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
            vs_random_score=0.8,
        )
        report = format_report(result)
        assert "test:model" in report
        assert "quick" in report
        assert "1300" in report
        assert "80.0%" in report  # vs_random_score

    def test_contains_opponent_table(self):
        """Should contain opponent results table."""
        result = EvaluationResult(
            model_id="test",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
            opponents={
                "random": OpponentResult(
                    opponent_id="random",
                    games=5,
                    wins=4,
                    draws=1,
                    losses=0,
                    score_rate=0.9,
                    win_rate=0.8,
                    reference_elo=800.0,
                ),
            },
        )
        report = format_report(result)
        assert "RESULTS BY OPPONENT" in report
        assert "random" in report


class TestFormatCompact:
    """Tests for format_compact function."""

    def test_compact_format(self):
        """Should produce single-line format."""
        result = EvaluationResult(
            model_id="test",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
        )
        compact = format_compact(result)
        assert "ELO=1300" in compact
        assert "W/D/L=5/3/2" in compact
        assert "\n" not in compact


class TestFormatMarkdown:
    """Tests for format_markdown function."""

    def test_markdown_format(self):
        """Should produce markdown format."""
        result = EvaluationResult(
            model_id="test:model",
            suite_name="quick",
            timestamp="2024-01-01T00:00:00",
            total_games=10,
            total_wins=5,
            total_draws=3,
            total_losses=2,
            score_rate=0.65,
            win_rate=0.5,
            estimated_elo=1300.0,
        )
        md = format_markdown(result)
        assert "## Evaluation: test:model" in md
        assert "**Suite:** quick" in md
        assert "| Opponent |" in md


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluateModel:
    """Integration tests for evaluate_model."""

    def test_evaluate_random_vs_random(self):
        """Should evaluate random policy against itself."""
        result = evaluate_model(
            model="random",
            opponents=["random"],
            games_per_opponent=2,
            verbose=False,
        )
        assert result.model_id == "random"
        assert result.total_games == 2

    def test_evaluate_stockfish_vs_random(self):
        """Should evaluate Stockfish against random."""
        result = evaluate_model(
            model="stockfish:1",
            opponents=["random"],
            games_per_opponent=2,
            base_seed=42,
            verbose=False,
        )
        assert result.model_id == "stockfish:1"
        assert result.total_games == 2
        # Stockfish should beat random
        assert result.score_rate >= 0.5

    def test_evaluate_with_suite(self):
        """Should use baseline suite."""
        result = evaluate_model(
            model="stockfish:1",
            suite="quick",
            games_per_opponent=1,
            verbose=False,
        )
        assert result.suite_name == "quick"
        # Quick suite has 2 opponents
        assert result.total_games == 2

    def test_vs_random_score_tracked(self):
        """Should track vs_random_score."""
        result = evaluate_model(
            model="stockfish:1",
            opponents=["random"],
            games_per_opponent=2,
            verbose=False,
        )
        assert result.vs_random_score is not None

    def test_custom_opponents(self):
        """Should accept custom opponents list."""
        result = evaluate_model(
            model="stockfish:1",
            opponents=["stockfish:1", "random"],
            games_per_opponent=1,
            verbose=False,
        )
        assert result.total_games == 2
        assert "random" in result.opponents

    def test_result_serializable(self):
        """Result should be JSON serializable."""
        result = evaluate_model(
            model="stockfish:1",
            opponents=["random"],
            games_per_opponent=1,
            verbose=False,
        )
        json_str = json.dumps(result.to_dict())
        assert json_str is not None
        # Should roundtrip
        restored = EvaluationResult.from_dict(json.loads(json_str))
        assert restored.model_id == result.model_id


class TestEvaluateModelPolicyInstance:
    """Tests for evaluate_model with Policy instances."""

    def test_accept_policy_instance(self):
        """Should accept Policy instance directly."""
        policy = load_policy("stockfish:1")
        result = evaluate_model(
            model=policy,
            opponents=["random"],
            games_per_opponent=1,
            verbose=False,
        )
        assert result.total_games == 1


@pytest.mark.slow
class TestEvaluateModelSlow:
    """Slow integration tests (marked for optional running)."""

    def test_full_quick_suite(self):
        """Test full quick suite evaluation."""
        result = evaluate_model(
            model="stockfish:3",
            suite="quick",
            games_per_opponent=5,
            verbose=True,
        )
        assert result.total_games == 10
        # Stockfish 3 should do well
        assert result.estimated_elo > 1000


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for CLI functionality."""

    def test_main_import(self):
        """Should be able to import main."""
        from chesster.eval.evaluate import main

        assert callable(main)

    def test_cli_help(self):
        """CLI should accept --help."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "chesster.eval.evaluate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--suite" in result.stdout
