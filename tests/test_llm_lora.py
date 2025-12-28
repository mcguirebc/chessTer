"""Tests for LoRA LLM policy."""
import tempfile
from pathlib import Path

import chess
import pytest

from chesster.policies.base import ChooseMoveParams


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from chesster.policies.llm_lora import LoRAConfig

        config = LoRAConfig()

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.bias == "none"

    def test_custom_config(self):
        """Custom config values should be respected."""
        from chesster.policies.llm_lora import LoRAConfig

        config = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1


class TestMoveWithLogProb:
    """Tests for MoveWithLogProb dataclass."""

    def test_creation(self):
        """Should create MoveWithLogProb with all fields."""
        import torch
        from chesster.policies.llm_lora import MoveWithLogProb

        result = MoveWithLogProb(
            uci="e2e4",
            san="e4",
            log_prob=torch.tensor(-1.5),
            info={"model": "test"},
        )

        assert result.uci == "e2e4"
        assert result.san == "e4"
        assert result.log_prob.item() == pytest.approx(-1.5)
        assert result.info["model"] == "test"


class TestLoRALLMPolicyUnit:
    """Unit tests for LoRALLMPolicy (no model loading)."""

    @pytest.fixture
    def policy_class(self):
        """Get policy class without instantiating."""
        from chesster.policies.llm_lora import LoRALLMPolicy

        return LoRALLMPolicy

    def test_policy_id(self, policy_class):
        """Policy should have correct ID."""
        assert policy_class.policy_id == "llm_lora"

    def test_lazy_loading(self, policy_class):
        """Model should not load until needed."""
        policy = policy_class(model_name="test-model")

        assert policy._loaded is False
        assert policy._model is None
        assert policy._tokenizer is None

    def test_prompt_building(self, policy_class):
        """Should build prompts correctly."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"
        policy._tokenizer = None  # No tokenizer in unit tests

        board = chess.Board()
        legal_moves = ["e2e4", "d2d4"]
        params = ChooseMoveParams()

        prompt = policy._build_prompt(board, legal_moves, params)

        # Simple prompt: "Pick from list"
        assert "e2e4" in prompt
        assert "d2d4" in prompt
        assert "Moves:" in prompt

    def test_prompt_with_elo_context(self, policy_class):
        """Should build valid prompt even with ELO context (context not used in simple prompt)."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"
        policy._tokenizer = None  # No tokenizer in unit tests

        board = chess.Board()
        legal_moves = ["e2e4"]
        params = ChooseMoveParams(
            opponent_elo=1200.0,
            self_elo=1500.0,
            opponent_is_bot=True,
        )

        prompt = policy._build_prompt(board, legal_moves, params)

        # Simple prompt format (ELO not used in simple version)
        assert "e2e4" in prompt
        assert "Moves:" in prompt

    def test_move_extraction_valid(self, policy_class):
        """Should extract valid UCI moves."""
        policy = policy_class.__new__(policy_class)

        board = chess.Board()
        result = policy._extract_move("The best move is e2e4.", board)

        assert result == "e2e4"

    def test_move_extraction_invalid(self, policy_class):
        """Should return None for invalid output."""
        policy = policy_class.__new__(policy_class)

        board = chess.Board()
        result = policy._extract_move("I don't know", board)

        assert result is None

    def test_move_extraction_illegal_move(self, policy_class):
        """Should skip illegal moves."""
        policy = policy_class.__new__(policy_class)

        board = chess.Board()
        # e1e2 is illegal, e2e4 is legal
        result = policy._extract_move("Try e1e2 or e2e4", board)

        assert result == "e2e4"


class TestDeviceSelection:
    """Tests for device auto-selection."""

    def test_explicit_device(self):
        """Should use explicit device."""
        from chesster.policies.llm_lora import _get_device

        assert _get_device("cuda") == "cuda"
        assert _get_device("cpu") == "cpu"
        assert _get_device("mps") == "mps"

    def test_auto_device(self):
        """Should auto-detect device."""
        from chesster.policies.llm_lora import _get_device

        device = _get_device(None)
        assert device in ("cuda", "mps", "cpu")


# Integration tests that require model download
@pytest.mark.slow
class TestLoRALLMPolicyIntegration:
    """Integration tests requiring model download.

    Run with: pytest -m slow
    """

    @pytest.mark.skip(reason="Requires model download (~3GB)")
    def test_forward_returns_log_prob(self):
        """forward() should return MoveWithLogProb."""
        from chesster.policies.llm_lora import LoRALLMPolicy

        policy = LoRALLMPolicy(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cpu",
        )

        board = chess.Board()
        params = ChooseMoveParams(temperature=0.1)

        result = policy.forward(board, params)

        assert result.uci in [m.uci() for m in board.legal_moves]
        assert hasattr(result, "log_prob")
        assert result.log_prob.dim() == 0  # Scalar tensor

        policy.unload()

    @pytest.mark.skip(reason="Requires model download (~3GB)")
    def test_only_lora_params_trainable(self):
        """Only LoRA parameters should be trainable."""
        from chesster.policies.llm_lora import LoRALLMPolicy

        policy = LoRALLMPolicy(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cpu",
        )
        policy._ensure_loaded()

        trainable = policy.trainable_parameters()
        total = policy.total_parameters()

        # LoRA should be <1% of total
        ratio = trainable / total
        assert ratio < 0.01

        policy.unload()

    @pytest.mark.skip(reason="Requires model download (~3GB)")
    def test_save_load_adapter(self):
        """Should save and load LoRA adapters."""
        from chesster.policies.llm_lora import LoRALLMPolicy

        policy = LoRALLMPolicy(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cpu",
        )
        policy._ensure_loaded()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            policy.save_adapter(adapter_path)

            # Verify files exist
            assert adapter_path.exists()
            assert (adapter_path / "adapter_config.json").exists()

            # Load adapter
            policy.load_adapter(adapter_path)

        policy.unload()
