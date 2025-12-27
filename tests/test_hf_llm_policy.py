"""Tests for HuggingFace LLM policy."""
import chess
import pytest

from chesster.policies.base import ChooseMoveParams


class TestPromptBuilding:
    """Tests for prompt construction (no model required)."""

    @pytest.fixture
    def policy_class(self):
        """Get the policy class without loading a model."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        return HuggingFaceLLMPolicy

    def test_prompt_building_simple(self, policy_class):
        """Basic prompt should include FEN and legal moves."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"

        board = chess.Board()
        legal_moves = ["e2e4", "d2d4", "g1f3"]

        prompt = policy._build_prompt(board, legal_moves)

        assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in prompt
        assert "e2e4" in prompt
        assert "d2d4" in prompt
        assert "White" in prompt

    def test_prompt_building_black_to_move(self, policy_class):
        """Prompt should indicate when Black is to move."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"

        board = chess.Board()
        board.push_san("e4")  # Now Black's turn
        legal_moves = ["e7e5", "d7d5"]

        prompt = policy._build_prompt(board, legal_moves)

        assert "Black" in prompt

    def test_prompt_building_with_elo_context(self, policy_class):
        """Prompt should include ELO context when provided."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"

        board = chess.Board()
        legal_moves = ["e2e4"]

        prompt = policy._build_prompt(
            board,
            legal_moves,
            opponent_elo=1200.0,
            self_elo=1500.0,
            opponent_is_bot=True,
        )

        assert "1200" in prompt
        assert "1500" in prompt
        assert "bot" in prompt.lower()

    def test_prompt_building_with_human_opponent(self, policy_class):
        """Prompt should indicate human opponent when specified."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"

        board = chess.Board()
        legal_moves = ["e2e4"]

        prompt = policy._build_prompt(
            board,
            legal_moves,
            opponent_elo=1400.0,
            opponent_is_bot=False,
        )

        assert "human" in prompt.lower()

    def test_prompt_includes_all_legal_moves(self, policy_class):
        """All legal moves should be in the prompt."""
        policy = policy_class.__new__(policy_class)
        policy._model_name = "test"

        board = chess.Board()
        legal_moves = list(board.legal_moves)
        legal_uci = [m.uci() for m in legal_moves]

        prompt = policy._build_prompt(board, legal_uci)

        for move in legal_uci[:5]:  # Check first 5
            assert move in prompt


class TestMoveExtraction:
    """Tests for UCI move extraction from LLM output."""

    @pytest.fixture
    def policy_class(self):
        """Get the policy class without loading a model."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        return HuggingFaceLLMPolicy

    def test_extract_simple_move(self, policy_class):
        """Should extract simple UCI move."""
        policy = policy_class.__new__(policy_class)
        board = chess.Board()

        result = policy._extract_move("e2e4", board)
        assert result == "e2e4"

    def test_extract_move_with_noise(self, policy_class):
        """Should extract move from verbose output."""
        policy = policy_class.__new__(policy_class)
        board = chess.Board()

        result = policy._extract_move(
            "I think the best move is e2e4 because it controls the center.",
            board,
        )
        assert result == "e2e4"

    def test_extract_promotion_move(self, policy_class):
        """Should extract promotion moves."""
        policy = policy_class.__new__(policy_class)
        # Set up a position where promotion is possible
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")

        result = policy._extract_move("The best move is a7a8q for promotion.", board)
        assert result == "a7a8q"

    def test_extract_move_invalid_returns_none(self, policy_class):
        """Should return None for invalid output."""
        policy = policy_class.__new__(policy_class)
        board = chess.Board()

        result = policy._extract_move("I don't know what move to make.", board)
        assert result is None

    def test_extract_move_illegal_move_skipped(self, policy_class):
        """Should skip illegal moves and find legal ones."""
        policy = policy_class.__new__(policy_class)
        board = chess.Board()

        # e1e2 is illegal (king blocked), but e2e4 is legal
        result = policy._extract_move("Try e1e2 or e2e4", board)
        assert result == "e2e4"

    def test_extract_first_legal_move(self, policy_class):
        """Should return the first legal move found."""
        policy = policy_class.__new__(policy_class)
        board = chess.Board()

        result = policy._extract_move("Options: d2d4 or e2e4", board)
        assert result == "d2d4"  # First legal move in text


class TestPolicyInitialization:
    """Tests for policy initialization."""

    def test_policy_has_correct_id(self):
        """Policy should have correct policy_id."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        policy = HuggingFaceLLMPolicy.__new__(HuggingFaceLLMPolicy)
        assert policy.policy_id == "hf_llm"

    def test_policy_has_description(self):
        """Policy should have a description."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        policy = HuggingFaceLLMPolicy.__new__(HuggingFaceLLMPolicy)
        assert len(policy.description) > 0

    def test_lazy_loading(self):
        """Model should not load until first use."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        # This should not download/load the model
        policy = HuggingFaceLLMPolicy(model_name="test-model")
        assert policy._loaded is False
        assert policy._model is None


class TestChooseMoveParams:
    """Tests for ChooseMoveParams with ELO context."""

    def test_params_with_elo_context(self):
        """Params should accept ELO context."""
        params = ChooseMoveParams(
            opponent_elo=1200.0,
            self_elo=1500.0,
            opponent_is_bot=True,
        )

        assert params.opponent_elo == 1200.0
        assert params.self_elo == 1500.0
        assert params.opponent_is_bot is True

    def test_params_default_none(self):
        """ELO params should default to None."""
        params = ChooseMoveParams()

        assert params.opponent_elo is None
        assert params.self_elo is None
        assert params.opponent_is_bot is None

    def test_params_immutable(self):
        """Params should be immutable (frozen dataclass)."""
        params = ChooseMoveParams(opponent_elo=1200.0)

        with pytest.raises(Exception):  # FrozenInstanceError
            params.opponent_elo = 1300.0


class TestFallbackBehavior:
    """Tests for fallback behavior on invalid LLM output."""

    def test_extract_returns_none_triggers_fallback_logic(self):
        """When _extract_move returns None, fallback should be used.

        This tests the logic flow - when extraction fails, we fall back to random.
        Full integration test would require model download.
        """
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        policy = HuggingFaceLLMPolicy.__new__(HuggingFaceLLMPolicy)
        board = chess.Board()

        # This invalid output should return None
        result = policy._extract_move("gibberish with no valid moves", board)
        assert result is None

        # Verify fallback move selection works
        from chesster.chess.board import choose_fallback_legal_move

        fallback = choose_fallback_legal_move(board, seed=42)
        assert fallback.uci() in [m.uci() for m in board.legal_moves]

    def test_fallback_is_deterministic_with_seed(self):
        """Fallback should be deterministic when seed is provided."""
        from chesster.chess.board import choose_fallback_legal_move

        board = chess.Board()

        move1 = choose_fallback_legal_move(board, seed=12345)
        move2 = choose_fallback_legal_move(board, seed=12345)

        assert move1.uci() == move2.uci()


class TestDeviceSelection:
    """Tests for device auto-selection."""

    def test_get_device_explicit(self):
        """Should use explicit device when provided."""
        from chesster.policies.hf_llm import _get_device

        assert _get_device("cuda") == "cuda"
        assert _get_device("cpu") == "cpu"
        assert _get_device("mps") == "mps"

    def test_get_device_auto(self):
        """Should auto-detect device when None."""
        from chesster.policies.hf_llm import _get_device

        device = _get_device(None)
        assert device in ("cuda", "mps", "cpu")


# Mark integration tests that require model download
@pytest.mark.slow
class TestIntegration:
    """Integration tests requiring actual model download.

    Run with: pytest -m slow
    """

    @pytest.mark.skip(reason="Requires model download (~3GB)")
    def test_inference_returns_legal_move(self):
        """Full inference should return a legal move."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        policy = HuggingFaceLLMPolicy(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            device="cpu",  # Use CPU for CI
        )

        board = chess.Board()
        params = ChooseMoveParams(temperature=0.1)

        result = policy.choose_move(board, params)

        assert result.uci in [m.uci() for m in board.legal_moves]
        policy.unload()

    @pytest.mark.skip(reason="Requires model download (~3GB)")
    def test_inference_with_elo_context(self):
        """Inference with ELO context should work."""
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy

        policy = HuggingFaceLLMPolicy(device="cpu")

        board = chess.Board()
        params = ChooseMoveParams(
            opponent_elo=1200.0,
            self_elo=1800.0,
            opponent_is_bot=True,
            temperature=0.1,
        )

        result = policy.choose_move(board, params)

        assert result.uci in [m.uci() for m in board.legal_moves]
        assert result.info.get("opponent_elo") == 1200.0
        policy.unload()
