"""Tests for LLM REINFORCE trainer."""
import tempfile
from pathlib import Path

import pytest


class TestLLMRLConfig:
    """Tests for LLM RL configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from chesster.train.llm_rl import LLMRLConfig

        config = LLMRLConfig()

        assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.games_per_batch == 10
        assert config.epochs == 1
        assert config.lr == 1e-4
        assert config.gamma == 0.99
        assert config.opponent == "random"

    def test_custom_config(self):
        """Custom config values should be respected."""
        from chesster.train.llm_rl import LLMRLConfig

        config = LLMRLConfig(
            model_name="test-model",
            games_per_batch=50,
            epochs=5,
            lr=5e-5,
        )

        assert config.model_name == "test-model"
        assert config.games_per_batch == 50
        assert config.epochs == 5
        assert config.lr == 5e-5


class TestGameTrajectory:
    """Tests for GameTrajectory dataclass."""

    def test_trajectory_creation(self):
        """Should create trajectory with computed num_moves."""
        import torch
        from chesster.train.llm_rl import GameTrajectory

        traj = GameTrajectory(
            moves=["e2e4", "d2d4", "g1f3"],
            log_probs=[torch.tensor(-1.0), torch.tensor(-1.5), torch.tensor(-2.0)],
            outcome=1.0,
        )

        assert traj.num_moves == 3
        assert traj.outcome == 1.0
        assert len(traj.moves) == len(traj.log_probs)


class TestDiscountedReturns:
    """Tests for discounted return computation."""

    def test_win_as_white(self):
        """White win should give positive returns."""
        from chesster.train.llm_rl import compute_discounted_returns

        returns = compute_discounted_returns(
            outcome=1.0,  # White wins
            num_moves=5,
            gamma=0.99,
            white_played=True,
        )

        assert len(returns) == 5
        # All returns should be positive
        assert all(r > 0 for r in returns)
        # Later moves should have higher returns (less discounting)
        assert returns[-1] > returns[0]

    def test_loss_as_white(self):
        """White loss should give negative returns."""
        from chesster.train.llm_rl import compute_discounted_returns

        returns = compute_discounted_returns(
            outcome=-1.0,  # Black wins
            num_moves=5,
            gamma=0.99,
            white_played=True,
        )

        assert len(returns) == 5
        assert all(r < 0 for r in returns)

    def test_win_as_black(self):
        """Black win (outcome=-1) should give positive returns to Black."""
        from chesster.train.llm_rl import compute_discounted_returns

        returns = compute_discounted_returns(
            outcome=-1.0,  # Black wins
            num_moves=5,
            gamma=0.99,
            white_played=False,  # We played Black
        )

        assert len(returns) == 5
        # Our returns should be positive (we won)
        assert all(r > 0 for r in returns)

    def test_draw(self):
        """Draw should give zero returns."""
        from chesster.train.llm_rl import compute_discounted_returns

        returns = compute_discounted_returns(
            outcome=0.0,
            num_moves=5,
            gamma=0.99,
            white_played=True,
        )

        assert len(returns) == 5
        assert all(r == 0.0 for r in returns)

    def test_gamma_effect(self):
        """Higher gamma should preserve returns further back."""
        from chesster.train.llm_rl import compute_discounted_returns

        # High gamma (0.99)
        returns_high = compute_discounted_returns(
            outcome=1.0,
            num_moves=10,
            gamma=0.99,
            white_played=True,
        )

        # Low gamma (0.5)
        returns_low = compute_discounted_returns(
            outcome=1.0,
            num_moves=10,
            gamma=0.5,
            white_played=True,
        )

        # First move with high gamma should retain more value
        assert returns_high[0] > returns_low[0]


class TestTrainingStats:
    """Tests for TrainingStats dataclass."""

    def test_stats_creation(self):
        """Should create stats with all fields."""
        from chesster.train.llm_rl import TrainingStats

        stats = TrainingStats(
            epoch=1,
            games_played=10,
            avg_loss=0.5,
            avg_return=0.7,
            win_rate=0.6,
            draw_rate=0.2,
            avg_game_length=25.5,
            valid_move_rate=0.15,
            elapsed_seconds=120.0,
        )

        assert stats.epoch == 1
        assert stats.games_played == 10
        assert stats.win_rate == 0.6
        assert stats.valid_move_rate == 0.15


# Integration tests requiring model download
@pytest.mark.slow
class TestLLMRLIntegration:
    """Integration tests requiring model download.

    Run with: pytest -m slow
    """

    @pytest.mark.skip(reason="Requires model download and GPU time")
    def test_play_training_game(self):
        """Should play a training game and collect trajectory."""
        from chesster.policies.llm_lora import LoRALLMPolicy
        from chesster.train.llm_rl import LLMRLConfig, play_training_game

        config = LLMRLConfig(max_moves=20)
        policy = LoRALLMPolicy(device="cpu")

        traj = play_training_game(
            policy=policy,
            opponent_policy=None,  # Random
            config=config,
            we_play_white=True,
            seed=42,
        )

        assert traj is not None
        assert len(traj.moves) > 0
        assert len(traj.moves) == len(traj.log_probs)
        assert traj.outcome in [-1.0, 0.0, 1.0]

        policy.unload()

    @pytest.mark.skip(reason="Requires model download and GPU time")
    def test_train_single_epoch(self):
        """Should complete a single training epoch."""
        import torch
        from chesster.policies.llm_lora import LoRALLMPolicy
        from chesster.train.llm_rl import LLMRLConfig, train_epoch

        config = LLMRLConfig(games_per_batch=2, max_moves=10)
        policy = LoRALLMPolicy(device="cpu")
        policy._ensure_loaded()

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        stats = train_epoch(policy, optimizer, config, epoch=0)

        assert stats.games_played >= 1
        assert stats.elapsed_seconds > 0
        assert isinstance(stats.avg_loss, float)

        policy.unload()

    @pytest.mark.skip(reason="Requires model download and GPU time")
    def test_full_training_run(self):
        """Should complete a minimal training run."""
        import tempfile
        from chesster.train.llm_rl import LLMRLConfig, train_llm_rl

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMRLConfig(
                games_per_batch=2,
                epochs=1,
                max_moves=10,
                output_dir=tmpdir,
            )

            result = train_llm_rl(config)

            # Check outputs
            output_dir = Path(tmpdir)
            assert (output_dir / "config.json").exists()
            assert (output_dir / "training_info.json").exists()
            assert (output_dir / "final_adapter").exists()
            assert result["best_win_rate"] >= 0.0
