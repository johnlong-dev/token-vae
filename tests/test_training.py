"""Tests for training utilities."""

from token_vae.training.config import TrainingConfig


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.vocab_size == 8000
        assert config.d_model == 256
        assert config.batch_size == 256
        assert config.epochs == 10
        assert config.kl_weight_final == 0.01

    def test_kl_annealing(self):
        """Test KL weight annealing."""
        config = TrainingConfig(kl_weight_final=0.01, kl_warmup_fraction=0.4)

        # At step 0, weight should be 0
        assert config.get_kl_weight(0, 1000) == 0.0

        # At 20% through warmup, weight should be 50% of final
        assert abs(config.get_kl_weight(200, 1000) - 0.005) < 1e-6

        # At 40% (end of warmup), weight should be final
        assert abs(config.get_kl_weight(400, 1000) - 0.01) < 1e-6

        # After warmup, weight should stay at final
        assert config.get_kl_weight(500, 1000) == 0.01
        assert config.get_kl_weight(1000, 1000) == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            vocab_size=1000,
            d_model=128,
            epochs=5,
        )

        assert config.vocab_size == 1000
        assert config.d_model == 128
        assert config.epochs == 5

    def test_kl_annealing_zero_warmup(self):
        """Zero warmup should immediately return final KL weight."""
        config = TrainingConfig(kl_weight_final=0.02, kl_warmup_fraction=0.0)
        assert config.get_kl_weight(0, 1000) == 0.02
