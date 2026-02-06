"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for Token VAE training."""

    # Model architecture
    vocab_size: int = 8000
    d_model: int = 256
    d_embed: int = 512

    # Training parameters
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    identity_vocab_repeats: int = 0

    # Loss weights
    kl_weight_final: float = 0.01
    skipgram_weight: float = 0.1
    interpolation_weight: float = 0.0
    interpolation_pairs: int = 32
    interpolation_entropy_target: float = 8.0
    interpolation_holdout_pairs_path: str | None = None
    interpolation_pair_max_resample: int = 10
    interpolation_track_pairs: bool = True

    # Inverse frequency weighting for reconstruction loss
    ifw_alpha: float = 0.0  # 0 = disabled (uniform), 1 = full inverse frequency

    # Prior diversity regularizer
    prior_reg_weight: float = 0.0
    prior_reg_samples: int = 512
    prior_reg_interval: int = 10
    prior_entropy_target: float = 3.0
    prior_max_freq_target: float = 0.05
    prior_max_freq_weight: float = 1.0
    prior_marginal_entropy_target: float = 0.0
    prior_marginal_entropy_weight: float = 1.0
    prior_hhi_target: float = 0.0  # 0 = disabled
    prior_hhi_weight: float = 1.0

    # KL annealing and free-bits
    kl_warmup_fraction: float = 0.4  # Anneal KL weight over first 40% of training
    kl_free_bits: float = 0.0  # Per-dimension KL floor (0 = disabled)

    # Co-occurrence parameters
    window_size: int = 5
    num_negatives: int = 5

    # Paths
    data_file: str = "data/fineweb-4M.txt"
    tokenizer_path: str = "artifacts/tokenizer/fw-4M-v16k"
    model_save_path: str = "artifacts/models/token_vae.pt"

    # Logging
    log_interval: int = 100  # Log every N batches
    save_interval: int = 1  # Save every N epochs

    # Reproducibility
    seed: int | None = None

    # Device (auto-detected)
    device: str = "auto"

    def __post_init__(self):
        """Ensure paths exist."""
        Path(self.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)

    def get_kl_weight(self, step: int, total_steps: int) -> float:
        """Get KL weight with linear annealing.

        Args:
            step: Current training step
            total_steps: Total training steps

        Returns:
            Current KL weight
        """
        warmup_steps = int(total_steps * self.kl_warmup_fraction)
        if warmup_steps <= 0:
            return self.kl_weight_final

        if step >= warmup_steps:
            return self.kl_weight_final

        # Linear annealing from 0 to kl_weight_final
        return self.kl_weight_final * (step / warmup_steps)
