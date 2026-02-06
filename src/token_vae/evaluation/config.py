"""Configuration for Token VAE evaluation."""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation tests."""

    # Prior sampling
    prior_num_samples: int = 10_000
    prior_batch_size: int = 512
    prior_top_k: int = 20

    # Interpolation
    interp_num_pairs: int = 50
    interp_num_steps: int = 20
    interp_max_js_threshold: float = 0.5  # JS distance (sqrt(JS divergence))
    interp_max_entropy_threshold: float = 8.5
    interp_top_k_cases: int = 5
    interp_max_js_top_k: int = 10
    entropy_vs_prior_delta_threshold: float = 1.0

    # Held-out interpolation pairs
    interp_holdout_pairs_path: str | None = None
    interp_holdout_num_pairs: int = 0
    interp_holdout_seed: int = 123
    interp_holdout_eval_pairs: int | None = None

    # Stratification
    stratify: bool = True
    common_frac: float = 0.2
    rare_frac: float = 0.2
    distance_quantiles: tuple[float, float] = (0.33, 0.66)

    # Off-manifold sweep
    off_manifold_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0)
    off_manifold_num_tokens: int = 200
    off_manifold_samples_per_token: int = 1

    # Integrity checks
    integrity_checks: bool = True
    integrity_prior_unique_min: float = 0.20
    integrity_prior_entropy_min: float = 1.0
    integrity_recon_high: float = 0.95

    # Diffusion walk test
    diffusion_walk_enabled: bool = True
    diffusion_num_walks: int = 100
    diffusion_num_steps: int = 50
    diffusion_beta: float = 0.01
    diffusion_beta_schedule: str = "constant"
    diffusion_non_special_threshold: float = 0.95
    diffusion_entropy_threshold: float = 8.0
    diffusion_min_unique_fraction_threshold: float = 0.10
    diffusion_min_mean_change_rate_threshold: float = 0.01
    diffusion_start_from: str = "prior"

    # Token frequency data
    data_file: str | None = None
    data_max_tokens: int | None = None

    # Evaluation reproducibility
    seed: int | None = 42
