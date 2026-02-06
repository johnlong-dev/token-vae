"""Loss functions for Token VAE training."""

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from token_vae.model import LOGVAR_MAX, LOGVAR_MIN
from token_vae.utils.pairs import make_pair_id

if TYPE_CHECKING:
    from token_vae.model import TokenVAE


def compute_ifw_weights(
    token_counts: np.ndarray,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute inverse frequency weights for reconstruction loss.

    Args:
        token_counts: Per-token occurrence counts, shape (vocab_size,)
        alpha: Exponent controlling IFW strength (0 = uniform, 1 = full inverse frequency)
        eps: Numerical stability constant

    Returns:
        Weight tensor of shape (vocab_size,) with mean == 1.0
    """
    counts = torch.tensor(token_counts, dtype=torch.float32)
    freq = counts / counts.sum()
    w = (1.0 / (freq + eps)) ** alpha
    w = w / w.mean()  # normalize so mean == 1.0
    return w


def reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute reconstruction loss (cross-entropy).

    Args:
        logits: Predicted logits, shape (batch_size, vocab_size)
        targets: Target token IDs, shape (batch_size,)
        reduction: Reduction method ('mean', 'sum', 'none')
        weight: Per-class weights for inverse frequency weighting, shape (vocab_size,)

    Returns:
        Reconstruction loss
    """
    return F.cross_entropy(logits, targets, reduction=reduction, weight=weight)


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Compute KL divergence from N(μ, σ²) to N(0, I).

    KL(q||p) = 0.5 * sum(μ² + σ² - log(σ²) - 1)
             = 0.5 * sum(μ² + exp(logvar) - logvar - 1)

    Args:
        mu: Mean of approximate posterior, shape (batch_size, d_model)
        logvar: Log variance of approximate posterior, shape (batch_size, d_model)
        reduction: Reduction method ('mean', 'sum', 'none')
        free_bits: Per-dimension KL floor (0 = disabled). Each dimension
            contributes max(kl_j, free_bits) instead of kl_j.

    Returns:
        KL divergence
    """
    logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)

    # KL per dimension
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    # Free-bits: clamp each dimension to a minimum value
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over latent dimensions
    kl_per_sample = kl_per_dim.sum(dim=-1)

    if reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    else:
        return kl_per_sample


def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.01,
    recon_weight: torch.Tensor | None = None,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss.

    Total loss = L_recon + kl_weight * L_KL

    Args:
        logits: Predicted logits, shape (batch_size, vocab_size)
        targets: Target token IDs, shape (batch_size,)
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        kl_weight: Weight for KL divergence term (β in β-VAE)
        recon_weight: Per-class weights for inverse frequency weighting
        free_bits: Per-dimension KL floor (0 = disabled)

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    recon_loss = reconstruction_loss(logits, targets, weight=recon_weight)
    kl_loss = kl_divergence(mu, logvar, free_bits=free_bits)

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def skipgram_loss(
    mu_center: torch.Tensor,
    mu_context: torch.Tensor,
    mu_negatives: torch.Tensor,
) -> torch.Tensor:
    """Compute skipgram loss for semantic co-occurrence.

    Loss = -log σ(μ_center · μ_context) - Σ_k log σ(-μ_center · μ_neg_k)

    This encourages tokens that co-occur to have similar embeddings,
    while pushing apart tokens that don't co-occur.

    Args:
        mu_center: Center token means, shape (batch_size, d_model)
        mu_context: Context token means, shape (batch_size, d_model)
        mu_negatives: Negative token means, shape (batch_size, num_negatives, d_model)

    Returns:
        Skipgram loss (scalar)
    """
    # Positive score: dot product of center and context
    # Shape: (batch_size,)
    pos_score = (mu_center * mu_context).sum(dim=-1)

    # Negative scores: dot product of center with each negative
    # mu_center: (batch, d_model) -> (batch, 1, d_model)
    # mu_negatives: (batch, num_neg, d_model)
    # Result: (batch, num_neg)
    neg_scores = torch.bmm(
        mu_negatives,
        mu_center.unsqueeze(-1)
    ).squeeze(-1)

    # Loss: -log σ(pos) - Σ log σ(-neg)
    pos_loss = -F.logsigmoid(pos_score)
    neg_loss = -F.logsigmoid(-neg_scores).sum(dim=-1)

    total_loss = (pos_loss + neg_loss).mean()

    return total_loss


def compute_skipgram_loss_from_ids(
    model: "TokenVAE",
    center_ids: torch.Tensor,
    context_ids: torch.Tensor,
    negative_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute skipgram loss given token IDs.

    Args:
        model: TokenVAE model
        center_ids: Center token IDs, shape (batch_size,)
        context_ids: Context token IDs, shape (batch_size,)
        negative_ids: Negative token IDs, shape (batch_size, num_negatives)

    Returns:
        Skipgram loss (scalar)
    """
    # Get μ for center tokens
    mu_center, _ = model.encode(center_ids)

    # Get μ for context tokens
    mu_context, _ = model.encode(context_ids)

    # Get μ for negative tokens
    batch_size, num_negatives = negative_ids.shape
    negative_ids_flat = negative_ids.view(-1)
    mu_negatives_flat, _ = model.encode(negative_ids_flat)
    mu_negatives = mu_negatives_flat.view(batch_size, num_negatives, -1)

    return skipgram_loss(mu_center, mu_context, mu_negatives)


def interpolation_entropy_loss(
    model: "TokenVAE",
    mu: torch.Tensor,
    num_pairs: int,
    entropy_target: float,
    eps: float = 1e-8,
    token_ids: torch.Tensor | None = None,
    holdout_pairs: set[int] | None = None,
    vocab_size: int | None = None,
    max_resample: int = 10,
    return_pair_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize high-entropy interpolations between token embeddings.

    Samples random token pairs from the current batch, interpolates between
    their μ vectors, decodes, and penalizes entropy above a target.

    Args:
        model: TokenVAE model
        mu: Mean embeddings for the current batch, shape (batch_size, d_model)
        num_pairs: Number of interpolation pairs to sample
        entropy_target: Target maximum entropy per interpolation
        eps: Numerical stability constant

    Returns:
        Tuple of (entropy_penalty_loss, mean_entropy)
    """
    if num_pairs <= 0 or mu.shape[0] < 2:
        zero = torch.tensor(0.0, device=mu.device)
        if return_pair_ids:
            return zero, zero, []
        return zero, zero

    if (holdout_pairs is not None or return_pair_ids) and vocab_size is None:
        raise ValueError("vocab_size is required when holdout_pairs or return_pair_ids is used")

    batch_size = mu.shape[0]
    token_list = token_ids.detach().cpu().tolist() if token_ids is not None else None

    idx_a: list[int] = []
    idx_b: list[int] = []
    pair_ids: list[int] = []
    seen_pairs: set[int] = set()

    attempts = 0
    max_attempts = max_resample * num_pairs
    while len(idx_a) < num_pairs and attempts < max_attempts:
        i = random.randrange(batch_size)
        j = random.randrange(batch_size)
        attempts += 1
        if i == j:
            continue

        pid = None
        if token_list is not None and vocab_size is not None:
            a = int(token_list[i])
            b = int(token_list[j])
            pid = make_pair_id(a, b, vocab_size)
            if holdout_pairs and pid in holdout_pairs:
                continue
            if pid in seen_pairs:
                continue
            seen_pairs.add(pid)

        idx_a.append(i)
        idx_b.append(j)
        if pid is not None:
            pair_ids.append(pid)

    if len(idx_a) == 0:
        zero = torch.tensor(0.0, device=mu.device)
        if return_pair_ids:
            return zero, zero, []
        return zero, zero

    idx_a_t = torch.tensor(idx_a, device=mu.device)
    idx_b_t = torch.tensor(idx_b, device=mu.device)

    alphas = torch.rand(len(idx_a), 1, device=mu.device)
    z = (1.0 - alphas) * mu[idx_a_t] + alphas * mu[idx_b_t]

    logits = model.decode(z)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)
    mean_entropy = entropy.mean()
    penalty = F.relu(entropy - entropy_target).mean()

    if return_pair_ids:
        return penalty, mean_entropy, pair_ids

    return penalty, mean_entropy


def prior_diversity_loss(
    model: "TokenVAE",
    num_samples: int,
    entropy_target: float,
    max_freq_target: float,
    max_freq_weight: float = 1.0,
    marginal_entropy_target: float = 0.0,
    marginal_entropy_weight: float = 1.0,
    hhi_target: float = 0.0,
    hhi_weight: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encourage diverse decoding from the prior.

    Penalizes low mean entropy, overly concentrated token mass, low
    marginal entropy, and high Herfindahl-Hirschman Index (HHI) on
    soft token mass.

    Args:
        model: TokenVAE model
        num_samples: Number of prior samples
        entropy_target: Minimum mean entropy target
        max_freq_target: Maximum allowed mean token mass for a single token
        max_freq_weight: Weight for max-frequency penalty
        marginal_entropy_target: Minimum marginal entropy target (0 = disabled)
        marginal_entropy_weight: Weight for marginal entropy penalty
        hhi_target: Maximum allowed HHI on soft token mass (0 = disabled)
        hhi_weight: Weight for HHI penalty
        eps: Numerical stability constant

    Returns:
        Tuple of (loss, mean_entropy, max_token_mass, marginal_entropy, hhi)
    """
    device = next(model.parameters()).device
    z = torch.randn(num_samples, model.d_model, device=device)
    logits = model.decode(z)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)
    mean_entropy = entropy.mean()

    # Token mass averaged across samples; max indicates concentration
    token_mass = probs.mean(dim=0)
    max_token_mass = token_mass.max()

    # Marginal entropy: entropy of the average token distribution
    marginal_entropy = -(token_mass * torch.log(token_mass + eps)).sum()

    # HHI: sum of squared token masses (concentration measure)
    hhi = (token_mass ** 2).sum()

    ent_penalty = F.relu(entropy_target - mean_entropy)
    freq_penalty = F.relu(max_token_mass - max_freq_target)
    marginal_penalty = F.relu(marginal_entropy_target - marginal_entropy)
    hhi_penalty = F.relu(hhi - hhi_target) if hhi_target > 0 else torch.tensor(0.0, device=device)

    loss = (
        ent_penalty
        + max_freq_weight * freq_penalty
        + marginal_entropy_weight * marginal_penalty
        + hhi_weight * hhi_penalty
    )

    return loss, mean_entropy, max_token_mass, marginal_entropy, hhi
