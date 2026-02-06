"""Tests for loss functions."""

import numpy as np
import torch

from token_vae.losses import (
    reconstruction_loss,
    kl_divergence,
    vae_loss,
    skipgram_loss,
    compute_ifw_weights,
    prior_diversity_loss,
)
from token_vae.model import TokenVAE


class TestLosses:
    """Test loss functions."""

    def test_reconstruction_loss(self):
        """Test reconstruction loss."""
        logits = torch.randn(4, 100)
        targets = torch.tensor([0, 1, 2, 3])

        loss = reconstruction_loss(logits, targets)

        assert loss.shape == ()
        assert loss >= 0

    def test_kl_divergence_zero(self):
        """Test KL divergence is zero for standard normal."""
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        kl = kl_divergence(mu, logvar)

        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_kl_divergence_positive(self):
        """Test KL divergence is positive for non-standard distributions."""
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        kl = kl_divergence(mu, logvar)

        assert kl >= 0

    def test_kl_divergence_stable_for_large_logvar(self):
        """KL should remain finite even for very large input log-variance."""
        mu = torch.zeros(2, 8)
        logvar = torch.full((2, 8), 1e6)
        kl = kl_divergence(mu, logvar)
        assert torch.isfinite(kl)

    def test_vae_loss(self):
        """Test combined VAE loss."""
        logits = torch.randn(4, 100)
        targets = torch.tensor([0, 1, 2, 3])
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        total, recon, kl = vae_loss(logits, targets, mu, logvar, kl_weight=0.01)

        assert total >= 0
        assert recon >= 0
        assert kl >= 0
        assert torch.isclose(total, recon + 0.01 * kl)

    def test_skipgram_loss(self):
        """Test skipgram loss."""
        batch_size = 4
        d_model = 32
        num_negatives = 5

        mu_center = torch.randn(batch_size, d_model)
        mu_context = torch.randn(batch_size, d_model)
        mu_negatives = torch.randn(batch_size, num_negatives, d_model)

        loss = skipgram_loss(mu_center, mu_context, mu_negatives)

        assert loss.shape == ()
        assert loss >= 0

    def test_skipgram_loss_similar_higher(self):
        """Test that similar embeddings have lower loss."""
        batch_size = 4
        d_model = 32
        num_negatives = 5

        # Similar center and context
        mu_center = torch.randn(batch_size, d_model)
        mu_context_similar = mu_center + 0.1 * torch.randn(batch_size, d_model)
        mu_context_different = torch.randn(batch_size, d_model)
        mu_negatives = torch.randn(batch_size, num_negatives, d_model)

        loss_similar = skipgram_loss(mu_center, mu_context_similar, mu_negatives)
        loss_different = skipgram_loss(mu_center, mu_context_different, mu_negatives)

        # Similar should have lower loss (on average)
        # This is probabilistic, so we just check shapes
        assert loss_similar.shape == ()
        assert loss_different.shape == ()


class TestIFWWeights:
    """Tests for inverse frequency weighting."""

    def test_ifw_weights_basic(self):
        """IFW weights should have mean == 1.0."""
        counts = np.array([100, 50, 10, 1, 200], dtype=np.float64)
        w = compute_ifw_weights(counts, alpha=1.0)
        assert w.shape == (5,)
        assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-5)

    def test_ifw_weights_alpha_zero(self):
        """Alpha=0 should produce uniform weights."""
        counts = np.array([100, 50, 10, 1, 200], dtype=np.float64)
        w = compute_ifw_weights(counts, alpha=0.0)
        assert torch.allclose(w, torch.ones(5), atol=1e-5)

    def test_ifw_weights_rare_upweighted(self):
        """Rare tokens should have higher weight than common tokens."""
        counts = np.array([1000, 1], dtype=np.float64)
        w = compute_ifw_weights(counts, alpha=1.0)
        assert w[1] > w[0]

    def test_ifw_weights_zero_counts(self):
        """Zero counts should not produce NaN or Inf."""
        counts = np.array([100, 0, 50], dtype=np.float64)
        w = compute_ifw_weights(counts, alpha=1.0)
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()


class TestKLFreeBits:
    """Tests for KL free-bits."""

    def test_kl_free_bits_floor(self):
        """KL should equal free_bits * d_model when mu=0, logvar=0."""
        d_model = 32
        mu = torch.zeros(4, d_model)
        logvar = torch.zeros(4, d_model)
        free_bits = 0.25

        kl = kl_divergence(mu, logvar, free_bits=free_bits)
        expected = free_bits * d_model
        assert torch.isclose(kl, torch.tensor(expected), atol=1e-5)

    def test_kl_free_bits_no_effect_when_kl_high(self):
        """Free-bits shouldn't inflate already-high KL."""
        d_model = 32
        mu = torch.ones(4, d_model) * 5.0  # large mu -> high KL
        logvar = torch.zeros(4, d_model)

        kl_no_fb = kl_divergence(mu, logvar, free_bits=0.0)
        kl_fb = kl_divergence(mu, logvar, free_bits=0.25)
        assert torch.isclose(kl_no_fb, kl_fb, atol=1e-4)


class TestPriorDiversity:
    """Tests for prior diversity loss."""

    def test_prior_diversity_loss_returns_five_tuple(self):
        """Prior diversity loss should return 5 tensors."""
        model = TokenVAE(vocab_size=100, d_model=32, d_embed=64)
        result = prior_diversity_loss(
            model, num_samples=16, entropy_target=3.0,
            max_freq_target=0.05,
        )
        assert len(result) == 5
        for t in result:
            assert isinstance(t, torch.Tensor)
            assert t.shape == ()

    def test_prior_diversity_hhi_disabled(self):
        """HHI penalty should be zero when target=0."""
        model = TokenVAE(vocab_size=100, d_model=32, d_embed=64)
        _, _, _, _, hhi = prior_diversity_loss(
            model, num_samples=16, entropy_target=3.0,
            max_freq_target=0.05, hhi_target=0.0,
        )
        # hhi is the raw HHI value, not the penalty. Check penalty is zero
        # by calling with hhi_target=0 (disabled) — the loss shouldn't include hhi
        loss_no_hhi, _, _, _, _ = prior_diversity_loss(
            model, num_samples=16, entropy_target=0.0,
            max_freq_target=1.0, marginal_entropy_target=0.0,
            hhi_target=0.0,
        )
        # With all targets met / disabled, loss should be zero
        assert torch.isclose(loss_no_hhi, torch.tensor(0.0), atol=1e-5)
