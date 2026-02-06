"""Tests for Token VAE model."""

import pytest
import torch

from token_vae.model import LOGVAR_MAX, LOGVAR_MIN, TokenVAE, get_device


class TestTokenVAE:
    """Test TokenVAE model."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return TokenVAE(vocab_size=100, d_model=32, d_embed=64)

    def test_init(self, model):
        """Test model initialization."""
        assert model.vocab_size == 100
        assert model.d_model == 32
        assert model.d_embed == 64

    def test_encode(self, model):
        """Test encoding."""
        tokens = torch.tensor([0, 1, 2, 3])
        mu, logvar = model.encode(tokens)

        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_encode_clamps_logvar(self, model):
        """Encoding should clamp log-variance to avoid numerical overflow."""
        with torch.no_grad():
            model.logvar_proj.weight.fill_(1e6)
            model.logvar_proj.bias.fill_(1e6)

        tokens = torch.tensor([0, 1, 2, 3])
        _, logvar = model.encode(tokens)
        assert torch.all(logvar <= LOGVAR_MAX)
        assert torch.all(logvar >= LOGVAR_MIN)

    def test_reparameterize_deterministic(self, model):
        """Test deterministic reparameterization."""
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        h = model.reparameterize(mu, logvar, deterministic=True)
        assert torch.allclose(h, mu)

    def test_reparameterize_stochastic(self, model):
        """Test stochastic reparameterization."""
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        h1 = model.reparameterize(mu, logvar, deterministic=False)
        h2 = model.reparameterize(mu, logvar, deterministic=False)

        # Should be different due to random sampling
        assert not torch.allclose(h1, h2)

        # But should have similar mean
        assert h1.shape == (4, 32)

    def test_decode(self, model):
        """Test decoding."""
        h = torch.randn(4, 32)
        logits = model.decode(h)

        assert logits.shape == (4, 100)

    def test_forward(self, model):
        """Test full forward pass."""
        tokens = torch.tensor([0, 1, 2, 3])
        logits, mu, logvar, h = model(tokens)

        assert logits.shape == (4, 100)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
        assert h.shape == (4, 32)

    def test_forward_deterministic(self, model):
        """Test deterministic forward pass."""
        tokens = torch.tensor([0, 1, 2, 3])

        logits1, mu1, _, h1 = model(tokens, deterministic=True)
        logits2, mu2, _, h2 = model(tokens, deterministic=True)

        # Should be identical
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(h1, mu1)

    def test_sample_prior(self, model):
        """Test sampling from prior."""
        h, logits = model.sample_prior(10)

        assert h.shape == (10, 32)
        assert logits.shape == (10, 100)

    def test_get_all_embeddings(self, model):
        """Test getting all embeddings."""
        all_mu, all_logvar = model.get_all_embeddings()

        assert all_mu.shape == (100, 32)
        assert all_logvar.shape == (100, 32)

    def test_interpolate(self, model):
        """Test interpolation."""
        h_interp, logits_interp = model.interpolate(0, 10, num_steps=5)

        assert h_interp.shape == (5, 32)
        assert logits_interp.shape == (5, 100)

    def test_device_handling(self, model):
        """Test that model works on available device."""
        device = get_device()
        model = model.to(device)

        tokens = torch.tensor([0, 1, 2], device=device)
        logits, mu, logvar, h = model(tokens)

        # Check device type (mps:0 and mps are equivalent)
        assert logits.device.type == device.type
        assert mu.device.type == device.type
