"""Tests for seed reproducibility."""

import torch

from token_vae.model import TokenVAE


def _make_model(seed: int) -> TokenVAE:
    """Create a model with a fixed seed."""
    torch.manual_seed(seed)
    return TokenVAE(vocab_size=100, d_model=32, d_embed=64)


class TestReproducibility:
    """Seed determinism tests."""

    def test_model_init_deterministic(self):
        """Same seed produces identical model weights."""
        m1 = _make_model(42)
        m2 = _make_model(42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_forward_deterministic(self):
        """Same seed produces identical stochastic forward pass."""
        model = _make_model(0)
        tokens = torch.tensor([1, 2, 3])

        torch.manual_seed(99)
        logits1, mu1, logvar1, h1 = model(tokens)

        torch.manual_seed(99)
        logits2, mu2, logvar2, h2 = model(tokens)

        assert torch.equal(logits1, logits2)
        assert torch.equal(h1, h2)

    def test_prior_sampling_deterministic(self):
        """Same seed produces identical prior samples."""
        model = _make_model(0)

        torch.manual_seed(99)
        h1, logits1 = model.sample_prior(10)

        torch.manual_seed(99)
        h2, logits2 = model.sample_prior(10)

        assert torch.equal(h1, h2)
        assert torch.equal(logits1, logits2)
