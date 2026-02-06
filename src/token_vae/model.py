"""Token VAE model architecture."""

import torch
import torch.nn as nn


LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0


def get_device() -> torch.device:
    """Get the best available device (MPS for M4 Mac, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TokenVAE(nn.Module):
    """Variational Autoencoder for token embeddings.

    Architecture:
        token_id → Embedding → Split → (μ, log_σ)
                                 ↓
                   Sample h = μ + σ⊙ε  (ε ~ N(0,I))
                                 ↓
                   Linear → logits

    The goal is to create a walkable latent manifold where:
    - Each token maps to a distribution q(h|t) = N(μ, σ²I)
    - The aggregate posterior is regularized toward N(0, I)
    - The space is decodable everywhere
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        d_embed: int = 512,
    ):
        """Initialize the Token VAE.

        Args:
            vocab_size: Size of the token vocabulary
            d_model: Dimension of the latent space (μ and σ each)
            d_embed: Dimension of the initial embedding (split into μ and log_σ)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed

        # Embedding layer: token_id → d_embed dimensional vector
        # This will be split into μ (d_model) and log_σ (d_model)
        self.embedding = nn.Embedding(vocab_size, d_embed)

        # Projection layers to get μ and log_σ from embedding
        self.mu_proj = nn.Linear(d_embed, d_model)
        self.logvar_proj = nn.Linear(d_embed, d_model)

        # Decoder: latent h → logits over vocabulary
        self.decoder = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Standard initialization for embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize projection layers
        for module in [self.mu_proj, self.logvar_proj, self.decoder]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode token IDs to distribution parameters.

        Args:
            token_ids: Tensor of token IDs, shape (batch_size,)

        Returns:
            Tuple of (μ, log_σ), each shape (batch_size, d_model)
        """
        # Get embeddings
        embed = self.embedding(token_ids)  # (batch, d_embed)

        # Project to μ and log_σ
        mu = self.mu_proj(embed)  # (batch, d_model)
        logvar = self.logvar_proj(embed)  # (batch, d_model)
        logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Reparameterization trick: sample h = μ + σ⊙ε.

        Args:
            mu: Mean of the distribution, shape (batch_size, d_model)
            logvar: Log variance of the distribution, shape (batch_size, d_model)
            deterministic: If True, return μ without sampling

        Returns:
            Sampled latent h, shape (batch_size, d_model)
        """
        if deterministic:
            return mu

        # σ = exp(0.5 * log_σ²) = exp(0.5 * logvar)
        logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)

        # Sample ε ~ N(0, I)
        eps = torch.randn_like(std)

        # h = μ + σ⊙ε
        return mu + std * eps

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to logits.

        Args:
            h: Latent vectors, shape (batch_size, d_model)

        Returns:
            Logits over vocabulary, shape (batch_size, vocab_size)
        """
        return self.decoder(h)

    def forward(
        self,
        token_ids: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            token_ids: Input token IDs, shape (batch_size,)
            deterministic: If True, use μ instead of sampling

        Returns:
            Tuple of (logits, μ, log_σ², h):
                - logits: shape (batch_size, vocab_size)
                - mu: shape (batch_size, d_model)
                - logvar: shape (batch_size, d_model)
                - h: shape (batch_size, d_model)
        """
        # Encode
        mu, logvar = self.encode(token_ids)

        # Sample latent
        h = self.reparameterize(mu, logvar, deterministic=deterministic)

        # Decode
        logits = self.decode(h)

        return logits, mu, logvar, h

    def sample_prior(
        self,
        num_samples: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the prior N(0, I) and decode.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Tuple of (h, logits):
                - h: Sampled latent vectors, shape (num_samples, d_model)
                - logits: Decoded logits, shape (num_samples, vocab_size)
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior
        h = torch.randn(num_samples, self.d_model, device=device)

        # Decode
        logits = self.decode(h)

        return h, logits

    def get_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get μ and log_σ for all tokens in vocabulary.

        Returns:
            Tuple of (all_mu, all_logvar), each shape (vocab_size, d_model)
        """
        device = next(self.parameters()).device
        all_tokens = torch.arange(self.vocab_size, device=device)

        with torch.no_grad():
            all_mu, all_logvar = self.encode(all_tokens)

        return all_mu, all_logvar

    def interpolate(
        self,
        token_id_a: int,
        token_id_b: int,
        num_steps: int = 10,
        use_mu: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between two tokens in latent space.

        Args:
            token_id_a: First token ID
            token_id_b: Second token ID
            num_steps: Number of interpolation steps
            use_mu: If True, interpolate between μ values; else sample

        Returns:
            Tuple of (h_interp, logits_interp):
                - h_interp: Interpolated latents, shape (num_steps, d_model)
                - logits_interp: Decoded logits, shape (num_steps, vocab_size)
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            # Get embeddings for both tokens
            tokens = torch.tensor([token_id_a, token_id_b], device=device)
            mu, logvar = self.encode(tokens)

            if use_mu:
                # Interpolate between means
                start = mu[0]
                end = mu[1]
            else:
                # Sample from each distribution
                start = self.reparameterize(mu[0:1], logvar[0:1]).squeeze(0)
                end = self.reparameterize(mu[1:2], logvar[1:2]).squeeze(0)

            # Linear interpolation
            alphas = torch.linspace(0, 1, num_steps, device=device)
            h_interp = torch.stack([
                (1 - alpha) * start + alpha * end
                for alpha in alphas
            ])

            # Decode all interpolated points
            logits_interp = self.decode(h_interp)

        return h_interp, logits_interp
