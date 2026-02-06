"""Visualization utilities for Token VAE."""

from pathlib import Path

import torch
import numpy as np
import sentencepiece as spm

from token_vae.model import TokenVAE
from token_vae.data.tokenizer import is_special_token


def create_visualizations(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    output_dir: str = "artifacts/reports/figures",
    num_prior_samples: int = 500,
    num_interpolations: int = 5,
) -> dict:
    """Create all visualizations for the Token VAE.

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        output_dir: Directory to save figures
        num_prior_samples: Number of prior samples to visualize
        num_interpolations: Number of interpolation paths to show

    Returns:
        Dictionary with paths to saved figures
    """
    # Import visualization libraries (optional dependencies)
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        try:
            from umap import UMAP
            has_umap = True
        except ImportError:
            has_umap = False
    except ImportError as e:
        print(f"Visualization dependencies not installed: {e}")
        print("Install with: uv sync --group viz")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    figures = {}

    # Get all token embeddings
    with torch.no_grad():
        all_mu, all_logvar = model.get_all_embeddings()
        all_mu_np = all_mu.cpu().numpy()
        all_logvar_np = all_logvar.cpu().numpy()

    # Get special token mask
    special_mask = np.array([
        is_special_token(tokenizer, i)
        for i in range(tokenizer.get_piece_size())
    ])

    # Sample from prior
    with torch.no_grad():
        prior_h, _ = model.sample_prior(num_prior_samples, device)
        prior_h_np = prior_h.cpu().numpy()

    # 1. PCA visualization
    print("Creating PCA visualization...")
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(all_mu_np[~special_mask])
    prior_pca = pca.transform(prior_h_np)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot token embeddings
    ax.scatter(
        mu_pca[:, 0], mu_pca[:, 1],
        alpha=0.3, s=5, c='blue', label='Token μ'
    )

    # Plot prior samples
    ax.scatter(
        prior_pca[:, 0], prior_pca[:, 1],
        alpha=0.5, s=20, c='red', marker='x', label='Prior samples'
    )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Token Embeddings (PCA)')
    ax.legend()

    pca_path = output_path / "embedding_pca.png"
    fig.savefig(pca_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['pca'] = str(pca_path)

    # 2. UMAP visualization (if available)
    if has_umap:
        print("Creating UMAP visualization...")
        # Subsample for UMAP (faster)
        n_subsample = min(2000, len(all_mu_np[~special_mask]))
        subsample_idx = np.random.choice(
            len(all_mu_np[~special_mask]), n_subsample, replace=False
        )
        mu_subsample = all_mu_np[~special_mask][subsample_idx]

        # Combine with prior samples for joint UMAP
        combined = np.vstack([mu_subsample, prior_h_np])

        umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        combined_umap = umap.fit_transform(combined)

        mu_umap = combined_umap[:n_subsample]
        prior_umap = combined_umap[n_subsample:]

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(
            mu_umap[:, 0], mu_umap[:, 1],
            alpha=0.3, s=5, c='blue', label='Token μ'
        )
        ax.scatter(
            prior_umap[:, 0], prior_umap[:, 1],
            alpha=0.5, s=20, c='red', marker='x', label='Prior samples'
        )

        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Token Embeddings (UMAP)')
        ax.legend()

        umap_path = output_path / "embedding_umap.png"
        fig.savefig(umap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures['umap'] = str(umap_path)

    # 3. Interpolation paths
    print("Creating interpolation visualization...")
    valid_tokens = [i for i in range(tokenizer.get_piece_size()) if not special_mask[i]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # PCA for interpolation paths
    for i in range(min(num_interpolations, len(axes) - 1)):
        ax = axes[i]
        token_a, token_b = np.random.choice(valid_tokens, 2, replace=False)

        with torch.no_grad():
            h_interp, logits_interp = model.interpolate(
                int(token_a), int(token_b), num_steps=20
            )
            h_interp_np = h_interp.cpu().numpy()

        # Transform to PCA space
        h_pca = pca.transform(h_interp_np)

        # Plot background
        ax.scatter(mu_pca[:, 0], mu_pca[:, 1], alpha=0.1, s=1, c='gray')

        # Plot interpolation path
        ax.plot(h_pca[:, 0], h_pca[:, 1], 'g-', linewidth=2, alpha=0.8)
        ax.scatter(h_pca[0, 0], h_pca[0, 1], c='green', s=100, marker='o', zorder=5)
        ax.scatter(h_pca[-1, 0], h_pca[-1, 1], c='green', s=100, marker='s', zorder=5)

        token_a_str = tokenizer.id_to_piece(int(token_a))
        token_b_str = tokenizer.id_to_piece(int(token_b))
        ax.set_title(f'{repr(token_a_str)} → {repr(token_b_str)}')

    # Use last axis for prior samples overlay
    ax = axes[-1]
    ax.scatter(mu_pca[:, 0], mu_pca[:, 1], alpha=0.2, s=3, c='blue', label='Tokens')
    ax.scatter(prior_pca[:, 0], prior_pca[:, 1], alpha=0.5, s=15, c='red', marker='x', label='Prior')
    ax.set_title('Prior Coverage')
    ax.legend()

    fig.suptitle('Interpolation Paths in Latent Space (PCA)', fontsize=14)
    plt.tight_layout()

    interp_path = output_path / "interpolations.png"
    fig.savefig(interp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['interpolations'] = str(interp_path)

    # 4. Variance distribution
    print("Creating variance distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot σ² distribution
    sigma_sq = np.exp(all_logvar_np[~special_mask])
    mean_sigma_sq = sigma_sq.mean(axis=1)

    axes[0].hist(mean_sigma_sq, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(1.0, color='red', linestyle='--', label='Prior σ²=1')
    axes[0].set_xlabel('Mean σ² per token')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Token Variances')
    axes[0].legend()

    # Plot μ norm distribution
    mu_norms = np.linalg.norm(all_mu_np[~special_mask], axis=1)
    axes[1].hist(mu_norms, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('||μ||')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of μ Norms')

    plt.tight_layout()
    var_path = output_path / "variance_distribution.png"
    fig.savefig(var_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['variance'] = str(var_path)

    # 5. Training loss curves (if available in model checkpoint)
    # This would need to be loaded from the checkpoint

    print(f"Saved {len(figures)} figures to {output_dir}")
    return figures


def plot_training_history(
    history: dict,
    output_dir: str = "artifacts/reports/figures",
) -> str:
    """Plot training loss curves.

    Args:
        history: Training history dict with loss arrays
        output_dir: Directory to save figure

    Returns:
        Path to saved figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['recon_loss']) + 1)

    # Reconstruction loss
    axes[0, 0].plot(epochs, history['recon_loss'], 'b-', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Reconstruction Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # KL divergence
    axes[0, 1].plot(epochs, history['kl_loss'], 'r-', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].grid(True, alpha=0.3)

    # Skipgram loss
    axes[1, 0].plot(epochs, history['skipgram_loss'], 'g-', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Skipgram Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Total loss
    axes[1, 1].plot(epochs, history['total_loss'], 'purple', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    loss_path = output_path / "training_loss.png"
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(loss_path)
