"""Training loop for Token VAE."""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from token_vae.model import TokenVAE, get_device
from token_vae.losses import (
    vae_loss,
    compute_skipgram_loss_from_ids,
    interpolation_entropy_loss,
    prior_diversity_loss,
    compute_ifw_weights,
)
from token_vae.training.config import TrainingConfig
from token_vae.utils.pairs import load_pairs, pairs_to_id_set


class Trainer:
    """Trainer for Token VAE."""

    def __init__(
        self,
        model: TokenVAE,
        identity_loader: DataLoader,
        cooccurrence_loader: DataLoader,
        config: TrainingConfig,
        token_counts: np.ndarray | None = None,
    ):
        """Initialize trainer.

        Args:
            model: TokenVAE model
            identity_loader: DataLoader for identity reconstruction
            cooccurrence_loader: DataLoader for co-occurrence learning
            config: Training configuration
            token_counts: Per-token occurrence counts for IFW (None = disabled)
        """
        self.model = model
        self.identity_loader = identity_loader
        self.cooccurrence_loader = cooccurrence_loader
        self.config = config

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

        # Setup device
        if config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Compute IFW weights if token counts provided and alpha > 0
        self.recon_weight: torch.Tensor | None = None
        if token_counts is not None and config.ifw_alpha > 0:
            self.recon_weight = compute_ifw_weights(
                token_counts, alpha=config.ifw_alpha
            ).to(self.device)

        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.interp_pairs_seen: set[int] = set()

        # Optional holdout pairs for interpolation regularizer
        self.interp_holdout_pairs: set[int] = set()
        if config.interpolation_holdout_pairs_path:
            try:
                pairs = load_pairs(config.interpolation_holdout_pairs_path)
                self.interp_holdout_pairs = pairs_to_id_set(pairs, config.vocab_size)
            except FileNotFoundError:
                print(
                    f"Warning: holdout pairs not found at {config.interpolation_holdout_pairs_path}",
                    flush=True,
                )

        # History for tracking
        self.history = {
            "recon_loss": [],
            "kl_loss": [],
            "skipgram_loss": [],
            "interp_loss": [],
            "interp_entropy": [],
            "interp_pairs_unique": [],
            "prior_reg_loss": [],
            "prior_entropy": [],
            "prior_max_freq": [],
            "prior_marginal_entropy": [],
            "prior_hhi": [],
            "total_loss": [],
            "kl_weight": [],
        }

    def train_epoch(self) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()

        epoch_losses = {
            "recon": 0.0,
            "kl": 0.0,
            "skipgram": 0.0,
            "interp": 0.0,
            "interp_entropy": 0.0,
            "prior_reg": 0.0,
            "prior_entropy": 0.0,
            "prior_max_freq": 0.0,
            "prior_marginal_entropy": 0.0,
            "prior_hhi": 0.0,
            "total": 0.0,
            "kl_weight": 0.0,
            "prior_reg_steps": 0,
        }
        num_batches = 0

        # Calculate total steps for KL annealing
        total_steps = self.config.epochs * len(self.identity_loader)

        # Create iterators
        identity_iter = iter(self.identity_loader)
        cooccurrence_iter = iter(self.cooccurrence_loader)

        pbar = tqdm(
            total=len(self.identity_loader),
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs}",
            file=sys.stdout,
            dynamic_ncols=True,
        )

        batch_idx = 0
        while True:
            # Get identity batch
            try:
                input_ids, target_ids = next(identity_iter)
            except StopIteration:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Get current KL weight
            kl_weight = self.config.get_kl_weight(self.global_step, total_steps)

            # Forward pass for VAE
            logits, mu, logvar, h = self.model(input_ids)

            # VAE loss
            total_vae_loss, recon_loss, kl_loss = vae_loss(
                logits, target_ids, mu, logvar,
                kl_weight=kl_weight,
                recon_weight=self.recon_weight,
                free_bits=self.config.kl_free_bits,
            )

            # Skipgram loss (get next co-occurrence batch)
            try:
                center_ids, context_ids, negative_ids = next(cooccurrence_iter)
            except StopIteration:
                # Reset co-occurrence iterator
                cooccurrence_iter = iter(self.cooccurrence_loader)
                center_ids, context_ids, negative_ids = next(cooccurrence_iter)

            center_ids = center_ids.to(self.device)
            context_ids = context_ids.to(self.device)
            negative_ids = negative_ids.to(self.device)

            skip_loss = compute_skipgram_loss_from_ids(
                self.model, center_ids, context_ids, negative_ids
            )

            interp_loss = torch.tensor(0.0, device=self.device)
            interp_entropy = torch.tensor(0.0, device=self.device)
            interp_pair_ids: list[int] = []
            if self.config.interpolation_weight > 0:
                return_pairs = self.config.interpolation_track_pairs or bool(self.interp_holdout_pairs)
                result = interpolation_entropy_loss(
                    self.model,
                    mu,
                    num_pairs=self.config.interpolation_pairs,
                    entropy_target=self.config.interpolation_entropy_target,
                    token_ids=input_ids,
                    holdout_pairs=self.interp_holdout_pairs or None,
                    vocab_size=self.config.vocab_size,
                    max_resample=self.config.interpolation_pair_max_resample,
                    return_pair_ids=return_pairs,
                )
                if return_pairs:
                    interp_loss, interp_entropy, interp_pair_ids = result
                else:
                    interp_loss, interp_entropy = result

                if self.config.interpolation_track_pairs and interp_pair_ids:
                    self.interp_pairs_seen.update(interp_pair_ids)

            prior_reg_loss = torch.tensor(0.0, device=self.device)
            prior_entropy = torch.tensor(0.0, device=self.device)
            prior_max_freq = torch.tensor(0.0, device=self.device)
            prior_marginal_entropy = torch.tensor(0.0, device=self.device)
            prior_hhi = torch.tensor(0.0, device=self.device)
            if self.config.prior_reg_weight > 0:
                if self.global_step % self.config.prior_reg_interval == 0:
                    prior_reg_loss, prior_entropy, prior_max_freq, prior_marginal_entropy, prior_hhi = prior_diversity_loss(
                        self.model,
                        num_samples=self.config.prior_reg_samples,
                        entropy_target=self.config.prior_entropy_target,
                        max_freq_target=self.config.prior_max_freq_target,
                        max_freq_weight=self.config.prior_max_freq_weight,
                        marginal_entropy_target=self.config.prior_marginal_entropy_target,
                        marginal_entropy_weight=self.config.prior_marginal_entropy_weight,
                        hhi_target=self.config.prior_hhi_target,
                        hhi_weight=self.config.prior_hhi_weight,
                    )
                    epoch_losses["prior_reg_steps"] += 1

            # Total loss
            total_loss = (
                total_vae_loss
                + self.config.skipgram_weight * skip_loss
                + self.config.interpolation_weight * interp_loss
                + self.config.prior_reg_weight * prior_reg_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Track losses
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["kl"] += kl_loss.item()
            epoch_losses["skipgram"] += skip_loss.item()
            epoch_losses["interp"] += interp_loss.item()
            epoch_losses["interp_entropy"] += interp_entropy.item()
            epoch_losses["prior_reg"] += prior_reg_loss.item()
            epoch_losses["prior_entropy"] += prior_entropy.item()
            epoch_losses["prior_max_freq"] += prior_max_freq.item()
            epoch_losses["prior_marginal_entropy"] += prior_marginal_entropy.item()
            epoch_losses["prior_hhi"] += prior_hhi.item()
            epoch_losses["total"] += total_loss.item()
            epoch_losses["kl_weight"] += kl_weight
            num_batches += 1

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                postfix = {
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'skip': f'{skip_loss.item():.4f}',
                    'kl_w': f'{kl_weight:.4f}',
                }
                if self.config.interpolation_weight > 0:
                    postfix['interp'] = f'{interp_loss.item():.4f}'
                    postfix['H'] = f'{interp_entropy.item():.2f}'
                if self.config.prior_reg_weight > 0:
                    postfix['prior'] = f'{prior_reg_loss.item():.4f}'
                pbar.set_postfix(postfix)

            self.global_step += 1
            batch_idx += 1
            pbar.update(1)

        pbar.close()

        if num_batches == 0:
            raise ValueError("Identity DataLoader produced zero batches; check dataset inputs.")

        # Compute averages
        for key in ["recon", "kl", "skipgram", "interp", "interp_entropy", "total", "kl_weight"]:
            epoch_losses[key] /= num_batches

        prior_steps = epoch_losses["prior_reg_steps"]
        if prior_steps > 0:
            for key in ["prior_reg", "prior_entropy", "prior_max_freq", "prior_marginal_entropy", "prior_hhi"]:
                epoch_losses[key] /= prior_steps
        else:
            for key in ["prior_reg", "prior_entropy", "prior_max_freq", "prior_marginal_entropy", "prior_hhi"]:
                epoch_losses[key] = 0.0

        return epoch_losses

    def train(self) -> dict:
        """Run full training.

        Returns:
            Training history
        """
        print(f"\nStarting training on {self.device}", flush=True)
        print(f"Identity dataset size: {len(self.identity_loader.dataset):,}", flush=True)
        print(f"Co-occurrence dataset size: {len(self.cooccurrence_loader.dataset):,}", flush=True)
        print(flush=True)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_losses = self.train_epoch()

            # Record history
            self.history["recon_loss"].append(epoch_losses["recon"])
            self.history["kl_loss"].append(epoch_losses["kl"])
            self.history["skipgram_loss"].append(epoch_losses["skipgram"])
            self.history["interp_loss"].append(epoch_losses["interp"])
            self.history["interp_entropy"].append(epoch_losses["interp_entropy"])
            self.history["interp_pairs_unique"].append(len(self.interp_pairs_seen))
            self.history["prior_reg_loss"].append(epoch_losses["prior_reg"])
            self.history["prior_entropy"].append(epoch_losses["prior_entropy"])
            self.history["prior_max_freq"].append(epoch_losses["prior_max_freq"])
            self.history["prior_marginal_entropy"].append(epoch_losses["prior_marginal_entropy"])
            self.history["prior_hhi"].append(epoch_losses["prior_hhi"])
            self.history["total_loss"].append(epoch_losses["total"])
            self.history["kl_weight"].append(epoch_losses["kl_weight"])

            # Print epoch summary
            print(f"\n=== Epoch {epoch + 1} Summary ===", flush=True)
            print(f"  Reconstruction Loss: {epoch_losses['recon']:.4f}", flush=True)
            print(f"  KL Divergence:       {epoch_losses['kl']:.4f}", flush=True)
            print(f"  Skipgram Loss:       {epoch_losses['skipgram']:.4f}", flush=True)
            if self.config.interpolation_weight > 0:
                print(f"  Interp Loss:         {epoch_losses['interp']:.4f}", flush=True)
                print(f"  Interp Entropy:      {epoch_losses['interp_entropy']:.2f}", flush=True)
                if self.config.interpolation_track_pairs:
                    print(
                        f"  Unique Interp Pairs: {len(self.interp_pairs_seen):,}",
                        flush=True,
                    )
            if self.config.prior_reg_weight > 0:
                print(f"  Prior Reg Loss:      {epoch_losses['prior_reg']:.4f}", flush=True)
                print(f"  Prior Entropy:       {epoch_losses['prior_entropy']:.2f}", flush=True)
                print(f"  Prior Max Freq:      {epoch_losses['prior_max_freq']:.2f}", flush=True)
                print(f"  Prior Marginal H:    {epoch_losses['prior_marginal_entropy']:.2f}", flush=True)
                print(f"  Prior HHI:           {epoch_losses['prior_hhi']:.6f}", flush=True)
            print(f"  Total Loss:          {epoch_losses['total']:.4f}", flush=True)
            print(flush=True)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s", flush=True)

        return self.history

    def save_checkpoint(self, path: str | None = None):
        """Save model checkpoint.

        Args:
            path: Optional override for save path
        """
        save_path = path or self.config.model_save_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "d_embed": self.config.d_embed,
            },
            "history": self.history,
            "seed": self.config.seed,
        }

        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}", flush=True)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint.get("history", self.history)

        print(f"Loaded checkpoint from {path}", flush=True)


def load_model(path: str, device: torch.device | None = None) -> TokenVAE:
    """Load a trained model from checkpoint.

    Args:
        path: Path to checkpoint
        device: Device to load model on

    Returns:
        Loaded TokenVAE model
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    model = TokenVAE(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        d_embed=config["d_embed"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model
