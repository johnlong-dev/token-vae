#!/usr/bin/env python3
"""Train the Token VAE model."""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from token_vae.model import TokenVAE, get_device
from token_vae.data.tokenizer import load_tokenizer
from token_vae.data.datasets import create_dataloaders
from token_vae.training.config import TrainingConfig
from token_vae.training.trainer import Trainer
from token_vae.utils.pairs import generate_pairs, get_valid_token_ids, save_pairs


def main():
    parser = argparse.ArgumentParser(description="Train Token VAE")
    parser.add_argument(
        "--data", "-d",
        default="data/fineweb-4M.txt",
        help="Path to training data",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default="artifacts/tokenizer/fw-4M-v16k.model",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/models/token_vae.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.01,
        help="Final KL divergence weight",
    )
    parser.add_argument(
        "--skipgram-weight",
        type=float,
        default=0.1,
        help="Skipgram loss weight",
    )
    parser.add_argument(
        "--identity-vocab-repeats",
        type=int,
        default=0,
        help="Repeat full vocab identity dataset this many times per epoch",
    )
    parser.add_argument(
        "--interp-weight",
        type=float,
        default=0.0,
        help="Interpolation entropy penalty weight",
    )
    parser.add_argument(
        "--interp-pairs",
        type=int,
        default=32,
        help="Interpolation pairs sampled per batch",
    )
    parser.add_argument(
        "--interp-entropy-target",
        type=float,
        default=8.0,
        help="Target max entropy for interpolation penalty",
    )
    parser.add_argument(
        "--interp-holdout-pairs",
        type=str,
        default=None,
        help="Path to holdout interpolation pairs (JSON)",
    )
    parser.add_argument(
        "--interp-holdout-num",
        type=int,
        default=0,
        help="Number of holdout interpolation pairs to generate",
    )
    parser.add_argument(
        "--interp-holdout-seed",
        type=int,
        default=123,
        help="Seed for holdout pair generation",
    )
    parser.add_argument(
        "--interp-pair-max-resample",
        type=int,
        default=10,
        help="Max resample attempts per interpolation batch",
    )
    parser.add_argument(
        "--no-interp-pair-tracking",
        action="store_true",
        help="Disable tracking unique interpolation pairs",
    )
    parser.add_argument(
        "--prior-reg-weight",
        type=float,
        default=0.0,
        help="Weight for prior diversity regularizer",
    )
    parser.add_argument(
        "--prior-reg-samples",
        type=int,
        default=512,
        help="Number of prior samples for diversity regularizer",
    )
    parser.add_argument(
        "--prior-reg-interval",
        type=int,
        default=10,
        help="Steps between prior diversity regularizer evaluations",
    )
    parser.add_argument(
        "--prior-entropy-target",
        type=float,
        default=3.0,
        help="Target minimum prior entropy",
    )
    parser.add_argument(
        "--prior-max-freq-target",
        type=float,
        default=0.05,
        help="Target maximum mean token mass from prior",
    )
    parser.add_argument(
        "--prior-max-freq-weight",
        type=float,
        default=1.0,
        help="Weight for max-frequency penalty inside prior regularizer",
    )
    parser.add_argument(
        "--prior-marginal-entropy-target",
        type=float,
        default=0.0,
        help="Target minimum marginal entropy for prior diversity (0 = disabled)",
    )
    parser.add_argument(
        "--prior-marginal-entropy-weight",
        type=float,
        default=1.0,
        help="Weight for marginal entropy penalty inside prior regularizer",
    )
    parser.add_argument(
        "--prior-hhi-target",
        type=float,
        default=0.0,
        help="Target maximum HHI on soft token mass (0 = disabled)",
    )
    parser.add_argument(
        "--prior-hhi-weight",
        type=float,
        default=1.0,
        help="Weight for HHI penalty inside prior regularizer",
    )
    parser.add_argument(
        "--ifw-alpha",
        type=float,
        default=0.0,
        help="Inverse frequency weight exponent (0 = disabled, 1 = full IFW)",
    )
    parser.add_argument(
        "--kl-free-bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor for free-bits (0 = disabled)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Latent dimension",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to use (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Seed everything if requested
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    console = Console()

    # Check files exist
    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)

    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data_path}[/red]")
        return 1

    if not tokenizer_path.exists():
        console.print(f"[red]Error: Tokenizer not found: {tokenizer_path}[/red]")
        console.print("[yellow]Run train_tokenizer.py first[/yellow]")
        return 1

    # Load tokenizer
    console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = load_tokenizer(str(tokenizer_path))
    vocab_size = tokenizer.get_piece_size()
    console.print(f"  Vocabulary size: {vocab_size}")

    # Ensure holdout interpolation pairs exist if requested
    holdout_pairs_path = args.interp_holdout_pairs
    if args.interp_holdout_num > 0:
        if holdout_pairs_path is None:
            holdout_pairs_path = "artifacts/interp/holdout_pairs.json"
        console.print("[bold]Generating holdout interpolation pairs...[/bold]")
        valid_tokens = get_valid_token_ids(tokenizer)
        pairs = generate_pairs(
            valid_tokens=valid_tokens,
            num_pairs=args.interp_holdout_num,
            seed=args.interp_holdout_seed,
            vocab_size=vocab_size,
        )
        save_pairs(holdout_pairs_path, pairs)
        console.print(f"  Saved holdout pairs to: {holdout_pairs_path}")

    # Create config
    config = TrainingConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight_final=args.kl_weight,
        skipgram_weight=args.skipgram_weight,
        identity_vocab_repeats=args.identity_vocab_repeats,
        interpolation_weight=args.interp_weight,
        interpolation_pairs=args.interp_pairs,
        interpolation_entropy_target=args.interp_entropy_target,
        interpolation_holdout_pairs_path=holdout_pairs_path,
        interpolation_pair_max_resample=args.interp_pair_max_resample,
        interpolation_track_pairs=not args.no_interp_pair_tracking,
        ifw_alpha=args.ifw_alpha,
        prior_reg_weight=args.prior_reg_weight,
        prior_reg_samples=args.prior_reg_samples,
        prior_reg_interval=args.prior_reg_interval,
        prior_entropy_target=args.prior_entropy_target,
        prior_max_freq_target=args.prior_max_freq_target,
        prior_max_freq_weight=args.prior_max_freq_weight,
        prior_marginal_entropy_target=args.prior_marginal_entropy_target,
        prior_marginal_entropy_weight=args.prior_marginal_entropy_weight,
        prior_hhi_target=args.prior_hhi_target,
        prior_hhi_weight=args.prior_hhi_weight,
        kl_free_bits=args.kl_free_bits,
        model_save_path=args.output,
        seed=args.seed,
    )

    # Create dataloaders
    console.print("[bold]Creating datasets...[/bold]")
    identity_loader, cooccurrence_loader = create_dataloaders(
        text_file=str(data_path),
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        window_size=config.window_size,
        num_negatives=config.num_negatives,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        identity_vocab_repeats=config.identity_vocab_repeats,
        seed=args.seed,
    )
    console.print(f"  DataLoader workers: {identity_loader.num_workers}")

    # Compute token counts for IFW if enabled
    token_counts = None
    if config.ifw_alpha > 0:
        console.print("[bold]Computing token counts for IFW...[/bold]")
        dataset = identity_loader.dataset
        # Handle ConcatDataset: access first sub-dataset's tokens
        from torch.utils.data import ConcatDataset
        if isinstance(dataset, ConcatDataset):
            tokens = dataset.datasets[0].tokens
        else:
            tokens = dataset.tokens
        counts = Counter(tokens)
        token_counts = np.zeros(vocab_size, dtype=np.float64)
        for tid, cnt in counts.items():
            token_counts[tid] = cnt
        console.print(f"  Token count range: {int(token_counts.min())} - {int(token_counts.max())}")

    # Create model
    console.print("[bold]Creating model...[/bold]")
    device = get_device()
    console.print(f"  Device: {device}")

    model = TokenVAE(
        vocab_size=vocab_size,
        d_model=config.d_model,
        d_embed=config.d_embed,
    )
    console.print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        identity_loader=identity_loader,
        cooccurrence_loader=cooccurrence_loader,
        config=config,
        token_counts=token_counts,
    )

    # Train
    trainer.train()

    # Final save
    trainer.save_checkpoint()

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"Model saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
