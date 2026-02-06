#!/usr/bin/env python3
"""Create visualizations for a trained Token VAE model."""

import argparse
from pathlib import Path

import torch
from rich.console import Console

from token_vae.model import get_device
from token_vae.data.tokenizer import load_tokenizer
from token_vae.training.trainer import load_model
from token_vae.evaluation.visualizations import create_visualizations, plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Visualize Token VAE")
    parser.add_argument(
        "--model", "-m",
        default="artifacts/models/token_vae.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default="artifacts/tokenizer/fw-4M-v16k.model",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/reports/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--num-prior-samples",
        type=int,
        default=500,
        help="Number of prior samples to visualize",
    )
    parser.add_argument(
        "--num-interpolations",
        type=int,
        default=5,
        help="Number of interpolation paths to show",
    )
    args = parser.parse_args()

    console = Console()

    # Check files exist
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)

    if not model_path.exists():
        console.print(f"[red]Error: Model not found: {model_path}[/red]")
        return 1

    if not tokenizer_path.exists():
        console.print(f"[red]Error: Tokenizer not found: {tokenizer_path}[/red]")
        return 1

    # Load model and tokenizer
    console.print("[bold]Loading model and tokenizer...[/bold]")
    device = get_device()
    model = load_model(str(model_path), device)
    tokenizer = load_tokenizer(str(tokenizer_path))

    # Create visualizations
    console.print("\n[bold]Creating visualizations...[/bold]")
    figures = create_visualizations(
        model, tokenizer,
        output_dir=args.output,
        num_prior_samples=args.num_prior_samples,
        num_interpolations=args.num_interpolations,
    )

    # Plot training history if available
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'history' in checkpoint and checkpoint['history'].get('recon_loss'):
        console.print("Creating training loss plot...")
        history_path = plot_training_history(
            checkpoint['history'],
            output_dir=args.output,
        )
        if history_path:
            figures['training_loss'] = history_path

    # Summary
    console.print("\n[bold green]Visualizations complete![/bold green]")
    console.print(f"Saved {len(figures)} figures to {args.output}:")
    for name, path in figures.items():
        console.print(f"  - {name}: {path}")

    return 0


if __name__ == "__main__":
    exit(main())
