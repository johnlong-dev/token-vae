#!/usr/bin/env python3
"""Evaluate a trained Token VAE model."""

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from token_vae.model import get_device
from token_vae.data.tokenizer import load_tokenizer
from token_vae.training.trainer import load_model
from token_vae.evaluation.tests import run_all_tests
from token_vae.evaluation.config import EvaluationConfig
from token_vae.evaluation.visualizations import create_visualizations, plot_training_history
from token_vae.evaluation.report import generate_report


def _seed_everything(seed: int | None) -> None:
    """Seed Python/NumPy/Torch for deterministic evaluation."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256(path: Path) -> str:
    """Compute file SHA256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def main():
    parser = argparse.ArgumentParser(description="Evaluate Token VAE")
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
        default="artifacts/reports",
        help="Output directory for report",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualizations",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to corpus for token frequency stratification",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to use for frequency stats (debugging)",
    )
    parser.add_argument(
        "--prior-samples",
        type=int,
        default=10_000,
        help="Number of prior samples",
    )
    parser.add_argument(
        "--prior-batch-size",
        type=int,
        default=512,
        help="Batch size for prior sampling",
    )
    parser.add_argument(
        "--prior-top-k",
        type=int,
        default=20,
        help="Top-k prior decoded tokens to report",
    )
    parser.add_argument(
        "--interp-pairs",
        type=int,
        default=50,
        help="Number of interpolation pairs to test",
    )
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=20,
        help="Interpolation steps per pair",
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
        "--interp-holdout-eval",
        type=int,
        default=None,
        help="Number of holdout pairs to evaluate (default: all)",
    )
    parser.add_argument(
        "--off-manifold-sigmas",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated sigmas for off-manifold sweep",
    )
    parser.add_argument(
        "--off-manifold-num-tokens",
        type=int,
        default=200,
        help="Number of tokens for off-manifold sweep",
    )
    parser.add_argument(
        "--off-manifold-samples",
        type=int,
        default=1,
        help="Samples per token for off-manifold sweep",
    )
    parser.add_argument(
        "--common-frac",
        type=float,
        default=0.2,
        help="Top fraction of tokens considered common",
    )
    parser.add_argument(
        "--rare-frac",
        type=float,
        default=0.2,
        help="Bottom fraction of tokens considered rare",
    )
    parser.add_argument(
        "--distance-quantiles",
        type=str,
        default="0.33,0.66",
        help="Comma-separated quantiles for distance buckets",
    )
    parser.add_argument(
        "--no-integrity-checks",
        action="store_true",
        help="Disable metric integrity checks",
    )
    parser.add_argument(
        "--diffusion-walks",
        type=int,
        default=100,
        help="Number of diffusion random walks",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Steps per diffusion walk",
    )
    parser.add_argument(
        "--diffusion-beta",
        type=float,
        default=0.01,
        help="Beta noise parameter for diffusion walk",
    )
    parser.add_argument(
        "--diffusion-beta-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Beta schedule for diffusion walk",
    )
    parser.add_argument(
        "--diffusion-min-unique-frac",
        type=float,
        default=0.10,
        help="Minimum per-step unique decoded token fraction for diffusion walk",
    )
    parser.add_argument(
        "--diffusion-min-change-rate",
        type=float,
        default=0.01,
        help="Minimum mean step-to-step token change rate for diffusion walk",
    )
    parser.add_argument(
        "--no-diffusion-walk",
        action="store_true",
        help="Disable diffusion walk test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global evaluation seed (set to -1 to disable seeding)",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to save raw test results as JSON",
    )
    args = parser.parse_args()
    eval_seed = None if args.seed == -1 else args.seed
    _seed_everything(eval_seed)

    console = Console()

    # Check files exist
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)

    if not model_path.exists():
        console.print(f"[red]Error: Model not found: {model_path}[/red]")
        console.print("[yellow]Run train_vae.py first[/yellow]")
        return 1

    if not tokenizer_path.exists():
        console.print(f"[red]Error: Tokenizer not found: {tokenizer_path}[/red]")
        return 1

    if args.data is not None and not Path(args.data).exists():
        console.print(f"[yellow]Warning: Data file not found: {args.data}[/yellow]")
        console.print("[yellow]Stratified metrics will be skipped[/yellow]")
        args.data = None

    # Load model and tokenizer
    console.print("[bold]Loading model and tokenizer...[/bold]")
    device = get_device()
    model = load_model(str(model_path), device)
    tokenizer = load_tokenizer(str(tokenizer_path))

    console.print(f"  Model loaded on: {device}")
    console.print(f"  Vocabulary size: {tokenizer.get_piece_size()}")

    # Run tests
    console.print("\n[bold]Running evaluation tests...[/bold]")
    # Build evaluation config
    sigmas = tuple(float(s.strip()) for s in args.off_manifold_sigmas.split(",") if s.strip())
    quantiles = tuple(float(q.strip()) for q in args.distance_quantiles.split(",") if q.strip())
    if len(quantiles) != 2:
        console.print("[red]Error: --distance-quantiles must have two values[/red]")
        return 1

    eval_config = EvaluationConfig(
        prior_num_samples=args.prior_samples,
        prior_batch_size=args.prior_batch_size,
        prior_top_k=args.prior_top_k,
        interp_num_pairs=args.interp_pairs,
        interp_num_steps=args.interp_steps,
        interp_holdout_pairs_path=args.interp_holdout_pairs,
        interp_holdout_num_pairs=args.interp_holdout_num,
        interp_holdout_seed=args.interp_holdout_seed,
        interp_holdout_eval_pairs=args.interp_holdout_eval,
        off_manifold_sigmas=sigmas,
        off_manifold_num_tokens=args.off_manifold_num_tokens,
        off_manifold_samples_per_token=args.off_manifold_samples,
        common_frac=args.common_frac,
        rare_frac=args.rare_frac,
        distance_quantiles=(quantiles[0], quantiles[1]),
        integrity_checks=not args.no_integrity_checks,
        diffusion_walk_enabled=not args.no_diffusion_walk,
        diffusion_num_walks=args.diffusion_walks,
        diffusion_num_steps=args.diffusion_steps,
        diffusion_beta=args.diffusion_beta,
        diffusion_beta_schedule=args.diffusion_beta_schedule,
        diffusion_min_unique_fraction_threshold=args.diffusion_min_unique_frac,
        diffusion_min_mean_change_rate_threshold=args.diffusion_min_change_rate,
        data_file=args.data,
        data_max_tokens=args.max_tokens,
        seed=eval_seed,
    )

    test_results = run_all_tests(model, tokenizer, verbose=True, config=eval_config)

    # Save raw results as JSON if requested
    if args.results_json:
        results_path = Path(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        def _sanitize(obj):
            """Make test result details JSON-serializable."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, float) and (obj != obj):  # NaN
                return None
            return obj

        json_results = []
        for r in test_results:
            json_results.append({
                "name": r.name,
                "passed": r.passed,
                "score": r.score,
                "threshold": r.threshold,
                "description": r.description,
                "details": _sanitize(r.details),
            })
        data_hash = None
        if args.data is not None and Path(args.data).exists():
            data_hash = _sha256(Path(args.data))
        payload = {
            "metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "device": str(device),
                "seed": eval_seed,
                "model": {
                    "path": str(model_path.resolve()),
                    "sha256": _sha256(model_path),
                },
                "tokenizer": {
                    "path": str(tokenizer_path.resolve()),
                    "sha256": _sha256(tokenizer_path),
                },
                "data": {
                    "path": str(Path(args.data).resolve()) if args.data is not None else None,
                    "sha256": data_hash,
                },
                "cli_args": vars(args),
                "evaluation_config": asdict(eval_config),
            },
            "results": json_results,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"Results JSON saved to: {results_path}")

    # Create visualizations
    figures = {}
    if not args.no_viz:
        console.print("\n[bold]Creating visualizations...[/bold]")
        try:
            figures = create_visualizations(
                model, tokenizer,
                output_dir=f"{args.output}/figures",
            )

            # Also plot training history if available in checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'history' in checkpoint and checkpoint['history'].get('recon_loss'):
                history_path = plot_training_history(
                    checkpoint['history'],
                    output_dir=f"{args.output}/figures",
                )
                if history_path:
                    figures['training_loss'] = history_path
        except ImportError as e:
            console.print(f"[yellow]Skipping visualizations: {e}[/yellow]")
            console.print("[yellow]Install viz dependencies: uv sync --group viz[/yellow]")

    # Generate report
    console.print("\n[bold]Generating report...[/bold]")

    # Load history from checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    history = checkpoint.get('history')

    report_path = generate_report(
        test_results=test_results,
        figures=figures,
        history=history,
        output_path=f"{args.output}/evaluation_report.md",
    )

    # Summary
    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print(f"Report saved to: {report_path}")

    all_passed = all(r.passed for r in test_results)
    if all_passed:
        console.print("\n[bold green]All evaluation criteria satisfied.[/bold green]")
        return 0
    else:
        failed = [r.name for r in test_results if not r.passed]
        console.print(f"\n[bold red]Failed: {', '.join(failed)}[/bold red]")
        return 1


if __name__ == "__main__":
    exit(main())
