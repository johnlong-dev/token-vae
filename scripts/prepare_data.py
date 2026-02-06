#!/usr/bin/env python3
"""Download and prepare training data from HuggingFace datasets.

Streams text from a HuggingFace dataset (default: FineWeb) and writes
a local text file of the desired size. This avoids downloading the
entire dataset.

Usage:
    # 1M tokens worth of text (rough estimate: ~4 chars per token)
    uv run scripts/prepare_data.py --num-chars 4000000

    # 10M tokens, save to custom path
    uv run scripts/prepare_data.py --num-chars 40000000 --output data/fineweb-10M.txt

    # Different dataset
    uv run scripts/prepare_data.py --dataset allenai/c4 --config en --split train
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from HuggingFace")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sample-10BT",
        help="Dataset config/subset (default: sample-10BT for FineWeb 10B token sample)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text field in the dataset",
    )
    parser.add_argument(
        "--num-chars",
        type=int,
        default=4_000_000,
        help="Approximate number of characters to download (~4 chars per token)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output text file path (default: data/fineweb-{size}.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for shuffling (requires downloading shuffle buffer first)",
    )
    args = parser.parse_args()

    console = Console()

    # Determine output path
    if args.output is None:
        size_label = _format_size(args.num_chars)
        args.output = f"data/fineweb-{size_label}.txt"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        console.print(f"[yellow]Output file already exists: {output_path}[/yellow]")
        console.print("[yellow]Delete it first to re-download, or use a different --output path[/yellow]")
        return 0

    # Import datasets (lazy — only needed if actually downloading)
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Error: 'datasets' package not installed[/red]")
        console.print("[yellow]Install it: uv add datasets[/yellow]")
        return 1

    console.print(f"[bold]Streaming from {args.dataset} ({args.config})[/bold]")
    console.print(f"  Target size: ~{args.num_chars:,} chars")
    console.print(f"  Output: {output_path}")
    console.print()

    # Stream dataset
    ds = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )

    if args.seed is not None:
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    chars_written = 0
    docs_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[chars]}"),
            TextColumn("{task.fields[docs]} docs"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Downloading...",
                total=args.num_chars,
                chars="0",
                docs="0",
            )

            for example in ds:
                text = example.get(args.text_field, "")
                if not text.strip():
                    continue

                f.write(text)
                f.write("\n")

                chars_written += len(text) + 1
                docs_written += 1

                progress.update(
                    task,
                    completed=min(chars_written, args.num_chars),
                    chars=f"{chars_written:,}",
                    docs=str(docs_written),
                )

                if chars_written >= args.num_chars:
                    break

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Characters: {chars_written:,}")
    console.print(f"  Documents: {docs_written:,}")
    console.print(f"  Estimated tokens: ~{chars_written // 4:,}")
    console.print(f"  Saved to: {output_path}")

    # Force exit — HF datasets streaming keeps background threads alive
    os._exit(0)


def _format_size(num_chars: int) -> str:
    """Format character count as a human-readable size label."""
    if num_chars >= 1_000_000_000:
        return f"{num_chars // 1_000_000_000}B"
    if num_chars >= 1_000_000:
        return f"{num_chars // 1_000_000}M"
    if num_chars >= 1_000:
        return f"{num_chars // 1_000}K"
    return str(num_chars)


if __name__ == "__main__":
    sys.exit(main())
