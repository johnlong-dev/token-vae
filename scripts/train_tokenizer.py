#!/usr/bin/env python3
"""Train a SentencePiece tokenizer on TinyStories data."""

import argparse
from pathlib import Path

from rich.console import Console

from token_vae.data.tokenizer import train_tokenizer, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--input", "-i",
        default="data/fineweb-4M.txt",
        help="Path to input text file",
    )
    parser.add_argument(
        "--output", "-o",
        default="artifacts/tokenizer/tinystories",
        help="Output path prefix (without extension)",
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=8000,
        help="Vocabulary size",
    )
    args = parser.parse_args()

    console = Console()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        return 1

    console.print("[bold]Training tokenizer[/bold]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Vocab size: {args.vocab_size}")
    console.print()

    # Train tokenizer
    model_path = train_tokenizer(
        input_file=str(input_path),
        model_prefix=args.output,
        vocab_size=args.vocab_size,
    )

    console.print(f"[green]Tokenizer saved to {model_path}[/green]")

    # Test the tokenizer
    console.print("\n[bold]Testing tokenizer...[/bold]")
    sp = load_tokenizer(model_path)

    test_text = "Once upon a time, there was a little girl named Lily."
    tokens = sp.encode(test_text)
    pieces = sp.encode_as_pieces(test_text)
    decoded = sp.decode(tokens)

    console.print(f"  Text: {test_text}")
    console.print(f"  Tokens: {tokens[:10]}... ({len(tokens)} total)")
    console.print(f"  Pieces: {pieces[:10]}...")
    console.print(f"  Decoded: {decoded}")
    console.print(f"  Vocab size: {sp.get_piece_size()}")

    return 0


if __name__ == "__main__":
    exit(main())
