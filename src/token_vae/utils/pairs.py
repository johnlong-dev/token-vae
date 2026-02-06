"""Helpers for interpolation pair handling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import sentencepiece as spm


def make_pair_id(a: int, b: int, vocab_size: int) -> int:
    """Create a canonical pair id for unordered token pairs."""
    if a > b:
        a, b = b, a
    return a * vocab_size + b


def pairs_to_id_set(pairs: Iterable[tuple[int, int]], vocab_size: int) -> set[int]:
    """Convert list of pairs into a set of canonical pair ids."""
    return {make_pair_id(a, b, vocab_size) for a, b in pairs}


def get_valid_token_ids(tokenizer: spm.SentencePieceProcessor) -> list[int]:
    """Get non-special token ids."""
    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(),
        tokenizer.unk_id(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
    }
    end_id = tokenizer.piece_to_id("<|endoftext|>")
    if end_id is not None and end_id >= 0:
        special_ids.add(end_id)
    return [i for i in range(vocab_size) if i not in special_ids]


def generate_pairs(
    valid_tokens: list[int],
    num_pairs: int,
    seed: int | None = None,
    exclude_pairs: set[int] | None = None,
    vocab_size: int | None = None,
) -> list[tuple[int, int]]:
    """Generate unique random token pairs."""
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()

    if exclude_pairs is not None and vocab_size is None:
        raise ValueError("vocab_size is required when exclude_pairs is provided")

    max_attempts = max(10_000, num_pairs * 20)
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        a, b = rng.sample(valid_tokens, 2)
        pid = make_pair_id(a, b, vocab_size) if vocab_size is not None else None
        if pid is not None:
            if pid in seen:
                attempts += 1
                continue
            if exclude_pairs and pid in exclude_pairs:
                attempts += 1
                continue
            seen.add(pid)
        pairs.append((a, b))
        attempts += 1

    return pairs


def save_pairs(path: str, pairs: list[tuple[int, int]]) -> str:
    """Save pairs to a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    return str(output)


def load_pairs(path: str) -> list[tuple[int, int]]:
    """Load pairs from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(int(a), int(b)) for a, b in data]
