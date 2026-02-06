"""Evaluation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import sentencepiece as spm

from token_vae.data.datasets import TokenIdentityDataset


def compute_token_frequencies(
    text_file: str,
    tokenizer: spm.SentencePieceProcessor,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Compute token frequency counts from a corpus."""
    dataset = TokenIdentityDataset(text_file, tokenizer, max_tokens=max_tokens)
    counts = np.bincount(dataset.tokens, minlength=tokenizer.get_piece_size())
    return counts


def classify_tokens(
    tokenizer: spm.SentencePieceProcessor,
    token_freqs: np.ndarray,
    common_frac: float,
    rare_frac: float,
) -> tuple[set[int], set[int], set[int]]:
    """Classify tokens into common, rare, and subword sets."""
    vocab_size = tokenizer.get_piece_size()
    ids = np.arange(vocab_size)
    freqs = token_freqs.astype(np.int64)

    # Only consider tokens with non-zero frequency for common/rare
    non_zero = freqs > 0
    ids_nz = ids[non_zero]
    freqs_nz = freqs[non_zero]

    if ids_nz.size == 0:
        return set(), set(), set()

    # Sort by frequency descending
    order = np.argsort(-freqs_nz)
    ids_sorted = ids_nz[order]

    common_k = max(1, int(len(ids_sorted) * common_frac))
    rare_k = max(1, int(len(ids_sorted) * rare_frac))

    common_ids = set(ids_sorted[:common_k].tolist())
    rare_ids = set(ids_sorted[-rare_k:].tolist())

    subword_ids: set[int] = set()
    for i in range(vocab_size):
        piece = tokenizer.id_to_piece(int(i))
        if not piece.startswith("▁"):
            subword_ids.add(int(i))

    return common_ids, rare_ids, subword_ids


def bucket_by_quantiles(
    values: Iterable[float],
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    """Compute bucket thresholds for near/medium/far."""
    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(arr, quantiles)
    return float(q1), float(q2)
