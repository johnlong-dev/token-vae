"""Evaluation tests for Token VAE."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon
from rich.console import Console
from rich.table import Table

from token_vae.model import TokenVAE
from token_vae.evaluation.config import EvaluationConfig
from token_vae.evaluation.utils import (
    bucket_by_quantiles,
    classify_tokens,
    compute_token_frequencies,
)
from token_vae.utils.pairs import (
    generate_pairs,
    get_valid_token_ids,
    load_pairs,
    save_pairs,
    pairs_to_id_set,
)
import sentencepiece as spm


def _seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch RNGs for deterministic evaluation."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TestResult:
    """Result of a single evaluation test."""
    name: str
    passed: bool
    score: float
    threshold: float
    details: dict
    description: str


def test_prior_decodability(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    num_samples: int = 1000,
    batch_size: int = 512,
    non_special_threshold: float = 0.85,
    unique_threshold: float = 0.30,
    top_k: int = 20,
) -> TestResult:
    """Test 1: Prior Decodability - Is the space walkable?

    Sample from N(0, I) and check if decoded tokens are valid.

    Pass criteria:
    - ≥85% decode to non-special tokens
    - ≥30% unique tokens among samples

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        num_samples: Number of prior samples
        non_special_threshold: Min fraction of non-special tokens
        unique_threshold: Min fraction of unique tokens

    Returns:
        TestResult with pass/fail and details
    """
    device = next(model.parameters()).device
    model.eval()

    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(), tokenizer.unk_id(),
        tokenizer.bos_id(), tokenizer.eos_id(),
        tokenizer.piece_to_id("<|endoftext|>"),
    }

    counts = np.zeros(vocab_size, dtype=np.int64)
    entropies: list[float] = []
    examples: list[str] = []
    non_special_count = 0
    samples_done = 0

    with torch.no_grad():
        while samples_done < num_samples:
            batch = min(batch_size, num_samples - samples_done)
            h = torch.randn(batch, model.d_model, device=device)
            logits = model.decode(h)

            preds = logits.argmax(dim=-1).cpu().numpy()
            counts += np.bincount(preds, minlength=vocab_size)

            non_special_count += sum(
                1 for t in preds if int(t) not in special_ids
            )

            if len(examples) < 10:
                for t in preds[: (10 - len(examples))]:
                    examples.append(tokenizer.id_to_piece(int(t)))

            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
            entropies.extend(entropy.tolist())

            samples_done += batch

    non_special_frac = non_special_count / num_samples
    unique_tokens = int((counts > 0).sum())
    unique_frac = unique_tokens / num_samples

    entropy_mean = float(np.mean(entropies)) if entropies else 0.0
    entropy_max = float(np.max(entropies)) if entropies else 0.0
    entropy_median = float(np.median(entropies)) if entropies else 0.0

    max_token_frequency = float(counts.max() / num_samples) if num_samples > 0 else 0.0
    tokens_seen_once = int((counts == 1).sum())
    tokens_seen_once_frac = float(tokens_seen_once / num_samples) if num_samples > 0 else 0.0

    # Gini coefficient on decoded token counts (diagnostic for concentration)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    total = sorted_counts.sum()
    gini = float((2 * np.sum(index * sorted_counts) - (n + 1) * total) / (n * total)) if total > 0 else 0.0

    # Top-k most frequent decoded tokens (exclude specials)
    counts_no_special = counts.copy()
    for sid in special_ids:
        if sid is not None and sid >= 0 and sid < vocab_size:
            counts_no_special[sid] = 0
    top_indices = np.argsort(-counts_no_special)[:top_k]
    top_tokens = [
        {
            "id": int(idx),
            "piece": tokenizer.id_to_piece(int(idx)),
            "count": int(counts_no_special[idx]),
            "fraction": float(counts_no_special[idx] / num_samples) if num_samples > 0 else 0.0,
        }
        for idx in top_indices
    ]

    passed = (non_special_frac >= non_special_threshold and
              unique_frac >= unique_threshold)

    return TestResult(
        name="Prior Decodability",
        passed=passed,
        score=non_special_frac,
        threshold=non_special_threshold,
        details={
            "non_special_fraction": non_special_frac,
            "unique_fraction": unique_frac,
            "unique_tokens": unique_tokens,
            "total_samples": num_samples,
            "examples": examples,
            "top_tokens": top_tokens,
            "max_token_frequency": max_token_frequency,
            "tokens_seen_once_fraction": tokens_seen_once_frac,
            "entropy_mean": entropy_mean,
            "entropy_max": entropy_max,
            "entropy_median": entropy_median,
            "log_vocab": float(np.log(vocab_size)) if vocab_size > 0 else 0.0,
            "gini_coefficient": gini,
        },
        description="Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens"
    )


def test_perturbation_stability(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    num_tokens: int = 100,
    sigma_levels: list[float] = [0.05, 0.1, 0.2],
    num_perturbations: int = 50,
    stability_threshold: float = 0.5,
    off_manifold_sigmas: list[float] | None = None,
    off_manifold_num_tokens: int = 200,
    off_manifold_samples_per_token: int = 1,
) -> TestResult:
    """Test 2: Perturbation Stability - Are there cliffs in the space?

    Perturb token embeddings and check if top-k predictions change smoothly.

    Pass criteria:
    - Top-k tokens should overlap significantly after small perturbations
    - Overlap should decrease gradually with larger σ

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        num_tokens: Number of tokens to test
        sigma_levels: Perturbation magnitudes to test
        num_perturbations: Perturbations per token per sigma
        stability_threshold: Min average overlap at σ=0.05

    Returns:
        TestResult with pass/fail and details
    """
    device = next(model.parameters()).device
    model.eval()

    # Sample random tokens (excluding special tokens)
    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(), tokenizer.unk_id(),
        tokenizer.bos_id(), tokenizer.eos_id(),
        tokenizer.piece_to_id("<|endoftext|>"),
    }
    valid_tokens = [i for i in range(vocab_size) if i not in special_ids]
    test_tokens = np.random.choice(valid_tokens, min(num_tokens, len(valid_tokens)), replace=False)

    k = 10  # Top-k to compare
    results_by_sigma = {}

    with torch.no_grad():
        for sigma in sigma_levels:
            overlaps = []

            for token_id in test_tokens:
                # Get original embedding
                token_tensor = torch.tensor([token_id], device=device)
                mu, logvar = model.encode(token_tensor)

                # Original top-k
                orig_logits = model.decode(mu)
                orig_topk = set(orig_logits.topk(k).indices[0].cpu().numpy())

                # Perturb and measure overlap
                for _ in range(num_perturbations):
                    noise = torch.randn_like(mu) * sigma
                    perturbed = mu + noise
                    pert_logits = model.decode(perturbed)
                    pert_topk = set(pert_logits.topk(k).indices[0].cpu().numpy())

                    overlap = len(orig_topk & pert_topk) / k
                    overlaps.append(overlap)

            results_by_sigma[sigma] = np.mean(overlaps)

    # Check if stability decreases gradually with sigma
    sigmas_sorted = sorted(results_by_sigma.keys())
    is_monotonic = all(
        results_by_sigma[sigmas_sorted[i]] >= results_by_sigma[sigmas_sorted[i+1]] - 0.1
        for i in range(len(sigmas_sorted) - 1)
    )

    # Pass if smallest sigma has high overlap and trend is reasonable
    smallest_sigma = min(sigma_levels)
    passed = (results_by_sigma[smallest_sigma] >= stability_threshold and is_monotonic)

    # Local sensitivity estimates (logit-space, not softmax-space)
    logit_sensitivity: list[float] = []
    margin_sensitivity: list[float] = []
    with torch.no_grad():
        lip_tokens = np.random.choice(valid_tokens, min(200, len(valid_tokens)), replace=False)
        lip_sigma = 0.05
        for token_id in lip_tokens:
            token_tensor = torch.tensor([token_id], device=device)
            mu_tok, _ = model.encode(token_tensor)
            orig_logits = model.decode(mu_tok)

            eps = torch.randn_like(mu_tok) * lip_sigma
            pert_logits = model.decode(mu_tok + eps)

            eps_norm = eps.norm().item()
            if eps_norm > 1e-10:
                # Logit sensitivity: ||Δlogits||₂ / ||ε||₂
                logit_diff = (pert_logits - orig_logits).norm().item()
                logit_sensitivity.append(logit_diff / eps_norm)

                # Margin sensitivity: Δ(logit_top1 − logit_top2) / ||ε||₂
                orig_top2 = orig_logits.topk(2).values[0]
                pert_top2 = pert_logits.topk(2).values[0]
                orig_margin = (orig_top2[0] - orig_top2[1]).item()
                pert_margin = (pert_top2[0] - pert_top2[1]).item()
                margin_sensitivity.append(abs(pert_margin - orig_margin) / eps_norm)

    def _percentile_stats(values: list[float]) -> dict:
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    logit_sens_stats = _percentile_stats(logit_sensitivity)
    margin_sens_stats = _percentile_stats(margin_sensitivity)

    # Optional off-manifold robustness sweep
    off_manifold_results = {}
    if off_manifold_sigmas:
        vocab_size = tokenizer.get_piece_size()
        special_ids = {
            tokenizer.pad_id(), tokenizer.unk_id(),
            tokenizer.bos_id(), tokenizer.eos_id(),
            tokenizer.piece_to_id("<|endoftext|>"),
        }
        valid_tokens = [i for i in range(vocab_size) if i not in special_ids]
        test_tokens = np.random.choice(
            valid_tokens, size=min(off_manifold_num_tokens, len(valid_tokens)), replace=False
        )

        with torch.no_grad():
            token_tensor = torch.tensor(test_tokens, device=device)
            mu, _ = model.encode(token_tensor)

            for sigma in off_manifold_sigmas:
                samples = off_manifold_samples_per_token
                mu_rep = mu.repeat_interleave(samples, dim=0)
                token_rep = token_tensor.repeat_interleave(samples)

                noise = torch.randn_like(mu_rep) * sigma
                z = mu_rep + noise
                logits = model.decode(z)

                preds = logits.argmax(dim=-1).cpu().numpy()
                pred_set = set(preds.tolist())

                safe_decode = sum(
                    1 for t in preds if int(t) not in special_ids
                ) / len(preds)
                unique_frac = len(pred_set) / len(preds) if len(preds) > 0 else 0.0
                top1_retention = (
                    (preds == token_rep.cpu().numpy()).mean() if len(preds) > 0 else 0.0
                )

                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()

                off_manifold_results[sigma] = {
                    "safe_decode_rate": float(safe_decode),
                    "unique_fraction": float(unique_frac),
                    "entropy_mean": float(np.mean(entropy)) if entropy.size > 0 else 0.0,
                    "entropy_max": float(np.max(entropy)) if entropy.size > 0 else 0.0,
                    "top1_retention": float(top1_retention),
                    "num_samples": int(len(preds)),
                }

    return TestResult(
        name="Perturbation Stability",
        passed=passed,
        score=results_by_sigma[smallest_sigma],
        threshold=stability_threshold,
        details={
            "overlaps_by_sigma": results_by_sigma,
            "is_monotonic": is_monotonic,
            "num_tokens_tested": len(test_tokens),
            "k": k,
            "off_manifold": off_manifold_results,
            "logit_sensitivity": logit_sens_stats,
            "margin_sensitivity": margin_sens_stats,
        },
        description="Measures decoder output stability under Gaussian perturbation of token embeddings"
    )


def test_interpolation_continuity(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    num_pairs: int = 50,
    num_steps: int = 20,
    max_js_threshold: float = 0.5,
    max_entropy_threshold: float = 8.0,
    top_k: int = 5,
    holdout_pairs: list[tuple[int, int]] | None = None,
    holdout_eval_pairs: int | None = None,
    exclude_pairs: set[int] | None = None,
    common_ids: set[int] | None = None,
    rare_ids: set[int] | None = None,
    subword_ids: set[int] | None = None,
    distance_quantiles: tuple[float, float] = (0.33, 0.66),
    prior_entropy_stats: dict | None = None,
    max_js_top_k: int = 10,
    cliff_threshold: float = 0.25,
    entropy_vs_prior_delta_threshold: float = 1.0,
    pair_seed: int | None = None,
) -> TestResult:
    """Test 3: Interpolation Continuity - Does traversal break decoding?

    Interpolate between token pairs and check for smooth probability changes.

    Pass criteria:
    - JS distance between adjacent steps should be bounded
    - Entropy should remain bounded (no degenerate distributions)

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        num_pairs: Number of token pairs to test
        num_steps: Interpolation steps per pair
        max_js_threshold: Max allowed JS distance between adjacent steps
        max_entropy_threshold: Max allowed entropy

    Returns:
        TestResult with pass/fail and details
    """
    device = next(model.parameters()).device
    model.eval()

    # Sample random token pairs
    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(), tokenizer.unk_id(),
        tokenizer.bos_id(), tokenizer.eos_id(),
        tokenizer.piece_to_id("<|endoftext|>"),
    }
    valid_tokens = [i for i in range(vocab_size) if i not in special_ids]

    def evaluate_pairs(pairs: list[tuple[int, int]]) -> dict:
        all_js_distances: list[float] = []
        all_entropies: list[float] = []
        max_js_observed = 0.0
        max_js_case = None
        top_entropy_cases: list[dict] = []
        pair_metrics: list[dict] = []

        with torch.no_grad():
            for token_a, token_b in pairs:
                token_a_piece = tokenizer.id_to_piece(int(token_a))
                token_b_piece = tokenizer.id_to_piece(int(token_b))

                # Compute distance between endpoints
                tokens = torch.tensor([int(token_a), int(token_b)], device=device)
                mu_pair, _ = model.encode(tokens)
                distance = float(torch.norm(mu_pair[0] - mu_pair[1]).item())

                # Interpolate
                h_interp, logits_interp = model.interpolate(
                    int(token_a), int(token_b), num_steps=num_steps
                )

                probs = F.softmax(logits_interp, dim=-1).cpu().numpy()
                probs = probs + 1e-10
                probs = probs / probs.sum(axis=-1, keepdims=True)
                h_norms = h_interp.norm(dim=-1).cpu().numpy()

                js_list: list[float] = []
                for i in range(len(probs) - 1):
                    js_distance = float(jensenshannon(probs[i], probs[i + 1]))
                    if np.isnan(js_distance):
                        continue
                    if js_distance < 0:
                        js_distance = 0.0
                    js_list.append(js_distance)
                    all_js_distances.append(js_distance)

                    if js_distance > max_js_observed:
                        max_js_observed = js_distance
                        alpha_from = i / (len(probs) - 1) if len(probs) > 1 else 0.0
                        alpha_to = (i + 1) / (len(probs) - 1) if len(probs) > 1 else 0.0

                        # Top-k probabilities at both steps
                        step_logits = logits_interp[i]
                        next_logits = logits_interp[i + 1]
                        step_probs = F.softmax(step_logits, dim=-1)
                        next_probs = F.softmax(next_logits, dim=-1)

                        step_top = step_probs.topk(max_js_top_k)
                        next_top = next_probs.topk(max_js_top_k)

                        step_top_list = [
                            {
                                "id": int(idx),
                                "piece": tokenizer.id_to_piece(int(idx)),
                                "prob": float(step_probs[idx].item()),
                            }
                            for idx in step_top.indices.cpu().numpy().tolist()
                        ]
                        next_top_list = [
                            {
                                "id": int(idx),
                                "piece": tokenizer.id_to_piece(int(idx)),
                                "prob": float(next_probs[idx].item()),
                            }
                            for idx in next_top.indices.cpu().numpy().tolist()
                        ]

                        max_js_case = {
                            "token_a": int(token_a),
                            "token_b": int(token_b),
                            "piece_a": token_a_piece,
                            "piece_b": token_b_piece,
                            "step_from": i,
                            "step_to": i + 1,
                            "alpha_from": float(alpha_from),
                            "alpha_to": float(alpha_to),
                            "js_distance": js_distance,
                            "js_divergence": js_distance ** 2,
                            "norm_from": float(h_norms[i]),
                            "norm_to": float(h_norms[i + 1]),
                            "topk_from": step_top_list,
                            "topk_to": next_top_list,
                        }

                entropies: list[float] = []
                for i, p in enumerate(probs):
                    entropy = float(-np.sum(p * np.log(p + 1e-10)))
                    entropies.append(entropy)
                    all_entropies.append(entropy)

                    if top_k > 0:
                        alpha = i / (len(probs) - 1) if len(probs) > 1 else 0.0
                        case = {
                            "token_a": int(token_a),
                            "token_b": int(token_b),
                            "piece_a": token_a_piece,
                            "piece_b": token_b_piece,
                            "step": i,
                            "alpha": float(alpha),
                            "entropy": float(entropy),
                            "latent_norm": float(h_norms[i]),
                        }
                        if len(top_entropy_cases) < top_k:
                            top_entropy_cases.append(case)
                            top_entropy_cases.sort(key=lambda x: x["entropy"], reverse=True)
                        elif entropy > top_entropy_cases[-1]["entropy"]:
                            top_entropy_cases.append(case)
                            top_entropy_cases.sort(key=lambda x: x["entropy"], reverse=True)
                            top_entropy_cases = top_entropy_cases[:top_k]

                top1 = logits_interp.argmax(dim=-1).cpu().numpy()
                flips = int(np.sum(top1[:-1] != top1[1:])) if len(top1) > 1 else 0
                switch_rate = float(flips / (len(top1) - 1)) if len(top1) > 1 else 0.0
                plateau_ratio = 1.0 - switch_rate if len(top1) > 1 else 1.0

                # Top-k Jaccard between consecutive steps
                topk_ids = logits_interp.topk(top_k, dim=-1).indices.cpu().numpy()
                jaccard_list = []
                for s in range(len(topk_ids) - 1):
                    a, b = set(topk_ids[s]), set(topk_ids[s + 1])
                    jaccard_list.append(len(a & b) / len(a | b) if (a | b) else 1.0)
                mean_topk_jaccard = float(np.mean(jaccard_list)) if jaccard_list else 1.0

                cliff_count = int(np.sum(np.array(js_list) > cliff_threshold)) if js_list else 0
                midpoint_entropy = entropies[len(entropies) // 2] if entropies else 0.0
                endpoint_entropy = (
                    (entropies[0] + entropies[-1]) / 2.0 if len(entropies) >= 2 else midpoint_entropy
                )

                pair_metrics.append({
                    "token_a": int(token_a),
                    "token_b": int(token_b),
                    "piece_a": token_a_piece,
                    "piece_b": token_b_piece,
                    "distance": distance,
                    "mean_js": float(np.mean(js_list)) if js_list else 0.0,
                    "max_js": float(np.max(js_list)) if js_list else 0.0,
                    "mean_js_divergence": (float(np.mean(js_list)) ** 2) if js_list else 0.0,
                    "max_js_divergence": (float(np.max(js_list)) ** 2) if js_list else 0.0,
                    "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
                    "max_entropy": float(np.max(entropies)) if entropies else 0.0,
                    "flip_count": flips,
                    "switch_rate": switch_rate,
                    "plateau_ratio": float(plateau_ratio),
                    "mean_topk_jaccard": mean_topk_jaccard,
                    "cliff_count": cliff_count,
                    "endpoint_entropy": float(endpoint_entropy),
                    "midpoint_entropy": float(midpoint_entropy),
                })

        mean_js = float(np.mean(all_js_distances)) if all_js_distances else 0.0
        max_js = float(max_js_observed) if all_js_distances else 0.0
        mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0
        max_entropy = float(np.max(all_entropies)) if all_entropies else 0.0
        endpoint_entropy_mean = (
            float(np.mean([m["endpoint_entropy"] for m in pair_metrics]))
            if pair_metrics else 0.0
        )
        midpoint_entropy_mean = (
            float(np.mean([m["midpoint_entropy"] for m in pair_metrics]))
            if pair_metrics else 0.0
        )

        mean_flip_count = (
            float(np.mean([m["flip_count"] for m in pair_metrics]))
            if pair_metrics else 0.0
        )
        mean_switch_rate = (
            float(np.mean([m["switch_rate"] for m in pair_metrics]))
            if pair_metrics else 0.0
        )
        mean_topk_jaccard = (
            float(np.mean([m["mean_topk_jaccard"] for m in pair_metrics]))
            if pair_metrics else 1.0
        )
        mean_plateau_ratio = (
            float(np.mean([m["plateau_ratio"] for m in pair_metrics]))
            if pair_metrics else 1.0
        )

        return {
            "mean_js": mean_js,
            "max_js": max_js,
            "mean_js_divergence": mean_js ** 2,
            "max_js_divergence": max_js ** 2,
            "mean_entropy": mean_entropy,
            "max_entropy": max_entropy,
            "endpoint_entropy_mean": endpoint_entropy_mean,
            "midpoint_entropy_mean": midpoint_entropy_mean,
            "mean_flip_count": mean_flip_count,
            "mean_switch_rate": mean_switch_rate,
            "mean_topk_jaccard": mean_topk_jaccard,
            "mean_plateau_ratio": mean_plateau_ratio,
            "pair_metrics": pair_metrics,
            "top_entropy_cases": top_entropy_cases,
            "max_js_case": max_js_case,
        }

    # Build train pairs (exclude holdout)
    train_pairs = generate_pairs(
        valid_tokens=valid_tokens,
        num_pairs=num_pairs,
        seed=pair_seed,
        exclude_pairs=exclude_pairs,
        vocab_size=vocab_size,
    )

    train_metrics = evaluate_pairs(train_pairs)

    # Holdout evaluation
    holdout_metrics = None
    if holdout_pairs:
        eval_count = holdout_eval_pairs or len(holdout_pairs)
        eval_pairs = holdout_pairs[:eval_count]
        holdout_metrics = evaluate_pairs(eval_pairs)

    # Stratified metrics (train pairs only)
    stratified = {}
    if train_metrics["pair_metrics"] and common_ids is not None and rare_ids is not None:
        def aggregate(metrics: list[dict]) -> dict:
            if not metrics:
                return {}
            return {
                "mean_js": float(np.mean([m["mean_js"] for m in metrics])),
                "max_js": float(np.max([m["max_js"] for m in metrics])),
                "mean_js_divergence": float(np.mean([m["mean_js_divergence"] for m in metrics])),
                "max_js_divergence": float(np.max([m["max_js_divergence"] for m in metrics])),
                "mean_entropy": float(np.mean([m["mean_entropy"] for m in metrics])),
                "max_entropy": float(np.max([m["max_entropy"] for m in metrics])),
                "flip_count_mean": float(np.mean([m["flip_count"] for m in metrics])),
                "switch_rate_mean": float(np.mean([m["switch_rate"] for m in metrics])),
                "cliff_count_mean": float(np.mean([m["cliff_count"] for m in metrics])),
                "plateau_ratio_mean": float(np.mean([m["plateau_ratio"] for m in metrics])),
                "mean_topk_jaccard": float(np.mean([m["mean_topk_jaccard"] for m in metrics])),
                "num_pairs": int(len(metrics)),
            }

        common_common = [
            m for m in train_metrics["pair_metrics"]
            if m["token_a"] in common_ids and m["token_b"] in common_ids
        ]
        rare_rare = [
            m for m in train_metrics["pair_metrics"]
            if m["token_a"] in rare_ids and m["token_b"] in rare_ids
        ]
        common_rare = [
            m for m in train_metrics["pair_metrics"]
            if (m["token_a"] in common_ids and m["token_b"] in rare_ids) or
               (m["token_b"] in common_ids and m["token_a"] in rare_ids)
        ]
        stratified["common_common"] = aggregate(common_common)
        stratified["rare_rare"] = aggregate(rare_rare)
        stratified["common_rare"] = aggregate(common_rare)

        if subword_ids is not None:
            subword_subword = [
                m for m in train_metrics["pair_metrics"]
                if m["token_a"] in subword_ids and m["token_b"] in subword_ids
            ]
            stratified["subword_subword"] = aggregate(subword_subword)

    # Distance buckets (train pairs only)
    distance_buckets = {}
    if train_metrics["pair_metrics"]:
        distances = [m["distance"] for m in train_metrics["pair_metrics"]]
        q1, q2 = bucket_by_quantiles(distances, distance_quantiles)

        def bucket_name(d: float) -> str:
            if d <= q1:
                return "near"
            if d <= q2:
                return "medium"
            return "far"

        buckets: dict[str, list[dict]] = {"near": [], "medium": [], "far": []}
        for m in train_metrics["pair_metrics"]:
            buckets[bucket_name(m["distance"])].append(m)

        for name, metrics in buckets.items():
            if metrics:
                distance_buckets[name] = {
                    "mean_js": float(np.mean([m["mean_js"] for m in metrics])),
                    "max_js": float(np.max([m["max_js"] for m in metrics])),
                    "mean_js_divergence": float(np.mean([m["mean_js_divergence"] for m in metrics])),
                    "max_js_divergence": float(np.max([m["max_js_divergence"] for m in metrics])),
                    "mean_entropy": float(np.mean([m["mean_entropy"] for m in metrics])),
                    "max_entropy": float(np.max([m["max_entropy"] for m in metrics])),
                    "flip_count_mean": float(np.mean([m["flip_count"] for m in metrics])),
                    "switch_rate_mean": float(np.mean([m["switch_rate"] for m in metrics])),
                    "cliff_count_mean": float(np.mean([m["cliff_count"] for m in metrics])),
                    "plateau_ratio_mean": float(np.mean([m["plateau_ratio"] for m in metrics])),
                    "mean_topk_jaccard": float(np.mean([m["mean_topk_jaccard"] for m in metrics])),
                    "num_pairs": int(len(metrics)),
                }
        distance_buckets["thresholds"] = {"near_max": q1, "medium_max": q2}

    entropy_vs_prior = None
    if prior_entropy_stats:
        mean_delta = float(train_metrics["mean_entropy"] - prior_entropy_stats.get("entropy_mean", 0.0))
        max_delta = float(train_metrics["max_entropy"] - prior_entropy_stats.get("entropy_max", 0.0))
        entropy_vs_prior = {
            "prior_mean": prior_entropy_stats.get("entropy_mean", 0.0),
            "prior_max": prior_entropy_stats.get("entropy_max", 0.0),
            "interp_mean": train_metrics["mean_entropy"],
            "interp_max": train_metrics["max_entropy"],
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "flag": abs(mean_delta) > entropy_vs_prior_delta_threshold or abs(max_delta) > entropy_vs_prior_delta_threshold,
        }

    train_mean_js = train_metrics["mean_js"]
    train_max_js = train_metrics["max_js"]
    train_mean_entropy = train_metrics["mean_entropy"]
    train_max_entropy = train_metrics["max_entropy"]

    train_pass = (train_max_js <= max_js_threshold and train_max_entropy <= max_entropy_threshold)
    holdout_pass = True
    if holdout_metrics is not None:
        holdout_pass = (
            holdout_metrics["max_js"] <= max_js_threshold
            and holdout_metrics["max_entropy"] <= max_entropy_threshold
        )

    passed = train_pass and holdout_pass
    score = max(
        train_mean_js,
        holdout_metrics["mean_js"] if holdout_metrics is not None else train_mean_js,
    )
    max_js_observed = max(
        train_max_js,
        holdout_metrics["max_js"] if holdout_metrics is not None else train_max_js,
    )
    max_entropy_observed = max(
        train_max_entropy,
        holdout_metrics["max_entropy"] if holdout_metrics is not None else train_max_entropy,
    )

    return TestResult(
        name="Interpolation Continuity",
        passed=passed,
        score=score,
        threshold=max_js_threshold,
        details={
            "mean_js_distance": train_mean_js,
            "max_js_distance": train_max_js,
            "mean_js_divergence": train_metrics["mean_js_divergence"],
            "max_js_divergence": train_metrics["max_js_divergence"],
            "mean_entropy": train_mean_entropy,
            "max_entropy": train_max_entropy,
            "gated_score": score,
            "gated_max_js_distance": max_js_observed,
            "gated_max_entropy": max_entropy_observed,
            "num_pairs": num_pairs,
            "num_steps": num_steps,
            "train_pass": train_pass,
            "holdout_pass": holdout_pass,
            "holdout_used_for_pass": holdout_metrics is not None,
            "mean_flip_count": train_metrics.get("mean_flip_count", 0.0),
            "mean_switch_rate": train_metrics.get("mean_switch_rate", 0.0),
            "mean_topk_jaccard": train_metrics.get("mean_topk_jaccard", 1.0),
            "mean_plateau_ratio": train_metrics.get("mean_plateau_ratio", 1.0),
            "top_entropy_cases": train_metrics["top_entropy_cases"],
            "max_js_case": train_metrics["max_js_case"],
            "train_pairs": train_pairs,
            "train_metrics": train_metrics,
            "holdout_metrics": holdout_metrics,
            "stratified_metrics": stratified,
            "distance_buckets": distance_buckets,
            "entropy_vs_prior": entropy_vs_prior,
            "endpoint_entropy_mean": train_metrics.get("endpoint_entropy_mean", 0.0),
            "midpoint_entropy_mean": train_metrics.get("midpoint_entropy_mean", 0.0),
        },
        description="Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds"
    )


def test_reconstruction_accuracy(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    accuracy_threshold: float = 0.90,
) -> TestResult:
    """Test 4: Reconstruction Accuracy - Can tokens reconstruct themselves?

    Encode all tokens using μ only (no sampling) and check reconstruction.

    Pass criteria:
    - Top-1 accuracy ≥ 90%

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        accuracy_threshold: Minimum required accuracy

    Returns:
        TestResult with pass/fail and details
    """
    device = next(model.parameters()).device
    model.eval()

    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(), tokenizer.unk_id(),
        tokenizer.bos_id(), tokenizer.eos_id(),
        tokenizer.piece_to_id("<|endoftext|>"),
    }

    correct = 0
    total = 0
    top5_correct = 0

    # Process in batches for efficiency
    batch_size = 256
    all_tokens = torch.arange(vocab_size, device=device)

    with torch.no_grad():
        for i in range(0, vocab_size, batch_size):
            batch = all_tokens[i:i + batch_size]

            # Forward pass with deterministic=True (use μ only)
            logits, mu, logvar, h = model(batch, deterministic=True)

            # Get predictions
            preds = logits.argmax(dim=-1)
            top5_preds = logits.topk(5, dim=-1).indices

            for j, token_id in enumerate(batch):
                token_id_int = token_id.item()
                if token_id_int in special_ids:
                    continue

                total += 1
                if preds[j].item() == token_id_int:
                    correct += 1
                if token_id_int in top5_preds[j].cpu().numpy():
                    top5_correct += 1

    accuracy = correct / total if total > 0 else 0
    top5_accuracy = top5_correct / total if total > 0 else 0

    # Posterior collapse diagnostic: per-dimension KL
    all_kl_per_dim: list[torch.Tensor] = []
    all_mu_norms: list[float] = []
    with torch.no_grad():
        for i in range(0, vocab_size, batch_size):
            batch = all_tokens[i:i + batch_size]
            non_special_mask = torch.tensor(
                [b.item() not in special_ids for b in batch], dtype=torch.bool
            )
            if not non_special_mask.any():
                continue
            batch_ns = batch[non_special_mask]
            mu_b, logvar_b = model.encode(batch_ns)
            kl_dim = 0.5 * (mu_b.pow(2) + logvar_b.exp() - logvar_b - 1)  # (B, d_model)
            all_kl_per_dim.append(kl_dim.cpu())
            all_mu_norms.extend(mu_b.norm(dim=-1).cpu().tolist())

    collapsed_dims = 0
    collapsed_fraction = 0.0
    mu_norm_mean = 0.0
    mu_norm_std = 0.0
    if all_kl_per_dim:
        kl_cat = torch.cat(all_kl_per_dim, dim=0)  # (total_tokens, d_model)
        mean_kl_per_dim = kl_cat.mean(dim=0)  # (d_model,)
        collapsed_dims = int((mean_kl_per_dim < 0.01).sum().item())
        collapsed_fraction = collapsed_dims / mean_kl_per_dim.shape[0]
        mu_norm_mean = float(np.mean(all_mu_norms))
        mu_norm_std = float(np.std(all_mu_norms))

    passed = accuracy >= accuracy_threshold

    # Find some failure cases
    failures = []
    with torch.no_grad():
        for i in range(0, min(1000, vocab_size)):
            if i in special_ids:
                continue
            token_tensor = torch.tensor([i], device=device)
            logits, _, _, _ = model(token_tensor, deterministic=True)
            pred = logits.argmax(dim=-1).item()
            if pred != i:
                failures.append({
                    "input": tokenizer.id_to_piece(i),
                    "predicted": tokenizer.id_to_piece(pred),
                    "input_id": i,
                    "predicted_id": pred,
                })
                if len(failures) >= 10:
                    break

    return TestResult(
        name="Reconstruction Accuracy",
        passed=passed,
        score=accuracy,
        threshold=accuracy_threshold,
        details={
            "top1_accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "correct": correct,
            "total": total,
            "example_failures": failures,
            "collapsed_dims": collapsed_dims,
            "collapsed_fraction": collapsed_fraction,
            "mu_norm_mean": mu_norm_mean,
            "mu_norm_std": mu_norm_std,
        },
        description="Verifies that tokens are recoverable when encoded to their posterior mean and decoded"
    )


def test_diffusion_walk(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    num_walks: int = 100,
    num_steps: int = 50,
    beta: float = 0.01,
    beta_schedule: str = "constant",
    non_special_threshold: float = 0.95,
    entropy_threshold: float = 8.0,
    min_unique_fraction_threshold: float = 0.10,
    min_mean_change_rate_threshold: float = 0.01,
    start_from: str = "prior",
) -> TestResult:
    """Test: Diffusion Random Walk — Is the space usable for diffusion?

    Simulates a forward diffusion random walk in latent space and checks
    that decoded tokens remain valid throughout the trajectory.

    Algorithm:
        1. Sample starting points from N(0,I) or token embeddings
        2. For T steps: z_{t+1} = sqrt(1-β_t) * z_t + sqrt(β_t) * ε
        3. At each step, decode and record non-special rate, entropy, etc.
        4. Pass if min non-special rate ≥ threshold AND max mean entropy ≤ threshold

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        num_walks: Number of random walks
        num_steps: Steps per walk
        beta: Noise schedule parameter
        beta_schedule: "constant", "linear", or "cosine"
        non_special_threshold: Min non-special rate at any step
        entropy_threshold: Max mean entropy at any step
        start_from: "prior" (sample N(0,I)) or "embeddings" (sample token μ)

    Returns:
        TestResult with per-step trajectory and aggregate stats
    """
    device = next(model.parameters()).device
    model.eval()

    vocab_size = tokenizer.get_piece_size()
    special_ids = {
        tokenizer.pad_id(), tokenizer.unk_id(),
        tokenizer.bos_id(), tokenizer.eos_id(),
        tokenizer.piece_to_id("<|endoftext|>"),
    }

    # Build beta schedule
    if beta_schedule == "linear":
        betas = torch.linspace(0.0, beta, num_steps, device=device)
    elif beta_schedule == "cosine":
        t = torch.linspace(0, 1, num_steps, device=device)
        betas = beta * (1 - torch.cos(t * torch.pi / 2))
    else:  # constant
        betas = torch.full((num_steps,), beta, device=device)

    # Sample starting points
    with torch.no_grad():
        if start_from == "embeddings":
            valid_tokens = [i for i in range(vocab_size) if i not in special_ids]
            chosen = np.random.choice(valid_tokens, size=num_walks, replace=True)
            token_tensor = torch.tensor(chosen, device=device)
            mu, _ = model.encode(token_tensor)
            z = mu.clone()
        else:  # prior
            z = torch.randn(num_walks, model.d_model, device=device)

        # Per-step metrics
        step_non_special_rates: list[float] = []
        step_entropy_means: list[float] = []
        step_unique_tokens: list[int] = []
        step_unique_fractions: list[float] = []
        step_norm_means: list[float] = []
        step_change_rates: list[float] = []
        prev_preds = None

        for t in range(num_steps):
            # Decode current z
            logits = model.decode(z)
            preds = logits.argmax(dim=-1).cpu().numpy()

            non_special = sum(1 for p in preds if int(p) not in special_ids) / len(preds)
            step_non_special_rates.append(float(non_special))

            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
            step_entropy_means.append(float(np.mean(entropy)))

            unique_tokens = int(len(set(preds.tolist())))
            step_unique_tokens.append(unique_tokens)
            step_unique_fractions.append(float(unique_tokens / len(preds)) if len(preds) > 0 else 0.0)
            step_norm_means.append(float(z.norm(dim=-1).mean().item()))
            if prev_preds is not None:
                change_rate = float(np.mean(preds != prev_preds))
                step_change_rates.append(change_rate)
            prev_preds = preds.copy()

            # Forward diffusion step
            beta_t = betas[t]
            z = torch.sqrt(1 - beta_t) * z + torch.sqrt(beta_t) * torch.randn_like(z)

    min_non_special = min(step_non_special_rates)
    max_entropy_mean = max(step_entropy_means)
    min_unique_fraction = min(step_unique_fractions) if step_unique_fractions else 0.0
    mean_change_rate = float(np.mean(step_change_rates)) if step_change_rates else 0.0
    passed = (
        min_non_special >= non_special_threshold
        and max_entropy_mean <= entropy_threshold
        and min_unique_fraction >= min_unique_fraction_threshold
        and mean_change_rate >= min_mean_change_rate_threshold
    )

    return TestResult(
        name="Diffusion Walk",
        passed=passed,
        score=min_non_special,
        threshold=non_special_threshold,
        details={
            "min_non_special_rate": min_non_special,
            "mean_non_special_rate": float(np.mean(step_non_special_rates)),
            "max_entropy_mean": max_entropy_mean,
            "mean_entropy_mean": float(np.mean(step_entropy_means)),
            "num_walks": num_walks,
            "num_steps": num_steps,
            "beta": beta,
            "beta_schedule": beta_schedule,
            "start_from": start_from,
            "non_special_threshold": non_special_threshold,
            "entropy_threshold": entropy_threshold,
            "min_unique_fraction": min_unique_fraction,
            "min_unique_fraction_threshold": min_unique_fraction_threshold,
            "mean_change_rate": mean_change_rate,
            "min_mean_change_rate_threshold": min_mean_change_rate_threshold,
            "step_non_special_rates": step_non_special_rates,
            "step_entropy_means": step_entropy_means,
            "step_unique_tokens": step_unique_tokens,
            "step_unique_fractions": step_unique_fractions,
            "step_change_rates": step_change_rates,
            "step_norm_means": step_norm_means,
        },
        description="Simulates forward-diffusion trajectories and verifies that decoded tokens remain non-degenerate"
    )


def test_metric_integrity(
    prior_details: dict,
    interp_details: dict,
    recon_details: dict,
    interp_entropy_threshold: float,
    config: EvaluationConfig,
) -> TestResult:
    """Cross-metric integrity checks to catch trivial or collapsed solutions."""
    triggers: list[str] = []

    prior_unique = prior_details.get("unique_fraction", 0.0)
    prior_entropy_mean = prior_details.get("entropy_mean", 0.0)
    recon_top1 = recon_details.get("top1_accuracy", 0.0)
    interp_max_entropy = interp_details.get("max_entropy", 0.0)

    if interp_max_entropy <= interp_entropy_threshold:
        if prior_unique < config.integrity_prior_unique_min:
            triggers.append(
                f"Interpolation looks good but prior unique fraction is low ({prior_unique:.3f})"
            )
        if prior_entropy_mean < config.integrity_prior_entropy_min:
            triggers.append(
                f"Interpolation looks good but prior entropy is low ({prior_entropy_mean:.3f})"
            )

    if recon_top1 >= config.integrity_recon_high and prior_unique < config.integrity_prior_unique_min:
        triggers.append(
            f"Reconstruction is high ({recon_top1:.3f}) while prior unique fraction is low ({prior_unique:.3f})"
        )

    passed = len(triggers) == 0

    return TestResult(
        name="Metric Integrity",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        details={
            "triggers": triggers,
            "prior_unique_fraction": prior_unique,
            "prior_entropy_mean": prior_entropy_mean,
            "recon_top1": recon_top1,
            "interp_max_entropy": interp_max_entropy,
        },
        description="Cross-validates metrics to detect inconsistencies indicative of posterior collapse or degenerate solutions"
    )


def run_all_tests(
    model: TokenVAE,
    tokenizer: spm.SentencePieceProcessor,
    verbose: bool = True,
    config: EvaluationConfig | None = None,
) -> list[TestResult]:
    """Run all evaluation tests.

    Args:
        model: Trained TokenVAE
        tokenizer: SentencePiece tokenizer
        verbose: Print results to console
        config: Evaluation configuration

    Returns:
        List of TestResults
    """
    config = config or EvaluationConfig()
    _seed_everything(config.seed)
    console = Console()

    if verbose:
        console.print("\n[bold blue]Running Evaluation Tests[/bold blue]\n")

    # Optional token frequency data
    token_freqs = None
    common_ids = rare_ids = subword_ids = None
    if config.data_file:
        token_freqs = compute_token_frequencies(
            config.data_file, tokenizer, max_tokens=config.data_max_tokens
        )
        common_ids, rare_ids, subword_ids = classify_tokens(
            tokenizer, token_freqs, config.common_frac, config.rare_frac
        )

    # Held-out interpolation pairs
    holdout_pairs = None
    exclude_pairs = None
    if config.interp_holdout_pairs_path or config.interp_holdout_num_pairs > 0:
        path = config.interp_holdout_pairs_path or "artifacts/interp/holdout_pairs.json"
        if Path(path).exists():
            holdout_pairs = load_pairs(path)
        else:
            valid_tokens = get_valid_token_ids(tokenizer)
            holdout_pairs = generate_pairs(
                valid_tokens=valid_tokens,
                num_pairs=config.interp_holdout_num_pairs,
                seed=config.interp_holdout_seed,
                vocab_size=tokenizer.get_piece_size(),
            )
            save_pairs(path, holdout_pairs)
        exclude_pairs = pairs_to_id_set(holdout_pairs, tokenizer.get_piece_size())

    results: list[TestResult] = []

    def run_and_log(name: str, fn: Callable[[], TestResult]) -> TestResult:
        if verbose:
            console.print(f"Running {name}...", end=" ")
        result = fn()
        results.append(result)
        if verbose:
            status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
            console.print(f"{status} (score: {result.score:.4f}, threshold: {result.threshold})")
        return result

    prior_result = run_and_log(
        "Prior Decodability",
        lambda: test_prior_decodability(
            model,
            tokenizer,
            num_samples=config.prior_num_samples,
            batch_size=config.prior_batch_size,
            top_k=config.prior_top_k,
        ),
    )

    run_and_log(
        "Perturbation Stability",
        lambda: test_perturbation_stability(
            model,
            tokenizer,
            off_manifold_sigmas=list(config.off_manifold_sigmas),
            off_manifold_num_tokens=config.off_manifold_num_tokens,
            off_manifold_samples_per_token=config.off_manifold_samples_per_token,
        ),
    )

    interp_result = run_and_log(
        "Interpolation Continuity",
        lambda: test_interpolation_continuity(
            model,
            tokenizer,
            num_pairs=config.interp_num_pairs,
            num_steps=config.interp_num_steps,
            max_js_threshold=config.interp_max_js_threshold,
            max_entropy_threshold=config.interp_max_entropy_threshold,
            holdout_pairs=holdout_pairs,
            holdout_eval_pairs=config.interp_holdout_eval_pairs,
            exclude_pairs=exclude_pairs,
            common_ids=common_ids,
            rare_ids=rare_ids,
            subword_ids=subword_ids,
            distance_quantiles=config.distance_quantiles,
            prior_entropy_stats=prior_result.details,
            max_js_top_k=config.interp_max_js_top_k,
            entropy_vs_prior_delta_threshold=config.entropy_vs_prior_delta_threshold,
            pair_seed=config.seed,
        ),
    )

    recon_result = run_and_log(
        "Reconstruction Accuracy",
        lambda: test_reconstruction_accuracy(model, tokenizer),
    )

    if config.integrity_checks:
        run_and_log(
            "Metric Integrity",
            lambda: test_metric_integrity(
                prior_result.details,
                interp_result.details,
                recon_result.details,
                interp_entropy_threshold=config.interp_max_entropy_threshold,
                config=config,
            ),
        )

    if config.diffusion_walk_enabled:
        run_and_log(
            "Diffusion Walk",
            lambda: test_diffusion_walk(
                model,
                tokenizer,
                num_walks=config.diffusion_num_walks,
                num_steps=config.diffusion_num_steps,
                beta=config.diffusion_beta,
                beta_schedule=config.diffusion_beta_schedule,
                non_special_threshold=config.diffusion_non_special_threshold,
                entropy_threshold=config.diffusion_entropy_threshold,
                min_unique_fraction_threshold=config.diffusion_min_unique_fraction_threshold,
                min_mean_change_rate_threshold=config.diffusion_min_mean_change_rate_threshold,
                start_from=config.diffusion_start_from,
            ),
        )

    if verbose:
        console.print()
        table = Table(title="Test Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")

        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            style = "green" if result.passed else "red"
            table.add_row(
                result.name,
                f"[{style}]{status}[/{style}]",
                f"{result.score:.4f}",
                f"{result.threshold:.4f}",
            )

        console.print(table)

        all_passed = all(r.passed for r in results)
        if all_passed:
            console.print("\n[bold green]All evaluation criteria satisfied.[/bold green]")
        else:
            failed = [r.name for r in results if not r.passed]
            console.print(f"\n[bold red]Failed tests: {', '.join(failed)}[/bold red]")

    return results
