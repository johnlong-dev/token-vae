"""Report generation for Token VAE evaluation."""

from datetime import datetime
from pathlib import Path

from token_vae.evaluation.tests import TestResult


def generate_report(
    test_results: list[TestResult],
    figures: dict[str, str],
    history: dict | None = None,
    output_path: str = "artifacts/reports/evaluation_report.md",
) -> str:
    """Generate a markdown evaluation report.

    Args:
        test_results: List of test results
        figures: Dictionary mapping figure names to paths
        history: Optional training history
        output_path: Path to save the report

    Returns:
        Path to the saved report
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build report content
    lines = [
        "# Token VAE Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This report presents quantitative evaluation of the learned latent space",
        "against criteria required for continuous diffusion over token embeddings.",
        "The evaluation tests whether the variationally-regularized embedding space",
        "satisfies four properties:",
        "",
        r"1. **Prior coverage** — Samples from the standard normal prior $\mathcal{N}(0, I)$ decode to non-degenerate vocabulary tokens",
        "2. **Local smoothness** — Small perturbations to embeddings produce gradual changes in decoder output",
        "3. **Interpolation continuity** — Linear paths between token embeddings remain decodable with bounded divergence",
        "4. **Reconstruction fidelity** — Tokens are recoverable from their posterior mean embeddings",
        "",
        "---",
        "",
        "## Test Results Summary",
        "",
    ]

    # Summary table
    all_passed = all(r.passed for r in test_results)
    overall_status = "**All evaluation criteria satisfied.**" if all_passed else "**One or more evaluation criteria not satisfied.**"
    lines.append(overall_status)
    lines.append("")

    lines.append("| Test | Status | Score | Threshold |")
    lines.append("|------|--------|-------|-----------|")

    for result in test_results:
        status = "Pass" if result.passed else "Fail"
        lines.append(f"| {result.name} | {status} | {result.score:.4f} | {result.threshold:.4f} |")

    lines.append("")

    # Detailed results
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Test Results")
    lines.append("")

    for result in test_results:
        lines.append(f"### {result.name}")
        lines.append("")
        lines.append(f"**Description**: {result.description}")
        lines.append("")
        lines.append(f"**Result**: {'PASSED' if result.passed else 'FAILED'}")
        lines.append(f"- Score: {result.score:.4f}")
        lines.append(f"- Threshold: {result.threshold:.4f}")
        lines.append("")

        # Add test-specific details
        lines.append("**Details**:")
        lines.append("")

        if result.name == "Prior Decodability":
            details = result.details
            lines.append(f"- Non-special token fraction: {details['non_special_fraction']:.2%}")
            lines.append(f"- Unique token fraction: {details['unique_fraction']:.2%}")
            lines.append(f"- Unique tokens decoded: {details['unique_tokens']} / {details['total_samples']}")
            if "max_token_frequency" in details:
                lines.append(f"- Max token frequency: {details['max_token_frequency']:.2%}")
            if "tokens_seen_once_fraction" in details:
                lines.append(f"- Tokens seen once: {details['tokens_seen_once_fraction']:.2%}")
            if "entropy_mean" in details:
                lines.append(
                    f"- Prior entropy (mean/median/max): "
                    f"{details['entropy_mean']:.2f} / "
                    f"{details['entropy_median']:.2f} / "
                    f"{details['entropy_max']:.2f}"
                )
            if "log_vocab" in details:
                lines.append(f"- log(V): {details['log_vocab']:.2f}")
            if "gini_coefficient" in details:
                lines.append(f"- Gini coefficient: {details['gini_coefficient']:.4f}")
            lines.append("")
            lines.append("Example decoded tokens from prior samples:")
            lines.append("```")
            for ex in details['examples'][:5]:
                lines.append(f"  {repr(ex)}")
            lines.append("```")
            if details.get("top_tokens"):
                lines.append("")
                lines.append("Top decoded tokens:")
                lines.append("```")
                for tok in details["top_tokens"]:
                    lines.append(
                        f"  {tok['piece']} ({tok['id']}): {tok['fraction']:.2%} ({tok['count']})"
                    )
                lines.append("```")

        elif result.name == "Perturbation Stability":
            details = result.details
            lines.append("Top-k overlap by perturbation σ:")
            lines.append("")
            for sigma, overlap in sorted(details['overlaps_by_sigma'].items()):
                lines.append(f"- σ={sigma}: {overlap:.2%} overlap")
            lines.append("")
            lines.append(f"- Monotonic decrease: {'Yes' if details['is_monotonic'] else 'No'}")
            if details.get("logit_sensitivity"):
                ls = details["logit_sensitivity"]
                ms = details["margin_sensitivity"]
                lines.append("")
                lines.append("Local sensitivity (logit-space, σ=0.05):")
                lines.append(f"- Logit sensitivity (||Δlogits||/||ε||): "
                             f"p50={ls['p50']:.2e}, p90={ls['p90']:.2e}, "
                             f"p99={ls['p99']:.2e}, max={ls['max']:.2e}")
                lines.append(f"- Margin sensitivity (|Δ(top1−top2)|/||ε||): "
                             f"p50={ms['p50']:.2e}, p90={ms['p90']:.2e}, "
                             f"p99={ms['p99']:.2e}, max={ms['max']:.2e}")
            if details.get("off_manifold"):
                lines.append("")
                lines.append("Off-manifold robustness sweep:")
                lines.append("")
                for sigma, stats in details["off_manifold"].items():
                    lines.append(
                        f"- σ={sigma}: safe={stats['safe_decode_rate']:.2%}, "
                        f"unique={stats['unique_fraction']:.2%}, "
                        f"H_mean={stats['entropy_mean']:.2f}, "
                        f"H_max={stats['entropy_max']:.2f}, "
                        f"top1_retention={stats['top1_retention']:.2%}"
                    )

        elif result.name == "Interpolation Continuity":
            details = result.details
            lines.append(f"- Mean JS distance: {details['mean_js_distance']:.4f}")
            lines.append(f"- Max JS distance: {details['max_js_distance']:.4f}")
            lines.append(f"- Mean JS divergence (derived): {details['mean_js_divergence']:.4f}")
            lines.append(f"- Max JS divergence (derived): {details['max_js_divergence']:.4f}")
            lines.append(f"- Mean entropy: {details['mean_entropy']:.2f}")
            lines.append(f"- Max entropy: {details['max_entropy']:.2f}")
            if "holdout_used_for_pass" in details:
                lines.append(f"- Holdout used for pass/fail: {'Yes' if details['holdout_used_for_pass'] else 'No'}")
            if "train_pass" in details:
                lines.append(f"- Train-set threshold pass: {'Yes' if details['train_pass'] else 'No'}")
            if "holdout_pass" in details and details.get("holdout_used_for_pass"):
                lines.append(f"- Holdout threshold pass: {'Yes' if details['holdout_pass'] else 'No'}")
            if "endpoint_entropy_mean" in details:
                lines.append(f"- Endpoint entropy mean: {details['endpoint_entropy_mean']:.2f}")
            if "midpoint_entropy_mean" in details:
                lines.append(f"- Midpoint entropy mean: {details['midpoint_entropy_mean']:.2f}")
            if "mean_switch_rate" in details:
                lines.append(f"- Mean argmax switch rate: {details['mean_switch_rate']:.4f}")
            if "mean_topk_jaccard" in details:
                lines.append(f"- Mean top-k Jaccard: {details['mean_topk_jaccard']:.4f}")
            if "mean_plateau_ratio" in details:
                lines.append(f"- Mean plateau ratio: {details['mean_plateau_ratio']:.4f}")
            if "mean_flip_count" in details:
                lines.append(f"- Mean flip count: {details['mean_flip_count']:.2f}")
            lines.append(f"- Pairs tested: {details['num_pairs']}")
            if details.get("holdout_metrics"):
                hm = details["holdout_metrics"]
                lines.append("")
                lines.append("Held-out interpolation pairs:")
                lines.append(f"- Mean JS: {hm['mean_js']:.4f}")
                lines.append(f"- Max JS: {hm['max_js']:.4f}")
                lines.append(f"- Mean entropy: {hm['mean_entropy']:.2f}")
                lines.append(f"- Max entropy: {hm['max_entropy']:.2f}")
            if details.get("entropy_vs_prior"):
                evp = details["entropy_vs_prior"]
                lines.append("")
                lines.append(
                    f"- Entropy vs prior (mean Δ / max Δ): "
                    f"{evp['mean_delta']:.2f} / {evp['max_delta']:.2f}"
                )
                if evp.get("flag"):
                    lines.append("- Entropy vs prior divergence flagged: Yes")
            if details.get("max_js_case"):
                case = details["max_js_case"]
                lines.append("")
                lines.append("Max JS divergence case:")
                lines.append("```")
                lines.append(
                    f"  {case['piece_a']} ({case['token_a']}) → {case['piece_b']} ({case['token_b']}), "
                    f"steps {case['step_from']}→{case['step_to']}, "
                    f"alpha {case['alpha_from']:.2f}→{case['alpha_to']:.2f}, "
                    f"JSdist={case.get('js_distance', 0.0):.4f}, "
                    f"norm {case['norm_from']:.2f}→{case['norm_to']:.2f}"
                )
                if case.get("topk_from") and case.get("topk_to"):
                    lines.append("  top-10 from:")
                    for tok in case["topk_from"]:
                        lines.append(f"    {tok['piece']} ({tok['id']}): {tok['prob']:.3f}")
                    lines.append("  top-10 to:")
                    for tok in case["topk_to"]:
                        lines.append(f"    {tok['piece']} ({tok['id']}): {tok['prob']:.3f}")
                lines.append("```")
            if details.get("top_entropy_cases"):
                lines.append("")
                lines.append("Highest entropy steps:")
                lines.append("```")
                for case in details["top_entropy_cases"]:
                    lines.append(
                        f"  {case['piece_a']} ({case['token_a']}) → {case['piece_b']} ({case['token_b']}), "
                        f"step {case['step']}, alpha={case['alpha']:.2f}, "
                        f"H={case['entropy']:.2f}, norm={case['latent_norm']:.2f}"
                    )
                lines.append("```")
            if details.get("stratified_metrics"):
                lines.append("")
                lines.append("Stratified interpolation metrics (train pairs):")
                lines.append("```")
                for name, stats in details["stratified_metrics"].items():
                    if not stats:
                        continue
                    lines.append(
                        f"  {name}: mean_js_dist={stats['mean_js']:.3f}, "
                        f"max_js_dist={stats['max_js']:.3f}, "
                        f"mean_H={stats['mean_entropy']:.2f}, "
                        f"max_H={stats['max_entropy']:.2f}, "
                        f"flip={stats['flip_count_mean']:.2f}, "
                        f"switch={stats.get('switch_rate_mean', 0):.3f}, "
                        f"jaccard={stats.get('mean_topk_jaccard', 0):.3f}, "
                        f"cliff={stats['cliff_count_mean']:.2f}, "
                        f"plateau={stats['plateau_ratio_mean']:.2f}, "
                        f"n={stats['num_pairs']}"
                    )
                lines.append("```")
            if details.get("distance_buckets"):
                lines.append("")
                lines.append("Distance bucket metrics (train pairs):")
                lines.append("```")
                buckets = details["distance_buckets"]
                thresholds = buckets.get("thresholds", {})
                if thresholds:
                    lines.append(
                        f"  thresholds: near<= {thresholds.get('near_max', 0):.3f}, "
                        f"medium<= {thresholds.get('medium_max', 0):.3f}"
                    )
                for name, stats in buckets.items():
                    if name == "thresholds" or not stats:
                        continue
                    lines.append(
                        f"  {name}: mean_js_dist={stats['mean_js']:.3f}, "
                        f"max_js_dist={stats['max_js']:.3f}, "
                        f"mean_H={stats['mean_entropy']:.2f}, "
                        f"max_H={stats['max_entropy']:.2f}, "
                        f"flip={stats['flip_count_mean']:.2f}, "
                        f"switch={stats.get('switch_rate_mean', 0):.3f}, "
                        f"jaccard={stats.get('mean_topk_jaccard', 0):.3f}, "
                        f"cliff={stats['cliff_count_mean']:.2f}, "
                        f"plateau={stats['plateau_ratio_mean']:.2f}, "
                        f"n={stats['num_pairs']}"
                    )
                lines.append("```")

        elif result.name == "Reconstruction Accuracy":
            details = result.details
            total = details['total']
            vocab_size = total + 5  # 5 specials: pad/unk/bos/eos/endoftext
            lines.append(f"- V={vocab_size:,} total tokens; 5 specials "
                         f"(pad/unk/bos/eos/endoftext) excluded → {total:,} evaluated")
            lines.append(f"- Top-1 accuracy: {details['top1_accuracy']:.2%}")
            lines.append(f"- Top-5 accuracy: {details['top5_accuracy']:.2%}")
            lines.append(f"- Correct: {details['correct']} / {details['total']}")
            if "collapsed_dims" in details:
                lines.append("")
                lines.append("Posterior collapse diagnostic:")
                lines.append(f"- Collapsed dims (KL < 0.01): {details['collapsed_dims']} ({details['collapsed_fraction']:.1%})")
                lines.append(f"- μ norm (mean ± std): {details['mu_norm_mean']:.2f} ± {details['mu_norm_std']:.2f}")

            if details['example_failures']:
                lines.append("")
                lines.append("Example failures:")
                lines.append("```")
                for failure in details['example_failures'][:5]:
                    lines.append(f"  {repr(failure['input'])} → {repr(failure['predicted'])}")
                lines.append("```")

        elif result.name == "Diffusion Walk":
            details = result.details
            lines.append(f"- Min non-special rate: {details['min_non_special_rate']:.2%}")
            lines.append(f"- Mean non-special rate: {details['mean_non_special_rate']:.2%}")
            lines.append(f"- Max mean entropy: {details['max_entropy_mean']:.2f}")
            lines.append(f"- Mean mean entropy: {details['mean_entropy_mean']:.2f}")
            if "min_unique_fraction" in details:
                lines.append(
                    f"- Min unique fraction: {details['min_unique_fraction']:.2%} "
                    f"(threshold {details['min_unique_fraction_threshold']:.2%})"
                )
            if "mean_change_rate" in details:
                lines.append(
                    f"- Mean step change rate: {details['mean_change_rate']:.2%} "
                    f"(threshold {details['min_mean_change_rate_threshold']:.2%})"
                )
            lines.append(f"- Walks: {details['num_walks']}, Steps: {details['num_steps']}")
            lines.append(f"- Beta: {details['beta']}, Schedule: {details['beta_schedule']}")
            lines.append(f"- Start from: {details['start_from']}")
            lines.append("")
            lines.append("Per-step trajectory (every 10th step):")
            lines.append("```")
            rates = details.get("step_non_special_rates", [])
            entropies = details.get("step_entropy_means", [])
            uniques = details.get("step_unique_tokens", [])
            norms = details.get("step_norm_means", [])
            lines.append(f"  {'step':>4}  {'non_special':>11}  {'entropy':>7}  {'unique':>6}  {'norm':>6}")
            for i in range(0, len(rates), 10):
                lines.append(
                    f"  {i:>4}  {rates[i]:>11.2%}  {entropies[i]:>7.2f}"
                    f"  {uniques[i] if i < len(uniques) else 0:>6}"
                    f"  {norms[i] if i < len(norms) else 0:>6.2f}"
                )
            lines.append("```")

        elif result.name == "Metric Integrity":
            details = result.details
            if details.get("triggers"):
                lines.append("Triggered integrity issues:")
                lines.append("```")
                for trig in details["triggers"]:
                    lines.append(f"  - {trig}")
                lines.append("```")
            else:
                lines.append("No integrity issues detected.")

        lines.append("")

    # Figures
    if figures:
        lines.append("---")
        lines.append("")
        lines.append("## Visualizations")
        lines.append("")

        figure_descriptions = {
            'pca': 'PCA projection of token embeddings overlaid with prior samples',
            'umap': 'UMAP projection of token embeddings overlaid with prior samples',
            'interpolations': 'Interpolation paths between random token pairs',
            'variance': 'Distribution of learned variances across tokens',
            'training_loss': 'Training loss over epochs',
        }

        for name, path in figures.items():
            desc = figure_descriptions.get(name, name)
            # Use relative path from report location
            rel_path = Path(path).relative_to(Path(output_path).parent)
            lines.append(f"### {desc.title()}")
            lines.append("")
            lines.append(f"![{desc}]({rel_path})")
            lines.append("")

    # Failure analysis
    failed_tests = [r for r in test_results if not r.passed]
    if failed_tests:
        lines.append("---")
        lines.append("")
        lines.append("## Failure Analysis")
        lines.append("")

        for result in failed_tests:
            lines.append(f"### {result.name}")
            lines.append("")

            if result.name == "Prior Decodability":
                lines.append("**Possible causes**:")
                lines.append("- Insufficient KL regularization — embeddings may not fill the prior support")
                lines.append("- Excessive KL regularization — posterior collapse, all tokens map to the origin")
                lines.append("")
                lines.append("**Recommendations**:")
                lines.append("- If unique fraction is low: increase KL weight or training epochs")
                lines.append("- If non-special fraction is low: check decoder capacity")

            elif result.name == "Perturbation Stability":
                lines.append("**Possible causes**:")
                lines.append("- Embeddings are too sparse — discontinuities between token regions")
                lines.append("- Learned variance too low — small perturbations cause large probability shifts")
                lines.append("")
                lines.append("**Recommendations**:")
                lines.append("- Increase KL weight to spread embeddings")
                lines.append("- Check if learned σ values are reasonable (should be ~1)")

            elif result.name == "Interpolation Continuity":
                lines.append("**Possible causes**:")
                lines.append("- Decoder has sharp decision boundaries")
                lines.append("- Embeddings clustered with gaps between clusters")
                lines.append("")
                lines.append("**Recommendations**:")
                lines.append("- Train longer with KL annealing")
                lines.append("- Increase skipgram weight for smoother semantic structure")

            elif result.name == "Reconstruction Accuracy":
                lines.append("**Possible causes**:")
                lines.append("- Excessive KL regularization — posterior collapse")
                lines.append("- Embedding dimension too small")
                lines.append("- Insufficient training epochs")
                lines.append("")
                lines.append("**Recommendations**:")
                lines.append("- Decrease KL weight if accuracy is very low")
                lines.append("- Increase d_model dimension")
                lines.append("- Train for more epochs")
            elif result.name == "Metric Integrity":
                lines.append("**Possible causes**:")
                lines.append("- Prior distribution collapsed despite strong reconstruction")
                lines.append("- Metrics improving in isolation but not jointly")
                lines.append("")
                lines.append("**Recommendations**:")
                lines.append("- Increase KL weight or prior regularization")
                lines.append("- Rebalance interpolation and skipgram losses")
                lines.append("- Check prior decodability diversity stats")

            lines.append("")

    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")

    if all_passed:
        lines.append("**All evaluation criteria are satisfied.** The latent space meets the tested")
        lines.append("prerequisites for continuous diffusion: prior coverage, local smoothness,")
        lines.append("interpolation continuity, and reconstruction fidelity.")
    else:
        lines.append("**One or more evaluation criteria are not satisfied.** Refer to the failure")
        lines.append("analysis above for diagnostics and recommendations.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by the Token VAE evaluation pipeline.*")

    # Write report
    content = "\n".join(lines)
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Report saved to {output_file}")
    return str(output_file)
