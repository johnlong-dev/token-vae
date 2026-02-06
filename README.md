# Token VAE

**Variationally-regularized token embeddings — a geometric prerequisite for continuous diffusion over discrete tokens.**

## Hypothesis

> A variationally-regularized token embedding space can serve as a continuous manifold suitable for diffusion and latent-space revision.

Concretely, if each token *t* is embedded as a distribution q(h|t) = N(mu\_t, sigma\_t^2 I) and these distributions are regularized toward a shared prior N(0, I), then the resulting space has:

1. **Prior coverage** — samples from the standard normal prior decode to non-degenerate vocabulary tokens
2. **Local smoothness** — small perturbations to embeddings produce gradual changes in decoder output
3. **Interpolation continuity** — linear paths between token embeddings remain decodable with bounded divergence
4. **Reconstruction fidelity** — tokens are recoverable from their posterior mean embeddings

These properties are necessary (though not sufficient) for continuous diffusion trajectories over discrete vocabulary.

## Quick Start

```bash
# Install dependencies
uv sync

# Prepare data (streams ~4M chars from FineWeb)
uv run scripts/prepare_data.py --num-chars 4000000

# Train tokenizer (16k vocab)
uv run scripts/train_tokenizer.py \
    --input data/fineweb-4M.txt \
    --output artifacts/tokenizer/fw-4M-v16k \
    --vocab-size 16000

# Train model
uv run scripts/train_vae.py \
    --interp-weight 0.05 --interp-pairs 64 \
    --interp-entropy-target 8.0 \
    --interp-holdout-num 200 \
    --interp-holdout-pairs artifacts/interp/holdout_pairs.json \
    --identity-vocab-repeats 5 \
    --kl-weight 0.03 --kl-free-bits 0.25 \
    --ifw-alpha 1.0 \
    --prior-reg-weight 0.50 --prior-reg-interval 1 \
    --prior-entropy-target 4.0 --prior-max-freq-target 0.05 \
    --prior-marginal-entropy-target 7.0 \
    --prior-hhi-target 0.001

# Evaluate
uv run scripts/evaluate.py
# (deterministic by default via --seed 42; use --seed -1 to disable)
```

## Full Validation

Run the complete hypothesis validation checklist (~11 hours on M-series Mac):

```bash
./run_validation.sh
```

This executes 21 training runs covering reproducibility (5 seeds), data scale (3 sizes), vocab scale (3 sizes), ablations (8 configs), and baselines (2 configs). Results are written to `validation_results/summary.md`.

Options:
- `--dry-run` — print all commands without executing
- `--section A|B|C|D|EF|G|H` — run a single section
- `--skip-existing` — skip runs that already have results.json

## Evaluation Tests

The evaluation suite runs 6 automated tests, each with explicit pass/fail thresholds:

| Test | What it measures | Pass condition |
|------|-----------------|----------------|
| **Prior Decodability** | Dense coverage of latent space | >= 85% valid, >= 30% unique from 10k prior samples |
| **Perturbation Stability** | Local Lipschitz continuity | >= 50% top-k overlap at sigma=0.05, monotonic decrease |
| **Interpolation Continuity** | Smooth traversal between tokens | Max JS distance <= 0.5, max entropy <= 8.5 (holdout-gated) |
| **Reconstruction Accuracy** | Identity recovery from embeddings | Top-1 accuracy >= 90% |
| **Metric Integrity** | Cross-metric consistency checks | No contradictory improvements |
| **Diffusion Walk** | Random walk decode validity | >=95% non-special, bounded entropy, and non-collapsed diversity/change |

## License

MIT
