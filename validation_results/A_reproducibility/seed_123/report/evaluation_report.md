# Token VAE Evaluation Report

Generated: 2026-02-06 14:16:44

## Overview

This report presents quantitative evaluation of the learned latent space
against criteria required for continuous diffusion over token embeddings.
The evaluation tests whether the variationally-regularized embedding space
satisfies four properties:

1. **Prior coverage** — Samples from the standard normal prior $\mathcal{N}(0, I)$ decode to non-degenerate vocabulary tokens
2. **Local smoothness** — Small perturbations to embeddings produce gradual changes in decoder output
3. **Interpolation continuity** — Linear paths between token embeddings remain decodable with bounded divergence
4. **Reconstruction fidelity** — Tokens are recoverable from their posterior mean embeddings

---

## Test Results Summary

**All evaluation criteria satisfied.**

| Test | Status | Score | Threshold |
|------|--------|-------|-----------|
| Prior Decodability | Pass | 1.0000 | 0.8500 |
| Perturbation Stability | Pass | 0.9046 | 0.5000 |
| Interpolation Continuity | Pass | 0.0763 | 0.5000 |
| Reconstruction Accuracy | Pass | 1.0000 | 0.9000 |
| Metric Integrity | Pass | 1.0000 | 1.0000 |
| Diffusion Walk | Pass | 1.0000 | 0.9500 |

---

## Detailed Test Results

### Prior Decodability

**Description**: Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.8500

**Details**:

- Non-special token fraction: 100.00%
- Unique token fraction: 56.27%
- Unique tokens decoded: 5627 / 10000
- Max token frequency: 0.21%
- Tokens seen once: 35.01%
- Prior entropy (mean/median/max): 4.10 / 4.33 / 6.34
- log(V): 9.68
- Gini coefficient: 0.7660

Example decoded tokens from prior samples:
```
  'M'
  'cludes'
  '▁del'
  '▁happen'
  '▁shaded'
```

Top decoded tokens:
```
  ▁all (216): 0.21% (21)
  ▁my (278): 0.17% (17)
  ▁at (121): 0.17% (17)
  ▁but (224): 0.17% (17)
  ▁" (250): 0.16% (16)
  The (1024): 0.15% (15)
  ▁not (179): 0.15% (15)
  ▁like (337): 0.15% (15)
  ▁from (172): 0.15% (15)
  ▁which (304): 0.13% (13)
  ▁been (353): 0.13% (13)
  ▁by (161): 0.13% (13)
  ▁can (189): 0.13% (13)
  ▁The (97): 0.13% (13)
  ▁he (100): 0.13% (13)
  F (15935): 0.12% (12)
  ▁one (283): 0.12% (12)
  ? (15964): 0.12% (12)
  ▁up (249): 0.11% (11)
  ▁their (263): 0.11% (11)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9046
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 90.46% overlap
- σ=0.1: 81.43% overlap
- σ=0.2: 62.96% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.17e+01, p99=3.65e+01, max=3.78e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.74e-01, p90=4.28e-01, p99=6.51e-01, max=7.61e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.17, H_max=7.45, top1_retention=98.00%
- σ=1.0: safe=100.00%, unique=99.50%, H_mean=0.94, H_max=5.32, top1_retention=88.00%
- σ=2.0: safe=100.00%, unique=98.00%, H_mean=1.14, H_max=3.31, top1_retention=16.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0763
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0763
- Max JS distance: 0.3779
- Mean JS divergence (derived): 0.0058
- Max JS divergence (derived): 0.1428
- Mean entropy: 0.90
- Max entropy: 7.32
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.01
- Midpoint entropy mean: 3.48
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8752
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0763
- Max JS: 0.3884
- Mean entropy: 0.88
- Max entropy: 7.78

- Entropy vs prior (mean Δ / max Δ): -3.20 / 0.98
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁integration (11577) → ▁asking (5324), steps 9→10, alpha 0.47→0.53, JSdist=0.3779, norm 4.84→4.77
  top-10 from:
    ▁integration (11577): 0.678
    ▁asking (5324): 0.119
    ▁and (37): 0.002
    ▁on (71): 0.002
    ▁to (32): 0.002
    ▁of (36): 0.002
    , (15915): 0.002
    . (15914): 0.002
    ▁a (6): 0.001
    ▁with (94): 0.001
  top-10 to:
    ▁asking (5324): 0.548
    ▁integration (11577): 0.205
    ▁and (37): 0.003
    ▁to (32): 0.002
    ▁of (36): 0.002
    ▁on (71): 0.002
    , (15915): 0.002
    . (15914): 0.002
    ▁a (6): 0.002
    ▁for (72): 0.002
```

Highest entropy steps:
```
  ▁rise (4895) → ▁mean (1765), step 9, alpha=0.47, H=7.32, norm=3.80
  ▁bitcoin (7019) → ▁phones (10821), step 10, alpha=0.53, H=7.28, norm=4.29
  ▁Edward (8772) → ▁much (705), step 11, alpha=0.58, H=7.22, norm=3.44
  ▁turned (3623) → ▁Books (9797), step 9, alpha=0.47, H=7.19, norm=4.01
  ▁slender (7467) → ▁fighting (8213), step 9, alpha=0.47, H=7.09, norm=4.24
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.888, medium<= 9.492
  near: mean_js_dist=0.082, max_js_dist=0.304, mean_H=1.31, max_H=7.32, flip=1.00, switch=0.053, jaccard=0.892, cliff=0.24, plateau=0.95, n=17
  medium: mean_js_dist=0.076, max_js_dist=0.331, mean_H=0.85, max_H=7.09, flip=1.00, switch=0.053, jaccard=0.890, cliff=1.25, plateau=0.95, n=16
  far: mean_js_dist=0.070, max_js_dist=0.378, mean_H=0.53, max_H=6.20, flip=1.00, switch=0.053, jaccard=0.845, cliff=1.76, plateau=0.95, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.9000

**Details**:

- V=16,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 15,995 evaluated
- Top-1 accuracy: 100.00%
- Top-5 accuracy: 100.00%
- Correct: 15995 / 15995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 6.64 ± 0.95

### Metric Integrity

**Description**: Cross-validates metrics to detect inconsistencies indicative of posterior collapse or degenerate solutions

**Result**: PASSED
- Score: 1.0000
- Threshold: 1.0000

**Details**:

No integrity issues detected.

### Diffusion Walk

**Description**: Simulates forward-diffusion trajectories and verifies that decoded tokens remain non-degenerate

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.9500

**Details**:

- Min non-special rate: 100.00%
- Mean non-special rate: 100.00%
- Max mean entropy: 4.17
- Mean mean entropy: 4.10
- Min unique fraction: 98.00% (threshold 10.00%)
- Mean step change rate: 20.02% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.01      99   15.91
    10      100.00%     4.13      99   15.91
    20      100.00%     4.13      99   15.91
    30      100.00%     4.12      99   15.95
    40      100.00%     4.17      99   15.97
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*