# Token VAE Evaluation Report

Generated: 2026-02-06 14:18:17

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
| Perturbation Stability | Pass | 0.9054 | 0.5000 |
| Interpolation Continuity | Pass | 0.0762 | 0.5000 |
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
- Unique token fraction: 57.12%
- Unique tokens decoded: 5712 / 10000
- Max token frequency: 0.20%
- Tokens seen once: 36.02%
- Prior entropy (mean/median/max): 4.09 / 4.33 / 6.26
- log(V): 9.68
- Gini coefficient: 0.7609

Example decoded tokens from prior samples:
```
  '▁we'
  '▁injured'
  '▁bodies'
  '▁sub'
  'anta'
```

Top decoded tokens:
```
  ▁at (121): 0.20% (20)
  ▁out (246): 0.16% (16)
  ▁The (97): 0.16% (16)
  ▁all (216): 0.16% (16)
  ▁may (413): 0.14% (14)
  ▁We (345): 0.14% (14)
  ▁are (119): 0.14% (14)
  ing (26): 0.13% (13)
  ▁not (179): 0.13% (13)
  ▁or (124): 0.13% (13)
  ▁two (473): 0.12% (12)
  ly (59): 0.12% (12)
  A (15921): 0.12% (12)
  ▁one (283): 0.12% (12)
  ▁had (366): 0.12% (12)
  ▁they (267): 0.12% (12)
  ▁you (82): 0.12% (12)
  ▁from (172): 0.11% (11)
  y (15911): 0.11% (11)
  ▁( (126): 0.11% (11)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9054
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 90.54% overlap
- σ=0.1: 81.70% overlap
- σ=0.2: 63.76% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.03e+01, p99=3.27e+01, max=3.39e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.61e-01, p90=4.39e-01, p99=5.46e-01, max=8.98e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.07, H_max=4.05, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.86, H_max=4.68, top1_retention=94.00%
- σ=2.0: safe=100.00%, unique=97.50%, H_mean=1.21, H_max=3.37, top1_retention=11.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0762
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0748
- Max JS distance: 0.3973
- Mean JS divergence (derived): 0.0056
- Max JS divergence (derived): 0.1578
- Mean entropy: 0.78
- Max entropy: 7.42
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.33
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8798
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0762
- Max JS: 0.3916
- Mean entropy: 0.89
- Max entropy: 7.76

- Entropy vs prior (mean Δ / max Δ): -3.32 / 1.16
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  DP (11034) → ▁intentions (12161), steps 9→10, alpha 0.47→0.53, JSdist=0.3973, norm 5.37→5.31
  top-10 from:
    DP (11034): 0.804
    ▁intentions (12161): 0.083
    ▁and (37): 0.001
    ▁to (32): 0.001
    , (15915): 0.001
    ▁is (66): 0.001
    ▁a (6): 0.001
    . (15914): 0.000
    ▁the (11): 0.000
    - (15922): 0.000
  top-10 to:
    ▁intentions (12161): 0.537
    DP (11034): 0.293
    ▁and (37): 0.002
    ▁to (32): 0.001
    , (15915): 0.001
    ▁is (66): 0.001
    ▁a (6): 0.001
    . (15914): 0.001
    ▁the (11): 0.001
    ▁of (36): 0.001
```

Highest entropy steps:
```
  ▁freezer (14673) → enture (5187), step 10, alpha=0.53, H=7.42, norm=4.59
  ▁manufacturer (5465) → ▁hop (1282), step 10, alpha=0.53, H=7.30, norm=4.52
  ▁freezer (14673) → enture (5187), step 9, alpha=0.47, H=7.23, norm=4.56
  arsaw (9727) → ▁elsewhere (8532), step 10, alpha=0.53, H=7.17, norm=4.45
  ▁significant (2242) → ▁Phil (3261), step 9, alpha=0.47, H=6.49, norm=4.10
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 9.137, medium<= 9.774
  near: mean_js_dist=0.074, max_js_dist=0.302, mean_H=0.81, max_H=6.49, flip=1.00, switch=0.053, jaccard=0.898, cliff=1.00, plateau=0.95, n=17
  medium: mean_js_dist=0.075, max_js_dist=0.325, mean_H=0.78, max_H=7.30, flip=1.00, switch=0.053, jaccard=0.881, cliff=1.50, plateau=0.95, n=16
  far: mean_js_dist=0.075, max_js_dist=0.397, mean_H=0.74, max_H=7.42, flip=1.00, switch=0.053, jaccard=0.861, cliff=1.65, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.58 ± 0.94

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
- Max mean entropy: 4.18
- Mean mean entropy: 4.07
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 21.57% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.00      99   15.96
    10      100.00%     4.01      98   16.02
    20      100.00%     4.05      97   16.04
    30      100.00%     4.10      98   16.03
    40      100.00%     4.16      98   16.05
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*