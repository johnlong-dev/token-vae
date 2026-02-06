# Token VAE Evaluation Report

Generated: 2026-02-06 14:20:58

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
| Perturbation Stability | Pass | 0.9181 | 0.5000 |
| Interpolation Continuity | Pass | 0.0753 | 0.5000 |
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
- Unique token fraction: 64.88%
- Unique tokens decoded: 6488 / 10000
- Max token frequency: 0.13%
- Tokens seen once: 43.73%
- Prior entropy (mean/median/max): 4.08 / 4.31 / 6.10
- log(V): 9.68
- Gini coefficient: 0.7046

Example decoded tokens from prior samples:
```
  'ide'
  'ertain'
  '▁Be'
  'ume'
  '▁Jack'
```

Top decoded tokens:
```
  ▁you (82): 0.13% (13)
  ▁from (172): 0.12% (12)
  ▁will (197): 0.11% (11)
  ▁at (121): 0.11% (11)
  ▁this (170): 0.11% (11)
  ▁could (541): 0.11% (11)
  ▁my (278): 0.11% (11)
  ▁provide (1276): 0.10% (10)
  ▁information (650): 0.10% (10)
  ▁but (224): 0.09% (9)
  The (1024): 0.09% (9)
  ’ (15940): 0.09% (9)
  ▁can (189): 0.08% (8)
  ▁also (347): 0.08% (8)
  ▁off (320): 0.08% (8)
  ▁make (481): 0.08% (8)
  ve (52): 0.08% (8)
  ▁10 (591): 0.08% (8)
  ly (59): 0.08% (8)
  ' (15939): 0.08% (8)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9181
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 91.81% overlap
- σ=0.1: 84.34% overlap
- σ=0.2: 67.10% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=3.07e+01, p90=3.45e+01, p99=3.99e+01, max=4.40e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.58e-01, p90=3.96e-01, p99=6.58e-01, max=8.18e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.07, H_max=5.13, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.90, H_max=5.22, top1_retention=88.50%
- σ=2.0: safe=100.00%, unique=100.00%, H_mean=1.21, H_max=3.14, top1_retention=14.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0753
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0753
- Max JS distance: 0.3712
- Mean JS divergence (derived): 0.0057
- Max JS divergence (derived): 0.1378
- Mean entropy: 0.77
- Max entropy: 7.06
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.20
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.9000
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0743
- Max JS: 0.4082
- Mean entropy: 0.76
- Max entropy: 7.43

- Entropy vs prior (mean Δ / max Δ): -3.31 / 0.96
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ALLY (14018) → ▁studies (3819), steps 10→11, alpha 0.53→0.58, JSdist=0.3712, norm 4.90→4.82
  top-10 from:
    ALLY (14018): 0.778
    ▁studies (3819): 0.157
    ▁for (72): 0.002
    s (15900): 0.001
    ▁and (37): 0.001
    , (15915): 0.001
    ▁to (32): 0.001
    ▁in (35): 0.001
    . (15914): 0.001
    ▁the (11): 0.001
  top-10 to:
    ▁studies (3819): 0.634
    ALLY (14018): 0.283
    ▁for (72): 0.002
    s (15900): 0.001
    ▁and (37): 0.001
    , (15915): 0.001
    ▁to (32): 0.001
    ▁in (35): 0.001
    ▁the (11): 0.001
    . (15914): 0.001
```

Highest entropy steps:
```
  ▁techniques (6206) → ination (1296), step 10, alpha=0.53, H=7.06, norm=3.76
  ▁techniques (6206) → ination (1296), step 9, alpha=0.47, H=6.80, norm=3.80
  On (10481) → 7. (1829), step 9, alpha=0.47, H=6.73, norm=3.63
  ▁gut (13228) → ▁each (716), step 11, alpha=0.58, H=6.49, norm=3.65
  ▁million (1429) → iche (9679), step 9, alpha=0.47, H=6.28, norm=3.96
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.826, medium<= 9.443
  near: mean_js_dist=0.078, max_js_dist=0.310, mean_H=0.97, max_H=6.73, flip=1.00, switch=0.053, jaccard=0.905, cliff=0.76, plateau=0.95, n=17
  medium: mean_js_dist=0.078, max_js_dist=0.360, mean_H=0.83, max_H=7.06, flip=1.00, switch=0.053, jaccard=0.900, cliff=1.50, plateau=0.95, n=16
  far: mean_js_dist=0.070, max_js_dist=0.371, mean_H=0.51, max_H=6.18, flip=1.00, switch=0.053, jaccard=0.895, cliff=1.71, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.53 ± 0.89

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
- Max mean entropy: 4.16
- Mean mean entropy: 4.00
- Min unique fraction: 99.00% (threshold 10.00%)
- Mean step change rate: 21.16% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     3.91     100   16.14
    10      100.00%     3.96     100   16.16
    20      100.00%     3.99     100   16.11
    30      100.00%     3.91     100   16.16
    40      100.00%     4.07     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*