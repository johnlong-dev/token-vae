# Token VAE Evaluation Report

Generated: 2026-02-06 14:24:01

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
| Perturbation Stability | Pass | 0.8975 | 0.5000 |
| Interpolation Continuity | Pass | 0.0767 | 0.5000 |
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
- Unique token fraction: 56.79%
- Unique tokens decoded: 5679 / 10000
- Max token frequency: 0.16%
- Tokens seen once: 36.20%
- Prior entropy (mean/median/max): 4.09 / 4.32 / 6.22
- log(V): 9.68
- Gini coefficient: 0.7640

Example decoded tokens from prior samples:
```
  '▁experience'
  '▁Ob'
  'nes'
  '▁forces'
  '▁rocket'
```

Top decoded tokens:
```
  ▁be (67): 0.16% (16)
  S (15920): 0.15% (15)
  ▁as (103): 0.14% (14)
  ▁from (172): 0.14% (14)
  ▁have (166): 0.14% (14)
  ▁like (337): 0.13% (13)
  ▁was (146): 0.13% (13)
  t (15895): 0.13% (13)
  ▁It (314): 0.13% (13)
  ▁all (216): 0.13% (13)
  ▁he (100): 0.12% (12)
  : (15944): 0.12% (12)
  ▁The (97): 0.12% (12)
  ▁my (278): 0.11% (11)
  ▁you (82): 0.11% (11)
  ▁first (434): 0.11% (11)
  ▁them (382): 0.11% (11)
  re (9): 0.11% (11)
  ▁by (161): 0.11% (11)
  ! (15967): 0.10% (10)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8975
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 89.75% overlap
- σ=0.1: 80.19% overlap
- σ=0.2: 61.20% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.87e+01, p90=3.07e+01, p99=3.17e+01, max=3.64e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.62e-01, p90=4.48e-01, p99=6.78e-01, max=8.07e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.06, H_max=3.38, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.96, H_max=5.30, top1_retention=86.50%
- σ=2.0: safe=100.00%, unique=98.50%, H_mean=1.21, H_max=3.32, top1_retention=12.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0767
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0736
- Max JS distance: 0.4087
- Mean JS divergence (derived): 0.0054
- Max JS divergence (derived): 0.1670
- Mean entropy: 0.74
- Max entropy: 7.05
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.14
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8728
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0767
- Max JS: 0.3803
- Mean entropy: 0.92
- Max entropy: 7.67

- Entropy vs prior (mean Δ / max Δ): -3.35 / 0.83
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  after (6575) → ▁manually (14790), steps 8→9, alpha 0.42→0.47, JSdist=0.4087, norm 5.42→5.49
  top-10 from:
    after (6575): 0.616
    ▁manually (14790): 0.278
    ▁in (35): 0.001
    ▁a (6): 0.001
    , (15915): 0.001
    ▁to (32): 0.001
    . (15914): 0.001
    ▁the (11): 0.001
    ▁is (66): 0.000
    ▁of (36): 0.000
  top-10 to:
    ▁manually (14790): 0.818
    after (6575): 0.114
    ▁in (35): 0.001
    ▁a (6): 0.000
    , (15915): 0.000
    ▁to (32): 0.000
    ▁the (11): 0.000
    . (15914): 0.000
    ▁is (66): 0.000
    ▁for (72): 0.000
```

Highest entropy steps:
```
  ▁Col (1193) → res (157), step 10, alpha=0.53, H=7.05, norm=3.72
  ▁Col (1193) → res (157), step 11, alpha=0.58, H=6.47, norm=3.69
  ▁Other (2541) → gers (3116), step 10, alpha=0.53, H=6.32, norm=4.11
  ▁networking (8888) → ▁oper (1003), step 10, alpha=0.53, H=6.31, norm=4.48
  ▁Other (2541) → gers (3116), step 9, alpha=0.47, H=6.29, norm=4.09
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.925, medium<= 9.660
  near: mean_js_dist=0.078, max_js_dist=0.293, mean_H=1.03, max_H=7.05, flip=1.00, switch=0.053, jaccard=0.901, cliff=0.59, plateau=0.95, n=17
  medium: mean_js_dist=0.071, max_js_dist=0.367, mean_H=0.63, max_H=5.45, flip=1.00, switch=0.053, jaccard=0.869, cliff=1.50, plateau=0.95, n=16
  far: mean_js_dist=0.071, max_js_dist=0.409, mean_H=0.55, max_H=6.31, flip=1.00, switch=0.053, jaccard=0.849, cliff=1.82, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.59 ± 0.93

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
- Max mean entropy: 4.26
- Mean mean entropy: 4.11
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 20.45% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.26      99   16.14
    10      100.00%     4.14      99   16.16
    20      100.00%     4.09     100   16.11
    30      100.00%     4.01     100   16.16
    40      100.00%     4.13     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*