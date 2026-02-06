# Token VAE Evaluation Report

Generated: 2026-02-06 14:22:53

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
| Perturbation Stability | Pass | 0.8985 | 0.5000 |
| Interpolation Continuity | Pass | 0.0766 | 0.5000 |
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
- Unique token fraction: 56.81%
- Unique tokens decoded: 5681 / 10000
- Max token frequency: 0.18%
- Tokens seen once: 35.72%
- Prior entropy (mean/median/max): 4.08 / 4.33 / 6.31
- log(V): 9.68
- Gini coefficient: 0.7629

Example decoded tokens from prior samples:
```
  '▁experience'
  '▁Ob'
  'nes'
  '▁forces'
  '▁ab'
```

Top decoded tokens:
```
  ▁all (216): 0.18% (18)
  : (15944): 0.14% (14)
  ▁be (67): 0.14% (14)
  ▁was (146): 0.13% (13)
  ▁It (314): 0.13% (13)
  ▁by (161): 0.13% (13)
  ▁but (224): 0.13% (13)
  ▁he (100): 0.12% (12)
  ’ (15940): 0.12% (12)
  ▁their (263): 0.12% (12)
  re (9): 0.12% (12)
  ▁The (97): 0.12% (12)
  S (15920): 0.12% (12)
  ! (15967): 0.12% (12)
  ▁has (221): 0.12% (12)
  t (15895): 0.12% (12)
  ▁many (605): 0.11% (11)
  ▁C (68): 0.11% (11)
  ▁up (249): 0.11% (11)
  ▁can (189): 0.11% (11)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8985
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 89.85% overlap
- σ=0.1: 80.55% overlap
- σ=0.2: 61.37% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.07e+01, p99=3.17e+01, max=3.64e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.57e-01, p90=4.29e-01, p99=6.41e-01, max=7.73e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.05, H_max=1.78, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.92, H_max=5.21, top1_retention=86.00%
- σ=2.0: safe=100.00%, unique=99.50%, H_mean=1.22, H_max=3.10, top1_retention=11.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0766
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0737
- Max JS distance: 0.3971
- Mean JS divergence (derived): 0.0054
- Max JS divergence (derived): 0.1577
- Mean entropy: 0.75
- Max entropy: 7.02
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.20
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8736
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0766
- Max JS: 0.3809
- Mean entropy: 0.91
- Max entropy: 7.62

- Entropy vs prior (mean Δ / max Δ): -3.33 / 0.71
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  abled (11739) → ▁|0 (5152), steps 9→10, alpha 0.47→0.53, JSdist=0.3971, norm 5.99→6.05
  top-10 from:
    abled (11739): 0.699
    ▁|0 (5152): 0.274
    ireless (6619): 0.000
    ▁to (32): 0.000
    ▁and (37): 0.000
    ▁a (6): 0.000
    . (15914): 0.000
    , (15915): 0.000
    ▁that (86): 0.000
    ▁absorb (14532): 0.000
  top-10 to:
    ▁|0 (5152): 0.810
    abled (11739): 0.166
    ireless (6619): 0.000
    ▁to (32): 0.000
    ▁and (37): 0.000
    ▁a (6): 0.000
    . (15914): 0.000
    , (15915): 0.000
    ▁absorb (14532): 0.000
    about (11186): 0.000
```

Highest entropy steps:
```
  ▁Col (1193) → res (157), step 10, alpha=0.53, H=7.02, norm=3.71
  ▁Other (2541) → gers (3116), step 9, alpha=0.47, H=6.79, norm=4.05
  ▁networking (8888) → ▁oper (1003), step 10, alpha=0.53, H=6.62, norm=4.44
  ▁Other (2541) → gers (3116), step 10, alpha=0.53, H=6.44, norm=4.08
  ▁Iowa (9754) → ▁family (1118), step 10, alpha=0.53, H=6.23, norm=3.77
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.939, medium<= 9.650
  near: mean_js_dist=0.079, max_js_dist=0.296, mean_H=1.08, max_H=7.02, flip=1.00, switch=0.053, jaccard=0.899, cliff=0.53, plateau=0.95, n=17
  medium: mean_js_dist=0.071, max_js_dist=0.363, mean_H=0.60, max_H=5.38, flip=1.00, switch=0.053, jaccard=0.874, cliff=1.62, plateau=0.95, n=16
  far: mean_js_dist=0.071, max_js_dist=0.397, mean_H=0.57, max_H=6.62, flip=1.00, switch=0.053, jaccard=0.847, cliff=1.65, plateau=0.95, n=17
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
- Max mean entropy: 4.28
- Mean mean entropy: 4.14
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 20.18% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.28     100   16.14
    10      100.00%     4.13      99   16.16
    20      100.00%     4.10     100   16.11
    30      100.00%     4.08      98   16.16
    40      100.00%     4.15     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*