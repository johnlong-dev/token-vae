# Token VAE Evaluation Report

Generated: 2026-02-06 14:26:14

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
| Prior Decodability | Pass | 0.9998 | 0.8500 |
| Perturbation Stability | Pass | 0.8336 | 0.5000 |
| Interpolation Continuity | Pass | 0.0783 | 0.5000 |
| Reconstruction Accuracy | Pass | 1.0000 | 0.9000 |
| Metric Integrity | Pass | 1.0000 | 1.0000 |
| Diffusion Walk | Pass | 1.0000 | 0.9500 |

---

## Detailed Test Results

### Prior Decodability

**Description**: Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens

**Result**: PASSED
- Score: 0.9998
- Threshold: 0.8500

**Details**:

- Non-special token fraction: 99.98%
- Unique token fraction: 62.99%
- Unique tokens decoded: 6299 / 10000
- Max token frequency: 0.63%
- Tokens seen once: 43.16%
- Prior entropy (mean/median/max): 4.06 / 4.30 / 6.09
- log(V): 9.68
- Gini coefficient: 0.7239

Example decoded tokens from prior samples:
```
  '▁proceed'
  '▁reun'
  'ighth'
  '▁enemy'
  'BID'
```

Top decoded tokens:
```
  ▁the (11): 0.63% (63)
  . (15914): 0.62% (62)
  , (15915): 0.59% (59)
  ▁and (37): 0.42% (42)
  ▁of (36): 0.36% (36)
  ▁in (35): 0.32% (32)
  ▁a (6): 0.27% (27)
  ▁to (32): 0.27% (27)
  ▁that (86): 0.22% (22)
  - (15922): 0.21% (21)
  s (15900): 0.16% (16)
  ' (15939): 0.14% (14)
  ▁this (170): 0.14% (14)
  ▁it (96): 0.13% (13)
  ’ (15940): 0.13% (13)
  ▁for (72): 0.13% (13)
  ▁by (161): 0.13% (13)
  ▁on (71): 0.12% (12)
  ▁at (121): 0.12% (12)
  ▁( (126): 0.12% (12)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8336
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 83.36% overlap
- σ=0.1: 68.99% overlap
- σ=0.2: 46.01% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.95e+01, p90=3.02e+01, p99=3.05e+01, max=3.09e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.89e-01, p90=4.56e-01, p99=6.91e-01, max=7.45e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.05, H_max=3.29, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=99.50%, H_mean=1.12, H_max=4.43, top1_retention=88.00%
- σ=2.0: safe=100.00%, unique=99.50%, H_mean=1.42, H_max=3.23, top1_retention=5.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0783
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0783
- Max JS distance: 0.3609
- Mean JS divergence (derived): 0.0061
- Max JS divergence (derived): 0.1303
- Mean entropy: 1.01
- Max entropy: 8.14
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 4.42
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8076
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0775
- Max JS: 0.3575
- Mean entropy: 0.96
- Max entropy: 7.76

- Entropy vs prior (mean Δ / max Δ): -3.05 / 2.05
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  aneous (5185) → ▁LDS (15226), steps 10→11, alpha 0.53→0.58, JSdist=0.3609, norm 4.66→4.71
  top-10 from:
    aneous (5185): 0.578
    ▁LDS (15226): 0.274
    ▁con (125): 0.000
    ▁growing (4296): 0.000
    azing (2648): 0.000
    ▁Another (3869): 0.000
    ▁fees (4198): 0.000
    ▁entrance (8510): 0.000
    ▁gent (8395): 0.000
    nity (8971): 0.000
  top-10 to:
    ▁LDS (15226): 0.740
    aneous (5185): 0.136
    ▁con (125): 0.000
    ▁growing (4296): 0.000
    ▁fees (4198): 0.000
    azing (2648): 0.000
    ▁gent (8395): 0.000
    ▁serves (7928): 0.000
    ▁navigation (14909): 0.000
    ▁entrance (8510): 0.000
```

Highest entropy steps:
```
  ▁(199 (9745) → ices (656), step 9, alpha=0.47, H=8.14, norm=3.93
  ▁(199 (9745) → ices (656), step 8, alpha=0.42, H=7.68, norm=3.98
  ▁SU (4038) → ▁convert (9490), step 9, alpha=0.47, H=7.48, norm=4.14
  ▁Other (2541) → gers (3116), step 9, alpha=0.47, H=7.44, norm=3.75
  ▁Enter (4678) → iders (6935), step 10, alpha=0.53, H=7.43, norm=3.99
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.527, medium<= 8.890
  near: mean_js_dist=0.078, max_js_dist=0.292, mean_H=1.06, max_H=8.14, flip=1.00, switch=0.053, jaccard=0.807, cliff=0.76, plateau=0.95, n=17
  medium: mean_js_dist=0.079, max_js_dist=0.311, mean_H=1.04, max_H=7.43, flip=1.00, switch=0.053, jaccard=0.812, cliff=1.06, plateau=0.95, n=16
  far: mean_js_dist=0.077, max_js_dist=0.361, mean_H=0.93, max_H=7.48, flip=1.00, switch=0.053, jaccard=0.805, cliff=1.18, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.14 ± 0.48

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
- Max mean entropy: 4.03
- Mean mean entropy: 3.98
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 20.57% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     3.99     100   16.14
    10      100.00%     3.98      97   16.16
    20      100.00%     4.02     100   16.11
    30      100.00%     3.96      98   16.16
    40      100.00%     3.99      99   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*