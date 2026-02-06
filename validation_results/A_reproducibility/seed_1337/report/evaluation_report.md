# Token VAE Evaluation Report

Generated: 2026-02-06 14:19:04

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
| Perturbation Stability | Pass | 0.9056 | 0.5000 |
| Interpolation Continuity | Pass | 0.0759 | 0.5000 |
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
- Unique token fraction: 56.38%
- Unique tokens decoded: 5638 / 10000
- Max token frequency: 0.14%
- Tokens seen once: 34.90%
- Prior entropy (mean/median/max): 4.08 / 4.32 / 6.20
- log(V): 9.68
- Gini coefficient: 0.7642

Example decoded tokens from prior samples:
```
  'reck'
  'ced'
  '▁corresponding'
  '▁gas'
  '▁turn'
```

Top decoded tokens:
```
  ▁have (166): 0.14% (14)
  ▁The (97): 0.14% (14)
  ' (15939): 0.13% (13)
  : (15944): 0.13% (13)
  ▁are (119): 0.13% (13)
  ) (15943): 0.12% (12)
  ed (29): 0.12% (12)
  ▁- (138): 0.12% (12)
  ▁was (146): 0.12% (12)
  ▁they (267): 0.11% (11)
  ▁B (92): 0.11% (11)
  ▁been (353): 0.11% (11)
  ▁not (179): 0.11% (11)
  D (15932): 0.11% (11)
  ▁It (314): 0.11% (11)
  K (15962): 0.10% (10)
  ▁an (104): 0.10% (10)
  ▁no (446): 0.10% (10)
  ▁but (224): 0.10% (10)
  ▁some (349): 0.10% (10)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9056
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 90.56% overlap
- σ=0.1: 82.17% overlap
- σ=0.2: 63.45% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.09e+01, p99=3.40e+01, max=3.58e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.56e-01, p90=4.89e-01, p99=7.34e-01, max=7.98e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.06, H_max=4.06, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.75, H_max=4.81, top1_retention=93.50%
- σ=2.0: safe=100.00%, unique=97.00%, H_mean=1.16, H_max=3.29, top1_retention=12.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0759
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0729
- Max JS distance: 0.3765
- Mean JS divergence (derived): 0.0053
- Max JS divergence (derived): 0.1417
- Mean entropy: 0.71
- Max entropy: 7.78
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 2.97
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8825
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0759
- Max JS: 0.3835
- Mean entropy: 0.88
- Max entropy: 7.66

- Entropy vs prior (mean Δ / max Δ): -3.37 / 1.58
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ullivan (4344) → ▁192 (13177), steps 10→11, alpha 0.53→0.58, JSdist=0.3765, norm 5.55→5.51
  top-10 from:
    ullivan (4344): 0.637
    ▁192 (13177): 0.262
    , (15915): 0.001
    ▁and (37): 0.001
    . (15914): 0.001
    ▁the (11): 0.000
    - (15922): 0.000
    ▁of (36): 0.000
    ▁in (35): 0.000
    ▁a (6): 0.000
  top-10 to:
    ▁192 (13177): 0.753
    ullivan (4344): 0.153
    , (15915): 0.001
    ▁and (37): 0.001
    . (15914): 0.001
    ▁the (11): 0.000
    - (15922): 0.000
    ▁of (36): 0.000
    ▁in (35): 0.000
    ▁a (6): 0.000
```

Highest entropy steps:
```
  ▁wor (209) → -1, (13036), step 9, alpha=0.47, H=7.78, norm=4.27
  ▁wor (209) → -1, (13036), step 10, alpha=0.53, H=7.49, norm=4.29
  ▁wor (209) → -1, (13036), step 8, alpha=0.42, H=6.72, norm=4.31
  ▁Italy (8742) → osa (11635), step 9, alpha=0.47, H=6.44, norm=4.51
  ▁innings (10879) → ance (297), step 10, alpha=0.53, H=6.29, norm=3.85
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 9.017, medium<= 9.541
  near: mean_js_dist=0.073, max_js_dist=0.325, mean_H=0.79, max_H=6.29, flip=1.00, switch=0.053, jaccard=0.901, cliff=1.00, plateau=0.95, n=17
  medium: mean_js_dist=0.074, max_js_dist=0.330, mean_H=0.76, max_H=7.78, flip=1.00, switch=0.053, jaccard=0.887, cliff=1.38, plateau=0.95, n=16
  far: mean_js_dist=0.071, max_js_dist=0.376, mean_H=0.58, max_H=6.44, flip=1.00, switch=0.053, jaccard=0.860, cliff=1.76, plateau=0.95, n=17
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
- Max mean entropy: 4.14
- Mean mean entropy: 4.04
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 20.04% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.14      98   16.02
    10      100.00%     4.05      97   16.05
    20      100.00%     4.01      97   15.99
    30      100.00%     3.89     100   16.01
    40      100.00%     4.09     100   16.01
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*