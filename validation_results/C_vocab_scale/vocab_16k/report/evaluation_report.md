# Token VAE Evaluation Report

Generated: 2026-02-06 14:22:18

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
| Interpolation Continuity | Pass | 0.0780 | 0.5000 |
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
- Unique token fraction: 56.56%
- Unique tokens decoded: 5656 / 10000
- Max token frequency: 0.15%
- Tokens seen once: 35.61%
- Prior entropy (mean/median/max): 4.09 / 4.33 / 6.26
- log(V): 9.68
- Gini coefficient: 0.7650

Example decoded tokens from prior samples:
```
  '▁experience'
  '▁Ob'
  '▁trying'
  'ten'
  '▁ab'
```

Top decoded tokens:
```
  ▁their (263): 0.15% (15)
  ▁like (337): 0.15% (15)
  ’ (15940): 0.15% (15)
  ▁up (249): 0.14% (14)
  ▁from (172): 0.14% (14)
  ▁all (216): 0.14% (14)
  ▁not (179): 0.13% (13)
  ▁have (166): 0.13% (13)
  ▁was (146): 0.13% (13)
  ▁them (382): 0.13% (13)
  t (15895): 0.12% (12)
  S (15920): 0.12% (12)
  ▁we (101): 0.12% (12)
  ▁has (221): 0.12% (12)
  ing (26): 0.12% (12)
  ! (15967): 0.12% (12)
  ▁but (224): 0.11% (11)
  ▁It (314): 0.11% (11)
  ed (29): 0.11% (11)
  ▁The (97): 0.11% (11)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8985
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 89.85% overlap
- σ=0.1: 80.53% overlap
- σ=0.2: 61.48% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.07e+01, p99=3.17e+01, max=3.64e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.65e-01, p90=4.22e-01, p99=5.99e-01, max=6.70e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.05, H_max=2.21, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.94, H_max=5.25, top1_retention=86.50%
- σ=2.0: safe=100.00%, unique=99.00%, H_mean=1.24, H_max=3.28, top1_retention=13.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0780
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0780
- Max JS distance: 0.3784
- Mean JS divergence (derived): 0.0061
- Max JS divergence (derived): 0.1432
- Mean entropy: 0.97
- Max entropy: 7.67
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.01
- Midpoint entropy mean: 3.94
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8800
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0756
- Max JS: 0.4036
- Mean entropy: 0.86
- Max entropy: 8.04

- Entropy vs prior (mean Δ / max Δ): -3.13 / 1.41
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁lets (9049) → cious (4808), steps 9→10, alpha 0.47→0.53, JSdist=0.3784, norm 5.25→5.29
  top-10 from:
    ▁lets (9049): 0.715
    cious (4808): 0.244
    ▁to (32): 0.000
    ▁and (37): 0.000
    , (15915): 0.000
    ▁in (35): 0.000
    . (15914): 0.000
    ▁the (11): 0.000
    ▁a (6): 0.000
    ▁of (36): 0.000
  top-10 to:
    cious (4808): 0.755
    ▁lets (9049): 0.206
    ▁to (32): 0.000
    ▁and (37): 0.000
    ▁in (35): 0.000
    , (15915): 0.000
    . (15914): 0.000
    ▁the (11): 0.000
    ▁a (6): 0.000
    ▁of (36): 0.000
```

Highest entropy steps:
```
  ▁gut (13228) → ▁each (716), step 12, alpha=0.63, H=7.67, norm=3.47
  5 (15948) → ▁ya (15115), step 8, alpha=0.42, H=7.37, norm=3.79
  ▁dresses (9896) → ▁ahead (4338), step 10, alpha=0.53, H=7.24, norm=4.07
  5 (15948) → ▁ya (15115), step 7, alpha=0.37, H=7.23, norm=3.64
  ▁gut (13228) → ▁each (716), step 13, alpha=0.68, H=7.16, norm=3.33
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.824, medium<= 9.533
  near: mean_js_dist=0.081, max_js_dist=0.294, mean_H=1.21, max_H=7.67, flip=1.00, switch=0.053, jaccard=0.904, cliff=0.53, plateau=0.95, n=17
  medium: mean_js_dist=0.078, max_js_dist=0.332, mean_H=0.95, max_H=7.37, flip=1.00, switch=0.053, jaccard=0.890, cliff=0.94, plateau=0.95, n=16
  far: mean_js_dist=0.075, max_js_dist=0.378, mean_H=0.74, max_H=7.09, flip=1.00, switch=0.053, jaccard=0.847, cliff=1.53, plateau=0.95, n=17
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
- Mean mean entropy: 4.16
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 21.80% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.28     100   16.14
    10      100.00%     4.17      99   16.16
    20      100.00%     4.17     100   16.11
    30      100.00%     4.06      99   16.16
    40      100.00%     4.17     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*