# Token VAE Evaluation Report

Generated: 2026-02-06 14:15:58

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
| Perturbation Stability | Pass | 0.8995 | 0.5000 |
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
- Unique token fraction: 56.70%
- Unique tokens decoded: 5670 / 10000
- Max token frequency: 0.17%
- Tokens seen once: 35.73%
- Prior entropy (mean/median/max): 4.09 / 4.33 / 6.31
- log(V): 9.68
- Gini coefficient: 0.7638

Example decoded tokens from prior samples:
```
  '30'
  '▁Ob'
  'nes'
  '▁forces'
  'atform'
```

Top decoded tokens:
```
  ▁all (216): 0.17% (17)
  ▁will (197): 0.15% (15)
  ▁from (172): 0.14% (14)
  ▁was (146): 0.14% (14)
  : (15944): 0.13% (13)
  S (15920): 0.13% (13)
  ▁but (224): 0.13% (13)
  ▁have (166): 0.13% (13)
  ▁like (337): 0.12% (12)
  ▁not (179): 0.12% (12)
  ▁The (97): 0.12% (12)
  t (15895): 0.12% (12)
  ! (15967): 0.11% (11)
  ▁you (82): 0.11% (11)
  / (15957): 0.11% (11)
  ▁he (100): 0.11% (11)
  ▁we (101): 0.11% (11)
  ▁work (323): 0.11% (11)
  ▁by (161): 0.11% (11)
  ' (15939): 0.10% (10)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8995
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 89.95% overlap
- σ=0.1: 80.74% overlap
- σ=0.2: 61.43% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.88e+01, p90=3.07e+01, p99=3.17e+01, max=3.64e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.68e-01, p90=4.29e-01, p99=5.97e-01, max=6.31e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.05, H_max=2.25, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.94, H_max=5.19, top1_retention=86.00%
- σ=2.0: safe=100.00%, unique=98.50%, H_mean=1.20, H_max=3.10, top1_retention=11.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0780
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0780
- Max JS distance: 0.3787
- Mean JS divergence (derived): 0.0061
- Max JS divergence (derived): 0.1434
- Mean entropy: 0.97
- Max entropy: 7.62
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.01
- Midpoint entropy mean: 3.91
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8818
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0755
- Max JS: 0.3816
- Mean entropy: 0.85
- Max entropy: 7.98

- Entropy vs prior (mean Δ / max Δ): -3.12 / 1.31
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁lets (9049) → cious (4808), steps 9→10, alpha 0.47→0.53, JSdist=0.3787, norm 5.32→5.34
  top-10 from:
    ▁lets (9049): 0.680
    cious (4808): 0.287
    ▁to (32): 0.000
    ▁and (37): 0.000
    , (15915): 0.000
    ▁in (35): 0.000
    . (15914): 0.000
    ▁the (11): 0.000
    ▁a (6): 0.000
    ▁of (36): 0.000
  top-10 to:
    cious (4808): 0.799
    ▁lets (9049): 0.174
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
  ▁gut (13228) → ▁each (716), step 12, alpha=0.63, H=7.62, norm=3.46
  ▁fallen (11960) → iev (7532), step 10, alpha=0.53, H=7.34, norm=4.34
  5 (15948) → ▁ya (15115), step 8, alpha=0.42, H=7.28, norm=3.77
  ▁dresses (9896) → ▁ahead (4338), step 10, alpha=0.53, H=7.21, norm=4.07
  ▁Distribution (13890) → ▁gross (5640), step 9, alpha=0.47, H=7.20, norm=4.46
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.834, medium<= 9.522
  near: mean_js_dist=0.082, max_js_dist=0.289, mean_H=1.24, max_H=7.62, flip=1.00, switch=0.053, jaccard=0.904, cliff=0.41, plateau=0.95, n=17
  medium: mean_js_dist=0.080, max_js_dist=0.329, mean_H=1.04, max_H=7.34, flip=1.00, switch=0.053, jaccard=0.892, cliff=1.12, plateau=0.95, n=16
  far: mean_js_dist=0.073, max_js_dist=0.379, mean_H=0.64, max_H=7.20, flip=1.00, switch=0.053, jaccard=0.850, cliff=1.71, plateau=0.95, n=17
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
- Mean mean entropy: 4.14
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 21.02% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.26     100   16.14
    10      100.00%     4.13      98   16.16
    20      100.00%     4.10      99   16.11
    30      100.00%     4.09      99   16.16
    40      100.00%     4.20     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*