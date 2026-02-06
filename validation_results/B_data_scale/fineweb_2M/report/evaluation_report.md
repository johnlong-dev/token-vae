# Token VAE Evaluation Report

Generated: 2026-02-06 14:19:40

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
| Perturbation Stability | Pass | 0.9058 | 0.5000 |
| Interpolation Continuity | Pass | 0.0897 | 0.5000 |
| Reconstruction Accuracy | Pass | 0.9914 | 0.9000 |
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
- Unique token fraction: 44.09%
- Unique tokens decoded: 4409 / 10000
- Max token frequency: 0.20%
- Tokens seen once: 25.39%
- Prior entropy (mean/median/max): 6.07 / 6.29 / 7.66
- log(V): 9.68
- Gini coefficient: 0.8418

Example decoded tokens from prior samples:
```
  '▁longer'
  'conom'
  '▁November'
  '▁responsible'
  '▁day'
```

Top decoded tokens:
```
  1 (15925): 0.20% (20)
  ▁years (596): 0.18% (18)
  ▁want (533): 0.17% (17)
  en (22): 0.16% (16)
  ▁through (503): 0.16% (16)
  ▁get (364): 0.15% (15)
  ▁U (228): 0.15% (15)
  ▁right (632): 0.15% (15)
  ate (122): 0.15% (15)
  R (15931): 0.14% (14)
  ." (846): 0.14% (14)
  ▁both (903): 0.14% (14)
  ▁should (675): 0.14% (14)
  ▁same (781): 0.14% (14)
  ▁just (397): 0.14% (14)
  in (7): 0.14% (14)
  ▁work (323): 0.13% (13)
  ar (34): 0.13% (13)
  ” (15966): 0.13% (13)
  ▁think (590): 0.13% (13)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9058
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 90.58% overlap
- σ=0.1: 82.18% overlap
- σ=0.2: 68.18% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.08e+01, p90=2.39e+01, p99=2.87e+01, max=3.09e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=9.91e-02, p90=2.62e-01, p99=5.10e-01, max=5.71e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=99.50%, H_mean=0.88, H_max=8.17, top1_retention=99.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=1.88, H_max=6.88, top1_retention=88.00%
- σ=2.0: safe=100.00%, unique=98.50%, H_mean=1.73, H_max=4.17, top1_retention=17.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0897
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0889
- Max JS distance: 0.2617
- Mean JS divergence (derived): 0.0079
- Max JS divergence (derived): 0.0685
- Mean entropy: 3.39
- Max entropy: 8.22
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.57
- Midpoint entropy mean: 6.95
- Mean argmax switch rate: 0.0832
- Mean top-k Jaccard: 0.9396
- Mean plateau ratio: 0.9168
- Mean flip count: 1.58
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0897
- Max JS: 0.3216
- Mean entropy: 3.11
- Max entropy: 8.19

- Entropy vs prior (mean Δ / max Δ): -2.69 / 0.56
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁WARR (10652) → ▁employer (11495), steps 10→11, alpha 0.53→0.58, JSdist=0.2617, norm 6.51→6.38
  top-10 from:
    ▁WARR (10652): 0.538
    ▁employer (11495): 0.168
    ▁and (37): 0.022
    ▁the (11): 0.016
    . (15914): 0.002
    pat (11069): 0.000
    aks (5361): 0.000
    wor (13087): 0.000
    ▁nav (8069): 0.000
    ▁Dis (2584): 0.000
  top-10 to:
    ▁employer (11495): 0.444
    ▁WARR (10652): 0.218
    ▁and (37): 0.026
    ▁the (11): 0.018
    . (15914): 0.003
    aks (5361): 0.001
    pat (11069): 0.000
    ▁Dis (2584): 0.000
    ▁Balt (8371): 0.000
    ▁nav (8069): 0.000
```

Highest entropy steps:
```
  gly (8933) → ▁? (6878), step 9, alpha=0.47, H=8.22, norm=4.65
  gly (8933) → ▁? (6878), step 10, alpha=0.53, H=8.22, norm=4.61
  ALLY (14018) → ▁studies (3819), step 11, alpha=0.58, H=8.14, norm=4.37
  ▁emer (3662) → ask (2291), step 11, alpha=0.58, H=8.12, norm=4.64
  gly (8933) → ▁? (6878), step 11, alpha=0.58, H=8.11, norm=4.64
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 9.415, medium<= 10.353
  near: mean_js_dist=0.087, max_js_dist=0.155, mean_H=4.46, max_H=8.09, flip=2.00, switch=0.105, jaccard=0.941, cliff=0.00, plateau=0.89, n=17
  medium: mean_js_dist=0.093, max_js_dist=0.165, mean_H=3.63, max_H=8.14, flip=1.69, switch=0.089, jaccard=0.939, cliff=0.00, plateau=0.91, n=16
  far: mean_js_dist=0.087, max_js_dist=0.262, mean_H=2.08, max_H=8.22, flip=1.06, switch=0.056, jaccard=0.939, cliff=0.12, plateau=0.94, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 0.9914
- Threshold: 0.9000

**Details**:

- V=16,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 15,995 evaluated
- Top-1 accuracy: 99.14%
- Top-5 accuracy: 99.81%
- Correct: 15857 / 15995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 7.18 ± 1.39

Example failures:
```
  '▁a' → '▁and'
  're' → '▁and'
  '▁the' → '▁and'
  'er' → '▁and'
  'or' → '▁and'
```

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
- Max mean entropy: 6.18
- Mean mean entropy: 6.09
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 18.41% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     6.08     100   16.14
    10      100.00%     6.06     100   16.16
    20      100.00%     6.11      99   16.11
    30      100.00%     6.09      99   16.16
    40      100.00%     6.04      99   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*