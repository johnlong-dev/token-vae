# Token VAE Evaluation Report

Generated: 2026-02-06 14:21:19

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
| Prior Decodability | Pass | 0.9999 | 0.8500 |
| Perturbation Stability | Pass | 0.8975 | 0.5000 |
| Interpolation Continuity | Pass | 0.0754 | 0.5000 |
| Reconstruction Accuracy | Pass | 1.0000 | 0.9000 |
| Metric Integrity | Pass | 1.0000 | 1.0000 |
| Diffusion Walk | Pass | 1.0000 | 0.9500 |

---

## Detailed Test Results

### Prior Decodability

**Description**: Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens

**Result**: PASSED
- Score: 0.9999
- Threshold: 0.8500

**Details**:

- Non-special token fraction: 99.99%
- Unique token fraction: 34.05%
- Unique tokens decoded: 3405 / 10000
- Max token frequency: 0.28%
- Tokens seen once: 9.70%
- Prior entropy (mean/median/max): 4.06 / 4.24 / 5.67
- log(V): 8.29
- Gini coefficient: 0.4686

Example decoded tokens from prior samples:
```
  '▁super'
  'board'
  '▁goals'
  '▁have'
  '▁vac'
```

Top decoded tokens:
```
  ▁to (32): 0.28% (28)
  s (3900): 0.26% (26)
  ▁of (36): 0.26% (26)
  ▁and (37): 0.26% (26)
  ▁a (6): 0.23% (23)
  ▁in (35): 0.23% (23)
  ▁is (66): 0.20% (20)
  - (3922): 0.19% (19)
  ▁I (50): 0.19% (19)
  ▁for (72): 0.18% (18)
  and (182): 0.18% (18)
  ▁from (172): 0.18% (18)
  ▁with (94): 0.17% (17)
  ing (26): 0.17% (17)
  ▁on (71): 0.17% (17)
  ’ (3940): 0.17% (17)
  ▁G (144): 0.16% (16)
  ed (29): 0.16% (16)
  : (3944): 0.14% (14)
  ▁we (101): 0.14% (14)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8975
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 89.75% overlap
- σ=0.1: 81.02% overlap
- σ=0.2: 65.39% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=1.28e+01, p90=1.30e+01, p99=1.32e+01, max=1.34e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.49e-01, p90=3.28e-01, p99=5.30e-01, max=5.86e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.09, H_max=2.85, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=99.00%, H_mean=0.94, H_max=4.93, top1_retention=91.00%
- σ=2.0: safe=100.00%, unique=97.50%, H_mean=1.26, H_max=3.60, top1_retention=16.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0754
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0742
- Max JS distance: 0.3051
- Mean JS divergence (derived): 0.0055
- Max JS divergence (derived): 0.0931
- Mean entropy: 0.80
- Max entropy: 6.05
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.01
- Midpoint entropy mean: 3.09
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.9396
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0754
- Max JS: 0.3345
- Mean entropy: 0.92
- Max entropy: 6.69

- Entropy vs prior (mean Δ / max Δ): -3.25 / 0.39
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  key (2666) → ▁Rousseau (2877), steps 9→10, alpha 0.47→0.53, JSdist=0.3051, norm 4.79→4.83
  top-10 from:
    key (2666): 0.565
    ▁Rousseau (2877): 0.336
    . (3914): 0.003
    , (3915): 0.003
    ▁the (11): 0.003
    ▁2015 (3846): 0.000
    ▁Dis (2584): 0.000
    ▁whose (3420): 0.000
    arch (606): 0.000
    go (2137): 0.000
  top-10 to:
    ▁Rousseau (2877): 0.742
    key (2666): 0.177
    , (3915): 0.003
    . (3914): 0.003
    ▁the (11): 0.002
    ▁2015 (3846): 0.000
    ▁Dis (2584): 0.000
    go (2137): 0.000
    arch (606): 0.000
    ▁of (36): 0.000
```

Highest entropy steps:
```
  ▁R (113) → ▁Then (2303), step 8, alpha=0.42, H=6.05, norm=3.30
  ▁& (641) → ives (886), step 9, alpha=0.47, H=5.72, norm=3.69
  ▁R (113) → ▁Then (2303), step 9, alpha=0.47, H=5.71, norm=3.41
  ▁Des (1555) → ▁Ch (327), step 10, alpha=0.53, H=5.68, norm=3.77
  her (127) → ▁if (388), step 10, alpha=0.53, H=5.49, norm=3.28
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.033, medium<= 8.672
  near: mean_js_dist=0.076, max_js_dist=0.263, mean_H=1.00, max_H=6.05, flip=1.00, switch=0.053, jaccard=0.944, cliff=0.18, plateau=0.95, n=17
  medium: mean_js_dist=0.075, max_js_dist=0.270, mean_H=0.80, max_H=5.68, flip=1.00, switch=0.053, jaccard=0.935, cliff=0.69, plateau=0.95, n=16
  far: mean_js_dist=0.072, max_js_dist=0.305, mean_H=0.61, max_H=4.81, flip=1.00, switch=0.053, jaccard=0.939, cliff=1.41, plateau=0.95, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.9000

**Details**:

- V=4,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 3,995 evaluated
- Top-1 accuracy: 100.00%
- Top-5 accuracy: 100.00%
- Correct: 3995 / 3995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 5.93 ± 0.79

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
- Max mean entropy: 3.97
- Mean mean entropy: 3.87
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 18.78% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     3.83      99   16.14
    10      100.00%     3.81      97   16.16
    20      100.00%     3.89      99   16.11
    30      100.00%     3.84      98   16.16
    40      100.00%     3.90     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*