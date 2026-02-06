# Token VAE Evaluation Report

Generated: 2026-02-06 14:21:43

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
| Prior Decodability | Pass | 0.9997 | 0.8500 |
| Perturbation Stability | Pass | 0.9465 | 0.5000 |
| Interpolation Continuity | Pass | 0.0750 | 0.5000 |
| Reconstruction Accuracy | Pass | 1.0000 | 0.9000 |
| Metric Integrity | Pass | 1.0000 | 1.0000 |
| Diffusion Walk | Pass | 0.9900 | 0.9500 |

---

## Detailed Test Results

### Prior Decodability

**Description**: Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens

**Result**: PASSED
- Score: 0.9997
- Threshold: 0.8500

**Details**:

- Non-special token fraction: 99.97%
- Unique token fraction: 51.16%
- Unique tokens decoded: 5116 / 10000
- Max token frequency: 0.20%
- Tokens seen once: 25.84%
- Prior entropy (mean/median/max): 4.07 / 4.27 / 6.01
- log(V): 8.99
- Gini coefficient: 0.5721

Example decoded tokens from prior samples:
```
  '2,'
  '▁don'
  '▁dust'
  '▁utility'
  'oint'
```

Top decoded tokens:
```
  ▁is (66): 0.20% (20)
  ▁for (72): 0.17% (17)
  - (7922): 0.16% (16)
  ing (26): 0.16% (16)
  ▁as (103): 0.16% (16)
  ▁that (86): 0.14% (14)
  ▁your (199): 0.13% (13)
  ed (29): 0.13% (13)
  ▁are (119): 0.13% (13)
  ▁I (50): 0.13% (13)
  ▁also (347): 0.13% (13)
  ▁be (67): 0.13% (13)
  ' (7939): 0.13% (13)
  ▁over (369): 0.12% (12)
  ▁in (35): 0.12% (12)
  ▁L (142): 0.12% (12)
  s (7900): 0.12% (12)
  ▁when (386): 0.12% (12)
  ▁you (82): 0.12% (12)
  ▁with (94): 0.12% (12)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9465
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 94.65% overlap
- σ=0.1: 89.63% overlap
- σ=0.2: 77.50% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=1.97e+01, p90=2.01e+01, p99=2.07e+01, max=2.12e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.60e-01, p90=3.64e-01, p99=5.56e-01, max=8.98e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.05, H_max=1.24, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.87, H_max=4.89, top1_retention=92.50%
- σ=2.0: safe=100.00%, unique=97.00%, H_mean=1.20, H_max=2.86, top1_retention=14.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0750
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0750
- Max JS distance: 0.3597
- Mean JS divergence (derived): 0.0056
- Max JS divergence (derived): 0.1294
- Mean entropy: 0.79
- Max entropy: 6.48
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.01
- Midpoint entropy mean: 3.33
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.9203
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0744
- Max JS: 0.4011
- Mean entropy: 0.78
- Max entropy: 6.87

- Entropy vs prior (mean Δ / max Δ): -3.28 / 0.47
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁First (2792) → arks (2281), steps 9→10, alpha 0.47→0.53, JSdist=0.3597, norm 4.92→4.94
  top-10 from:
    ▁First (2792): 0.653
    arks (2281): 0.263
    ▁of (36): 0.002
    ▁and (37): 0.002
    ▁the (11): 0.002
    ▁to (32): 0.001
    . (7914): 0.001
    , (7915): 0.001
    ▁a (6): 0.001
    ▁in (35): 0.000
  top-10 to:
    arks (2281): 0.740
    ▁First (2792): 0.183
    ▁of (36): 0.002
    ▁and (37): 0.001
    ▁the (11): 0.001
    . (7914): 0.001
    ▁to (32): 0.001
    , (7915): 0.001
    ▁a (6): 0.001
    ▁bitcoin (7019): 0.000
```

Highest entropy steps:
```
  ON (1278) → ▁space (1768), step 10, alpha=0.53, H=6.48, norm=3.53
  ▁up (249) → ▁art (772), step 9, alpha=0.47, H=6.42, norm=3.11
  ▁up (249) → ▁art (772), step 8, alpha=0.42, H=6.09, norm=3.04
  ▁up (249) → ▁art (772), step 10, alpha=0.53, H=6.01, norm=3.21
  ON (1278) → ▁space (1768), step 11, alpha=0.58, H=5.91, norm=3.53
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.638, medium<= 9.010
  near: mean_js_dist=0.080, max_js_dist=0.279, mean_H=1.11, max_H=6.48, flip=1.00, switch=0.053, jaccard=0.930, cliff=0.41, plateau=0.95, n=17
  medium: mean_js_dist=0.075, max_js_dist=0.320, mean_H=0.76, max_H=5.64, flip=1.00, switch=0.053, jaccard=0.926, cliff=1.00, plateau=0.95, n=16
  far: mean_js_dist=0.070, max_js_dist=0.360, mean_H=0.51, max_H=5.15, flip=1.00, switch=0.053, jaccard=0.905, cliff=1.76, plateau=0.95, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.9000

**Details**:

- V=8,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 7,995 evaluated
- Top-1 accuracy: 100.00%
- Top-5 accuracy: 100.00%
- Correct: 7995 / 7995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 6.17 ± 0.82

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
- Score: 0.9900
- Threshold: 0.9500

**Details**:

- Min non-special rate: 99.00%
- Mean non-special rate: 99.84%
- Max mean entropy: 4.13
- Mean mean entropy: 4.04
- Min unique fraction: 99.00% (threshold 10.00%)
- Mean step change rate: 19.10% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.06     100   16.14
    10      100.00%     3.99     100   16.16
    20      100.00%     3.99     100   16.11
    30      100.00%     4.03      99   16.16
    40      100.00%     4.13     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*