# Token VAE Evaluation Report

Generated: 2026-02-06 14:17:30

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
| Perturbation Stability | Pass | 0.9099 | 0.5000 |
| Interpolation Continuity | Pass | 0.0751 | 0.5000 |
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
- Unique token fraction: 57.27%
- Unique tokens decoded: 5727 / 10000
- Max token frequency: 0.16%
- Tokens seen once: 36.37%
- Prior entropy (mean/median/max): 4.10 / 4.33 / 6.32
- log(V): 9.68
- Gini coefficient: 0.7603

Example decoded tokens from prior samples:
```
  '▁discussed'
  '▁Highway'
  '▁around'
  'night'
  '▁you'
```

Top decoded tokens:
```
  ' (15939): 0.16% (16)
  ▁from (172): 0.16% (16)
  ▁by (161): 0.16% (16)
  D (15932): 0.15% (15)
  ▁you (82): 0.15% (15)
  ▁as (103): 0.14% (14)
  G (15942): 0.14% (14)
  ’ (15940): 0.13% (13)
  ▁your (199): 0.13% (13)
  ▁like (337): 0.13% (13)
  ▁but (224): 0.12% (12)
  ▁or (124): 0.12% (12)
  ▁our (284): 0.12% (12)
  ▁some (349): 0.12% (12)
  ▁It (314): 0.11% (11)
  ▁B (92): 0.11% (11)
  ) (15943): 0.11% (11)
  ▁was (146): 0.11% (11)
  ▁any (365): 0.11% (11)
  ▁into (404): 0.10% (10)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9099
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 90.99% overlap
- σ=0.1: 82.90% overlap
- σ=0.2: 65.69% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.86e+01, p90=3.06e+01, p99=3.34e+01, max=3.46e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.63e-01, p90=3.94e-01, p99=7.29e-01, max=8.69e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.14, H_max=7.67, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.89, H_max=5.28, top1_retention=88.50%
- σ=2.0: safe=100.00%, unique=98.50%, H_mean=1.19, H_max=3.27, top1_retention=11.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0751
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0751
- Max JS distance: 0.3816
- Mean JS divergence (derived): 0.0056
- Max JS divergence (derived): 0.1456
- Mean entropy: 0.82
- Max entropy: 7.51
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.52
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8803
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0747
- Max JS: 0.4283
- Mean entropy: 0.81
- Max entropy: 7.95

- Entropy vs prior (mean Δ / max Δ): -3.28 / 1.19
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  oln (9640) → untary (11272), steps 10→11, alpha 0.53→0.58, JSdist=0.3816, norm 6.28→6.16
  top-10 from:
    oln (9640): 0.876
    untary (11272): 0.074
    uine (8044): 0.000
    ▁enth (6757): 0.000
    ENSO (13122): 0.000
    ▁Similar (11436): 0.000
    . (15914): 0.000
    ▁quil (13357): 0.000
    onsin (10615): 0.000
    adium (4659): 0.000
  top-10 to:
    untary (11272): 0.516
    oln (9640): 0.387
    uine (8044): 0.001
    ▁enth (6757): 0.001
    ▁quil (13357): 0.000
    ENSO (13122): 0.000
    . (15914): 0.000
    ▁Similar (11436): 0.000
    ▁and (37): 0.000
    onsin (10615): 0.000
```

Highest entropy steps:
```
  uling (10624) → ▁car (665), step 10, alpha=0.53, H=7.51, norm=3.68
  uling (10624) → ▁car (665), step 11, alpha=0.58, H=7.46, norm=3.57
  ▁teaches (12815) → Q (15973), step 10, alpha=0.53, H=7.35, norm=3.97
  ▁Employ (10302) → || (593), step 10, alpha=0.53, H=6.81, norm=3.92
  ▁rise (4895) → ▁Diego (13415), step 9, alpha=0.47, H=6.79, norm=4.24
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.997, medium<= 9.608
  near: mean_js_dist=0.082, max_js_dist=0.343, mean_H=1.20, max_H=7.51, flip=1.00, switch=0.053, jaccard=0.904, cliff=0.65, plateau=0.95, n=17
  medium: mean_js_dist=0.071, max_js_dist=0.343, mean_H=0.62, max_H=6.79, flip=1.00, switch=0.053, jaccard=0.894, cliff=1.62, plateau=0.95, n=16
  far: mean_js_dist=0.073, max_js_dist=0.382, mean_H=0.64, max_H=6.24, flip=1.00, switch=0.053, jaccard=0.844, cliff=1.65, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.64 ± 0.95

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
- Mean mean entropy: 4.04
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 20.96% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.11      98   15.99
    10      100.00%     4.06     100   16.00
    20      100.00%     4.04      99   16.02
    30      100.00%     4.00      99   15.98
    40      100.00%     4.03      98   16.02
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*