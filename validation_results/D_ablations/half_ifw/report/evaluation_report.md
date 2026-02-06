# Token VAE Evaluation Report

Generated: 2026-02-06 14:25:40

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
| Perturbation Stability | Pass | 0.8408 | 0.5000 |
| Interpolation Continuity | Pass | 0.0735 | 0.5000 |
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
- Unique token fraction: 63.60%
- Unique tokens decoded: 6360 / 10000
- Max token frequency: 0.45%
- Tokens seen once: 43.61%
- Prior entropy (mean/median/max): 4.08 / 4.30 / 6.17
- log(V): 9.68
- Gini coefficient: 0.7184

Example decoded tokens from prior samples:
```
  '▁Irving'
  '▁Me'
  '▁Power'
  'oos'
  '▁je'
```

Top decoded tokens:
```
  ▁the (11): 0.45% (45)
  , (15915): 0.43% (43)
  . (15914): 0.39% (39)
  ▁and (37): 0.35% (35)
  ▁of (36): 0.34% (34)
  ▁for (72): 0.20% (20)
  - (15922): 0.20% (20)
  ▁to (32): 0.19% (19)
  ▁that (86): 0.17% (17)
  s (15900): 0.16% (16)
  ▁is (66): 0.15% (15)
  ▁in (35): 0.15% (15)
  ▁he (100): 0.15% (15)
  ' (15939): 0.14% (14)
  ▁I (50): 0.14% (14)
  ▁a (6): 0.14% (14)
  ▁you (82): 0.13% (13)
  ▁out (246): 0.13% (13)
  ▁from (172): 0.12% (12)
  ▁was (146): 0.12% (12)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8408
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 84.08% overlap
- σ=0.1: 70.21% overlap
- σ=0.2: 47.88% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.96e+01, p90=3.03e+01, p99=3.11e+01, max=3.14e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.82e-01, p90=5.00e-01, p99=7.53e-01, max=8.21e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.03, H_max=1.28, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=99.50%, H_mean=0.88, H_max=5.23, top1_retention=88.50%
- σ=2.0: safe=100.00%, unique=98.50%, H_mean=1.25, H_max=3.14, top1_retention=10.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0735
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0730
- Max JS distance: 0.3852
- Mean JS divergence (derived): 0.0053
- Max JS divergence (derived): 0.1484
- Mean entropy: 0.69
- Max entropy: 6.23
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 3.35
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8099
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0735
- Max JS: 0.4120
- Mean entropy: 0.72
- Max entropy: 7.62

- Entropy vs prior (mean Δ / max Δ): -3.39 / 0.06
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  abled (11739) → ▁|0 (5152), steps 8→9, alpha 0.42→0.47, JSdist=0.3852, norm 5.37→5.44
  top-10 from:
    abled (11739): 0.695
    ▁|0 (5152): 0.249
    oln (9640): 0.000
    hovah (15298): 0.000
    minist (2790): 0.000
    usalem (11847): 0.000
    ukocy (5738): 0.000
    ▁Dian (14237): 0.000
    kele (11114): 0.000
    ▁soph (6598): 0.000
  top-10 to:
    ▁|0 (5152): 0.770
    abled (11739): 0.187
    oln (9640): 0.000
    hovah (15298): 0.000
    usalem (11847): 0.000
    minist (2790): 0.000
    ukocy (5738): 0.000
    ▁Dian (14237): 0.000
    kele (11114): 0.000
    ▁experien (2853): 0.000
```

Highest entropy steps:
```
  ▁Other (2541) → gers (3116), step 9, alpha=0.47, H=6.23, norm=3.97
  ▁capt (1969) → ▁surg (15432), step 10, alpha=0.53, H=5.86, norm=4.45
  ▁fun (965) → rought (2702), step 9, alpha=0.47, H=5.77, norm=4.21
  ▁Other (2541) → gers (3116), step 10, alpha=0.53, H=5.70, norm=3.99
  ▁Enter (4678) → iders (6935), step 10, alpha=0.53, H=5.67, norm=4.30
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 8.698, medium<= 9.480
  near: mean_js_dist=0.075, max_js_dist=0.307, mean_H=0.82, max_H=6.23, flip=1.00, switch=0.053, jaccard=0.811, cliff=1.24, plateau=0.95, n=17
  medium: mean_js_dist=0.073, max_js_dist=0.352, mean_H=0.68, max_H=5.67, flip=1.00, switch=0.053, jaccard=0.816, cliff=1.62, plateau=0.95, n=16
  far: mean_js_dist=0.071, max_js_dist=0.385, mean_H=0.56, max_H=5.86, flip=1.00, switch=0.053, jaccard=0.802, cliff=1.88, plateau=0.95, n=17
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
- μ norm (mean ± std): 6.47 ± 0.67

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
- Max mean entropy: 4.13
- Mean mean entropy: 3.94
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 19.69% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.13      99   16.14
    10      100.00%     3.87      99   16.16
    20      100.00%     3.87      98   16.11
    30      100.00%     3.87     100   16.16
    40      100.00%     3.98     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*