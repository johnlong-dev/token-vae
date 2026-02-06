# Token VAE Evaluation Report

Generated: 2026-02-06 14:27:22

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

**One or more evaluation criteria not satisfied.**

| Test | Status | Score | Threshold |
|------|--------|-------|-----------|
| Prior Decodability | Fail | 1.0000 | 0.8500 |
| Perturbation Stability | Pass | 0.7349 | 0.5000 |
| Interpolation Continuity | Pass | 0.0695 | 0.5000 |
| Reconstruction Accuracy | Pass | 1.0000 | 0.9000 |
| Metric Integrity | Fail | 0.0000 | 1.0000 |
| Diffusion Walk | Pass | 1.0000 | 0.9500 |

---

## Detailed Test Results

### Prior Decodability

**Description**: Evaluates whether samples from the standard normal prior decode to non-degenerate vocabulary tokens

**Result**: FAILED
- Score: 1.0000
- Threshold: 0.8500

**Details**:

- Non-special token fraction: 100.00%
- Unique token fraction: 14.46%
- Unique tokens decoded: 1446 / 10000
- Max token frequency: 2.66%
- Tokens seen once: 6.68%
- Prior entropy (mean/median/max): 0.31 / 0.10 / 2.31
- log(V): 9.68
- Gini coefficient: 0.9740

Example decoded tokens from prior samples:
```
  ','
  'e'
  '▁the'
  '▁game'
  '▁t'
```

Top decoded tokens:
```
  ▁the (11): 2.66% (266)
  . (15914): 2.15% (215)
  , (15915): 2.14% (214)
  ▁that (86): 1.54% (154)
  ▁of (36): 1.30% (130)
  ▁and (37): 1.25% (125)
  ▁to (32): 1.18% (118)
  ▁in (35): 1.12% (112)
  s (15900): 1.05% (105)
  ’ (15940): 1.05% (105)
  ▁it (96): 1.02% (102)
  ▁for (72): 0.95% (95)
  ▁a (6): 0.93% (93)
  ▁I (50): 0.89% (89)
  ▁is (66): 0.88% (88)
  ▁on (71): 0.84% (84)
  ▁from (172): 0.80% (80)
  ▁or (124): 0.79% (79)
  ▁have (166): 0.75% (75)
  ▁be (67): 0.72% (72)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.7349
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 73.49% overlap
- σ=0.1: 52.92% overlap
- σ=0.2: 29.72% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=8.29e+01, p90=8.46e+01, p99=8.77e+01, max=9.08e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=6.26e-01, p90=1.90e+00, p99=3.33e+00, max=3.51e+00

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=82.00%, H_mean=0.68, H_max=3.33, top1_retention=17.50%
- σ=1.0: safe=100.00%, unique=80.50%, H_mean=0.35, H_max=1.77, top1_retention=0.50%
- σ=2.0: safe=100.00%, unique=74.50%, H_mean=0.14, H_max=1.66, top1_retention=0.00%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0695
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0682
- Max JS distance: 0.4006
- Mean JS divergence (derived): 0.0046
- Max JS divergence (derived): 0.1605
- Mean entropy: 0.39
- Max entropy: 6.05
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 2.11
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8075
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0695
- Max JS: 0.4279
- Mean entropy: 0.43
- Max entropy: 6.79

- Entropy vs prior (mean Δ / max Δ): 0.08 / 3.75
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁Recently (12071) → ▁X (1684), steps 9→10, alpha 0.47→0.53, JSdist=0.4006, norm 2.42→2.33
  top-10 from:
    ▁Recently (12071): 0.558
    ▁X (1684): 0.240
    ▁in (35): 0.006
    ▁light (1414): 0.003
    ▁28 (3941): 0.003
    ▁assessment (6674): 0.002
    et (62): 0.001
    ▁pattern (4553): 0.001
    eman (8964): 0.001
    ▁options (2639): 0.001
  top-10 to:
    ▁X (1684): 0.761
    ▁Recently (12071): 0.105
    ▁in (35): 0.003
    ▁light (1414): 0.002
    ▁28 (3941): 0.002
    ▁assessment (6674): 0.001
    et (62): 0.001
    ▁pattern (4553): 0.000
    ▁Ch (327): 0.000
    eman (8964): 0.000
```

Highest entropy steps:
```
  ▁Saf (6917) → com (525), step 11, alpha=0.58, H=6.05, norm=2.50
  ▁shop (4746) → ▁belie (1312), step 9, alpha=0.47, H=4.85, norm=2.67
  ▁Saf (6917) → com (525), step 12, alpha=0.63, H=4.22, norm=2.32
  ▁Saf (6917) → com (525), step 10, alpha=0.53, H=4.18, norm=2.70
  ▁Pal (4511) → ▁throw (4017), step 9, alpha=0.47, H=4.15, norm=2.23
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 4.762, medium<= 5.167
  near: mean_js_dist=0.069, max_js_dist=0.376, mean_H=0.42, max_H=4.15, flip=1.00, switch=0.053, jaccard=0.812, cliff=2.00, plateau=0.95, n=17
  medium: mean_js_dist=0.067, max_js_dist=0.401, mean_H=0.34, max_H=3.98, flip=1.00, switch=0.053, jaccard=0.800, cliff=1.94, plateau=0.95, n=16
  far: mean_js_dist=0.069, max_js_dist=0.394, mean_H=0.40, max_H=6.05, flip=1.00, switch=0.053, jaccard=0.811, cliff=1.94, plateau=0.95, n=17
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
- μ norm (mean ± std): 3.63 ± 0.70

### Metric Integrity

**Description**: Cross-validates metrics to detect inconsistencies indicative of posterior collapse or degenerate solutions

**Result**: FAILED
- Score: 0.0000
- Threshold: 1.0000

**Details**:

Triggered integrity issues:
```
  - Interpolation looks good but prior unique fraction is low (0.145)
  - Interpolation looks good but prior entropy is low (0.308)
  - Reconstruction is high (1.000) while prior unique fraction is low (0.145)
```

### Diffusion Walk

**Description**: Simulates forward-diffusion trajectories and verifies that decoded tokens remain non-degenerate

**Result**: PASSED
- Score: 1.0000
- Threshold: 0.9500

**Details**:

- Min non-special rate: 100.00%
- Mean non-special rate: 100.00%
- Max mean entropy: 0.42
- Mean mean entropy: 0.33
- Min unique fraction: 78.00% (threshold 10.00%)
- Mean step change rate: 16.98% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     0.30      78   16.14
    10      100.00%     0.34      86   16.16
    20      100.00%     0.25      82   16.11
    30      100.00%     0.33      82   16.16
    40      100.00%     0.31      89   16.17
```

---

## Failure Analysis

### Prior Decodability

**Possible causes**:
- Insufficient KL regularization — embeddings may not fill the prior support
- Excessive KL regularization — posterior collapse, all tokens map to the origin

**Recommendations**:
- If unique fraction is low: increase KL weight or training epochs
- If non-special fraction is low: check decoder capacity

### Metric Integrity

**Possible causes**:
- Prior distribution collapsed despite strong reconstruction
- Metrics improving in isolation but not jointly

**Recommendations**:
- Increase KL weight or prior regularization
- Rebalance interpolation and skipgram losses
- Check prior decodability diversity stats

---

## Conclusion

**One or more evaluation criteria are not satisfied.** Refer to the failure
analysis above for diagnostics and recommendations.

---

*Generated by the Token VAE evaluation pipeline.*