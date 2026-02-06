# Token VAE Evaluation Report

Generated: 2026-02-06 14:27:56

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
| Perturbation Stability | Pass | 0.8694 | 0.5000 |
| Interpolation Continuity | Pass | 0.0709 | 0.5000 |
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
- Unique token fraction: 69.25%
- Unique tokens decoded: 6925 / 10000
- Max token frequency: 0.16%
- Tokens seen once: 46.89%
- Prior entropy (mean/median/max): 6.09 / 6.22 / 7.42
- log(V): 9.68
- Gini coefficient: 0.6664

Example decoded tokens from prior samples:
```
  'ath'
  '▁firm'
  'l'
  '▁Special'
  '▁je'
```

Top decoded tokens:
```
  . (15914): 0.16% (16)
  ▁the (11): 0.06% (6)
  ▁out (246): 0.06% (6)
  ▁as (103): 0.06% (6)
  ular (616): 0.06% (6)
  ▁Court (4677): 0.06% (6)
  ▁chemical (9928): 0.06% (6)
  ▁asked (2386): 0.06% (6)
  ▁points (1728): 0.06% (6)
  ly (59): 0.05% (5)
  ▁bully (8431): 0.05% (5)
  ▁areas (1778): 0.05% (5)
  ▁kind (1409): 0.05% (5)
  ▁added (1812): 0.05% (5)
  ▁for (72): 0.05% (5)
  ▁Jehovah (15804): 0.05% (5)
  ▁chest (8432): 0.05% (5)
  2 (15926): 0.05% (5)
  ▁in (35): 0.05% (5)
  ▁search (2066): 0.05% (5)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.8694
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 86.94% overlap
- σ=0.1: 75.97% overlap
- σ=0.2: 57.18% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.24e+01, p90=2.47e+01, p99=2.86e+01, max=3.01e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.53e-01, p90=3.41e-01, p99=5.03e-01, max=5.78e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.01, H_max=0.78, top1_retention=100.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.23, H_max=5.38, top1_retention=100.00%
- σ=2.0: safe=100.00%, unique=100.00%, H_mean=1.32, H_max=4.21, top1_retention=48.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0709
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0709
- Max JS distance: 0.4114
- Mean JS divergence (derived): 0.0050
- Max JS divergence (derived): 0.1693
- Mean entropy: 0.60
- Max entropy: 8.34
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.00
- Midpoint entropy mean: 2.78
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8133
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0695
- Max JS: 0.4700
- Mean entropy: 0.52
- Max entropy: 7.00

- Entropy vs prior (mean Δ / max Δ): -5.49 / 0.92
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁ob (755) → unicipal (10839), steps 9→10, alpha 0.47→0.53, JSdist=0.4114, norm 7.66→7.82
  top-10 from:
    ▁ob (755): 0.702
    unicipal (10839): 0.290
    ▁Walt (10192): 0.000
    ▁difficulties (12186): 0.000
    ▁mysterious (10446): 0.000
    ▁packing (12795): 0.000
    ▁Arc (13188): 0.000
    liminary (15777): 0.000
    ▁FatW (15351): 0.000
    ▁Update (15679): 0.000
  top-10 to:
    unicipal (10839): 0.845
    ▁ob (755): 0.148
    ▁Walt (10192): 0.000
    ▁difficulties (12186): 0.000
    ▁mysterious (10446): 0.000
    liminary (15777): 0.000
    ▁Update (15679): 0.000
    ▁beneficiaries (14972): 0.000
    entially (4226): 0.000
    ▁FatW (15351): 0.000
```

Highest entropy steps:
```
  cop (5579) → itutional (4557), step 8, alpha=0.42, H=8.34, norm=5.03
  ▁1 (111) → ▁Hold (12437), step 10, alpha=0.53, H=8.29, norm=4.91
  ▁1 (111) → ▁Hold (12437), step 11, alpha=0.58, H=7.82, norm=4.99
  ▁fallen (11960) → iev (7532), step 8, alpha=0.42, H=7.79, norm=5.30
  ▁fallen (11960) → iev (7532), step 9, alpha=0.47, H=7.76, norm=5.32
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 11.611, medium<= 12.466
  near: mean_js_dist=0.070, max_js_dist=0.356, mean_H=0.62, max_H=8.29, flip=1.00, switch=0.053, jaccard=0.823, cliff=1.29, plateau=0.95, n=17
  medium: mean_js_dist=0.070, max_js_dist=0.360, mean_H=0.55, max_H=7.79, flip=1.00, switch=0.053, jaccard=0.806, cliff=1.62, plateau=0.95, n=16
  far: mean_js_dist=0.073, max_js_dist=0.411, mean_H=0.63, max_H=8.34, flip=1.00, switch=0.053, jaccard=0.810, cliff=1.76, plateau=0.95, n=17
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
- μ norm (mean ± std): 8.67 ± 0.97

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
- Max mean entropy: 6.14
- Mean mean entropy: 6.01
- Min unique fraction: 97.00% (threshold 10.00%)
- Mean step change rate: 21.10% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     6.08      98   16.14
    10      100.00%     6.11     100   16.16
    20      100.00%     6.05      98   16.11
    30      100.00%     5.92     100   16.16
    40      100.00%     5.88     100   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*