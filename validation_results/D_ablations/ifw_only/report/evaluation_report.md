# Token VAE Evaluation Report

Generated: 2026-02-06 14:26:47

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
| Perturbation Stability | Pass | 0.9219 | 0.5000 |
| Interpolation Continuity | Pass | 0.0823 | 0.5000 |
| Reconstruction Accuracy | Pass | 0.9988 | 0.9000 |
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
- Unique token fraction: 48.73%
- Unique tokens decoded: 4873 / 10000
- Max token frequency: 0.17%
- Tokens seen once: 28.25%
- Prior entropy (mean/median/max): 4.88 / 5.13 / 6.91
- log(V): 9.68
- Gini coefficient: 0.8127

Example decoded tokens from prior samples:
```
  '▁outside'
  '&'
  '▁award'
  '▁…'
  '▁subt'
```

Top decoded tokens:
```
  ▁her (387): 0.17% (17)
  ▁ (15893): 0.17% (17)
  ▁well (520): 0.16% (16)
  ▁its (410): 0.15% (15)
  R (15931): 0.15% (15)
  We (1507): 0.14% (14)
  ▁now (566): 0.14% (14)
  W (15933): 0.14% (14)
  ▁Google (1187): 0.13% (13)
  L (15941): 0.13% (13)
  ▁me (235): 0.13% (13)
  ). (568): 0.13% (13)
  ▁any (365): 0.13% (13)
  ▁We (345): 0.12% (12)
  ▁after (554): 0.12% (12)
  ▁said (383): 0.12% (12)
  ▁because (655): 0.12% (12)
  ▁right (632): 0.12% (12)
  ▁W (98): 0.12% (12)
  ers (116): 0.12% (12)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9219
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 92.19% overlap
- σ=0.1: 85.47% overlap
- σ=0.2: 72.31% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.50e+01, p90=2.67e+01, p99=2.83e+01, max=2.86e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.48e-01, p90=3.65e-01, p99=5.64e-01, max=6.32e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.16, H_max=6.89, top1_retention=99.00%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.85, H_max=5.36, top1_retention=95.50%
- σ=2.0: safe=100.00%, unique=98.00%, H_mean=1.36, H_max=3.42, top1_retention=15.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0823
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0823
- Max JS distance: 0.3607
- Mean JS divergence (derived): 0.0068
- Max JS divergence (derived): 0.1301
- Mean entropy: 1.19
- Max entropy: 6.93
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.02
- Midpoint entropy mean: 4.30
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8704
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0807
- Max JS: 0.4107
- Mean entropy: 1.15
- Max entropy: 6.98

- Entropy vs prior (mean Δ / max Δ): -3.69 / 0.02
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  ▁capt (1969) → ▁surg (15432), steps 9→10, alpha 0.47→0.53, JSdist=0.3607, norm 6.05→6.14
  top-10 from:
    ▁capt (1969): 0.694
    ▁surg (15432): 0.238
    ▁and (37): 0.001
    ▁or (124): 0.001
    ▁on (71): 0.001
    ing (26): 0.001
    ▁be (67): 0.001
    ▁( (126): 0.001
    s (15900): 0.001
    ▁of (36): 0.001
  top-10 to:
    ▁surg (15432): 0.720
    ▁capt (1969): 0.215
    ▁and (37): 0.001
    ▁or (124): 0.001
    ▁on (71): 0.001
    ing (26): 0.001
    ▁be (67): 0.001
    ▁of (36): 0.001
    ▁( (126): 0.001
    s (15900): 0.001
```

Highest entropy steps:
```
  ogy (11067) → ▁McM (14125), step 8, alpha=0.42, H=6.93, norm=4.91
  ▁Whether (6489) → itution (2149), step 10, alpha=0.53, H=6.92, norm=4.67
  ogy (11067) → ▁McM (14125), step 7, alpha=0.37, H=6.59, norm=4.90
  ▁Whether (6489) → itution (2149), step 11, alpha=0.58, H=6.58, norm=4.68
  ▁against (1205) → ▁continu (8813), step 9, alpha=0.47, H=6.56, norm=4.02
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 9.705, medium<= 10.750
  near: mean_js_dist=0.088, max_js_dist=0.307, mean_H=1.65, max_H=6.56, flip=1.00, switch=0.053, jaccard=0.877, cliff=0.12, plateau=0.95, n=17
  medium: mean_js_dist=0.080, max_js_dist=0.308, mean_H=1.02, max_H=6.92, flip=1.00, switch=0.053, jaccard=0.876, cliff=0.75, plateau=0.95, n=16
  far: mean_js_dist=0.078, max_js_dist=0.361, mean_H=0.88, max_H=6.93, flip=1.00, switch=0.053, jaccard=0.859, cliff=1.24, plateau=0.95, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 0.9988
- Threshold: 0.9000

**Details**:

- V=16,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 15,995 evaluated
- Top-1 accuracy: 99.88%
- Top-5 accuracy: 99.93%
- Correct: 15976 / 15995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 7.32 ± 1.33

Example failures:
```
  'ed' → ','
  '▁in' → '▁the'
  '▁be' → '▁the'
  '▁on' → '▁the'
  '▁with' → '▁the'
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
- Max mean entropy: 4.82
- Mean mean entropy: 4.71
- Min unique fraction: 95.00% (threshold 10.00%)
- Mean step change rate: 19.20% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.67      97   16.14
    10      100.00%     4.70      99   16.16
    20      100.00%     4.70      98   16.11
    30      100.00%     4.68      98   16.16
    40      100.00%     4.74      98   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*