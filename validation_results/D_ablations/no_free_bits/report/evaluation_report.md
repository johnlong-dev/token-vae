# Token VAE Evaluation Report

Generated: 2026-02-06 14:25:07

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
| Perturbation Stability | Pass | 0.9184 | 0.5000 |
| Interpolation Continuity | Pass | 0.0825 | 0.5000 |
| Reconstruction Accuracy | Pass | 0.9992 | 0.9000 |
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
- Unique token fraction: 48.96%
- Unique tokens decoded: 4896 / 10000
- Max token frequency: 0.18%
- Tokens seen once: 28.39%
- Prior entropy (mean/median/max): 4.87 / 5.12 / 6.98
- log(V): 9.68
- Gini coefficient: 0.8112

Example decoded tokens from prior samples:
```
  'list'
  '▁Ob'
  '▁trying'
  '▁love'
  '▁rocket'
```

Top decoded tokens:
```
  ▁said (383): 0.18% (18)
  ▁how (453): 0.16% (16)
  ▁no (446): 0.16% (16)
  L (15941): 0.16% (16)
  al (30): 0.15% (15)
  ▁such (627): 0.15% (15)
  ▁there (384): 0.14% (14)
  am (73): 0.14% (14)
  ▁This (390): 0.14% (14)
  ▁people (444): 0.14% (14)
  ▁M (87): 0.13% (13)
  ▁made (667): 0.12% (12)
  ▁go (299): 0.12% (12)
  ▁L (142): 0.12% (12)
  ▁when (386): 0.12% (12)
  ▁system (688): 0.12% (12)
  ▁may (413): 0.12% (12)
  on (10): 0.12% (12)
  ▁There (792): 0.12% (12)
  ▁A (61): 0.11% (11)
```

### Perturbation Stability

**Description**: Measures decoder output stability under Gaussian perturbation of token embeddings

**Result**: PASSED
- Score: 0.9184
- Threshold: 0.5000

**Details**:

Top-k overlap by perturbation σ:

- σ=0.05: 91.84% overlap
- σ=0.1: 84.98% overlap
- σ=0.2: 70.66% overlap

- Monotonic decrease: Yes

Local sensitivity (logit-space, σ=0.05):
- Logit sensitivity (||Δlogits||/||ε||): p50=2.48e+01, p90=2.63e+01, p99=2.70e+01, max=3.11e+01
- Margin sensitivity (|Δ(top1−top2)|/||ε||): p50=1.48e-01, p90=3.33e-01, p99=4.85e-01, max=5.94e-01

Off-manifold robustness sweep:

- σ=0.5: safe=100.00%, unique=100.00%, H_mean=0.11, H_max=6.92, top1_retention=99.50%
- σ=1.0: safe=100.00%, unique=100.00%, H_mean=0.82, H_max=5.88, top1_retention=92.00%
- σ=2.0: safe=100.00%, unique=99.00%, H_mean=1.31, H_max=3.61, top1_retention=19.50%

### Interpolation Continuity

**Description**: Assesses continuity along linear interpolation paths via Jensen-Shannon distance and entropy bounds

**Result**: PASSED
- Score: 0.0825
- Threshold: 0.5000

**Details**:

- Mean JS distance: 0.0793
- Max JS distance: 0.4120
- Mean JS divergence (derived): 0.0063
- Max JS divergence (derived): 0.1697
- Mean entropy: 1.04
- Max entropy: 6.84
- Holdout used for pass/fail: Yes
- Train-set threshold pass: Yes
- Holdout threshold pass: Yes
- Endpoint entropy mean: 0.02
- Midpoint entropy mean: 3.62
- Mean argmax switch rate: 0.0526
- Mean top-k Jaccard: 0.8651
- Mean plateau ratio: 0.9474
- Mean flip count: 1.00
- Pairs tested: 50

Held-out interpolation pairs:
- Mean JS: 0.0825
- Max JS: 0.3911
- Mean entropy: 1.24
- Max entropy: 7.47

- Entropy vs prior (mean Δ / max Δ): -3.83 / -0.14
- Entropy vs prior divergence flagged: Yes

Max JS divergence case:
```
  abled (11739) → ▁|0 (5152), steps 9→10, alpha 0.47→0.53, JSdist=0.4120, norm 7.09→7.20
  top-10 from:
    abled (11739): 0.740
    ▁|0 (5152): 0.219
    ▁all (216): 0.000
    ▁are (119): 0.000
    ▁that (86): 0.000
    ▁and (37): 0.000
    ▁from (172): 0.000
    ing (26): 0.000
    ▁an (104): 0.000
    ▁for (72): 0.000
  top-10 to:
    ▁|0 (5152): 0.774
    abled (11739): 0.189
    ▁all (216): 0.000
    ▁are (119): 0.000
    ▁and (37): 0.000
    ▁that (86): 0.000
    ▁from (172): 0.000
    ▁an (104): 0.000
    ▁for (72): 0.000
    ing (26): 0.000
```

Highest entropy steps:
```
  ▁governments (11009) → ▁Mir (10580), step 10, alpha=0.53, H=6.84, norm=4.82
  ▁governments (11009) → ▁Mir (10580), step 11, alpha=0.58, H=6.62, norm=4.87
  ▁Other (2541) → gers (3116), step 10, alpha=0.53, H=6.41, norm=4.18
  ▁sleep (3351) → ▁Everything (10977), step 9, alpha=0.47, H=6.39, norm=4.56
  ▁Col (1193) → res (157), step 10, alpha=0.53, H=6.37, norm=3.84
```

Distance bucket metrics (train pairs):
```
  thresholds: near<= 9.551, medium<= 10.968
  near: mean_js_dist=0.087, max_js_dist=0.288, mean_H=1.60, max_H=6.41, flip=1.00, switch=0.053, jaccard=0.877, cliff=0.12, plateau=0.95, n=17
  medium: mean_js_dist=0.077, max_js_dist=0.330, mean_H=0.89, max_H=6.84, flip=1.00, switch=0.053, jaccard=0.872, cliff=0.81, plateau=0.95, n=16
  far: mean_js_dist=0.073, max_js_dist=0.412, mean_H=0.63, max_H=6.08, flip=1.00, switch=0.053, jaccard=0.846, cliff=1.53, plateau=0.95, n=17
```

### Reconstruction Accuracy

**Description**: Verifies that tokens are recoverable when encoded to their posterior mean and decoded

**Result**: PASSED
- Score: 0.9992
- Threshold: 0.9000

**Details**:

- V=16,000 total tokens; 5 specials (pad/unk/bos/eos/endoftext) excluded → 15,995 evaluated
- Top-1 accuracy: 99.92%
- Top-5 accuracy: 99.96%
- Correct: 15982 / 15995

Posterior collapse diagnostic:
- Collapsed dims (KL < 0.01): 0 (0.0%)
- μ norm (mean ± std): 7.32 ± 1.33

Example failures:
```
  'ed' → '▁of'
  '▁with' → '▁of'
  '▁an' → '▁and'
  '▁was' → '▁a'
  '▁this' → '▁of'
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
- Max mean entropy: 4.85
- Mean mean entropy: 4.77
- Min unique fraction: 96.00% (threshold 10.00%)
- Mean step change rate: 19.67% (threshold 1.00%)
- Walks: 100, Steps: 50
- Beta: 0.01, Schedule: constant
- Start from: prior

Per-step trajectory (every 10th step):
```
  step  non_special  entropy  unique    norm
     0      100.00%     4.84      98   16.14
    10      100.00%     4.82      96   16.16
    20      100.00%     4.78      98   16.11
    30      100.00%     4.76      99   16.16
    40      100.00%     4.79      97   16.17
```

---

## Conclusion

**All evaluation criteria are satisfied.** The latent space meets the tested
prerequisites for continuous diffusion: prior coverage, local smoothness,
interpolation continuity, and reconstruction fidelity.

---

*Generated by the Token VAE evaluation pipeline.*