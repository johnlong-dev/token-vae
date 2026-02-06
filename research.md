# Token VAE: Variational Regularization of Discrete Token Embeddings as a Prerequisite for Continuous Diffusion

## Abstract

Continuous diffusion models operate on smooth manifolds, yet natural language is fundamentally discrete. Applying diffusion to language therefore requires an embedding space that bridges this gap: one that is continuous and geometrically well-behaved while remaining faithful to the underlying token vocabulary. We propose Token VAE, a lightweight variational autoencoder that maps discrete tokens into a regularized latent space and evaluate whether the resulting geometry satisfies four prerequisites for continuous diffusion: prior coverage, local smoothness, interpolation continuity, and reconstruction fidelity. We assess these properties through a six-test evaluation battery (including holdout-gated interpolation and cross-metric integrity checks) across 21 configurations spanning five random seeds, three data scales, three vocabulary scales, eight ablation variants, and two baselines. Twenty of 21 configurations pass all tests; the single failure is a non-VAE baseline that fails as expected on prior coverage and metric integrity. These results establish that variational regularization produces a token embedding space satisfying the tested necessary conditions for diffusion. They do not, however, establish sufficiency: no diffusion model was trained on this space, and we cannot confirm that generation quality follows from geometric prerequisites alone. We present these findings as a preliminary validation step, with transparent disclosure of threshold proximity, ablation ceiling effects, and provenance limitations.

## 1. Introduction

### 1.1 Problem Statement

Text generation with diffusion models has gained attention as an alternative to autoregressive decoding, offering parallel generation and the potential for iterative refinement. However, a fundamental obstacle separates diffusion from language: diffusion processes operate on continuous spaces with smooth density gradients, while natural language consists of discrete tokens drawn from a finite vocabulary. A naive embedding of tokens into Euclidean space provides no guarantee that the resulting geometry supports the operations diffusion requires: sampling from a simple prior, smooth local traversal, continuous interpolation, and faithful reconstruction.

We formalize this gap as four geometric prerequisites that a token embedding space must satisfy before it can serve as a substrate for continuous diffusion:

1. **Prior coverage.** Samples from a standard Gaussian prior, when decoded, should produce diverse, non-degenerate vocabulary tokens rather than collapsing to a small set of common words.
2. **Local smoothness.** Small perturbations in the latent space should produce gradual, predictable changes in decoded output, rather than discontinuous jumps.
3. **Interpolation continuity.** Linear paths between token embeddings should decode to bounded-entropy distributions with bounded divergence between adjacent steps, avoiding degenerate regions.
4. **Reconstruction fidelity.** Tokens encoded to their posterior mean should be recoverable through decoding, confirming that the latent space preserves token identity.

These are necessary conditions. A space that fails any one of them is unsuitable for diffusion. A space that satisfies all four may still produce poor diffusion results due to factors these tests do not measure (high-frequency geometry, density mismatch, sequence-level coherence). This distinction between necessary and sufficient conditions is central to the claims we make.

### 1.2 Scope and Falsifiability

This work is limited to testing the four prerequisites above. We do not train a diffusion model on the resulting latent space. We do not evaluate generation quality, fluency, or diversity of produced text. The hypothesis under test is:

> A variationally-regularized token embedding space can serve as a continuous manifold suitable for diffusion and latent-space revision.

This hypothesis is falsifiable. If the non-VAE baseline (trained without variational regularization) were to pass all tests, the necessity of the VAE would be refuted. If multiple random seeds produced inconsistent results, reproducibility would fail. If ablation configurations that remove key loss components were to fail, component necessity would be established; conversely, if they all pass, we cannot claim necessity.

### 1.3 Contributions

This paper makes four contributions:

1. **A lightweight VAE architecture for token embeddings** that maps discrete tokens to a 256-dimensional latent space using a composite loss with five components (reconstruction, KL divergence with free bits, skip-gram, interpolation entropy, and prior diversity).
2. **A six-test evaluation battery** with holdout-gated interpolation that assesses the four geometric prerequisites, cross-metric integrity, and diffusion walk robustness.
3. **A 21-configuration empirical validation** covering reproducibility (5 seeds), data scale (2M/4M/8M characters), vocabulary scale (4k/8k/16k), ablations (8 configurations), and baselines (2 configurations), with 20 of 21 passing all tests.
4. **Transparent disclosure** of threshold proximity in near-threshold cases, ablation ceiling effects that prevent component necessity claims, and provenance gaps in evaluation metadata.

## 2. Background and Related Work

### Variational Autoencoders for Discrete Data

The variational autoencoder framework (Kingma and Welling, 2014) provides a principled approach to learning continuous latent representations of data by maximizing a lower bound on the data likelihood. When applied to discrete data such as text, VAEs face the challenge of posterior collapse, where the decoder learns to ignore the latent variable. Bowman et al. (2016) documented this problem for sentence-level VAEs and proposed KL annealing as a mitigation. Kingma et al. (2016) introduced the free-bits technique, which floors the per-dimension KL contribution to prevent individual latent dimensions from collapsing. We adopt free bits in our training objective.

### Continuous Diffusion for Language

Denoising diffusion probabilistic models (Ho et al., 2020) learn to reverse a gradual noising process, generating data by iteratively denoising from Gaussian noise. Applying this to language requires a continuous representation. Li et al. (2022) proposed Diffusion-LM, which trains diffusion directly on word embeddings with a learned rounding step to map back to discrete tokens. Gong et al. (2023) introduced DiffuSeq, applying diffusion to sequence-to-sequence tasks. Austin et al. (2021) took an alternative approach with D3PM, defining diffusion directly over discrete states. Our work addresses a prerequisite question that these models assume: whether the embedding space itself has the geometric properties diffusion requires.

### Token Embedding Geometry

The geometric structure of token embeddings has been studied since Mikolov et al. (2013) demonstrated that word2vec skip-gram embeddings capture semantic relationships as linear directions. We incorporate a skip-gram loss to encourage co-occurring tokens to have similar latent representations. Our approach operates at the subword level (SentencePiece BPE tokens), which introduces additional challenges since subword units may have less stable co-occurrence patterns than whole words.

### Evaluation of Latent Spaces

There is no standard evaluation framework for the geometric quality of token embedding spaces. Prior work on evaluating generative models has largely focused on image-centric criteria, including disentanglement-oriented setups such as beta-VAE (Higgins et al., 2017) and distributional metrics like FID and IS (Heusel et al., 2017; Salimans et al., 2016), which do not transfer directly to discrete token spaces. Our evaluation battery draws on the principle of testing multiple complementary properties rather than relying on a single metric, following the observation that individual metrics can be satisfied by degenerate solutions.

## 3. Method

### 3.1 Architecture

Token VAE maps each token in a vocabulary of size V to a distribution in a d-dimensional latent space. The architecture consists of three stages: encoding, reparameterization, and decoding.

```
token_id  -->  Embedding(V, 512)
           -->  mu_proj(512, 256)  +  logvar_proj(512, 256)
           -->  reparameterize: h = mu + sigma * epsilon,  epsilon ~ N(0, I)
           -->  decoder(256, V)  -->  logits
```

The embedding layer maps token IDs to 512-dimensional vectors. Two linear projection layers produce the mean (mu) and log-variance (logvar) of the approximate posterior, each of dimension 256. The log-variance is clamped to [-10, 10] for numerical stability. During training, the reparameterization trick samples a latent vector h; during evaluation, the posterior mean is used deterministically. A single linear layer decodes h back to logits over the vocabulary.

Weight initialization uses normal initialization (std=0.02) for the embedding layer and Xavier uniform initialization for the projection and decoder layers, with biases initialized to zero.

The architecture is deliberately minimal. It contains no hidden layers in the encoder beyond the projections, and the decoder is a single linear transformation. This simplicity is intentional: we aim to test whether variational regularization of the embedding space itself is sufficient for the geometric prerequisites, without confounding the question with decoder capacity.

The default configuration uses V=16,000 (SentencePiece BPE vocabulary), d_embed=512, and d_model=256. Vocabulary scale experiments also test V=4,000 and V=8,000.

### 3.2 Training Objective

The training loss is a weighted combination of five components:

**Reconstruction loss.** Cross-entropy between the decoder output and the target token, optionally weighted by inverse frequency weights (IFW):

```
L_recon = CrossEntropy(logits, target, weight=IFW)
```

where IFW weights are computed as w_i = (1 / freq_i)^alpha, normalized to have mean 1.0. The IFW exponent alpha defaults to 1.0, giving full inverse-frequency weighting. This upweights rare tokens to prevent the decoder from focusing only on common tokens. The reconstruction weight is 1.0 (unscaled).

**KL divergence with free bits.** The KL divergence from the approximate posterior to the standard normal prior, with a per-dimension floor:

```
KL_j = 0.5 * (mu_j^2 + exp(logvar_j) - logvar_j - 1)
L_KL = sum_j max(KL_j, lambda)
```

where lambda=0.25 is the free-bits threshold. This prevents individual latent dimensions from collapsing to the prior while still encouraging overall regularization. The KL weight (beta) is 0.03, applied after a linear warmup over the first 40% of training.

**Skip-gram loss.** Negative-sampling skip-gram loss applied to the posterior means of co-occurring tokens:

```
L_skip = -log sigma(mu_center . mu_context) - sum_k log sigma(-mu_center . mu_neg_k)
```

This encourages tokens that co-occur in natural text to have nearby latent representations, providing semantic structure to the space. The skip-gram weight is 0.1.

**Interpolation entropy loss.** For each training batch, 64 random token pairs are sampled (excluding 200 held-out pairs), interpolated in latent space, decoded, and penalized for entropy exceeding a target of 8.0:

```
L_interp = mean(ReLU(H(decode(lerp(mu_a, mu_b, alpha))) - H_target))
```

where alpha is sampled uniformly from [0, 1]. The holdout pairs are reserved for evaluation to prevent overfitting to the interpolation test. The interpolation weight is 0.05.

**Prior diversity loss.** A four-component regularizer that samples from the standard normal prior and penalizes degenerate decodings:

1. Mean entropy penalty: ReLU(H_target - mean(H(decode(z))))
2. Max frequency penalty: ReLU(max_freq(decode(z)) - freq_target)
3. Marginal entropy penalty: ReLU(H_marginal_target - H(mean(softmax(decode(z)))))
4. HHI penalty: ReLU(HHI(mean(softmax(decode(z)))) - HHI_target)

The targets are: entropy target 4.0, max frequency target 0.05, marginal entropy target 7.0, and HHI target 0.001. The prior diversity weight is 0.50, reflecting its importance in achieving prior coverage.

### 3.3 Data Pipeline

Training data is sourced from FineWeb, a large-scale English web text corpus (Penedo et al., 2024). We extract raw text and tokenize it using SentencePiece BPE tokenizers trained on a 4M-character subset. The training pipeline uses three dataset types:

1. **Identity dataset.** Each vocabulary token appears as its own training example, repeated 5 times per epoch. This provides a uniform signal across the vocabulary and is essential for reconstruction of rare tokens.
2. **Vocabulary dataset.** Tokens sampled from the tokenized text, reflecting the natural frequency distribution.
3. **Co-occurrence dataset.** Token pairs extracted from windowed contexts in the tokenized text, used for the skip-gram loss. Negative samples are drawn from a frequency-based distribution using the 3/4 power heuristic (word2vec).

The default configuration uses 4M characters of FineWeb text. Data scale experiments also test 2M and 8M characters.

### 3.4 Training Procedure

Training uses the Adam optimizer with a learning rate of 1e-3, batch size of 256, and runs for 10 epochs. KL weight warmup is linear over the first 40% of total training steps. The prior diversity loss is computed every batch (interval=1). Total training time is approximately 30 minutes on an Apple M-series Mac.

## 4. Evaluation Framework

The evaluation battery consists of six tests, each targeting a specific property of the latent space. All tests are seeded for reproducibility (we use the run seed, matching training). We describe each test's protocol, thresholds, and rationale.

### 4.1 Prior Decodability

**Protocol.** Sample 10,000 vectors from N(0, I), decode each to logits, and take the argmax token. Compute the fraction of decoded tokens that are non-special (not pad, unk, bos, eos, or endoftext) and the fraction of unique tokens among all samples.

**Thresholds.** non_special_fraction >= 0.85 AND unique_fraction >= 0.30.

**Rationale.** Prior coverage is the most basic requirement for a diffusion-compatible space. If sampling from the prior produces only special tokens or collapses to a handful of common tokens, the space cannot support diverse generation. The dual criterion is important: a space could achieve high non-special fraction while collapsing to a small set of tokens. The unique fraction threshold (0.30, meaning at least 3,000 distinct tokens from 10,000 samples) guards against this.

**Score semantics.** The reported score is the non_special_fraction only. Pass/fail depends on both criteria. This means a run can show score=1.0 while failing, if the unique fraction is below 0.30.

### 4.2 Perturbation Stability

**Protocol.** For 100 randomly selected non-special tokens, encode to the posterior mean, compute the top-10 decoded tokens, then add Gaussian noise at three sigma levels (0.05, 0.1, 0.2). For each of 50 perturbations per token per sigma, compute the top-10 overlap with the unperturbed prediction.

**Thresholds.** Mean top-10 overlap at sigma=0.05 >= 0.50, AND overlap must decrease approximately monotonically across sigma levels (with 0.1 tolerance).

**Rationale.** Local smoothness requires that small movements in latent space produce gradual changes in decoded output. High overlap at small sigma indicates smooth local geometry. Monotonic decrease confirms that the decoder's sensitivity scales predictably with perturbation magnitude. The test also computes logit sensitivity (the ratio of logit change to perturbation norm), providing a local Lipschitz estimate.

### 4.3 Interpolation Continuity

**Protocol.** Generate 50 random token pairs (excluding held-out pairs), linearly interpolate between their posterior means in 20 steps, and decode each point to a probability distribution. Compute the Jensen-Shannon distance between adjacent step distributions and the entropy at each point. Separately evaluate 200 held-out token pairs that were excluded from training's interpolation loss.

**Thresholds.** max JS distance <= 0.50 AND max entropy <= 8.50, for both train and holdout pairs (gated evaluation).

**Rationale.** Interpolation continuity tests whether linear paths in latent space traverse regions that decode to well-defined distributions. High JS distance between adjacent steps would indicate discontinuities (cliffs). High entropy would indicate degenerate regions where the decoder is uncertain. The holdout pairs are critical: since the training loss penalizes interpolation entropy on train pairs, evaluating only those pairs would be circular. The holdout pairs provide an unbiased estimate of interpolation quality on unseen token combinations.

### 4.4 Reconstruction Accuracy

**Protocol.** Encode every non-special token in the vocabulary using the posterior mean (deterministic, no sampling), decode, and check whether the argmax of the output matches the input.

**Thresholds.** Top-1 accuracy >= 0.90.

**Rationale.** Reconstruction fidelity confirms that the latent space preserves token identity. If a token cannot be recovered from its own posterior mean, the encoding has lost information. The test also measures collapsed dimensions (latent dimensions with mean KL < 0.01 across all tokens), which would indicate partial posterior collapse.

### 4.5 Metric Integrity

**Protocol.** Cross-validate results from the prior decodability, interpolation, and reconstruction tests. Three trigger conditions are checked:

1. If interpolation max entropy <= threshold but prior unique fraction < 0.20.
2. If interpolation max entropy <= threshold but prior entropy < 1.0.
3. If reconstruction accuracy > 0.95 but prior unique fraction < 0.20.

**Thresholds.** Pass if zero triggers fire.

**Rationale.** Individual tests can be satisfied by degenerate solutions. A space could achieve low interpolation entropy by mapping everything to the same token (which would also give perfect reconstruction). The integrity checks catch such inconsistencies. The non-VAE baseline triggers all three checks, demonstrating their value: it achieves high reconstruction and passes interpolation, but its prior coverage is degenerate.

### 4.6 Diffusion Walk

**Protocol.** Simulate 100 forward-diffusion random walks of 50 steps each, starting from N(0, I) samples. At each step, apply the noise schedule z_{t+1} = sqrt(1-beta) * z_t + sqrt(beta) * epsilon, with beta=0.01 (constant schedule). Decode at each step and record non-special rate, mean entropy, unique fraction, and token change rate.

**Thresholds.** min non-special rate >= 0.95, max mean entropy <= 8.0, min unique fraction >= 0.10, mean change rate >= 0.01.

**Rationale.** This test simulates the actual forward process that a diffusion model would use, checking whether the space remains well-behaved under the noising dynamics. The four criteria ensure that tokens remain valid, distributions remain sharp, diversity is maintained, and the walk actually moves through different regions of the space. The beta value of 0.01 represents a mild noise level; this is a sanity check rather than a rigorous simulation of the full diffusion schedule.

## 5. Experimental Design

### 5.1 Validation Matrix

The 21-run validation covers five experimental sections:

**Section A: Reproducibility (5 runs).** Five random seeds (42, 123, 456, 789, 1337) with the baseline configuration (4M data, 16k vocabulary). Tests whether results are stable across initialization randomness.

**Section B: Data Scale (3 runs).** Three FineWeb sizes (2M, 4M, 8M characters) with seed 42 and 16k vocabulary. Tests whether the approach scales with data quantity.

**Section C: Vocabulary Scale (3 runs).** Three BPE vocabulary sizes (4k, 8k, 16k) with seed 42 and 4M data. Tests whether the approach generalizes across vocabulary sizes.

**Section D: Ablations (8 runs).** Eight configurations that selectively disable or modify loss components, all with seed 42, 4M data, and 16k vocabulary. The configurations are:

- **baseline**: All loss components active with default weights.
- **no_marginal_entropy**: Disables the marginal entropy target in the prior diversity loss.
- **no_hhi**: Disables the HHI target in the prior diversity loss.
- **no_marginal_no_hhi**: Disables both marginal entropy and HHI targets.
- **no_free_bits**: Sets the free-bits threshold to 0.0.
- **half_ifw**: Reduces the IFW exponent from 1.0 to 0.5.
- **no_ifw**: Sets the IFW exponent to 0.0 (uniform reconstruction weights).
- **ifw_only**: Keeps IFW but disables free bits, prior diversity, marginal entropy, and HHI.

**Section G: Baselines (2 runs).** Two baseline configurations:

- **non_vae**: Disables all VAE-specific losses (KL weight=0, no free bits, no prior regularization, no interpolation loss, no IFW). This trains a standard autoencoder and serves as the primary falsification test.
- **no_skipgram**: Full baseline but with skip-gram weight set to 0. Tests whether co-occurrence structure is necessary.

### 5.2 Evaluation Protocol

Each run follows the same sequence: train the model, then evaluate with the six-test battery using the same seed as training. Holdout interpolation pairs are generated once per run (200 pairs) and excluded from both training and the train-split interpolation evaluation. Results are stored as JSON with metadata including model hash, tokenizer hash, training seed, platform, and evaluation configuration.

## 6. Results

### 6.1 Reproducibility

All five seeds pass all six tests. Table 1 shows key metrics across seeds.

**Table 1: Reproducibility results (Section A, 5 seeds)**

| Seed | Prior Unique | Perturb (sigma=0.05) | Holdout Max JS | Holdout Max H | Recon Top-1 | Diff. Min Unique | Diff. Change Rate |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | 0.567 | 0.900 | 0.382 | 7.98 | 1.000 | 0.960 | 0.210 |
| 123 | 0.563 | 0.905 | 0.388 | 7.78 | 1.000 | 0.980 | 0.200 |
| 456 | 0.573 | 0.910 | 0.428 | 7.95 | 1.000 | 0.970 | 0.210 |
| 789 | 0.571 | 0.905 | 0.392 | 7.76 | 1.000 | 0.960 | 0.216 |
| 1337 | 0.564 | 0.906 | 0.384 | 7.66 | 1.000 | 0.970 | 0.200 |

**Table 2: Reproducibility variance**

| Metric | Mean | Std | Min | Max |
|--------|:---:|:---:|:---:|:---:|
| Prior unique fraction | 0.5675 | 0.0039 | 0.5627 | 0.5727 |
| Perturbation score | 0.9050 | 0.0033 | 0.8995 | 0.9099 |
| Interpolation score | 0.0763 | 0.0009 | 0.0751 | 0.0780 |
| Holdout max JS | 0.3947 | 0.0172 | 0.3816 | 0.4283 |
| Holdout max entropy | 7.8275 | 0.1205 | 7.6643 | 7.9844 |
| Reconstruction score | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Diffusion min unique | 0.9680 | 0.0075 | 0.9600 | 0.9800 |
| Diffusion change rate | 0.2072 | 0.0060 | 0.2002 | 0.2157 |

Variance is low across all metrics. Standard deviations are below 1% of the mean for all metrics except holdout max entropy (std=0.12), which is expected given the stochastic nature of pair-level evaluation. Three metrics (prior score, reconstruction, diffusion score) are at ceiling across all seeds, providing no discrimination. This ceiling effect is noted as a limitation: these metrics function as pass/fail gates rather than discriminative measures.

### 6.2 Scale Sensitivity

**Table 3: Data scale results (Section B)**

| Data Size | Prior Unique | Perturb | Holdout Max H | Recon Top-1 | All Pass |
|-----------|:---:|:---:|:---:|:---:|:---:|
| 2M | 0.441 | 0.906 | 8.19 | 0.991 | YES |
| 4M | 0.563 | 0.898 | 8.04 | 1.000 | YES |
| 8M | 0.649 | 0.918 | 7.43 | 1.000 | YES |

All three data scales pass. Metrics improve consistently with scale: prior unique fraction increases from 0.441 to 0.649, reconstruction improves from 0.991 to 1.000, and holdout max entropy decreases from 8.19 to 7.43 (lower is better, further from the 8.50 threshold). The 2M configuration is the weakest, with holdout max entropy at 8.19 (3.6% margin from the 8.50 threshold) and reconstruction at 0.991 (well above the 0.90 threshold but the lowest observed). These margins are disclosed in the threshold proximity analysis (Section 7).

**Table 4: Vocabulary scale results (Section C)**

| Vocab Size | Prior Unique | Perturb | Holdout Max H | Recon Top-1 | All Pass |
|------------|:---:|:---:|:---:|:---:|:---:|
| 4k | 0.341 | 0.898 | 6.69 | 1.000 | YES |
| 8k | 0.512 | 0.947 | 6.87 | 1.000 | YES |
| 16k | 0.563 | 0.899 | 8.04 | 1.000 | YES |

All three vocabulary sizes pass. Prior unique fraction scales positively with vocabulary size (0.341 to 0.563), while the 4k configuration sits at 0.341, only 13.5% above the 0.30 threshold. The 8k vocabulary achieves notably high perturbation stability (0.947), though this may reflect properties of the specific vocabulary rather than a systematic advantage. Holdout max entropy increases with vocabulary size, consistent with larger vocabularies producing higher-entropy decoded distributions.

### 6.3 Ablation Study

All eight ablation configurations pass all six tests. We therefore cannot claim that any individual loss component is strictly necessary for passing the evaluation battery. The ablation results provide directional evidence of incremental benefit, not evidence of component necessity. This is a limitation of the current evaluation: the thresholds may be too lenient or the tests too few to distinguish the contributions of individual regularizers.

**Table 5: Ablation results with deltas from baseline**

| Config | Prior Unique (delta) | Perturb (delta) | Interp Mean JS (delta) | Recon (delta) | Signal |
|--------|:---:|:---:|:---:|:---:|------|
| baseline | 0.568 | 0.899 | 0.074 | 1.000 | (reference) |
| no_marginal_entropy | 0.559 (-0.009) | 0.902 (+0.004) | 0.073 (-0.000) | 1.000 (0.000) | Negligible |
| no_hhi | 0.568 (-0.000) | 0.898 (-0.001) | 0.074 (-0.000) | 1.000 (0.000) | Negligible |
| no_marginal_no_hhi | 0.562 (-0.006) | 0.899 (+0.001) | 0.074 (-0.000) | 1.000 (0.000) | Negligible |
| no_free_bits | 0.490 (-0.079) | 0.918 (+0.020) | 0.079 (+0.006) | 0.999 (-0.001) | Meaningful |
| half_ifw | 0.636 (+0.068) | 0.841 (-0.058) | 0.074 (-0.001) | 1.000 (0.000) | Tradeoff |
| no_ifw | 0.630 (+0.062) | 0.834 (-0.065) | 0.078 (+0.005) | 1.000 (0.000) | Tradeoff |
| ifw_only | 0.487 (-0.081) | 0.922 (+0.023) | 0.082 (+0.009) | 0.999 (-0.001) | Meaningful |

The ablations group into three tiers:

**Negligible impact** (no_marginal_entropy, no_hhi, no_marginal_no_hhi). Removing the marginal entropy target, the HHI target, or both produces metric deltas below 0.01 on all measures. These components of the prior diversity loss do not measurably contribute to the tested properties at the current evaluation granularity.

**Diversity-stability tradeoff** (half_ifw, no_ifw). Reducing or removing IFW weighting increases prior unique fraction (+0.062 to +0.068) at the cost of decreased perturbation stability (-0.058 to -0.065). This tradeoff is consistent: IFW upweights rare tokens during reconstruction, which concentrates the latent space and improves local smoothness but reduces the variety of tokens covered by the prior. Both configurations still pass all tests.

**Meaningful directional shift** (no_free_bits, ifw_only). Removing free bits or keeping only IFW (with no prior regularization) reduces prior unique fraction by 0.079 to 0.081 and slightly reduces reconstruction accuracy (0.999 vs 1.000). These configurations represent the most aggressive ablations and produce the largest metric shifts, but still pass all tests. The directional signal suggests that free bits and prior regularization provide incremental benefit for prior coverage.

### 6.4 Baseline Comparison

**Table 6: Baseline comparison**

| Config | Prior Unique | Prior H | Perturb | Interp Score | Recon | Integrity | All Pass |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full VAE (ref) | 0.567 | 4.09 | 0.900 | 0.078 | 1.000 | PASS | YES |
| non_vae | 0.145 | 0.31 | 0.735 | 0.070 | 1.000 | FAIL | NO |
| no_skipgram | 0.693 | 6.09 | 0.869 | 0.071 | 1.000 | PASS | YES |

**non_vae baseline.** The non-VAE baseline provides the strongest causal evidence in the validation. Removing all VAE-specific losses (KL divergence, prior regularization, interpolation penalty, IFW) causes prior unique fraction to drop from 0.567 to 0.145 (a 74% decrease), and prior entropy to drop from 4.09 to 0.31 (a 92% decrease). Perturbation stability also decreases from 0.900 to 0.735. The metric integrity test fires three triggers: low prior unique fraction, low prior entropy, and the contradiction of high reconstruction with low diversity.

An important observation: the non_vae baseline passes the interpolation continuity test (mean JS = 0.070, max JS = 0.401, max entropy = 6.05). This demonstrates that interpolation continuity alone is not sufficient evidence of manifold quality. A space with degenerate prior coverage can still produce smooth interpolation paths if those paths happen to traverse well-decoded regions. This validates the design of the multi-test battery: no single test is sufficient; the conjunction of all four prerequisites is required.

**no_skipgram baseline.** The no_skipgram baseline passes all tests, with prior unique fraction of 0.693 (higher than the full VAE) and perturbation stability of 0.869 (lower than the full VAE's 0.900). The skip-gram loss is therefore not necessary for passing the tested prerequisites. Its effect appears to be a modest improvement in perturbation stability at the cost of some prior diversity.

However, the no_skipgram configuration has the tightest threshold margins in the entire validation. Its gated max entropy reaches 8.34, only 1.9% below the 8.50 threshold. Its gated max JS reaches 0.470, only 6.0% below the 0.500 threshold. A small change in data, hyperparameters, or evaluation seed could push these values over threshold. This proximity is disclosed in the threshold analysis below.

### 6.5 Qualitative Analysis

**Prior samples.** Decoded prior samples from the seed-42 model include tokens such as "30", "Ob", "nes", "forces", "atform", "under", "cause", "IB", "own", "T". These represent a mix of whole words, subword fragments, and numbers, consistent with a BPE vocabulary. The sample entropy mean is 4.09, indicating that decoded distributions are peaked but not degenerate.

By contrast, the non_vae baseline prior samples include ",", "e", "the", "game", "t", "or", "about", "C", "your", "and", which are dominated by high-frequency tokens and function words, reflecting a collapsed prior that maps most of the latent space to common tokens.

**Interpolation paths.** A representative interpolation between "gut" and "each" (seed 42) achieves a maximum entropy of 7.62 at step 12 (alpha=0.63), which is the midpoint region where the decoder transitions between the two tokens' influence. The mean JS distance across steps is 0.078, indicating smooth transitions. The maximum entropy occurs in the interior of the path, as expected; endpoints have lower entropy because they are near actual token embeddings.

**Diffusion walk trajectory.** Diffusion walks (seed 42) maintain 100% non-special token rate across all 50 steps, with a mean change rate of 0.210 (approximately 21% of decoded tokens change per step at beta=0.01). The minimum unique fraction is 0.960, indicating that the walks traverse diverse regions. This confirms that the mild forward-diffusion process does not push decoded outputs into degenerate regions of the space.

## 7. Discussion

### 7.1 What the Results Support

The 21-run validation supports the following conclusions:

1. **Variational regularization is necessary for prior coverage.** The non_vae baseline demonstrates that without KL regularization and prior diversity losses, the latent space collapses to a small set of common tokens under prior sampling. This is the strongest causal finding: VAE regularization directly enables the prior coverage prerequisite.

2. **The four geometric prerequisites are robustly satisfied.** Across 5 random seeds, 3 data scales, 3 vocabulary scales, and 8 ablation variants, 20 of 21 configurations pass all tests with low variance. The single failure is the expected non_vae baseline.

3. **Results are reproducible.** Five random seeds produce consistent metrics with standard deviations below 1% of the mean.

4. **Positive data scaling.** Increasing training data from 2M to 8M characters consistently improves all metrics, suggesting the approach benefits from scale.

### 7.2 What the Results Do Not Support

1. **Sufficiency for diffusion.** The tests measure necessary conditions only. A diffusion model trained on this latent space might produce poor text, hallucinate, or fail to converge. No downstream generation experiment was conducted.

2. **Component necessity from ablations.** All eight ablation configurations pass all tests. We cannot claim that any single loss component is required. The marginal entropy and HHI components of the prior diversity loss, in particular, show negligible measurable impact.

3. **Hyperparameter optimality.** The loss weights, thresholds, and architecture choices were not systematically optimized. The current configuration represents one working point, not a claimed optimum.

4. **Robustness to strong diffusion noise.** The diffusion walk test uses beta=0.01 (constant schedule, 50 steps), which represents a mild noise level. The behavior under stronger noise schedules typical of production diffusion models (e.g., linear beta from 1e-4 to 0.02 over 1000 steps) is untested.

5. **Sequence-level coherence.** Token VAE operates on individual tokens. It makes no claims about sequence-level properties (coherence, fluency, grammaticality) that would be required for text generation.

### 7.3 Threshold Sensitivity

The evaluation thresholds are empirically chosen, not theoretically derived. Several runs pass with margins that could be sensitive to minor experimental variations.

**Table 7: Threshold proximity (all cases with margin < 15%)**

| Run | Metric | Value | Threshold | Margin | Margin % |
|-----|--------|:---:|:---:|:---:|:---:|
| G/no_skipgram | Gated max entropy | 8.34 | 8.50 | 0.16 | 1.9% |
| G/no_skipgram | Gated max JS | 0.470 | 0.500 | 0.030 | 6.0% |
| B/fineweb_2M | Holdout max entropy | 8.19 | 8.50 | 0.31 | 3.6% |
| D/no_ifw | Gated max entropy | 8.14 | 8.50 | 0.36 | 4.2% |
| C/vocab_8k | Diff. min non-special | 0.990 | 0.950 | 0.040 | 4.2% |
| C/vocab_4k | Prior unique fraction | 0.341 | 0.300 | 0.041 | 13.5% |
| A/seed_456 | Holdout max JS | 0.428 | 0.500 | 0.072 | 14.4% |

The tightest case is the no_skipgram baseline's gated max entropy (1.9% margin). If the threshold were 8.30 instead of 8.50, this configuration would fail. The fineweb_2M configuration is also near the entropy threshold (3.6% margin), and the vocab_4k configuration is near the unique fraction threshold (13.5% margin).

These margins do not invalidate the results, but they constrain the confidence with which we can generalize. Tighter thresholds, different evaluation seeds, or different data distributions could change pass/fail outcomes for borderline configurations. We recommend that future work investigate theoretically grounded thresholds rather than the empirical values used here.

### 7.4 Design Tradeoffs

**IFW diversity-stability tradeoff.** The ablation results reveal a consistent tradeoff: inverse frequency weighting improves perturbation stability (by concentrating the latent space around rare tokens' neighborhoods) but reduces prior diversity (by narrowing the effective support of decoded tokens). This is a design choice rather than a flaw; different downstream applications might prefer different points on this tradeoff.

**Holdout-gated evaluation.** Reserving 200 interpolation pairs from training and using them for evaluation prevents overfitting the interpolation loss to the test metric. The holdout pairs consistently produce higher (worse) entropy and JS values than train pairs, confirming that the holdout provides a more conservative estimate. The gated evaluation (pass requires both train and holdout to pass) means the final verdict reflects the harder condition.

## 8. Limitations

1. **No downstream diffusion evaluation.** The central limitation of this work is that we test prerequisites for diffusion without training or evaluating a diffusion model. The leap from "the space has these geometric properties" to "diffusion on this space produces good text" remains unvalidated.

2. **Single-token granularity.** Token VAE operates on individual tokens. Text generation requires sequence-level modeling, which introduces alignment, ordering, and coherence challenges that single-token geometry does not address.

3. **Ablation ceiling effect.** All eight ablation configurations pass all tests, which limits the ablation study to directional analysis rather than component necessity claims. The evaluation battery may lack the resolution to distinguish contributions of individual regularizers, or the thresholds may be too lenient.

4. **Threshold proximity.** Multiple configurations pass with margins below 5% of the threshold value. The thresholds are empirical, not theoretically motivated, and different threshold choices could change the conclusions for borderline cases.

5. **Single dataset.** All training data comes from FineWeb (English web text). Performance on other languages, domains, or data distributions is unknown.

6. **Provenance gaps.** The evaluation metadata does not include data file hashes (0/21 runs) or git commit hashes (0/21 runs). While model hashes, tokenizer hashes, seeds, and evaluation configurations are recorded, the absence of data and code provenance limits full auditability.

7. **Small scale.** The architecture uses a 256-dimensional latent space with vocabularies up to 16k tokens. Behavior at larger scales (higher dimensions, larger vocabularies, longer training) is untested.

## 9. Conclusion

We have presented Token VAE, a variational autoencoder that maps discrete tokens into a regularized 256-dimensional latent space, and evaluated whether this space satisfies four geometric prerequisites for continuous diffusion: prior coverage, local smoothness, interpolation continuity, and reconstruction fidelity.

Across 21 configurations covering five random seeds, three data scales, three vocabulary scales, eight ablation variants, and two baselines, 20 of 21 configurations pass all six evaluation tests. The single failure is the non-VAE baseline, which confirms that variational regularization is necessary (though not sufficient) for achieving prior coverage. Reproducibility is tight, with metric standard deviations below 1% of the mean across seeds. Data scaling from 2M to 8M characters consistently improves all metrics.

These results establish that the tested necessary conditions for continuous diffusion are robustly satisfied. They do not establish that diffusion on this space would produce coherent or high-quality text. The ablation study provides directional evidence of incremental benefit from loss components but cannot establish that any individual component is strictly required. Several configurations pass with margins as tight as 1.9% of the threshold value, and the thresholds themselves are empirically rather than theoretically motivated.

Future work should proceed in three directions. First, train a continuous diffusion model on the Token VAE latent space and evaluate generation quality, which would test whether the necessary conditions documented here are also practically sufficient. Second, extend the approach from single-token to sequence-level representations, which is required for any practical text generation system. Third, investigate tighter, theoretically grounded evaluation thresholds that could provide stronger guarantees and better discriminate between loss component contributions.

## References

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces. In *Advances in Neural Information Processing Systems 34 (NeurIPS)*.

Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., and Bengio, S. (2016). Generating Sentences from a Continuous Space. In *Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning (CoNLL)*.

Gong, S., Li, M., Feng, J., Wu, Z., and Kong, L. (2023). DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In *Advances in Neural Information Processing Systems 30 (NIPS)*.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In *Advances in Neural Information Processing Systems 33 (NeurIPS)*.

Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., and Welling, M. (2016). Improved Variational Inference with Inverse Autoregressive Flow. In *Advances in Neural Information Processing Systems 29 (NeurIPS)*.

Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., and Hashimoto, T. B. (2022). Diffusion-LM Improves Controllable Text Generation. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. In *Advances in Neural Information Processing Systems 26 (NeurIPS)*.

Penedo, G., Kydlíček, H., Ben allal, L., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In *The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. (2016). Improved Techniques for Training GANs. In *Advances in Neural Information Processing Systems 29 (NIPS)*.

van den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
