# Hypothesis Validation Summary

## A. Reproducibility (5 seeds)

| seed | Prior Decodability | Perturbation Stability | Interpolation Continuity | Reconstruction Accuracy | Metric Integrity | Diffusion Walk | all_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 1.0000 P | 0.8995 P | 0.0780 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 123 | 1.0000 P | 0.9046 P | 0.0763 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 456 | 1.0000 P | 0.9099 P | 0.0751 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 789 | 1.0000 P | 0.9054 P | 0.0762 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 1337 | 1.0000 P | 0.9056 P | 0.0759 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |

## B. Data Scale

| data_size | Prior Decodability | Perturbation Stability | Interpolation Continuity | Reconstruction Accuracy | Metric Integrity | Diffusion Walk | all_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2M | 1.0000 P | 0.9058 P | 0.0897 P | 0.9914 P | 1.0000 P | 1.0000 P | YES |
| 4M | 1.0000 P | 0.8982 P | 0.0783 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 8M | 1.0000 P | 0.9181 P | 0.0753 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |

## C. Vocab Scale

| vocab_size | Prior Decodability | Perturbation Stability | Interpolation Continuity | Reconstruction Accuracy | Metric Integrity | Diffusion Walk | all_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4k | 0.9999 P | 0.8975 P | 0.0754 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| 8k | 0.9997 P | 0.9465 P | 0.0750 P | 1.0000 P | 1.0000 P | 0.9900 P | YES |
| 16k | 1.0000 P | 0.8985 P | 0.0780 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |

## D. Ablations

| config | Prior Decodability | Perturbation Stability | Interpolation Continuity | Reconstruction Accuracy | Metric Integrity | Diffusion Walk | all_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.0000 P | 0.8985 P | 0.0766 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| no_marginal_entropy | 1.0000 P | 0.9024 P | 0.0767 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| no_hhi | 1.0000 P | 0.8975 P | 0.0767 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| no_marginal_no_hhi | 1.0000 P | 0.8994 P | 0.0766 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| no_free_bits | 1.0000 P | 0.9184 P | 0.0825 P | 0.9992 P | 1.0000 P | 1.0000 P | YES |
| half_ifw | 1.0000 P | 0.8408 P | 0.0735 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| no_ifw | 0.9998 P | 0.8336 P | 0.0783 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |
| ifw_only | 1.0000 P | 0.9219 P | 0.0823 P | 0.9988 P | 1.0000 P | 1.0000 P | YES |

## G. Baselines

| config | Prior Decodability | Perturbation Stability | Interpolation Continuity | Reconstruction Accuracy | Metric Integrity | Diffusion Walk | all_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non_vae | 1.0000 F | 0.7349 P | 0.0695 P | 1.0000 P | 0.0000 F | 1.0000 P | NO |
| no_skipgram | 1.0000 P | 0.8694 P | 0.0709 P | 1.0000 P | 1.0000 P | 1.0000 P | YES |

## Overall Summary

- **Total runs with results**: 21
- **All tests passing**: 20/21
