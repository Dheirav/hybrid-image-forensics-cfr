## 1. Does B add information? Fusion weights and probes fitted on COCO-Forge validation, scored on unseen data

A alone is the Model A softmax. `fixed avg` is the 0.5/0.5 probability average used by the headline pairs. `stack` is a logistic regression on the two logits (A_s with B_s of the same seed). `probe A` and `probe A+B` are logistic regressions on frozen penultimate embeddings (C chosen by 5-fold CV on validation). AUROC, mean over 3 seeds (per seed in brackets).

| evaluation set | A alone | B alone | fixed avg | stack A+B | probe A | probe A+B |
|---|---|---|---|---|---|---|
| COCO-Forge seen tools | 0.994 (0.997, 0.993, 0.991) | 0.911 (0.914, 0.907, 0.911) | 0.988 (0.992, 0.986, 0.986) | 0.994 (0.997, 0.993, 0.992) | 0.994 (0.996, 0.993, 0.991) | 0.994 (0.996, 0.994, 0.992) |
| COCO-Forge removal (unseen) | 0.795 (0.820, 0.825, 0.739) | 0.682 (0.680, 0.696, 0.670) | 0.715 (0.730, 0.739, 0.675) | 0.796 (0.820, 0.828, 0.740) | 0.787 (0.820, 0.807, 0.734) | 0.794 (0.821, 0.820, 0.740) |
| hard: splice_feather | 0.994 (0.994, 0.992, 0.996) | 0.913 (0.911, 0.912, 0.918) | 0.989 (0.989, 0.988, 0.989) | 0.994 (0.994, 0.992, 0.996) | 0.994 (0.993, 0.992, 0.996) | 0.994 (0.993, 0.993, 0.996) |
| hard: copy_move_feather | 0.990 (0.990, 0.991, 0.988) | 0.883 (0.883, 0.881, 0.883) | 0.981 (0.984, 0.978, 0.982) | 0.990 (0.990, 0.991, 0.989) | 0.990 (0.989, 0.991, 0.989) | 0.990 (0.990, 0.991, 0.989) |
| hard: splice_poisson | 0.516 (0.524, 0.512, 0.510) | 0.492 (0.496, 0.494, 0.486) | 0.496 (0.499, 0.498, 0.492) | 0.515 (0.524, 0.511, 0.510) | 0.518 (0.522, 0.523, 0.509) | 0.512 (0.523, 0.509, 0.505) |
| hard: copy_move_poisson | 0.514 (0.514, 0.515, 0.512) | 0.497 (0.498, 0.497, 0.497) | 0.500 (0.501, 0.501, 0.499) | 0.513 (0.515, 0.515, 0.510) | 0.518 (0.513, 0.527, 0.515) | 0.513 (0.511, 0.516, 0.511) |
| hard: self_resample | 0.895 (0.907, 0.885, 0.894) | 0.838 (0.838, 0.834, 0.843) | 0.897 (0.900, 0.889, 0.902) | 0.897 (0.907, 0.887, 0.897) | 0.890 (0.904, 0.874, 0.891) | 0.897 (0.902, 0.890, 0.900) |

Stack weights on (logit A, logit B), per seed: A 0.53, B -0.01; A 0.51, B 0.11; A 0.50, B 0.10

Paired bootstrap over test images on seed-averaged scores (A+B minus A alone), 95 percent interval:

| evaluation set | stack A+B minus A | probe A+B minus probe A | probe A+B minus A alone |
|---|---|---|---|
| COCO-Forge seen tools | -0.0004 [-0.0013, +0.0003] | +0.0001 [-0.0006, +0.0009] | -0.0003 [-0.0014, +0.0008] |
| COCO-Forge removal (unseen) | +0.0030 [-0.0033, +0.0091] | +0.0018 [-0.0135, +0.0177] | -0.0028 [-0.0176, +0.0108] |
| hard: splice_feather | +0.0001 [-0.0005, +0.0008] | -0.0002 [-0.0011, +0.0006] | -0.0003 [-0.0017, +0.0010] |
| hard: copy_move_feather | +0.0008 [+0.0000, +0.0017] | +0.0011 [-0.0007, +0.0033] | +0.0018 [-0.0000, +0.0040] |
| hard: splice_poisson | -0.0027 [-0.0093, +0.0040] | -0.0062 [-0.0278, +0.0139] | -0.0089 [-0.0245, +0.0062] |
| hard: copy_move_poisson | +0.0011 [-0.0056, +0.0078] | -0.0062 [-0.0260, +0.0131] | -0.0003 [-0.0156, +0.0148] |
| hard: self_resample | +0.0014 [-0.0014, +0.0045] | +0.0032 [-0.0034, +0.0102] | -0.0001 [-0.0062, +0.0060] |

## 2. Is B right where A is unsure? (COCO-Forge test, seen tools, pooled over 3 seeds)

| seed | n | a_wrong | b_wrong | both | b_fixes | a_fixes | n_unsure | auc_a_unsure | auc_b_unsure | err_phi | logit_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 417 | 16 | 75 | 9 | 7 | 66 | 36 | 0.649 | 0.538 | 0.199 | 0.722 |
| 1 | 417 | 15 | 72 | 9 | 6 | 63 | 30 | 0.777 | 0.406 | 0.218 | 0.726 |
| 2 | 417 | 14 | 69 | 8 | 6 | 61 | 32 | 0.81 | 0.45 | 0.204 | 0.743 |

`b_fixes` counts images A gets wrong and B gets right at 0.5; `a_fixes` the reverse. `unsure` = A probability between 0.05 and 0.95. `err_phi` is the correlation of the two error indicators (0 = independent errors, 1 = identical errors).

## 3. Do A and B learn different representations? Linear CKA on test (seen tools) embeddings

1.0 means the two embedding spaces are linearly equivalent, near 0 means unrelated. References are the same architecture trained with another seed.

| pair | CKA |
|---|---|
| A seed0 vs A seed1 (reference) | 0.877 |
| B seed0 vs B seed1 (reference) | 0.924 |
| A vs B, trained alone (seed 0) | 0.563 |
| A vs B inside indep (seed 0) | 0.497 |
| A vs B inside ml (seed 0) | 0.631 |
| A vs B inside ml_hem (seed 0) | 0.339 |
| A vs B inside cfr_adaptive (seed 0) | 0.66 |
| A vs B inside ml_lam4 (seed 0) | 0.795 |

B inside the aligned pairs is compared with the A it was aligned to; if alignment made B a copy of A the pair CKA would rise toward the A-vs-A reference.
