## Headline: fused test AUROC, authentic + splice + copy_move (primary)

| variant | seeds | primary AUROC mean ± sd | per-seed | removal (test) | ood_removal | val AUROC | test ECE | FPR authentic | Dice | pair disagreement | corr(A,B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single_a | 3 | 0.9936 ± 0.0027 | 0.9966, 0.9931, 0.9912 | 0.795 | 0.789 | 0.9872 | 0.140 | 0.029 | 0.663 | nan | nan |
| single_b | 3 | 0.9108 ± 0.0037 | 0.9141, 0.9069, 0.9115 | 0.682 | 0.707 | 0.8908 | 0.070 | 0.143 | 0.489 | nan | nan |
| indep | 3 | 0.9841 ± 0.0029 | 0.9866, 0.9808, 0.9848 | 0.704 | 0.734 | 0.9797 | 0.060 | 0.033 | 0.617 | 0.199 | 0.674 |
| ml | 3 | 0.9791 ± 0.0023 | 0.9793, 0.9812, 0.9767 | 0.716 | 0.759 | 0.9685 | 0.061 | 0.053 | 0.616 | 0.166 | 0.788 |
| ml_hem | 3 | 0.9844 ± 0.0011 | 0.9844, 0.9855, 0.9832 | 0.723 | 0.758 | 0.9744 | 0.083 | 0.029 | 0.621 | 0.188 | 0.733 |
| cfr_adaptive | 3 | 0.9824 ± 0.0010 | 0.9817, 0.9835, 0.9821 | 0.713 | 0.749 | 0.9740 | 0.057 | 0.053 | 0.624 | 0.166 | 0.761 |
| cfr_sudden | 3 | 0.9832 ± 0.0026 | 0.9854, 0.9803, 0.9838 | 0.725 | 0.770 | 0.9741 | 0.046 | 0.059 | 0.625 | 0.161 | 0.770 |
| cfr_fixed | 3 | 0.9836 ± 0.0026 | 0.9861, 0.9808, 0.9839 | 0.731 | 0.774 | 0.9743 | 0.052 | 0.048 | 0.634 | 0.173 | 0.741 |
| cfr_scratch | 3 | 0.9850 ± 0.0009 | 0.9859, 0.9850, 0.9840 | 0.708 | 0.745 | 0.9734 | 0.069 | 0.036 | 0.587 | 0.200 | 0.684 |

## Paired comparisons (A minus B), primary metric

| comparison | shared seeds | per-seed difference | mean | 95% paired bootstrap over test images (seed-averaged probs) | positive in every seed |
|---|---|---|---|---|---|
| cfr_adaptive - ml | 3 | +0.0024, +0.0023, +0.0055 | +0.0034 | [-0.0012, +0.0066] | yes |
| cfr_adaptive - ml_hem | 3 | -0.0027, -0.0020, -0.0011 | -0.0019 | [-0.0089, +0.0029] | no |
| ml - indep | 3 | -0.0073, +0.0004, -0.0081 | -0.0050 | [-0.0118, -0.0004] | no |
| ml_hem - ml | 3 | +0.0051, +0.0043, +0.0066 | +0.0053 | [-0.0001, +0.0114] | yes |
| cfr_adaptive - cfr_sudden | 3 | -0.0037, +0.0032, -0.0017 | -0.0007 | [-0.0041, +0.0009] | no |
| cfr_adaptive - cfr_fixed | 3 | -0.0044, +0.0027, -0.0018 | -0.0012 | [-0.0049, +0.0001] | no |
| cfr_adaptive - cfr_scratch | 3 | -0.0042, -0.0015, -0.0019 | -0.0026 | [-0.0102, -0.0012] | no |
| cfr_adaptive - indep | 3 | -0.0049, +0.0027, -0.0027 | -0.0016 | [-0.0077, +0.0014] | no |
| cfr_adaptive - single_a | 3 | -0.0149, -0.0096, -0.0091 | -0.0112 | [-0.0190, -0.0042] | no |

## Same comparisons on unseen-tool removal (test removal + authentic; secondary, headroom exists here)

| comparison | per-seed difference | mean |
|---|---|---|
| cfr_adaptive - ml | +0.0108, -0.0053, -0.0169 | -0.0038 |
| cfr_adaptive - ml_hem | +0.0144, -0.0469, +0.0004 | -0.0107 |
| ml - indep | -0.0035, +0.0200, +0.0204 | +0.0123 |
| ml_hem - ml | -0.0036, +0.0416, -0.0173 | +0.0069 |
| cfr_adaptive - cfr_sudden | -0.0061, -0.0142, -0.0185 | -0.0129 |
| cfr_adaptive - cfr_fixed | -0.0280, -0.0196, -0.0071 | -0.0182 |
| cfr_adaptive - cfr_scratch | +0.0361, -0.0533, +0.0304 | +0.0044 |
| cfr_adaptive - indep | +0.0074, +0.0148, +0.0035 | +0.0086 |
| cfr_adaptive - single_a | -0.0990, -0.1114, -0.0358 | -0.0820 |

## Transition controller

- cfr_adaptive seed 0: trigger epoch 7 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_adaptive seed 1: trigger epoch 6 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_adaptive seed 2: trigger epoch 8 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_sudden seed 0: trigger epoch 7 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_sudden seed 1: trigger epoch 10 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_sudden seed 2: trigger epoch 8 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_fixed seed 0: trigger epoch 5 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_fixed seed 1: trigger epoch 5 (0-indexed epoch at which CFR weight first applies; None = never triggered)
- cfr_fixed seed 2: trigger epoch 5 (0-indexed epoch at which CFR weight first applies; None = never triggered)
