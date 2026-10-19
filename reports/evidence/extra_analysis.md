## 1. Detection by manipulated-region size (test, splice + copy-move vs all authentic)

Terciles of mask area cut at 0.091 and 0.137 of the frame. AUROC of each stratum against all 250 authentic images, and recall at 0.5.

| variant | small AUROC | medium AUROC | large AUROC | small recall | medium recall | large recall |
|---|---|---|---|---|---|---|
| single_a | 0.990 | 0.991 | 1.000 | 0.923 | 0.945 | 0.994 |
| single_b | 0.879 | 0.906 | 0.947 | 0.708 | 0.776 | 0.863 |
| indep | 0.974 | 0.981 | 0.997 | 0.917 | 0.921 | 1.000 |
| ml | 0.969 | 0.976 | 0.993 | 0.875 | 0.867 | 0.958 |
| ml_hem | 0.977 | 0.979 | 0.996 | 0.833 | 0.842 | 0.976 |
| cfr_adaptive | 0.972 | 0.980 | 0.995 | 0.923 | 0.909 | 0.994 |

## 2. Selective prediction (test, seen tools), accuracy at coverage, mean over seeds

Two abstention rules for the fused score: lowest confidence |p - 0.5| first, and branch disagreement first (ties broken by confidence). Coverage 0.8 means the 20 percent most uncertain images are withheld. The random-abstention accuracy is the full-coverage accuracy.

| variant | acc @1.0 | confidence @0.8 | confidence @0.6 | disagreement-first @0.8 | AURC (confidence) |
|---|---|---|---|---|---|
| indep | 0.958 | 0.979 | 0.995 | 0.979 | 0.009 |
| ml | 0.928 | 0.973 | 0.996 | 0.973 | 0.014 |
| ml_hem | 0.936 | 0.976 | 0.991 | 0.974 | 0.011 |
| cfr_adaptive | 0.945 | 0.976 | 0.991 | 0.976 | 0.013 |
| cfr_sudden | 0.939 | 0.983 | 0.989 | 0.983 | 0.012 |
| cfr_fixed | 0.954 | 0.980 | 0.989 | 0.979 | 0.013 |
| cfr_scratch | 0.950 | 0.980 | 0.993 | 0.977 | 0.010 |

## 3. Does alignment make errors harder to correct (test, seen tools, mean over seeds)

`B fixes A's miss`: of images A gets wrong, the share B gets right. `A fixes B's miss`: the reverse. A model that has copied its partner would show both near zero. The independent pair is the reference.

| variant | A macro-F1 | B macro-F1 | A wrong | B fixes A's miss | B wrong | A fixes B's miss | fused macro-F1 |
|---|---|---|---|---|---|---|---|
| indep | 0.958 | 0.815 | 0.040 | 0.417 | 0.178 | 0.870 | 0.957 |
| ml | 0.958 | 0.805 | 0.040 | 0.231 | 0.183 | 0.832 | 0.925 |
| ml_hem | 0.962 | 0.774 | 0.037 | 0.327 | 0.199 | 0.879 | 0.933 |
| cfr_adaptive | 0.962 | 0.825 | 0.037 | 0.307 | 0.169 | 0.848 | 0.943 |
| cfr_sudden | 0.963 | 0.835 | 0.035 | 0.315 | 0.159 | 0.849 | 0.937 |
| cfr_fixed | 0.964 | 0.823 | 0.034 | 0.325 | 0.171 | 0.865 | 0.952 |
| cfr_scratch | 0.961 | 0.784 | 0.038 | 0.443 | 0.197 | 0.891 | 0.948 |

## 4. Validation metrics either side of the CFR trigger (val split, from the training history)

`before` is the epoch just before the CFR weight first applied, `final` the last epoch. Weights are the blend gamma between the mutual-learning and CFR terms. Validation, never test.

| run | trigger epoch | gamma by epoch | fused val AUROC before / final | A before / final | B before / final | disagreement before / final |
|---|---|---|---|---|---|---|
| cfr_adaptive s0 | 7 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 | 0.955 / 0.974 | 0.974 / 0.988 | 0.847 / 0.891 | 0.228 / 0.168 |
| cfr_adaptive s1 | 6 | 0.00 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 | 0.962 / 0.972 | 0.972 / 0.985 | 0.830 / 0.885 | 0.310 / 0.176 |
| cfr_adaptive s2 | 8 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 | 0.964 / 0.975 | 0.984 / 0.987 | 0.863 / 0.896 | 0.210 / 0.180 |
| cfr_sudden s0 | 7 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 1.00 | 0.956 / 0.973 | 0.976 / 0.988 | 0.847 / 0.888 | 0.204 / 0.182 |
| cfr_sudden s1 | 10 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 | 0.964 / 0.970 | 0.982 / 0.986 | 0.880 / 0.883 | 0.196 / 0.190 |
| cfr_sudden s2 | 8 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 | 0.961 / 0.978 | 0.983 / 0.989 | 0.872 / 0.897 | 0.194 / 0.184 |
| cfr_fixed s0 | 5 | 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 | 0.961 / 0.970 | 0.974 / 0.985 | 0.821 / 0.890 | 0.382 / 0.192 |
| cfr_fixed s1 | 5 | 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 | 0.958 / 0.976 | 0.975 / 0.989 | 0.816 / 0.879 | 0.256 / 0.192 |
| cfr_fixed s2 | 5 | 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 | 0.957 / 0.976 | 0.970 / 0.983 | 0.784 / 0.884 | 0.290 / 0.198 |

## 5. Compute-matched control (variant J)

| model | parameters (M) | GMACs per 384 px image |
|---|---|---|
| Model A (EfficientNet-B0 + decoder) | 4.21 | 1.61 |
| Model B (forensic CNN + decoder) | 1.40 | 1.84 |
| A + B pair | 5.62 | 3.45 |
| two Model A (the compute-matched alternative) | 8.43 | 3.22 |

Post-hoc ensemble of two independently trained Model A runs (seed pairs 0+1, 1+2, 2+0), averaged probabilities, primary metric (test, seen tools):

- single Model A: 0.9936 (per seed 0.9966, 0.9931, 0.9912)
- two-Model-A ensemble: 0.9951 (per pair 0.9961, 0.9933, 0.9960)
- best A+B pair variant for reference (mean over seeds): see the headline table in `comparison.md`.

This control gets about 2x A's compute from a stronger visual model instead of a weak forensic one; it is not trained jointly, so it is a ceiling on 'more of A', not a matched training procedure.

## 6. Extra variants (3 seeds each where finished)

| variant | seeds | primary AUROC mean ± sd | per seed | removal (test) | FPR authentic | test ECE | Dice |
|---|---|---|---|---|---|---|---|
| single_a | 3 | 0.9936 ± 0.0027 | 0.9966, 0.9931, 0.9912 | 0.795 | 0.029 | 0.140 | 0.663 |
| ml | 3 | 0.9791 ± 0.0023 | 0.9793, 0.9812, 0.9767 | 0.716 | 0.053 | 0.061 | 0.616 |
| cfr_scratch | 3 | 0.9850 ± 0.0009 | 0.9859, 0.9850, 0.9840 | 0.708 | 0.036 | 0.069 | 0.587 |
| cfr_adaptive | 3 | 0.9824 ± 0.0010 | 0.9817, 0.9835, 0.9821 | 0.713 | 0.053 | 0.057 | 0.624 |
| joint_concat | 3 | 0.9929 ± 0.0025 | 0.9903, 0.9953, 0.9930 | 0.766 | 0.031 | 0.147 | 0.608 |
| joint_gate | 3 | 0.9908 ± 0.0008 | 0.9901, 0.9905, 0.9917 | 0.791 | 0.029 | 0.140 | 0.611 |
| cfr_scratch_noalign | 3 | 0.9874 ± 0.0006 | 0.9880, 0.9874, 0.9869 | 0.688 | 0.040 | 0.051 | 0.603 |
| ml_lam0.25 | 3 | 0.9858 ± 0.0019 | 0.9868, 0.9837, 0.9870 | 0.728 | 0.031 | 0.069 | 0.617 |
| ml_lam4 | 3 | 0.9568 ± 0.0068 | 0.9587, 0.9626, 0.9493 | 0.705 | 0.089 | 0.055 | 0.611 |

Paired comparisons on the primary metric (per-seed difference, mean, 95% paired bootstrap over test images on seed-averaged probabilities). These are descriptive; the pre-declared decision rule in `02` applies only to cfr_adaptive vs ml and ml_hem.

| A minus B | shared seeds | per-seed | mean | 95% bootstrap |
|---|---|---|---|---|
| joint_concat - single_a | 3 | -0.0062, +0.0022, +0.0018 | +0.0005 | [-0.0009, +0.0021] |
| joint_concat - indep | 3 | +0.0038, +0.0145, +0.0082 | +0.0085 | [+0.0033, +0.0150] |
| joint_gate - single_a | 3 | -0.0065, -0.0026, +0.0005 | -0.0034 | [-0.0102, +0.0010] |
| joint_gate - indep | 3 | +0.0035, +0.0097, +0.0069 | +0.0045 | [-0.0032, +0.0112] |
| cfr_scratch - cfr_scratch_noalign | 3 | -0.0021, -0.0023, -0.0029 | -0.0021 | [-0.0057, +0.0012] |
| ml_lam0.25 - ml | 3 | +0.0075, +0.0024, +0.0103 | +0.0067 | [+0.0010, +0.0131] |
| ml_lam4 - ml | 3 | -0.0206, -0.0187, -0.0274 | -0.0224 | [-0.0317, -0.0143] |
| ml_lam0.25 - indep | 3 | +0.0002, +0.0029, +0.0022 | +0.0013 | [-0.0028, +0.0057] |
| ml_lam4 - indep | 3 | -0.0279, -0.0183, -0.0355 | -0.0278 | [-0.0408, -0.0169] |

Learned gate, mean weight on Model A over test images (per seed): 0.994, 0.991, 0.997

