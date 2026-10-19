## Branch behaviour on test (authentic + splice + copy_move), mean over seeds

| variant | n seeds | aucA | aucB | errA | errB | both_wrong | B_err_shared_with_A | agree | logit_corr | mean_abs_prob_gap |
|---|---|---|---|---|---|---|---|---|---|---|
| single_a | 3 | 0.994 | 0.994 | 0.036 | 0.036 | 0.036 | 1.000 | 1.000 | 1.000 | 0.000 |
| single_b | 3 | 0.911 | 0.911 | 0.173 | 0.173 | 0.173 | 1.000 | 1.000 | 1.000 | 0.000 |
| indep | 3 | 0.993 | 0.899 | 0.040 | 0.178 | 0.023 | 0.130 | 0.828 | 0.729 | 0.261 |
| ml | 3 | 0.988 | 0.899 | 0.040 | 0.183 | 0.030 | 0.168 | 0.838 | 0.837 | 0.152 |
| ml_hem | 3 | 0.991 | 0.878 | 0.037 | 0.199 | 0.024 | 0.121 | 0.812 | 0.783 | 0.167 |
| cfr_adaptive | 3 | 0.990 | 0.910 | 0.037 | 0.169 | 0.026 | 0.152 | 0.846 | 0.836 | 0.183 |
| cfr_sudden | 3 | 0.991 | 0.914 | 0.035 | 0.159 | 0.024 | 0.151 | 0.854 | 0.838 | 0.183 |
| cfr_fixed | 3 | 0.992 | 0.907 | 0.034 | 0.171 | 0.023 | 0.135 | 0.841 | 0.815 | 0.195 |
| cfr_scratch | 3 | 0.992 | 0.881 | 0.038 | 0.197 | 0.021 | 0.109 | 0.807 | 0.721 | 0.227 |

## Transition: first epoch at which the CFR weight applied, and what blocked eligibility before it

- cfr_adaptive s0: trigger epoch 7, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,calibrated_enough; e3: val_stable,calibrated_enough; e4: val_stable
- cfr_adaptive s1: trigger epoch 6, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e3: val_stable,aligned_not_collapsed
- cfr_adaptive s2: trigger epoch 8, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e2: past_warmup,val_stable,aligned_not_collapsed; e3: val_stable; e4: val_stable; e5: val_stable
- cfr_sudden s0: trigger epoch 7, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,val_stable,aligned_not_collapsed; e2: past_warmup,val_stable,calibrated_enough; e3: val_stable; e4: val_stable
- cfr_sudden s1: trigger epoch 10, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e3: val_stable,aligned_not_collapsed; e5: val_stable; e7: val_stable
- cfr_sudden s2: trigger epoch 8, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,aligned_not_collapsed; e3: val_stable; e4: val_stable; e5: val_stable
- cfr_fixed s0: trigger epoch 5, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed; e1: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e3: val_stable; e4: val_stable,aligned_not_collapsed,calibrated_enough
- cfr_fixed s1: trigger epoch 5, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e1: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e3: val_stable,calibrated_enough
- cfr_fixed s2: trigger epoch 5, failed checks per epoch before/at trigger: e0: past_warmup,both_learn_signal,val_stable,aligned_not_collapsed,calibrated_enough; e1: past_warmup,val_stable,aligned_not_collapsed,calibrated_enough; e2: past_warmup,aligned_not_collapsed; e3: val_stable; e5: val_stable

## Training-time pair statistics at the last epoch (validation)

- indep: disagreement 0.194, prob corr 0.694
- ml: disagreement 0.182, prob corr 0.803
- ml_hem: disagreement 0.170, prob corr 0.802
- cfr_adaptive: disagreement 0.175, prob corr 0.765
- cfr_sudden: disagreement 0.185, prob corr 0.763
- cfr_fixed: disagreement 0.194, prob corr 0.739
- cfr_scratch: disagreement 0.197, prob corr 0.723
