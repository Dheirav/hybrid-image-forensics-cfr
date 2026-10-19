## CFR and hard-example weights on the training images, end-of-training checkpoint, mean over 3 seeds

`challenged_5pct` is the share of training images where the partner is more right than the model by more than 0.05 (the only images CFR up-weights). `cfr_ess` and `hem_ess` are the effective sample size as a fraction of n (1 = uniform weights). `spearman_cfr_hem` is the rank correlation of the two weight vectors on the same checkpoint. `gate_share` is the share of images where the alignment weight is above its floor (the partner is more right). `epochs_gamma_pos` and `epochs_gamma_1` count epochs with any CFR weight and with the full CFR weight (of 12).

| variant | model | train_auroc | train_acc | challenged_5pct | challenged_50pct | cfr_w_max | cfr_ess | hem_ess | spearman_cfr_hem | gate_share | align_mean | epochs_gamma_pos | epochs_gamma_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfr_adaptive | A | 1.0 | 0.9978 | 0.0028 | 0.0003 | 2.0986 | 0.9993 | 0.9911 | -0.3143 | 0.0864 | 0.3135 | 5.0 | 2.0 |
| cfr_adaptive | B | 0.9273 | 0.8445 | 0.7524 | 0.0918 | 1.9131 | 0.9327 | 0.9218 | 0.9777 | 0.9133 | 0.8854 | 5.0 | 2.0 |
| cfr_scratch | A | 0.9998 | 0.9955 | 0.0074 | 0.0001 | 1.9233 | 0.9989 | 0.9895 | -0.0929 | 0.0519 | 0.2866 | 12.0 | 12.0 |
| cfr_scratch | B | 0.8615 | 0.7697 | 0.8814 | 0.151 | 1.8025 | 0.9394 | 0.9413 | 0.958 | 0.9477 | 0.902 | 12.0 | 12.0 |
| ml | A | 0.9997 | 0.9967 | 0.0224 | 0.0 | 1.7098 | 0.999 | 0.9831 | -0.0981 | 0.1068 | 0.3228 | 0.0 | 0.0 |
| ml | B | 0.8989 | 0.8062 | 0.724 | 0.0341 | 1.8632 | 0.9523 | 0.9401 | 0.9136 | 0.8926 | 0.8145 | 0.0 | 0.0 |
| ml_hem | A | 1.0 | 0.9972 | 0.0214 | 0.0 | 1.5746 | 0.999 | 0.9885 | -0.0008 | 0.0868 | 0.3076 | 0.0 | 0.0 |
| ml_hem | B | 0.8773 | 0.7686 | 0.7838 | 0.056 | 1.8096 | 0.95 | 0.9505 | 0.9296 | 0.9129 | 0.8219 | 0.0 | 0.0 |
