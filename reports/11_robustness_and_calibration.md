# 11 Calibration, uncertainty and robustness

One checkpoint (`runs/cfr_adaptive_s0`), one seed. These are checks of behaviour on the test split, not a multi-seed study.

## Calibration

Temperature scaling was fitted on the validation split only (`calibrate.py`): T = 0.518. A value below 1 sharpens the
probabilities, because validation showed the fused output was under-confident (ECE 0.105).

| split | ECE before | ECE after | NLL before | NLL after |
|---|---|---|---|---|
| val (fitted here) | 0.105 | 0.053 | 0.253 | 0.209 |
| test (all tools) | 0.058 | 0.089 | 0.403 | 0.480 |
| ood_removal | 0.177 | 0.265 | 0.731 | 1.048 |

The fit improves the data it was fitted on and makes the held-out splits worse. This is the useful finding: calibration on a
validation split drawn from the same tools did not transfer. Validation contains only splice and copy-move, so it is
easy; the test split contains removal, where the model is wrong with confidence, and sharpening makes those errors costlier. So the application's
optional calibration file should not be applied for the shipped checkpoint, and the uncalibrated test ECE of 0.058 (three-seed mean for `cfr_adaptive` 0.057) is the honest figure. The e2e run in `10` did use the file, so its probabilities are the sharpened ones; no decision there depends on it.

Confidence is useful as a ranking (AUROC) and a rough probability on seen tools. It is not reliable on unseen manipulation: on `ood_removal`
ECE is 0.18 even uncalibrated, and the model is confidently wrong on missed removals (`10`, p = 0.014 for a removal).

## Uncertainty output

The only uncertainty signals in the application are the disagreement flag between the two branches (`branches_agree`) and the
distance of the probability from 0.5. Neither is a calibrated uncertainty. I did not test whether disagreement predicts error beyond the small examples in `10`;
for the mean error rates see `07` (under `ml` the branches disagree on about 16 percent of seen-tool images, while A alone is wrong on about 4 percent).
A proper selective-prediction evaluation (accuracy against coverage by branch agreement) is not done.

## Robustness to degradation

Degradations are applied to both classes, so a model that merely flags "processed" images would pay through the authentic
false-positive rate. The test split contains removal, so the clean row is the all-tools AUROC (0.895), not the seen-tool primary metric (0.982).
These are not unseen-tool tests; the unseen-tool result is in `12` and `05`.

| condition | AUROC | macro-F1 | FPR on authentic | Dice |
|---|---|---|---|---|
| clean | 0.895 | 0.836 | 0.064 | 0.636 |
| JPEG q70 | 0.888 | 0.812 | 0.076 | 0.591 |
| JPEG q50 | 0.865 | 0.817 | 0.092 | 0.583 |
| recompress q90 then q60 | 0.865 | 0.804 | 0.092 | 0.587 |
| resize 0.75 | 0.877 | 0.844 | 0.116 | 0.599 |
| resize 1.25 | 0.830 | 0.778 | 0.224 | 0.610 |
| Gaussian blur sigma 1 | 0.631 | 0.578 | **0.592** | 0.416 |
| Gaussian noise sigma 5 | 0.855 | 0.798 | 0.052 | 0.555 |
| resize 0.75 + JPEG 70 + blur 0.8 | 0.618 | 0.574 | **0.516** | 0.366 |

- JPEG recompression and mild resizing cost 1 to 3 points of AUROC, which is acceptable, but upscaling by 1.25 raises false positives on authentic images to 22 percent.
- **Blur is the failure.** A sigma-1 blur drops AUROC to 0.63 and flags 59 percent of authentic images as manipulated. A chain that includes a small blur does the same.
  The training data contain no blur, so the model reads "less sharp" as an editing trace. This is a real limitation for any social-media-style input
  and needs to be stated whenever the system is demonstrated.
- No threshold was tuned on any of these conditions.

## Update: AI-edited and AI-generated images, mask size, selective prediction

### AI-edited photographs and fully generated images (zero-shot)

The external test-only set (`04` addendum): 400 real, 400 AI-edited (flux, sd15, sd3, sdxl, with masks), 400 AI-generated (flux, sd2, sd3, sdxl), 512 px JPEG.
No model saw anything like it in training. Threshold 0.5, no calibration, checkpoint chosen on COCO-Forge validation, edge-anchored tiling, image score = max over windows. Mean over 3 seeds.
Tables: `reports/evidence/external_summary.md` (all 14 variants). Kept separate from degradation and from unseen-tool (removal) tests, as the brief requires.

| variant | FPR on real | AI-edited AUROC | AI-edited recall | AI-edited Dice | AI-generated AUROC | AI-generated recall |
|---|---|---|---|---|---|---|
| single_a | 0.047 | 0.593 | 0.081 | 0.023 | 0.566 | 0.077 |
| single_b | 0.353 | 0.509 | 0.322 | 0.035 | 0.563 | 0.414 |
| indep | 0.070 | 0.534 | 0.099 | 0.016 | 0.584 | 0.099 |
| ml | 0.141 | 0.528 | 0.126 | 0.015 | 0.605 | 0.203 |
| ml_hem | 0.075 | 0.536 | 0.077 | 0.012 | 0.581 | 0.102 |
| cfr_adaptive | 0.132 | 0.552 | 0.140 | 0.014 | 0.591 | 0.191 |
| cfr_scratch | 0.090 | 0.542 | 0.111 | 0.014 | 0.593 | 0.148 |
| joint_concat | 0.062 | 0.593 | 0.092 | 0.013 | 0.583 | 0.077 |
| joint_gate | 0.073 | 0.586 | 0.109 | 0.018 | 0.576 | 0.092 |

**The models do not detect AI-edited or AI-generated images.** AUROC is 0.51 to 0.62. For the proposed system (`cfr_adaptive`, seed-averaged scores, bootstrap over images) it is 0.554 for AI-edited (95 percent interval 0.514 to 0.591) and 0.594 for AI-generated
(0.557 to 0.634), so it is statistically above 0.5 but of no practical use: at the 0.5 threshold it finds 14 and 19 percent of the fakes while raising a false alarm on 13 percent of real photographs.
Model A alone and the learned-fusion systems have similar AUROC (0.57 to 0.59) with fewer false alarms and fewer detections (about 8 percent), which is a different operating point, not a better detector. Dice for the AI-edited masks is
0.01 to 0.04, so the localisation does not find AI edits either. By generator, AI-edited AUROC ranges from 0.49 (sd15, `cfr_adaptive`) to 0.63 (sd3, `single_a`), and AI-generated from 0.47 (sd3, `single_a`) to 0.66 (sd2, `cfr_adaptive`);
with 100 images per group and three seeds these per-generator differences are not reliable.

This is the expected result for a detector trained only on synthetic paste, copy-move and removal on COCO photographs. It is a measured gap, not a failure of the code: the same code gets 0.98 on held-out data of the kinds it trained on.
Two cautions about the number: the real images are all from one folder and the file-size cue is weak but present for generated images (AUROC 0.69 from file size alone, `04`), so a detector can be above chance here for the wrong reason.

Controls for the evaluation itself: scoring only the central 384 px (no padding, one window) gives lower false-positive rates (0.06 for `cfr_adaptive`) and the same AUROC (0.57 and 0.61); the first version of the tiling, which padded 512 px images with mirrored borders,
gave 0.19 (`reports/evidence/external_summary_padded.md`). That difference is what led to the tiling fix in `10`. The AUROC conclusion does not depend on the tiling.

### Mask size

Test, splice and copy-move images split into terciles by mask area (cut at 9.1 and 13.7 percent of the frame), AUROC against all authentic images, and recall at 0.5. Mean over seeds (`extra_analysis.md`, section 1).

| variant | small AUROC | medium | large | small recall | medium | large |
|---|---|---|---|---|---|---|
| single_a | 0.990 | 0.991 | 1.000 | 0.923 | 0.945 | 0.994 |
| single_b | 0.879 | 0.906 | 0.947 | 0.708 | 0.776 | 0.863 |
| indep | 0.974 | 0.981 | 0.997 | 0.917 | 0.921 | 1.000 |
| ml | 0.969 | 0.976 | 0.993 | 0.875 | 0.867 | 0.958 |
| ml_hem | 0.977 | 0.979 | 0.996 | 0.833 | 0.842 | 0.976 |
| cfr_adaptive | 0.972 | 0.980 | 0.995 | 0.923 | 0.909 | 0.994 |

Larger edits are easier for every model, as expected, and the effect is largest for B (recall 71 percent small against 86 percent large). The range of areas in this benchmark is narrow (the middle third spans only 9 to 14 percent
of the frame), so "small" here is not tiny; the 2 percent case in `10` was missed. The pair variants lose a little on small regions compared with A alone.

### Selective prediction

Accuracy at the 0.5 threshold when the most uncertain images are withheld (test, seen tools, mean over seeds; `extra_analysis.md`, section 2).
Confidence rule: withhold lowest |p - 0.5| first. Disagreement rule: withhold images where the two branches disagree first.

| variant | accuracy at coverage 1.0 | confidence rule at 0.8 | at 0.6 | disagreement rule at 0.8 | risk-coverage area (lower is better) |
|---|---|---|---|---|---|
| indep | 0.958 | 0.979 | 0.995 | 0.979 | 0.009 |
| ml | 0.928 | 0.973 | 0.996 | 0.973 | 0.014 |
| ml_hem | 0.936 | 0.976 | 0.991 | 0.974 | 0.011 |
| cfr_adaptive | 0.945 | 0.976 | 0.991 | 0.976 | 0.013 |
| cfr_sudden | 0.939 | 0.983 | 0.989 | 0.983 | 0.012 |
| cfr_fixed | 0.954 | 0.980 | 0.989 | 0.979 | 0.013 |
| cfr_scratch | 0.950 | 0.980 | 0.993 | 0.977 | 0.010 |

Withholding the least confident 20 percent raises accuracy by 2 to 5 points, so the score is useful as a ranking of reliability **on seen tools**. Branch disagreement is no better than plain confidence (it equals or trails it in every row), so the
`branches_agree` flag in the application is a convenience, not a validated reliability estimate. None of this carries over to unseen manipulations: the removal and AI-image results show confident wrong answers there.
