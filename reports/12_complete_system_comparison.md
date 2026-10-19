# 12 Complete system comparison

Status: **a three-seed pilot on a small synthetic benchmark. Exploratory, not a definitive evaluation.** 9 headline variants x 3 seeds = 27 runs,
plus 15 stage 3 runs (learned fusion, non-aligned CFR, alignment-strength sweep) and the three learning-rate checks described below, 12 epochs each, identical data, splits, optimiser and schedule. Seeds 0, 1 and 2 change initialisation
of the new layers, data order and augmentation. The ImageNet weights for Model A are the same in every run.
Primary metric (declared before any CFR result, `02`): fused test AUROC on authentic + splice + copy-move, mean over seeds. Model selection used
validation AUROC only; the test split was evaluated once per run with the best-validation checkpoint. Full tables, with all seeds,
bootstrap intervals and the unseen-tool secondary comparison: `reports/evidence/comparison.md`.

## Brief's variants and what was run

| brief | variant | run name | status |
|---|---|---|---|
| A | single visual model | `single_a` | run, 3 seeds |
| B | single forensic model | `single_b` | run, 3 seeds |
| C | independent ensemble | `indep` | run, 3 seeds |
| D | hybrid fusion without mutual learning | `indep` (fixed average), `joint_concat`, `joint_gate` | the first pass had the fixed average only; concatenation with a joint head and an input-dependent gate were added in stage 3 (`06` update 2) |
| E | mutual learning throughout | `ml` | run, 3 seeds |
| F | CFR throughout | `cfr_scratch` | run, 3 seeds |
| G | mutual learning then aligned CFR with the adaptive controller (the proposed system) | `cfr_adaptive` | run, 3 seeds |
| H | mutual learning plus hard-example mining | `ml_hem` | run, 3 seeds |
| I | adaptive schedule with sudden transition | `cfr_sudden` | run, 3 seeds |
| J | parameter- or compute-matched control | two-Model-A ensemble (post hoc) | built from the existing `single_a` seeds, no new training: 8.4 M parameters and 3.2 GMACs against the pair's 5.6 M and 3.4 GMACs. Not a jointly trained control (see below). |
| extra | CFR switched on at fixed epoch 5 | `cfr_fixed` | run, 3 seeds |

## Headline (fused test AUROC, seen tools; mean +/- sd over 3 seeds)

| variant | primary | test ECE | FPR authentic | removal AUROC (unseen tool) | Dice |
|---|---|---|---|---|---|
| single_a | **0.9936 +/- 0.0027** | 0.140 | 0.029 | 0.795 | 0.663 |
| single_b | 0.9108 +/- 0.0037 | 0.070 | 0.143 | 0.682 | 0.489 |
| indep | 0.9841 +/- 0.0029 | 0.060 | 0.033 | 0.704 | 0.617 |
| ml | 0.9791 +/- 0.0023 | 0.061 | 0.053 | 0.716 | 0.616 |
| ml_hem | 0.9844 +/- 0.0011 | 0.083 | 0.029 | 0.723 | 0.621 |
| cfr_adaptive | 0.9824 +/- 0.0010 | 0.057 | 0.053 | 0.713 | 0.624 |
| cfr_sudden | 0.9832 +/- 0.0026 | 0.046 | 0.059 | 0.725 | 0.625 |
| cfr_fixed | 0.9836 +/- 0.0026 | 0.052 | 0.048 | 0.731 | 0.634 |
| cfr_scratch | 0.9850 +/- 0.0009 | 0.069 | 0.036 | 0.708 | 0.587 |

## Reading, in order of importance

1. **The strongest relevant baseline is Model A alone, and the complete system does not beat it.** `cfr_adaptive` minus `single_a` is
   -0.0112 (negative in every seed, bootstrap [-0.0190, -0.0042]). Every hybrid variant is between 0.979 and 0.985, below A alone at 0.994.
   Part of that gap is not the pair's fault: A is near the ceiling of this benchmark, and every pair variant averages in a much weaker B (`06`).
   But it means the plain claim "the hybrid is better than a visual model" is not supported here.
2. **Adaptive CFR against its controls: inconclusive.** The pre-declared rule gives *not supported* (`09`): ahead of `ml` by +0.0034 in all
   three seeds but with an interval of [-0.0012, +0.0066]; behind `ml_hem` by -0.0019 in all three seeds with an interval of [-0.0089, +0.0029].
   It is also not better than the sudden switch (-0.0007), the fixed-epoch switch (-0.0012) or CFR from the first step (-0.0026, interval excludes zero).
3. **The ordering among the pair variants is small.** All seven pair variants lie within 0.0059 AUROC of each other, and the largest
   seed-to-seed sd is 0.003. With 417 test images and three seeds, effects below roughly 0.005 are not resolvable here.
4. **Mutual learning alone did not help.** `ml` is below `indep` by 0.0050 (interval [-0.0118, -0.0004]). Hard-example mining added on top recovers 0.0053 (interval [-0.0001, +0.0114], positive in every seed).
   The most consistent positive effect in the whole study is therefore hard-example weighting, not partner alignment or the CFR challenge.
5. **Unseen-tool removal (secondary):** the hybrids are 6 to 9 points of AUROC below A alone (cfr_adaptive minus single_a -0.082, negative in all three seeds).
   Among the hybrids, no variant is consistently better; seeds disagree in sign.
6. **Calibration and false positives** vary with no pattern strong enough to matter: ECE 0.046 to 0.083 and authentic FPR 0.029 to 0.059 across pair variants (single A has a much higher ECE, 0.140, at a lower FPR).
7. **Localisation:** pair variants reach Dice 0.59 to 0.63, below A alone (0.66), with the pairs' mask being an average of both decoders' outputs.

## Is the comparison fair

- Same data, splits, schedule and optimiser for all variants; no variant-specific tuning. I tried three learning rates only for `indep` seed 0 (validation fused AUROC 0.974 to 0.980, a spread within noise), and kept the default for all.
  The CFR hyperparameters (beta = 2, floor = 0.25, w_max = 4) and controller thresholds were set before results and not changed.
- The pairs get about 1.3x the parameters and 2.1x the compute of A alone (5.6 M and 3.5 GMACs against 4.2 M and 1.6 GMACs, 384 px input). The compute-matched control is two Model A runs averaged: 0.9951 (per seed pair 0.9961, 0.9933, 0.9960), against 0.9936 for one A. It beats every A+B pair variant, so the extra compute is better spent on a second A than on the forensic branch.
- Model A starts from ImageNet weights and Model B from random initialisation. That is a design choice of the project, but it is a large part of why B is weaker, and it limits what mutual learning can transmit.
- 12 epochs is short, and the adaptive controller only fires at epoch 6 to 8 (`08`).

## What this supports and what it does not

Supported: all pipelines train and evaluate on real held-out data; A is a strong detector for the manipulations it has seen; B learns real signal, but weaker;
mutual learning keeps the models aligned without collapse; hard-example weighting gives a small, fairly consistent gain over plain mutual learning.
Not supported: that the hybrid beats the visual model, that alignment makes B better, that adaptive CFR beats its controls, or any claim about AI-edited or AI-generated images.
Not tested: whether any of this holds on a benchmark where A is not at ceiling.


## Update: variants added in stage 3

Source: `reports/evidence/extra_analysis.md` (section 6) and `reports/evidence/comparison.md`. Same data, splits, schedule and seeds. Mean over 3 seeds.

| variant | primary AUROC | FPR authentic | removal AUROC | Dice | note |
|---|---|---|---|---|---|
| single_a | 0.9936 +/- 0.0027 | 0.029 | 0.795 | 0.663 | strongest single model |
| two-Model-A ensemble (post hoc) | 0.9951 | n/a | n/a | n/a | compute-matched control (variant J) |
| joint_concat | 0.9929 +/- 0.0025 | 0.031 | 0.766 | 0.608 | learned fusion; level with A (+0.0005, [-0.0009, +0.0021]) |
| joint_gate | 0.9908 +/- 0.0008 | 0.029 | 0.791 | 0.611 | gate puts 99 percent weight on A |
| indep (fixed average) | 0.9841 +/- 0.0029 | 0.033 | 0.704 | 0.617 | |
| ml, weight 0.25 | 0.9858 +/- 0.0019 | 0.031 | 0.728 | 0.617 | better than `ml` weight 1 by +0.0067 |
| ml, weight 1 (headline) | 0.9791 +/- 0.0023 | 0.053 | 0.716 | 0.616 | |
| ml, weight 4 | 0.9568 +/- 0.0068 | 0.089 | 0.705 | 0.611 | clearly worse |
| cfr_scratch, no alignment | 0.9874 +/- 0.0006 | 0.040 | 0.688 | 0.603 | |
| cfr_scratch | 0.9850 +/- 0.0009 | 0.036 | 0.708 | 0.587 | |
| cfr_adaptive (proposed system) | 0.9824 +/- 0.0010 | 0.053 | 0.713 | 0.624 | |

What changes in the reading above:

1. **The best A+B system is now level with A alone, not below it.** Learned concatenation (0.9929) is statistically indistinguishable from A (0.9936), and clearly above the fixed average. This corrects the strength of point 1 above: the earlier gap
   was partly the cost of fixed-weight averaging. The fusion still gets nothing from B (`06` update 2: destroying B changes nothing), and the two-Model-A ensemble beats every A+B system.
2. **The adaptive CFR system (0.9824) is below the learned-fusion systems (0.9908, 0.9929) and below A.** The main comparison against the "strongest relevant baseline" is therefore: strongest baseline = Model A (or two A), proposed system = 1.1 points of AUROC lower (interval excludes zero, `comparison.md`).
3. **Competition without alignment is nominally better than with it** (`09` update); the alignment term adds nothing measurable. High alignment weights are harmful (`07` update).
4. The pre-declared decision rule for CFR (adaptive against `ml` and `ml_hem`) is unchanged and is not re-run with the new variants. It still gives inconclusive. The weight-0.25 mutual-learning result is a post-hoc finding and is reported
   next to the rule, not substituted for it: `ml` at weight 0.25 is ahead of `cfr_adaptive` by 0.0034 (per seed +0.0051, +0.0001, +0.0049), a descriptive number I have not bootstrapped. It means a tuned mutual-learning control is at least as strong as the proposed system on this benchmark.

Not done: a jointly trained, parameter-matched control (for example a single wider network with the pair's parameter count), and non-hard-example fusion ablations. Listed in `15`.

Zero-shot AI-edited and AI-generated images (external, test only, `11`): every variant is close to chance (AUROC 0.51 to 0.62); the proposed system 0.554 and 0.594. No variant is a useful detector there, so
nothing in this comparison supports a claim about AI-edited or AI-generated images, for CFR or for any control.
