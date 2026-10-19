# 09 CFR: definition and validation

Part 1 (definition) was fixed in `src/hybrid_forensics/losses.py` (`cfr_terms`) and tested
(`tests/test_cfr_terms.py`) before any CFR run finished. Part 2 (results) follows the definition and was written after all 24 runs finished.

## Part 1. Definition

**Name.** Competitive Forensic Reasoning here is *competitive hard-example correction with ground-truth-gated
alignment*. It is not a formal game with an equilibrium concept; calling it one would claim more than the objective supports.

Notation: models M in {A, B}, partner P is the other model, y the ground-truth label, p_t^M(i) the probability model M
gives to the true class of sample i (detached when used as a weight), CE_i the per-sample cross-entropy, and
KL_i(P || M) the temperature-T KL from the partner's softened distribution to the model's, times T^2 (partner detached).

**Competitive challenge.** The partner challenges M on sample i when it is more right about the ground truth:
c_M(i) = max(0, p_t^P(i) - p_t^M(i)). A challenge exposes a weak prediction of M. It is evaluated against the label,
not against disagreement: two models that disagree where both are wrong produce no challenge.

**Difficult-sample selection.** Samples with large c_M(i). The weight is
w_M(i) = clip(1 + beta * c_M(i), max 4), then divided by its batch mean so the average weight is 1 and the weighting cannot
rescale the loss or reward overconfidence (beta = 2).

**Correction.** M is trained on the same ground-truth cross-entropy, but challenged samples count more.
The label, not the partner, is what corrects: a challenge only says where to look.

**Alignment retained, and gated by ground truth.** The alignment weight for M on sample i is
a_M(i) = floor + (1 - floor) * g_M(i), with g_M(i) = p_t^P(i) if p_t^P(i) > p_t^M(i) else 0, and floor = 0.25.
M is pulled strongly toward the partner only where the partner is more correct; where the partner is wrong the pull
falls to the floor. This is how Model B stays aligned with Model A without copying A's mistakes. The floor keeps the alignment loss nonzero everywhere.

**Per-model objective.** L_M = mean_i [ w_M(i) * CE_i + lambda * a_M(i) * KL_i(P || M) ] + segmentation loss, lambda = 1.
During the transition the loss is blended with the mutual-learning loss by gamma in [0, 1]:
L_M = (1 - gamma) * L_M^ML + gamma * L_M^CFR, where L_M^ML has weights 1 and a_M = 1.

**Degeneracy guards.** Ground-truth CE is always present; weights are clipped, mean-normalised and use detached probabilities;
alignment never drops below the floor; the controller and the logs track disagreement, KL, correlation and each
model's own AUROC, so collapse (identical models) or divergence (permanent disagreement) is visible in the run history.

**What the two controls change.** `ml_hem` weights samples by the model's *own* error (1 - p_t^M) and aligns on every sample
(ungated). So CFR against `ml_hem` isolates two things: partner-relative challenge instead of self-hardness, and
ground-truth-gated alignment instead of ungated alignment. `ml` has neither.

## Part 2. Results

Decision rule (declared in `02` before any CFR result existed): "adaptive CFR improves on mutual learning and on hard-example
mining" is **supported** only if the per-seed paired AUROC difference is positive in every seed for `cfr_adaptive` against each of
`ml` and `ml_hem`, the 95 percent paired bootstrap interval excludes 0, and ECE and authentic false-positive rate are no worse.
A negative result of the same strength means unsupported; anything else is inconclusive. Primary metric: fused test AUROC on
authentic + splice + copy_move, three seeds. Full table: `reports/evidence/comparison.md`.

| variant | primary AUROC (mean +/- sd) | test ECE | FPR authentic | removal AUROC | Dice |
|---|---|---|---|---|---|
| ml | 0.9791 +/- 0.0023 | 0.061 | 0.053 | 0.716 | 0.616 |
| ml_hem | 0.9844 +/- 0.0011 | 0.083 | 0.029 | 0.723 | 0.621 |
| cfr_adaptive | 0.9824 +/- 0.0010 | 0.057 | 0.053 | 0.713 | 0.624 |

| comparison | per-seed difference | mean | 95% paired bootstrap | positive in all seeds |
|---|---|---|---|---|
| cfr_adaptive - ml | +0.0024, +0.0023, +0.0055 | +0.0034 | [-0.0012, +0.0066] | yes |
| cfr_adaptive - ml_hem | -0.0027, -0.0020, -0.0011 | -0.0019 | [-0.0089, +0.0029] | no (negative in all) |

**Against `ml`:** CFR is ahead in all three seeds and its ECE is slightly lower (0.057 against 0.061) with the same false-positive
rate, but the interval includes zero, so the rule's second condition fails. Inconclusive, leaning positive.

**Against `ml_hem`:** CFR is behind in all three seeds, and its false-positive rate is higher (0.053 against 0.029) though its ECE is lower
(0.057 against 0.083). The interval includes zero, so this is not "unsupported with the same strength" either. Inconclusive, leaning negative.

**Outcome under the pre-declared rule: inconclusive, so the hypothesis is not shown.** "Supported" needs both comparisons to be
positive and clear of zero, and neither is; the two also point in opposite directions. The apparent gain over `ml` is matched by hard-example mining alone with a simple self-weighting,
which suggests that whatever help CFR gives over plain mutual learning is the hard-example weighting and not the partner-relative
challenge or the gated alignment. I would not call this a refutation: the intervals are wide enough (about +/-0.005) that a true effect of
0.003 to 0.005 AUROC is compatible with the data, and the benchmark has little headroom on seen tools.

**Corrections that damage correct predictions** (the brief asks for this). Counting test images where one system is right and
the other wrong, per seed, on the 417 seen-tool images: CFR breaks 7 / 4 / 2 predictions that `ml` got right and fixes 11 / 12 / 11 that `ml`
got wrong. Against `ml_hem` it breaks 13 / 6 / 3 and fixes 15 / 12 / 6. So net movement is small and mildly favourable
against `ml`, and about even against `ml_hem`. It is not a case of CFR damaging a large number of correct predictions.

**Unseen-tool removal (secondary, where headroom exists).** CFR minus `ml` is +0.011, -0.005, -0.017 over seeds (mean -0.004); CFR minus
`ml_hem` is +0.014, -0.047, +0.000 (mean -0.011). Seeds disagree, so there is no evidence either way of an out-of-distribution benefit.

**Alignment retained.** Under `cfr_adaptive`, B's logit correlation with A is 0.84 and their disagreement stays around 17 percent (`07`),
so B stays aligned with A without collapsing into it, and the alignment loss was nonzero by construction (floor 0.25). That B stayed aligned is not the same as B
becoming useful: B's own AUROC under CFR is 0.910, no better than B trained alone (0.911).

## Update: alignment-preserving against non-aligned competition, and the brief's "verify that CFR does not" list

### Aligned against non-aligned competitive training

`cfr_scratch_noalign` is `cfr_scratch` (CFR weights from the first step) with the alignment weight set to 0, so the partner-relative challenge weights act
and the KL term does not (`runs_extra/cfr_scratch_noalign_s*`, three seeds). It isolates what alignment adds to competition.

| variant | primary AUROC (mean +/- sd) | per seed | removal AUROC | FPR authentic | test ECE |
|---|---|---|---|---|---|
| cfr_scratch (aligned) | 0.9850 +/- 0.0009 | 0.9859, 0.9850, 0.9840 | 0.708 | 0.036 | 0.069 |
| cfr_scratch_noalign | 0.9874 +/- 0.0006 | 0.9880, 0.9874, 0.9869 | 0.688 | 0.040 | 0.051 |

Aligned minus non-aligned: -0.0021 (per seed -0.0021, -0.0023, -0.0029; interval [-0.0057, +0.0012]). **Preserving alignment did not help.** The non-aligned competitive
variant is nominally ahead in all three seeds and better calibrated, and is level in the statistical sense. Together with the alignment-strength result in `07` (more alignment is worse above weight 0.25),
this says the alignment term in the objective is not what produces any benefit here. The requirement that B stays aligned with A is met by the aligned runs (`07`), but it is not buying accuracy.

### Checks from the brief ("verify that CFR does not ...")

| check | evidence | result |
|---|---|---|
| increase confidence without improving correctness | mean confidence max(p, 1-p) against accuracy, test seen tools, 3 seeds: indep 0.858 / 0.958, ml 0.780 / 0.928, ml_hem 0.754 / 0.936, cfr_adaptive 0.834 / 0.945 | no. Every variant is under-confident (confidence below accuracy); CFR is nearer the independent pair than `ml` is. |
| cause permanent disagreement | disagreement on validation before and after the switch (`08`) | no. It falls to 0.17 to 0.20 in every run. |
| destroy alignment | logit correlation 0.84 and probability gap 0.18 for `cfr_adaptive` (`07`) | no. Alignment is about the same as under `ml` (correlation 0.84 against 0.84; probability gap 0.18 against 0.15), but nothing collapses or diverges. |
| make the models collapse | no run has identical predictions: disagreement never below 0.15 | no |
| improve only training accuracy | validation AUROC at the last epoch (0.974) and test AUROC (0.982) agree; train loss is not comparable across variants because the weighted terms differ | no sign of an overfit gap, but I have no training-set AUROC to compare directly |
| introduce data leakage | CFR uses the same training data and labels as every other variant; the weights are computed from training batches only | none |
| depend on a single random seed | 3 seeds for every compared variant, per-seed differences reported | the result is not seed-specific: it is consistently small |

### The same four-line summary, after the extra variants

Mutual learning with a large weight hurts; with a small weight it is level with no alignment. Hard-example weighting gives a small gain. Partner-relative CFR weights give a gain of the same size as
self-hardness weights, and removing the alignment term from CFR does not hurt. Neither of the two things that make CFR different from hard-example mining (partner-relative challenge, gated alignment) has shown a measurable benefit on this benchmark.
