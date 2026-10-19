# 06 Hybrid integration

Seed 0, run `runs/indep_s0` (A and B trained side by side on the same batches with no alignment, i.e. the
independent-ensemble control), analysed by `fusion_analysis.py` (`runs/indep_s0/fusion_analysis.json`). One seed.
Fusion weights and the stacker were fitted on validation and scored on test, seen tools.

## Fusion comparison

| fusion | test AUROC (seen tools) | macro-F1 |
|---|---|---|
| A alone | 0.9936 | 0.967 |
| B alone | 0.9009 | 0.814 |
| fixed 0.5 average (what the application uses) | 0.9866 | 0.957 |
| best fixed weight chosen on validation | 0.9936 (w_A = 1.00) | 0.967 |
| logistic stacker on the two logits, fitted on validation | 0.9935 | 0.965 |

The validation-chosen weight is 1.0 on Model A, meaning validation prefers to give Model B no weight at all. The
stacker's coefficients are +0.56 for A and -0.06 for B: it also ignores B. **The hybrid does not outperform the
strongest single branch.** The 0.5 average is worse than A alone (0.9866 against 0.9936), because it mixes in a weaker, noisier model.

## Branch reliance (diagnostic interventions, not proof of usefulness)

Input of one branch replaced by uniform noise at evaluation, test seen tools:

| condition | fused AUROC | A AUROC | B AUROC |
|---|---|---|---|
| both intact | 0.9866 | 0.9936 | 0.9009 |
| A input destroyed | 0.9009 | 0.503 | 0.9009 |
| B input destroyed | 0.9915 | 0.9936 | 0.471 |

Destroying A drops the fused score to B's level, so A carries the system. Destroying B slightly raises the fused score
(0.9866 to 0.9915), so B is contributing noise, not complementary information, in this pairing. Both branches are live
(each collapses to chance when its own input is destroyed), so neither is a dead branch; the issue is that B adds nothing on top of A.

## Not done

- Two-branch feature concatenation with a joint head, and input-dependent learned gating: not implemented. The
  stacker above is a three-parameter, input-independent learned fusion. Given that validation gives B zero weight, a
  gating network would have little to work with here; I have not tested that claim.
- Gradient flow through both branches in a joint model is covered by the unit test (every trainable weight receives gradient)
  and by both branches learning in the pair runs.
- "Checkpoints reproduce the same evaluation predictions": the 384 px test in `tests/test_infer.py` shows the application path
  matches the evaluation path for the same weights. A reload-and-compare of a trained checkpoint against `probs.npz` is
  done in report 10.

## Reading

On this benchmark the forensic branch is weaker than the visual branch and not complementary to it. That does not mean forensic
features are useless in general: the data are 384 px COCO crops edited at one JPEG quality draw, and Model A is near ceiling.
What it means for the project is that any benefit of mutual learning or CFR will have to come from B becoming a better
model, or from regularising A, not from B adding information at fusion time.

## Update: three seeds, and the application check

The fixed 0.5 average of the independent pair scores 0.9841 +/- 0.0029 over seeds 0 to 2 (per seed 0.9866, 0.9808, 0.9848).
Model A alone scores 0.9936 +/- 0.0027 (0.9966, 0.9931, 0.9912). The pair is below A alone in every seed, so the seed-0 result
was not a fluke of one run. The same holds for every trained pair variant in `12` (all of them land at 0.979 to 0.985, below A alone).
I did not repeat the validation-fitted weight and stacker analysis on seeds 1 and 2, so the "validation gives B zero weight" finding
rests on seed 0 only.

The reload check promised above was run: the application path (`infer.Predictor`, tiling code, no calibration) was applied to all 500
test images of `runs/cfr_adaptive_s0/best.pt` and compared with the evaluation `probs.npz` for the same weights.
Maximum absolute difference in fused probability 0.0058 (mean 0.0003), no label changes at 0.5. The residual is bf16
non-determinism between a batch of 16 and a batch of 1. Details in `10`; data in `reports/evidence/app_vs_eval_probs.json`.

## Update 2: learned fusion (concatenation and gating), three seeds

The brief asks for two-branch concatenation and input-dependent gating, which the first pass did not build. Both were added as `FusionHead`
(`src/hybrid_forensics/models.py`) and trained in `runs_extra/joint_concat_s*` and `runs_extra/joint_gate_s*` with the same data, schedule and seeds as the
other variants. Each branch keeps its own cross-entropy and mask loss, and the joint output gets a third cross-entropy term, so neither branch can die behind
the fusion. No alignment term. Gradients reach both branches (unit test `test_fusion_head_gradients_reach_both_branches_and_gate_starts_even`).
`concat` layer-normalises the two embeddings (1280 and 256 wide) and feeds a two-layer classifier. `gate` computes weights over the two branch logits from
both embeddings, initialised at 0.5 and 0.5. Source: `reports/evidence/extra_analysis.md`, section 6.

| fusion | primary AUROC (mean +/- sd) | per seed | removal AUROC | FPR authentic | Dice |
|---|---|---|---|---|---|
| Model A alone | 0.9936 +/- 0.0027 | 0.9966, 0.9931, 0.9912 | 0.795 | 0.029 | 0.663 |
| fixed 0.5 average, independent pair (`indep`) | 0.9841 +/- 0.0029 | 0.9866, 0.9808, 0.9848 | 0.704 | 0.033 | 0.617 |
| concatenation, joint head | 0.9929 +/- 0.0025 | 0.9903, 0.9953, 0.9930 | 0.766 | 0.031 | 0.608 |
| input-dependent gate | 0.9908 +/- 0.0008 | 0.9901, 0.9905, 0.9917 | 0.791 | 0.029 | 0.611 |

Paired differences on the primary metric (descriptive; 95 percent bootstrap over test images on seed-averaged probabilities):

| comparison | per seed | mean | interval |
|---|---|---|---|
| concat - A alone | -0.0062, +0.0022, +0.0018 | +0.0005 | [-0.0009, +0.0021] |
| gate - A alone | -0.0065, -0.0026, +0.0005 | -0.0034 | [-0.0102, +0.0010] |
| concat - fixed average | +0.0038, +0.0145, +0.0082 | +0.0085 | [+0.0033, +0.0150] |
| gate - fixed average | +0.0035, +0.0097, +0.0069 | +0.0045 | [-0.0032, +0.0112] |

**Learned fusion fixes the fixed-average problem but still does not beat Model A.** Concatenation is clearly better than the fixed 0.5 average
(+0.0085, interval excludes zero, positive in all seeds) and statistically level with A alone. The gate is level with A as well. So the
loss in the first pass came from averaging in a weak model with a fixed weight, and a learned fusion removes that cost. It does not turn B into a gain.

**Why: the fusion ignores B.** Branch-reliance test (one input replaced by uniform noise at evaluation, test seen tools, fused AUROC; `runs_extra/*/reliance.json`):

| run | both intact | A destroyed | B destroyed |
|---|---|---|---|
| concat s0 / s1 / s2 | 0.990 / 0.995 / 0.993 | 0.787 / 0.769 / 0.891 | 0.992 / 0.995 / 0.992 |
| gate s0 / s1 / s2 | 0.990 / 0.991 / 0.992 | 0.695 / 0.672 / 0.596 | 0.990 / 0.991 / 0.992 |

Destroying B changes nothing (differences under 0.002), destroying A costs 10 to 40 points. The learned gate gives Model A a mean weight of 0.994, 0.991
and 0.997 across the three seeds, so the gate learned to switch B off. The concatenation head behaves the same way in effect. Read as a diagnostic intervention, not proof about
forensic features in general: on this benchmark, with this B, the optimiser finds no use for B once A is available. That answers the brief's question
"does fusion add information": no.

Items from the brief's list that are now covered: both fusion types implemented and compared; gradients reach both branches; no dead branch in the sense
of a branch with no gradient (the gate's near-zero weight on B is a learned outcome, not a bug: with B's weight fixed at 0.5 B would add noise, as in the first table); class order is the same
as in every other variant (index 1 is manipulated). The reload check against the evaluation probabilities was done for the application path in `10`; it was not repeated
for the joint heads, which the application does not serve.
