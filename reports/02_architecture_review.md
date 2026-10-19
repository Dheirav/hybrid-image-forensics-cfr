# 02 Architecture review

This is a desk review of the design in `PROMPT.md`. No code exists to review, so nothing here is a result. Each point is a claim about the design that the later phases should test.

## Is the flow internally consistent?

The flow is: image, preprocessing, visual model and forensic model, mutual learning, validation-triggered transition, competitive correction with alignment retained, fusion, classification and localisation, confidence.

It is consistent on paper, with three gaps.

1. **Fusion is used before it is defined.** Phase 4 compares concatenation, fixed-weight fusion and learned gating, but Phase 5 to 7 train two models that each output class probabilities. A feature-level concatenation model has one head, so mutual learning between "A" and "B" only makes sense for the prediction-level variants. The brief has to say which fusion the CFR system uses, otherwise variants D to G are not comparable.
2. **Where localisation lives is not stated.** If the decoder sits on Model A only, Model B never sees mask supervision and alignment says nothing about localisation. If it sits on the fused features, it is outside the mutual-learning loop. Pick one and say so.
3. **"Confidence and reliability" has no training signal.** Softmax confidence after KL alignment is pulled towards the partner's confidence, which can make it less informative. Reliability should come from calibration on validation data plus selective prediction, as Phase 9 says, not from a separate learned head.

## Components that look unnecessary or over-built

- **Learned gating (4.5)** on a small dataset adds parameters that the comparison then has to be matched against (variant J). Keep it as optional.
- **A formal game-theoretic CFR objective.** The brief itself allows a "competitive hard-example correction" fallback. Given a final-year timeline, that fallback should be the default design, named honestly.
- **Ten variants (A to J) at three seeds each** is 30 training runs plus controls. On an 8 GB laptop GPU shared with other jobs, that is not realistic at full size. See the risk register.

## Things that cannot be trained or verified with the stated labels

- **Three classes (authentic, edited, generated) with one mask type.** Only edited images have masks. Segmentation loss must be masked per sample, and Dice/IoU can only be reported on edited images. Generated images have no meaningful "manipulated region".
- **Conventional manipulation** (splicing, copy-move, removal) is a fourth data category in the brief. A three-class head would have to merge it with something. This needs a decision before the label schema is fixed.
- **"Which manipulation type"** (stated aim) is only as good as the type labels in the data. Without per-sample tool labels it reduces to the class split.

## The mutual-learning formulation

`L_A = CE(p_A, y) + lambda * KL(p_B^T || p_A^T)` is standard deep mutual learning. Points to settle before coding:

- KL direction and which side is detached. In `KL(p_B^T || p_A^T)` for the loss of A, `p_B` must be a detached target. State this in code and test it.
- Temperature `T` needs the usual `T^2` rescaling so the KL term does not vanish with `T`.
- Heterogeneous branches have different convergence speeds. A forensic branch on SRM residuals often learns slowly at first, so early alignment may drag it towards the visual branch's shortcuts instead of letting it find its own signal. The brief's own test "does Model B retain useful forensic information" is the right check, and a warm-up with `lambda = 0` is a cheap control.

## CFR: what has to be pinned down

The brief asks for a mathematical definition before implementation. Minimum content:

1. A **challenge** is a sample where the partner is wrong and the model is right, or the models disagree. Selection should use ground truth, otherwise it is just disagreement maximisation, which the brief rules out.
2. A **valid correction** is a gradient step on samples the model currently gets wrong, weighted by how confidently its partner got them right.
3. **Alignment retained** means the KL term stays on, but is down-weighted on samples where the partner is wrong (ground-truth gated). This is the mechanism that answers the critical requirement that B does not copy A's mistakes.
4. **Degeneracy guards:** a floor on agreement, a ceiling on disagreement, and a check that predictions are not constant.

Written this way, CFR is close to **ground-truth-gated mutual learning with hard-example reweighting**. That is a legitimate method, but it overlaps with variant H (mutual learning plus hard-example mining). The honest research question is whether the gate adds anything beyond mining, and the design should expect that the answer may be no.

## Claims the system cannot support

- It cannot show an image is authentic. The output is a model prediction with limits.
- It cannot be expected to detect every AI edit, nor generalise to generators not represented in training. That is why a held-out generator split is a hard requirement.
- A working demo is not evidence that CFR works.

## Compute fit (RTX 4060 8 GB, shared)

EfficientNet-B0 plus a small forensic CNN at 256 to 512 px fits in 8 GB with a modest batch and mixed precision. CLIP-L at 448 px, as used in the neighbouring repo, does not train comfortably next to other jobs. With roughly 6 GB of VRAM usable and most CPU cores taken by other processes, plan for the smallest backbone, and for pilots with one seed before spending on three.

---

# Addendum, 2026-10-18: scope decisions, traceability and the pre-declared decision rule

Written before any model was trained on real data, so the rule below cannot be fitted to the results.

## Scope that the available data supports

Data: COCO val2017 photographs with synthetic conventional manipulations built by `make_dataset.py`
(splice, copy-move, object removal), exact masks, source ids, source-level splits. Network speed (about
100 KB/s per connection) ruled out CASIA/DEFACTO-sized downloads and there is no generator for AI edits on this machine.

Consequences, stated as claims the project may and may not make:

- May claim: binary detection (authentic vs manipulated) and localisation of synthetic conventional edits, with a source-disjoint test split, and a cross-tool test on removal, which is never seen in training.
- May not claim: anything about AI-edited or fully AI-generated images, real-world forgeries, or social-media-style processing beyond the degradations tested.
- Because the edits are synthetic, a detector can reach high accuracy by finding the pipeline's own seams (paste feathering, inpainting texture). That is a property of this benchmark, and is why the unseen-tool split and the degradation tests matter.

Label schema: 2 classes (0 authentic, 1 manipulated) with manipulation type kept as metadata. Mask decoder: one per branch, fused at the output by the same probability average, so the decoders stay inside the mutual-learning loop.

## Traceability

| objective | implementation | test | metric | acceptance criterion |
|---|---|---|---|---|
| Detect manipulated images | `VisualBranch`, `ForensicBranch`, fusion | `train.py`, held-out test | AUROC, macro-F1 | fused test AUROC clearly above the metadata-only and 32 px patch baselines, and above 0.5 with seed spread excluded |
| Forensic features give signal | SRM/Bayar/DCT in Model B | single-B run; B-only vs A-only | AUROC | B above 0.5 on val and test, and above the patch baseline |
| Localise manipulated region | per-branch decoder, `masked_seg_loss` | Dice/IoU on manipulated test images | Dice, IoU | reported as is; no threshold set, since there is no prior for this data |
| Hybrid helps | probability fusion | fused vs best single branch | paired AUROC difference over seeds | positive in every seed to call it a gain |
| Mutual learning helps | `ml` vs `indep` | same seeds | paired difference | positive in every seed |
| Model B stays aligned with A, without copying A's errors | KL alignment, GT-gated in CFR | agreement, KL, complementary-error rate, B accuracy on samples A gets wrong | disagreement in (0.02, 0.35), B-alone AUROC not below `single_b` by more than the seed std | both hold |
| Adaptive transition works as specified | `TransitionController` | log of eligibility, trigger, gamma per epoch | trigger epoch, before/after val metrics | trigger recorded or "not eligible" recorded with reasons; never forced |
| CFR adds value | `cfr_adaptive` vs `ml`, `ml_hem` | paired seeds, bootstrap over test images | paired AUROC difference | see decision rule |
| Generalises | `ood_removal` split, degradations | cross-tool and robustness runs | AUROC, F1, FPR on authentic | reported, no pass mark; a large drop is a finding |
| Calibrated, reportable confidence | temperature fitted on val | ECE, NLL, Brier on test | ECE | reported before/after; fitted on val only |
| End-to-end app | `infer.py` | 12 case matrix, 384 px equivalence test | completion, errors, seconds | all failure cases rejected with a message, none crash |

## Primary metric and decision rule (fixed now)

Primary metric: AUROC of the fused prediction on the **test** split restricted to authentic, splice and copy-move images (the tools seen in training), averaged over seeds. The test split also holds removal images (a third of its manipulated samples, never trained on); they are reported separately with `ood_removal` as the unseen-tool result. This restriction was added on 2026-10-18 after the dataset was built and before any model was trained, because the builder puts removal in the test split.
Secondary: macro-F1 at threshold 0.5, authentic false-positive rate, Dice on manipulated test images, AUROC on `ood_removal`.

"Adaptive CFR improves on mutual learning and on hard-example mining" is declared **supported** only if,
for the headline comparison (`cfr_adaptive` against each of `ml` and `ml_hem`):
1. the per-seed paired AUROC difference is positive in every seed, and
2. the 95 percent paired bootstrap interval over test images, computed from the seed-averaged probabilities, excludes 0, and
3. the gain is not paid for by worse calibration (ECE) or a higher false-positive rate on authentic images.

If the sign is negative with the same strength, the verdict is **unsupported**. Anything else, including a
positive mean with an interval that includes 0 or a seed that disagrees, is **inconclusive**, not a win.
The number of seeds is small, so inconclusive is the likely outcome for effects below roughly 0.01 AUROC.

## Traceability outcome (added after the experiments; the table above and the decision rule are unchanged)

| objective | outcome | where |
|---|---|---|
| Detect manipulated images | met on seen tools: A 0.994, pair variants 0.979 to 0.993 test AUROC; far above the metadata (0.55) and patch (0.51) baselines | `05`, `12` |
| Forensic features give signal | met: B 0.911 on seen tools, 0.14 authentic FPR; weaker than A, not complementary | `05`, `06` |
| Localise manipulated region | reported: Dice 0.66 (A), 0.49 (B), 0.59 to 0.63 (pairs); poor on unseen removal | `05`, `12` |
| Hybrid helps | **not met**: no fusion beats Model A alone; learned concatenation is level with it, fixed average is below it | `06` |
| Mutual learning helps | **not met**: `ml` is below `indep` (-0.0050, interval excludes 0); weight 0.25 is level with `indep`; weight 4 is clearly worse | `07` |
| Model B stays aligned with A without copying A's errors | alignment met (disagreement 0.17, correlation 0.84; B AUROC 0.910 against 0.911 alone). Copying: B corrects 31 percent of A's misses under `cfr_adaptive` against 42 percent unaligned, so a small amount of error sharing remains | `07` |
| Adaptive transition works as specified | met as a mechanism (logged, never forced, triggers at epoch 6 to 8); no benefit over sudden or fixed switch | `08` |
| CFR adds value | **inconclusive** under the pre-declared rule; no benefit over hard-example mining or over non-aligned competition | `09`, `12` |
| Generalises | unseen removal tool: pairs 0.71 to 0.73 AUROC, A 0.795; degradations: a few points for JPEG and resize, collapse under blur; zero-shot AI-edited and AI-generated images: near chance | `11` |
| Calibrated, reportable confidence | partly: ECE 0.058 on test uncalibrated; the validation-fitted temperature made it worse; selective prediction evaluated, no gain from branch disagreement | `11` |
| End-to-end app | met: all listed cases completed or rejected with a message; application output equals evaluation output on 500 images | `10` |
| New: detect AI-edited and AI-generated images | **not met** zero-shot (external set, test only) | `11` |
| New: learned fusion (concat, gate) | implemented, tested, 3 seeds each | `06` |
