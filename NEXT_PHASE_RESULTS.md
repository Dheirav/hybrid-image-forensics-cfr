# Next phase results (interim, written 2026-10-19 when the work was stopped)

Stage 1 (verification, diagnosis, tool-disjoint test of all 42 existing runs) is finished. Stage 2 (retraining the headline comparison on a harder
benchmark, `forge2`) was **stopped by request after one of 36 runs**, so there is no Stage 2 result and nothing below claims one.
Plan and pre-declared criteria: `NEXT_PHASE_PLAN.md`. Raw tables: `reports/evidence/next_phase/`.

## Experiment A: baseline verification

All six COCO-Forge numbers in the brief match the stored evidence to four decimals (`baseline_verification.md`): A 0.9936, B 0.9108, concat 0.9929, mutual learning 0.9791,
mutual learning plus hard-example mining 0.9844, adaptive CFR 0.9824. Each is the three-seed mean of the primary metric (test AUROC, authentic + splice + copy-move).
The external figures 0.554 and 0.594 are the AUROC of seed-averaged scores for `cfr_adaptive`; the mean of the per-seed AUROCs is 0.552 and 0.591. They are on a different dataset
(400 real against 400 AI-edited, or against 400 AI-generated, 512 px, zero-shot) and are not comparable with the COCO-Forge numbers.
Re-scoring five checkpoints through a second code path reproduced the stored probabilities exactly (0 label flips of 500). Retraining `single_a` seed 0 gave 0.9961 against 0.9966,
inside the seed spread (sd 0.0027).

## Experiment B: shortcut and leakage checks, and a new test set

`forge_hard` (built by `make_hard.py`) applies five editing tools to the same 250 test sources and 250 validation sources, one authentic crop per source,
so every tool is compared against identical authentic images. `audit_hard.md`: 0 unreadable files, 0 source overlap with COCO-Forge training, 0 near duplicates against training authentic crops,
JPEG quality independent of label (p = 0.30), file size and quality at chance (0.46 to 0.54 AUROC) for every tool, edits stay inside their masks (0.1 percent of pixels change far outside).
A bug was caught while building it: OpenCV's `seamlessClone` alters about as many pixels outside the mask as inside it, so the Poisson tools now take only the masked pixels from the clone.

## Diagnoses

**D1. Why is Model A so strong on COCO-Forge?** It reads the hard edge of the training tool, not scene content and not a metadata shortcut.
Rebuilt training tools score the same as before (A 0.994 on `splice_feather`, 0.990 on `copy_move_feather`; COCO-Forge test 0.994), so the new pipeline is comparable.
The same sources edited with Poisson blending (same scene semantics, no hard seam) give A **0.516 and 0.514, chance**. A content-preserving re-paste of an object over itself (`self_resample`, no new scene content)
still gives 0.895. By the pre-declared criteria A is seam-driven (drop of 0.48, far above 0.10) and is not content-driven (0.895 is above 0.80). The earlier concern that A mostly detects "an object that does not belong" is not supported: it fails on exactly the edits that keep the semantics and hide the seam.
A simple hand-built patch-statistics baseline also sees Poisson edits only weakly (0.55 to 0.57) against 0.84 to 0.85 for the feathered tools, so those edits are subtle, not mislabelled.

**D2. Why is B weaker and why does fusion not help?** B underfits: its training accuracy is 0.77 to 0.84 against 0.997 for A, so it is limited by what it can learn, not by overfitting.
It holds no information A lacks. A logistic stack fitted on validation puts weight 0.5 on A and between -0.01 and 0.11 on B, and its test AUROC differs from A alone by -0.0004 [-0.0013, +0.0003] (seen tools),
with no interval clearly excluding zero on any of the seven evaluation sets (one borderline case, `copy_move_feather`, +0.0008 [+0.0000, +0.0017], is too small to matter), including both Poisson sets and `self_resample`. Linear probes on A embeddings against A+B embeddings agree. On the images A gets wrong, B fixes 6 to 7 per seed while A fixes 61 to 66 of B's errors.
Among images where A is unsure, B's AUROC is 0.41 to 0.54, which is chance. B is also at chance on both Poisson tools (0.49, 0.50).

**D3. Different evidence or different predictions?** Different representations, not complementary ones. Linear CKA between A and B trained alone is 0.56, against 0.88 and 0.92 between seeds of the same architecture, and error indicators correlate at only 0.2.
So they differ, but B's different errors are noise, because B is wrong far more often. Alignment raises CKA modestly (0.63 for `ml`, 0.66 for `cfr_adaptive`, 0.80 for weight 4); `ml_hem` is lower (0.34).

**D4. Why does CFR not beat hard-example mining?** On the trained checkpoints (`diag_cfr.md`):
- Model A is challenged on 0.3 percent of training images under `cfr_adaptive`, so CFR's weights for A are uniform (effective sample size 0.999). Competition only ever acts on B, because A beats B everywhere.
- For B, the CFR and hard-example weights are almost the same ranking (Spearman 0.98 for `cfr_adaptive`, 0.96 for `cfr_scratch`) with similar spread (effective sample size 0.93 against 0.92).
- Alignment: A is pulled toward B only where B is more right (9 percent of images), so A's alignment weight sits near its floor (0.31); B is pulled toward A on 91 percent.
- The adaptive controller spends 2 epochs at full CFR (5 with any CFR weight) of 12.
- The pre-declared reading ("effectively hard-example mining if the Spearman correlation is above 0.9 for both models") is met for B and not formally for A, because A's two weight vectors are both near-uniform so their ranks are noise. The honest summary is that CFR differs from hard-example mining only in an inert term for A and in the alignment gate.
- Gradient flow was checked by the existing tests (partner detached, ground-truth cross-entropy always present) and shows no defect.

**D5. Why does it fail on AI-edited and AI-generated images?**
- Not a shift in what "real" looks like: false-positive rate on external real photos with one window is 0.025 to 0.062, the same as on COCO-Forge authentic (0.029 to 0.053).
- Not edit size: AI-edit AUROC is 0.55 to 0.61 for small, medium and large regions alike.
- Not window aggregation: tiled max and a single central window agree within 0.02.
- The features are not empty: leave-one-generator-out linear probes on frozen embeddings reach 0.81 (edited) and 0.82 (generated) for the fine-tuned Model A, against 0.57 and 0.59 from the trained head. The ImageNet-pretrained EfficientNet that was **never fine-tuned** probes higher still (0.93 and 0.89), and Model B's features are weak (0.64 and 0.68).
  So fine-tuning on COCO-Forge narrowed the features toward seam detection and lost separability that the generic features had. This meets the pre-declared "head or domain shift" reading. Caveat: the probe is trained on external labels and uses folder names as labels, so it measures separability that may be content or style rather than forensic traces, and it says nothing about detection ability.

## Updated comparison (three seeds each, `reports/evidence/next_phase/hard_auroc.md` and `hard_other_metrics.md`)

AUROC against the same 250 authentic test images, threshold 0.5 for the other metrics. Left block: COCO-Forge primary metric. Right block: forge_hard tools no model has seen.

| variant | COCO-Forge primary | splice_feather (control) | copy_move_feather (control) | splice_poisson | copy_move_poisson | self_resample |
|---|---|---|---|---|---|---|
| Model A alone | 0.9936 | 0.994 | 0.990 | 0.516 | 0.514 | 0.895 |
| Model B alone | 0.9108 | 0.913 | 0.883 | 0.492 | 0.497 | 0.838 |
| learned concat fusion | 0.9929 | 0.994 | 0.987 | 0.502 | 0.511 | 0.890 |
| mutual learning (weight 1) | 0.9791 | 0.979 | 0.967 | 0.493 | 0.496 | 0.885 |
| mutual learning (weight 0.25) | 0.9858 | 0.985 | 0.977 | 0.501 | 0.503 | 0.889 |
| mutual learning + hard-example mining | 0.9844 | 0.980 | 0.973 | 0.495 | 0.494 | 0.881 |
| adaptive CFR | 0.9824 | 0.985 | 0.974 | 0.495 | 0.496 | 0.892 |

On the Poisson tools every variant has recall of 3 to 13 percent, balanced accuracy 0.48 to 0.51 and Dice of 0.03 or less, which is a detector that has not seen the edit, not one that is close. Authentic false-positive rate at 0.5 is 0.028 (A), 0.032 (concat), 0.033 (mutual learning 0.25), 0.035 (with hard-example mining), 0.061 (adaptive CFR), 0.163 (B).
Using a validation-chosen threshold instead of 0.5 changes macro-F1 by at most about 0.02, which is the size of the gaps between the pair variants, so their order is not stable. The pair variants trail A by up to 0.03 on the two feathered tools (concat is level) and are within 0.015 of it on `self_resample`.

## Error analysis, main failure modes

1. **Tool specificity.** Every trained variant detects the feathered paste it was trained on and is at chance on the same edit made with Poisson blending. The 0.99 headline measures one tool.
2. **Unseen removal and re-paste.** Removal 0.71 to 0.80, content-preserving re-paste 0.84 to 0.90.
3. **Blur.** Sigma-1 blur flags 59 percent of authentic images (report 11).
4. **AI-edited and AI-generated images.** Near chance zero-shot (0.55 to 0.61), although generic pretrained features separate them with a probe.
5. **Model B.** Underfits (training accuracy 0.77 to 0.84), is wrong far more than A, and is not used by any learned fusion.

## Decision

On the evidence that exists: **do not keep the current CFR formulation as a claimed contribution.** It has not demonstrated its benefit. It does not beat mutual learning at weight 0.25 or hard-example mining on COCO-Forge, CFR acts only on the weaker model, its weights are nearly those of hard-example mining there,
and none of the 14 variants beats Model A alone on any tool. Revising CFR is not worth doing until two things are settled: whether Model B can learn anything A cannot, and whether any method gains on a benchmark with headroom. Neither question is answered yet, which is why Stage 2 matters.
The better-supported direction today is a strong visual model trained on a diverse set of editing tools, with the forensic branch and CFR as ablations to be earned back. This is a recommendation from Stage 1, not a Stage 2 finding.

## The exact next experiment

Run Stage 2 as planned: `tools/queue_forge2.sh` (36 runs, about 3 hours on a free machine). It trains on four tools including Poisson blending, tests on seven tools of which three were never seen, and applies the rule in `NEXT_PHASE_PLAN.md` to
`single_a`, `ml_lam0.25`, `ml`, `ml_hem`, `cfr_adaptive`, `joint_concat` over five seeds. The reason is that on forge_hard the headroom is visible for the first time (0.51 on Poisson, 0.89 on re-paste), so it is the first benchmark where a
difference between methods, or a benefit from Model B, can show up. First check, after the first two runs, that Model A is not already at ceiling on the four seen tools; if it is, the benchmark needs harder tools before any comparison.

## What was and was not done

- Done: Experiments A, B, C and D as diagnoses on the existing runs, Experiment E stage 1 and Experiment F (`diag_external.md`) on the existing runs, five diagnoses, the new tool-disjoint test set, per-tool metrics for all 42 runs.
- Not done: Experiment E stage 2 and the comparison it would give (`runs_f2/` holds one finished `single_a` seed 0 that was never analysed, and an interrupted `ml_lam0.25` seed 0 that is not usable).
  No paired CFR-against-mutual-learning test on forge_hard was run, because those models were not trained on its tools.
- Memory: `train.py` kept every test mask in float32, which needed more than 6 GB for the 2000-image forge2 test split on a 7.8 GB machine. It now stores float16 masks and uint8 ground truth in `predict` (a Dice can change only when a probability is within 5e-4 of 0.5). The 42 earlier runs used the old float32 code.
