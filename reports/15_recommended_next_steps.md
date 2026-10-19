# 15 Recommended next steps

In order. Each step says what question it answers, so the cheapest one that changes a decision comes first.

1. **Does Model A rely on content or on forensic traces? (about half a day, no new data.)** Build a content-matched test: for each spliced test image,
   compare against a composite where the pasted object is a plausible one for the scene versus an implausible one, and against the
   same edit after a global forensic scrub (re-render the whole image through the same JPEG pipeline). If A's
   score depends on the object, not the seam, the 0.994 is a semantic result and the forensic branch is being asked to do a job
   that the benchmark does not need.
2. **A benchmark where A is not at ceiling.** Options, from cheapest: (a) more unseen tools in the existing generator (feathered or Poisson-blended splice, other inpainting, colour-matched pasting) with the same
   leakage and shortcut audits; (b) a public conventional-forgery set (CASIA v2, Columbia, CoMoFoD, DEFACTO) once a link that is faster than the
   100 KB/s seen here is available, with the source-leakage audit repeated; (c) AI-edited and AI-generated images **for training**, which are the actual
   subject of the title. A 1200-image zero-shot test is now in place (`data/external`) and shows near-chance detection (AUROC 0.55 to 0.59); the next step is a train/validation/test set of that kind with source-disjoint splits. Without (c) the title overstates what can be claimed.
3. **Fix the blur weakness and the window-aggregation rule.** The image score is the maximum over windows, which raises false alarms on larger images (0.15 on 512 px real photographs against 0.06 for one window); evaluate mean, top-k and a calibrated per-window threshold on validation data of the target domain, never on the test set.
   For the blur weakness: add blur and noise to training augmentation of both classes, then repeat `eval_robustness`. This is a cheap, well-defined engineering fix and the one limitation that would hurt any live demonstration.
4. **Decide the role of Model B.** Try a pretrained backbone fed the forensic residual, one seed first. (A learned gate and a concatenation head were already tried and both ignore B.) If B is still not
   complementary on the harder benchmark from step 2, drop it from the claim and present CFR as a regulariser for A.
5. **Rerun the CFR comparison with enough power.** `single_a`, `ml` at weight 0.25 and 1, `ml_hem`, `cfr_adaptive`, `cfr_scratch_noalign`, at least ten seeds, on the benchmark from step 2, using the same pre-declared rule (`02`). Add a jointly trained parameter-matched control (a single wider network; the two-Model-A ensemble already exists as a post-hoc compute control) and ablate the challenge term and the gated alignment separately, because `ml_hem` and tuned mutual learning are the controls CFR has to beat.
6. **Longer schedules for the controller.** 12 epochs left the controller 4 to 6 epochs of CFR. If the adaptive claim is kept, test 30 epochs, otherwise drop the controller.
7. **Calibration.** Fit calibration on a set that includes the manipulation types expected at deployment, or report uncalibrated scores with the AUROC. Evaluate selective prediction (accuracy against coverage) using the branch-disagreement flag.
8. **Before any novelty claim:** a literature check that the CFR objective (partner-relative challenge, ground-truth-gated alignment with a floor) is not already published under other names.

## For the final-year write-up, safe claims today

- The pipeline works end to end on conventional manipulations and is tested.
- On this benchmark, a pretrained visual model is a strong detector, and the forensic branch is a weaker, non-complementary one.
- Mutual learning keeps the branches aligned without collapse; it does not improve either on its own.
- Adaptive CFR is not distinguishable from hard-example mining at 3 seeds, 12 epochs, and a benchmark with little headroom.

Unsafe claims: that CFR improves detection, that the hybrid beats the visual model, anything about AI-generated or AI-edited images, or any real-world performance figure.
