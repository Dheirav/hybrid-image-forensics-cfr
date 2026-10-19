# 05 Single-model validation

Three seeds (0, 1, 2), 12 epochs, EfficientNet-B0 (ImageNet weights) for Model A and the SRM+Bayar+DCT CNN for Model B
(from scratch), batch 16, AdamW lr 5e-4 with a one-cycle schedule, bf16 autocast with the forensic transforms
kept in fp32. Results are in `runs/single_{a,b}_s{0,1,2}`. Tables show the mean over the three seeds, so the spread
between seeds is now measured: Model A 0.9936 +/- 0.0027 and Model B 0.9108 +/- 0.0037 on the primary metric (sd over seeds).
Three seeds is still a small sample, so these are estimates of the spread, not tight ones.

## Software checks (not evidence of detection)

All pass under `.venv/bin/python -m pytest -q`: SRM output shape and range; Bayar constraint holds after the
weights are perturbed; block DCT matches `scipy.fft.dctn` and changing one 8 x 8 block changes only that cell;
both branches return the same output dict with the right shapes; every trainable weight receives a gradient;
SRM and DCT carry no trainable weights; the mask loss ignores samples without a mask; the KL term is detached from
the partner, has the right direction and is zero when the models agree; a 4-image batch can be overfitted.
One real bug was found by a test along the way: reflect-padding in inference failed on images smaller than the pad.

Found while reading my own code, not by a test: bf16 autocast would have quantised pixel values above 128 to steps of 2
inside the SRM and Bayar filters, which removes exactly the signal Model B is supposed to read. The forensic
transforms now run in fp32 under autocast.

## Held-out results (real data, source-disjoint splits)

"Test, seen tools" is the declared primary population: authentic + splice + copy_move. Removal is never trained on.

| model | val AUROC | test AUROC, seen tools | test AUROC, removal vs authentic | ood_removal AUROC | FPR on authentic (test) | Dice (test) |
|---|---|---|---|---|---|---|
| A visual | 0.987 | 0.994 (per seed 0.997, 0.993, 0.991) | 0.795 | 0.789 | 0.029 | 0.663 |
| B forensic | 0.891 | 0.911 (per seed 0.914, 0.907, 0.912) | 0.682 | 0.707 | 0.143 | 0.489 |

Source: `reports/evidence/comparison.md`. The seed-0 values quoted in earlier drafts (A 0.997, B 0.914) sit at the top of the seed range.
`ood_removal` is a separate set of sources with authentic and removal only; removal is never trained on.

Validation AUROC over epochs (seed 0): A 0.922 at epoch 1 rising to 0.987; B 0.516 at epoch 1, 0.726 at epoch 3, then 0.89 at epoch 10 to 12.
B started at chance and learned slowly, so it does learn from its inputs rather than sitting on a shortcut: the data audit
found no metadata or off-mask signal for it to use.

## What this says

- Both branches learn useful signal on held-out sources. Neither is at chance, and both clear the metadata (0.55) and local-patch (0.50 to 0.83) baselines from `04`.
- Model A is much stronger than Model B on every metric, and nearly saturates the seen-tool test (0.994). The benchmark leaves A almost no room to be improved on seen tools.
- Both lose a lot on removal (A 0.994 to 0.795, B 0.911 to 0.682, three-seed means). The synthetic seams of paste editing are what they learned; inpainting is a different signature.
- Model B's false-positive rate on authentic images (0.14) is high at the 0.5 threshold. No threshold was tuned.
- Localisation works to a moderate degree (Dice 0.66 for A on the test manipulated images, of which removal is a third) and poorly on unseen removal (Dice about 0.24 for seed 0 on ood_removal).
- I did not investigate whether Model A is relying on semantic cues (an object that does not belong) rather than forensic traces. Splice is detectable that way, and the removal drop is consistent with it. That would need a content-matched test and is listed in `15`.

## Gate 3.4

Both branches and the decoders execute and learn on held-out real data, so the gate to the integrated system is passed.
It is passed for this synthetic benchmark only.
