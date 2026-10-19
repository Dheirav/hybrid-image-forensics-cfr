# 04 Dataset audit

Run 2026-10-18 (IST) on `data/forge/manifest.csv`. Raw numbers: `reports/evidence/dataset_audit.json`
(produced by `python -m hybrid_forensics.audit_dataset`). Everything below was executed in this pass.

## What the data is, and what it is not

COCO val2017 photographs (2000 selected, `tools/fetch_coco_subset.py`) turned into **synthetic conventional
manipulations** by `make_dataset.py`. Each source image gives a pair from the same 384 x 384 crop window:
the untouched crop (authentic) and one edited copy, so the two differ only by the edit.

| tool (`manipulation`) | what it does | mask |
|---|---|---|
| splice | object from another source image of the same split pasted in, feathered edge | pasted region |
| copy_move | the crop's own object copied to a non-overlapping place | pasted region |
| removal | the object inpainted away with OpenCV Telea | object region, dilated 7 px |

It contains **no AI-edited and no fully AI-generated images**. The brief's four categories reduce to two
here (authentic, conventional manipulation), and the AI categories are out of scope, not "tested and passed".
Labels are 0 authentic, 1 manipulated, with the tool kept as metadata.

## Splits (all by source image)

| split | authentic | splice | copy_move | removal | purpose |
|---|---|---|---|---|---|
| train | 1300 | 650 | 650 | 0 | training |
| val | 250 | 125 | 125 | 0 | model selection, controller, calibration |
| test | 250 | 84 | 83 | 83 | final evaluation (removal part is unseen-tool) |
| ood_removal | 200 | 0 | 0 | 200 | unseen-tool test, separate sources |

Removal never appears in train or val, so it serves as the held-out editing tool.

## Checks and results

| check | result |
|---|---|
| Missing or unreadable files, missing masks | 0 of 4000 images; every manipulated image has a mask |
| Dimensions | all 384 x 384; every mask matches its image |
| Masks empty / authentic images with a mask | 0 / 0 |
| Mask area (fraction of frame, median) | splice 9.7%, copy_move 9.7%, removal 11.5%; min 1.7%, max 27% |
| Source overlap between any two splits | 0 for all six pairs |
| Splice donors outside the split of the image they are pasted into | 0 |
| Sources that are not exactly one authentic + one manipulated | 0 |
| Class balance per split | 0.50 manipulated in every split |
| Near duplicates (64-bit dHash, Hamming at most 6) across splits / within splits | 0 / 0 |
| JPEG quality by class (75, 85, 95 drawn independently of label) | chi-square p = 0.34, no association |

## Mask alignment

For each pair, pixels that changed by more than 30 grey levels between the authentic crop and the edited crop:

| tool | changed inside mask | changed far (more than 15 px) outside mask | images with more than 1% changed far outside |
|---|---|---|---|
| splice | 90.0% | 0.12% | 19 of 859 |
| copy_move | 83.6% | 0.11% | 11 of 858 |
| removal | 69.6% | 0.12% | 5 of 283 |

So the masks sit on the edits and almost nothing changes elsewhere. The 30 to 40 percent of unchanged pixels
inside removal and copy-move masks are places where the pasted or inpainted content is close to what was there,
which is expected. The few images with changes far outside the mask are the cases where the two JPEGs
of a pair were saved at very different qualities (75 against 95), not misaligned masks. I did not exclude them.

## Shortcut audit (trained on train, scored on val; chance is 0.5)

| baseline | val AUROC |
|---|---|
| JPEG quality only | 0.517 |
| JPEG quality + file size | 0.547 |
| 32 x 32 patch statistics, patches taken at least 15 px away from the mask (image level) | 0.505 |
| 32 x 32 patch statistics, patches inside the mask (positive control, image level) | 0.832 |

The metadata baselines are near chance. File size gives a small lift (0.547 on 500 images, about 1.5 standard errors),
which fits pasted content adding a little detail, and is not a decision-grade shortcut. Patches away from the edit carry
no information (0.505), so the edit is local and there is no global fingerprint of the pipeline. The positive control
shows the baseline can detect something when the edit is in view, so the near-chance results are not a broken probe.

## Licence and terms

COCO annotations are CC BY 4.0. Images carry per-image Flickr licences, kept in the manifest column `coco_license`
(1 BY-NC-SA 1156, 3 BY-NC-ND 1052, 4 BY 718, 2 BY-NC 498, 5 BY-SA 352, 6 BY-ND 222, 7 no known restrictions 2).
Roughly 64 percent are NC and 64 percent include an ND condition; an edited copy is a derivative of an ND image. I treat
this as private, non-commercial research use only: **do not redistribute `data/forge`**, and a public release would need
a licence-filtered subset. Terms: https://cocodataset.org/#termsofuse.

## Gate 2.5

Passed for the claim this data can support: data integrity, mask validity, label validity and split separation are
verified, and the metadata and local-patch shortcut baselines are at chance.

It does not pass for anything beyond that:
- The edits are synthetic, so the dataset says nothing about forgeries made by people or by AI tools.
- 2000 sources is small. With 250 authentic and 250 manipulated test images, the AUROC standard error is about 0.01 to 0.02.
- Splice has a semantic cue (an object appears where it does not belong) that a visual model can use without any forensic trace.

## Addendum: external AI-edited and AI-generated test set (test only, zero-shot)

Why: the brief asks for AI-edited photographs and fully generated images, which COCO-Forge does not contain, and the network (about 100 KB/s) ruled out downloading any.
The only copy on this machine was a slice of OpenSDI inside the old `Deepfake_Detection` project's `data_sources/heldout`. I used it **read-only, as a test set only**: nothing was trained,
tuned or selected on it, and no labels, code or numbers from that project were used (the folder names are the labels). `tools/build_external.py` copies the files into `data/external/`
so later evaluation does not depend on that folder. I am flagging this because the project was meant to be standalone; the dependency is on files, not on the old project's results,
and it is limited to this one test.

| kind | generators | images | mask |
|---|---|---|---|
| real | one folder (sd2/real) | 400 | none |
| AI-edited (partial inpainting) | flux, sd15, sd3, sdxl | 100 each, 400 | yes, median 10.5 percent of the frame (range 1 to 89 percent) |
| AI-generated (entire image) | flux, sd2, sd3, sdxl | 100 each, 400 | none |

Checks run (`data/external/manifest.csv`, same columns as the main manifest):

- Readability: all 1200 files open. All are 512 x 512 JPEG with the same quantisation table (so no JPEG-quality shortcut between classes).
- Source leakage: images whose COCO id matches any of the 2000 source images used to build COCO-Forge are excluded (2 edited images were dropped). Real images come from a different folder than the generated and edited ones, so I cannot rule out that the same photographs appear both as a real image and as the source of an edit; that would not leak into COCO-Forge training either way, since nothing from this set was trained on.
- Metadata-only baseline: file size alone gives AUROC 0.57 for edited against real and 0.69 for generated against real (generated files are smaller on average). So a weak file-size cue exists for the generated class. Image size and quantisation table carry no information.
- Not checked: whether the folder labels are correct (no independent ground truth), near-duplicates inside this set, and whether OpenSDI's real images come from the same distribution as its fakes beyond the file-size cue.
- Licence: OpenSDI terms apply, and its real images are COCO photographs with mixed licences. Private research use only; not for redistribution.

The set supports one kind of claim: how the models trained on conventional synthetic edits behave, without adaptation, on AI-edited and AI-generated 512 px images. Results are in `11` and `12`.
It does not support training or any claim about detectors trained on AI-generated data.
