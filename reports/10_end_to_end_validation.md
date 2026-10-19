# 10 End-to-end application test

What was run: `tools/e2e_matrix.py` against the real trained checkpoint `runs/cfr_adaptive_s0/best.pt` through `infer.Predictor` (validation, tiling,
both branches, fusion, heat map, mask, JSON and overlay export). Inputs are test-split images (never trained on), edited copies of them, and four images from the
external AI set (`04` addendum). Raw results: `reports/evidence/e2e_results.json`; exported masks and overlays: `outputs/e2e_results/`.
One checkpoint, one seed. This tests that the pipeline behaves, not how accurate it is; accuracy is in `12`. The matrix was run twice: first with the
validation-fitted temperature (the first version of this report), and again after the tiling change described below, **without** the temperature, because `11` shows it makes calibration worse.
The table is the second run.

## Cases

| case | outcome | result |
|---|---|---|
| authentic crop | completed | authentic, p(manipulated) 0.074 |
| splice | completed | manipulated, p 0.987, Dice 0.94 |
| copy-move | completed | manipulated, p 0.998, Dice 0.93 |
| removal (tool never trained on) | completed | **missed**: authentic, p 0.099, Dice 0.00 |
| small region (2.3 percent of frame) | completed | **missed**: authentic, p 0.288, branches disagree, Dice 0.15 |
| large region (26.8 percent) | completed | manipulated, p 0.894, Dice 0.94 |
| splice recompressed at JPEG q40 | completed | manipulated, p 0.989, Dice 0.90 |
| splice resized to 288 px | completed | manipulated, p 0.993, Dice 0.83 |
| splice, Gaussian blur sigma 1.5 | completed | manipulated but p 0.618 and branches disagree, Dice 0.83 |
| 1152 x 768 mosaic (5 authentic tiles, 1 spliced) | completed, 6 windows, 0.25 s | manipulated, p 0.989, Dice 0.83 |
| 1152 x 384 authentic strip (3:1) | completed, 3 windows | authentic, p 0.075 |
| .gif | rejected with a clear message | unsupported file type |
| truncated JPEG | rejected with a clear message | could not be decoded |
| 720 x 120 | rejected | side shorter than 128 px (it is also past the aspect limit; the size check fires first) |
| 64 x 64 | rejected | outside 128 to 4096 px |
| 5000 x 300 | rejected | outside 128 to 4096 px |
| **AI-edited, SDXL (external, 512 px)** | completed, 4 windows | **missed**: authentic, p 0.065, Dice 0.00 |
| **AI-edited, FLUX (external)** | completed | flagged manipulated, p 0.614, branches disagree, but the mask misses the edited region (Dice 0.00) |
| **AI-generated, SD3 (external)** | completed | **missed**: authentic, p 0.271 |
| real photograph (external) | completed | authentic, p 0.153 |

Per-image inference takes about 0.04 to 0.06 s for a 384 px image once the GPU is warm (the first call took 1.2 s), and about 0.12 s for a 512 px image (4 windows).

The four AI cases are single hand-picked examples, the first image of each group in the manifest, and say nothing about rates. The rates over 1200 images are in `11`: the models are close to chance on this kind of image.

## What this shows

- The pipeline runs from file to label, probability, mask, overlay and JSON for supported inputs, and rejects unsupported input with
  an explanatory error instead of a crash or a silent wrong answer.
- The pipeline does not resize. Large images are tiled at 384 px, the image score is the maximum over windows, and a spliced tile in a larger mosaic
  was found and localised.
- The misses are the same ones the held-out evaluation predicts: unseen manipulation type (removal), very small regions, and now AI-edited and AI-generated images, which are not detected in a way worth reporting (`11`).
- The blurred splice is still called manipulated but with much lower confidence and with the two branches disagreeing, which is the behaviour
  a user would want from the `branches_agree` flag. It is also the beginning of the blur weakness reported in `11`.

## Bug found by this test

The first run of the matrix returned "authentic, p 0.006" for a **truncated JPEG** instead of rejecting it. OpenCV fills in the
missing part with grey and reports no error, so a damaged file would have been scored as if it were real content. `validate()` now decodes with PIL first,
which raises on truncation; a unit test was added (`tests/test_infer.py::test_rejects_truncated_jpeg`, 23 tests pass). The matrix was rerun and now rejects it.
The same run also showed my own test case "3:1 aspect" was really 4.3:1 (it was rejected, correctly); I fixed the case, not the limit.

## Does the application equal the evaluation

All 500 test images (authentic, splice, copy-move and removal) of `runs/cfr_adaptive_s0` were passed through the application path with
no calibration and compared with the evaluation `probs.npz` for the same weights (`reports/evidence/app_vs_eval_probs.json`):

| output | max abs difference | mean abs difference | label changes at 0.5 |
|---|---|---|---|
| fused | 0.0058 | 0.0003 | 0 |
| Model A | 0.0116 | 0.0006 | 0 |
| Model B | 0.0009 | 0.00001 | 0 |

The differences come from bf16 arithmetic: evaluation used batches of 16, the application one image at a time. They do not change any decision.
The row order of the saved probabilities was checked against the manifest labels before comparing.

## Not covered

- AI-generated or AI-edited inputs as a systematic test: only four single examples here; see `11` for 1200 images.
- Non-JPEG/PNG formats such as WebP, TIFF, HEIC, which are rejected by design.
- Images with EXIF rotation, CMYK or 16-bit PNG, alpha channels. Not tested.
- A browser or app front end. There is only the command-line and Python interface (`python -m hybrid_forensics.infer`).

## Second bug found by the AI cases: padding seams

Running 512 px external images through the application showed a false-positive rate on real photographs of 0.19, against 0.06 when only the central 384 px was scored.
Cause: images were padded symmetrically to a multiple of 384 px, which puts mirrored seams inside the image, and the forensic branch reads seams as edits. The application now anchors the last window to the image edge,
with overlapping windows and no padding (`infer.tile_starts`; 2 new tests, 26 pass). 384 px inputs are unchanged, which is why the equivalence check above is unaffected, and I reran it after the change
(`tools/app_vs_eval.py`: same numbers, no label changes over 500 images). The rate dropped to 0.15 with the new tiling, not to 0.06: the image score is the maximum over windows, so a four-window image has four chances to cross 0.5.
I left the aggregation alone; changing it after looking at this set would be tuning on it.
