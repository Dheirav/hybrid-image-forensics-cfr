# 13 Failure analysis

Each entry: what was observed, the suspected cause, the diagnostic, the correction, and whether it resolved the issue.
The brief's checklist is covered in the second table.

## Failures and corrections, in the order they happened

| # | observed | suspected cause | diagnostic | correction | resolved |
|---|---|---|---|---|---|
| 1 | GPU wheels (about 4 GB) were arriving at 100 to 170 KB/s, an estimated 6 hours, and one connection reset | slow network, not the package index | measured rates; tried segmented parallel download (no gain) | copied the identical pinned package files from the other project's venv into `.venv` (`tools/seed_venv_from_local.py`); `requirements.txt` records the real versions. Disclosed at the time. | yes. `.venv` works and has no later dependency on the old one. |
| 2 | `infer` crashed on images smaller than the padding size | reflect-padding needs more pixels than the image has | unit test with a small image failed | symmetric padding in `_tiles` | yes (test passes) |
| 3 | the application ignored the calibration file | the temperature was loaded but never applied in the forward path | found by reading `infer.py` against `calibrate.py` | apply it to the fused logit | yes |
| 4 | no symptom yet; found by reading the code: the forensic filters would have seen pixels rounded to bf16 | bf16 autocast quantises values above 128 to steps of 2, which erases the faint residual that Model B reads | found by reading the code, then confirmed by arithmetic | SRM, Bayar and DCT run in fp32 under autocast | yes, before any reported run |
| 5 | primary-metric ambiguity: the builder puts removal in the test split, but removal is never trained on | would have mixed an unseen-tool test into the headline number | noticed before any training | primary metric restricted to authentic + splice + copy-move, written into `02` before any result; removal reported as a secondary, unseen-tool result | yes |
| 6 | free memory fell to 83 MiB during one final evaluation (`ml_hem_s0`), at risk of the machine killing the job | about 16 persistent data-loader workers across train and evaluation loaders | checked `free` and process list | only the training loader keeps persistent workers; evaluation loaders use 2 non-persistent ones. Applies to runs from `ml_s1` onward and does not change numerics. | yes (3.5 GiB used afterwards, queue finished) |
| 7 | the shell died twice (exit 144) | my own `pkill -f` and `pgrep -f` patterns matched my command line | repeated | bracketed `ps ... awk` patterns instead | yes |
| 8 | a background watcher was killed by the harness for low memory | the memory problem in 6 | notification | not restarted on my own; restarted when the user asked | yes |
| 9 | after training, the truncated-JPEG e2e case returned "authentic, p 0.006" rather than an error | OpenCV decodes a damaged JPEG into a grey-filled partial image with no error | e2e matrix (`10`) | decode with PIL first, which raises; unit test added | yes (23 tests pass) |
| 10 | my "3:1 aspect" e2e case was rejected | the case was really 4.3:1 | read the error message | fixed the case, not the limit | yes |
| 11 | validation-fitted temperature made calibration worse on test (ECE 0.058 to 0.089) | validation has no removal and is easier than test | `calibrate.py` outputs, `11` | recommend not applying it; reported both numbers | not fixed: I have no representative calibration set. Recorded as a limitation. |
| 12 | blur sigma 1 flags 59 percent of authentic images | no blur in training, so softness reads as editing | `eval_robustness` (`11`) | none. Not corrected, no retraining with blur. | no. Open limitation. |

## Checklist from the brief

| item | what was done | result |
|---|---|---|
| label mapping errors | audit (`04`); saved probabilities checked against manifest labels in every analysis (an assertion in `analysis.subset`); application output compared with evaluation output | no mismatch |
| incorrect masks | mask alignment audit: 90 / 84 / 70 percent of pixels changed inside the mask for splice / copy-move / removal, about 0.12 percent far outside | masks align; removal masks are looser (70 percent) |
| train/test leakage | source overlap between splits, near-duplicates, donors in the wrong split | 0 in each |
| preprocessing inconsistencies | application against evaluation on all 500 test images | no label changes, max probability difference 0.006 |
| incorrect loss reduction | unit tests for the mask loss and the KL term | pass |
| KL direction or detach errors | unit test: partner detached, direction correct, zero when equal | pass |
| gradient instability | training loss in all 324 logged epochs is finite; maximum 4.55 (epoch 0) | none seen. I did not check gradient norms. |
| learning-rate mismatch | 3 rates on one seed: validation AUROC 0.974, 0.978, 0.980 | within noise; kept the default |
| model collapse | agreement, correlation and each model's own AUROC logged each epoch (`07`) | no collapse; B stays 15 to 20 percent different from A |
| class imbalance | balance 0.5 by construction | none |
| data shortcuts | JPEG quality, file metadata and off-mask patch baselines (`04`) | metadata-only AUROC 0.55, JPEG-only 0.52, outside-mask patches 0.51, positive control 0.83 |
| incorrect checkpoint selection | selected by validation AUROC; test evaluated once at the end | no test use in selection |
| threshold selection using test data | 0.5 throughout; the temperature was fitted on validation only | none |
| accidental use of test data in training | splits are by source; the `ood_removal` split and removal are never trained on | none found |
| fusion ignores one branch | validation-fitted fusion weight is 1.0 on A for seed 0; the stacker gives B -0.06 (`06`) | **yes**, in the sense that B adds nothing |
| CFR corrections damage correct predictions | paired flip counts (`09`) | small: breaks 2 to 7 against `ml`, fixes 11 or 12 |
| application differs from evaluation | reload comparison (`10`) | no |

## What is still unexplained

- Why `cfr_scratch` (CFR from the first step) beats the adaptive schedule by 0.0026 AUROC with a clear interval, when the premise of the schedule is the opposite.
  I did not run a diagnostic; it may be real, or the way the weights behave when B is still at chance, or small-sample luck.
- Whether Model A relies on semantic cues (an object that does not belong) rather than forensic traces. The drop from 0.994 to 0.795 on removal fits that, but it does not prove it.
- Why the pair variants are 0.9 to 1.5 points below A alone beyond the effect of averaging in a weaker model; the logistic stacker for seed 0 recovers A's score, which says fusion weights explain most of it, but I did not check this on the other seeds.

## Update: findings from the stage 3 work

| # | observed | suspected cause | diagnostic | correction | resolved |
|---|---|---|---|---|---|
| 13 | on 512 px external images the application's false-positive rate on real photographs was 0.19, against 0.06 when only the central 384 px window was scored | symmetric padding to a multiple of 384 put mirror seams inside the image, and the forensic branch reads seams | the same checkpoints scored on the central crop with no padding (`external_crop.json`), then with edge-anchored windows | windows are now anchored to the image edge and overlap, padding is used only for images smaller than one window (`infer.tile_starts`, 2 new tests). False-positive rate with the new tiling for `cfr_adaptive` seed 0: 0.15. | partly. The seams are gone, but 0.15 is still above the 0.06 of a single window: the image score is the maximum over windows, so a 4-window image has four chances to cross 0.5. Not changed; changing the aggregation rule after seeing this set would be tuning on it. Open limitation. |
| 14 | the fixed 0.5 average of the pair scored below Model A alone in every seed | averaging a weak, noisy model into a strong one | learned concatenation and gating (`06` update 2) | learned fusion removes the loss (concat level with A) | resolved as a loss; B still adds nothing |
| 15 | the learned gate ended at a mean weight of 0.99 on Model A | B has no signal that A lacks on this benchmark | branch-reliance (`06` update 2) | none needed; reported | not a bug |
| 16 | alignment weight 1 for headline mutual learning is on the harmful side of the weight curve | weight chosen a priori, not tuned | sweep at 0.25 and 4 (`07` update) | reported next to the headline; headline not changed after the fact | open: `ml` is a weaker control than it could be |
| 17 | zero-shot AI-edited and AI-generated images are not detected (AUROC 0.52 to 0.61) | the models only ever saw synthetic paste, copy-move and removal | external set (`04` addendum, `11`) | none; scope statement | not fixable without training data of that kind |
| 18 | my earlier report said the application check used "417 seen-tool images"; it used all 500 test images | I mixed up the two counts | re-read the script and `app_vs_eval_probs.json` (n = 500) | reports corrected | yes |
| 19 | rebuilding `extra_analysis.md` for the repo gave different `indep` numbers from the stored file | `extra_analysis.all_runs` globbed `runs*/*/results.json`, so the learning-rate probe `runs_lr2e-4/indep_s0` silently replaced the headline `indep` seed 0 (the other probe and the `indep` seeds 1 and 2 were not affected) | diffed a rerun against the stored evidence | glob restricted to `runs/` and `runs_extra/`, evidence regenerated, quoted figures corrected in `06`, `07`, `11`, `02`, `14` (for example concat minus the fixed average is now +0.0085 [+0.0033, +0.0150], was +0.0075; B fixes 42 percent of A's misses under `indep`, was 44) | yes, no conclusion changes. `compare.md`, and so every headline number, was unaffected (it reads `runs/` only) |
