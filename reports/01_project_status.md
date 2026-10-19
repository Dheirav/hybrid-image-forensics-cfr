# 01 Project status

Updated 2026-10-19 after the stage 3 runs (42 training runs in total) and the zero-shot AI-image tests. Scope: `/home/dheirav/Code/nitin-work` only, a standalone project. The first
version of this file was written when no code or data existed; that version and its open decisions are superseded.
The decisions that closed them: standalone, not a fork of the neighbouring repository; dataset built here from COCO (`04`);
binary labels; one mask decoder per branch; EfficientNet-B0 for Model A.

## Component status

| component | status | evidence |
|---|---|---|
| Dataset and manifest | built and audited | `04` |
| Input validation and preprocessing | implemented, tested | `10`, `tests/test_infer.py` |
| Model A, visual branch (EfficientNet-B0) | trained, 3 seeds | `05` |
| Model B, forensic branch (SRM, Bayar, DCT) | trained, 3 seeds; weaker than A | `05` |
| Localisation decoders | trained with the branches | `05` (Dice) |
| Hybrid fusion | fixed average, learned concatenation and learned gate all built; none beats A alone (concat is level) and the learned fusions ignore B | `06` |
| Mutual learning | implemented, run, 3 seeds | `07` |
| Validation-triggered transition controller | implemented, unit-tested, run | `08` |
| CFR | defined, unit-tested, run | `09` |
| Calibration, ECE, selective prediction | done; validation-fitted temperature did not transfer; branch disagreement adds nothing over confidence | `11` |
| Evaluation harness (macro-F1, AUROC, Dice, IoU, seeds, paired bootstrap) | done | `12`, `reports/evidence/comparison.md` |
| Application | command-line and Python only | `10` |
| Compute-matched control | post-hoc two-Model-A ensemble; a jointly trained parameter-matched control is **not built** | `12` |
| AI-edited and AI-generated images | **zero-shot test set only** (1200 images, from an OpenSDI slice found on the machine); the models are near chance on it; no training data of that kind | `04` addendum, `11` |

Software tests: 26 pass (`.venv/bin/python -m pytest -q`). They show the code is correct, not that detection works.

## Where the project stands

The pipeline works. The research claim is not shown: see `14` for the verdicts and `15` for the next steps.
