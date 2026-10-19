# 03 Risk register

Likelihood and impact are my judgement from the desk review, not measurements. Updated as evidence arrives.

| # | risk | likelihood | impact | mitigation / first check |
|---|---|---|---|---|
| R1 | No codebase or data for this project exists in `nitin-work`; Phase 2 onward cannot start | certain (verified) | blocks everything | decide base repo and dataset (see `01`, open decisions) |
| R2 | The audit reference reports were written for a different repo and get quoted as results | medium | misleading deliverable | keep `reference/` marked; quote nothing from it |
| R3 | Dataset lacks conventional manipulation, so a four-category claim is unsupported | high with OpenSDI only | claim must be narrowed | narrow to authentic vs AI-edited vs AI-generated, or add CASIA/Columbia/CoMoFoD |
| R4 | Generator or source leakage across splits (edited, recompressed or resized derivatives of one source in different splits) | medium | inflates every metric | group split by `source_id`; run the pair and ID-overlap checks before any training |
| R5 | Metadata or format shortcut (JPEG quantisation table, file size, resolution differ by class) | medium | model learns the wrong thing | metadata-only and small-crop baselines before the main run (2.4) |
| R6 | Masks soft-edged, misaligned or missing for some samples | medium | wrong Dice/IoU | mask audit with a fixed threshold, recorded |
| R7 | Forensic branch does not learn beyond the visual one, or learns a dataset shortcut | medium | hybrid adds nothing | train B alone first; corrupt-input reliance tests |
| R8 | Mutual learning collapses both models to the same predictions, or B copies A's errors | medium | alignment claim fails | monitor agreement and complementary errors; ground-truth-gated KL |
| R9 | CFR is indistinguishable from mutual learning plus hard-example mining | high | research benefit unproven | variant H is mandatory; report a null result plainly |
| R10 | Improvements fall inside seed noise | high at 1 seed | cannot conclude | at least 3 seeds for headline rows, paired uncertainty; label pilots exploratory |
| R11 | 30 or more training runs exceed available GPU time | high | partial comparison | smallest backbone, pilot first, drop optional variants (gating, parameter-matched control) in that order |
| R12 | Shared machine: other jobs hold most CPU and about 1.7 GB of VRAM, and the harness may kill GPU jobs started from its shell | high | stalled or lost runs | start detached with `setsid nohup`, add a progress reader in `tools/` with each run |
| R13 | Threshold or checkpoint chosen on test data | low if guarded | invalid test result | select on validation only; freeze primary metrics before the test run |
| R14 | Demo preprocessing or class order differs from evaluation | medium | demo contradicts the paper | one shared preprocessing module; a test that app output equals eval output |
| R15 | Confidence is read as truth | medium | overclaiming | report calibrated probabilities with ECE; always show limitations text |

## Gate status (updated 2026-10-18)

| gate | state |
|---|---|
| Repository to audit exists | superseded: standalone project started; no old repo used (`01`) |
| 2.5 Data integrity | passed (`04`) |
| 3.4 Single-model validation | passed (`05`) |
| Integrated system | built; does not beat the visual model alone (`06`, `12`) |
| CFR benefit | inconclusive under the pre-declared rule (`09`, `12`) |
| End to end | passed on the real checkpoint (`10`) |

## Risk outcomes

R1 and R2 resolved by starting standalone. R3 materialised: no AI-edited or AI-generated training data, so the claim is narrowed to conventional edits; a zero-shot test set shows the models do not transfer to AI images (`11`).
R4 and R5 checked and not found. R7 partly materialised: B learns real signal but adds nothing in fusion. R8 not observed (no collapse; B copies a little of A's errors).
R9 materialised: CFR is not distinguishable from plain hard-example mining. R10 materialised: effects are inside seed noise. R11 avoided by using a small benchmark. R12 occurred (memory, `13`). New: the external set was taken from another project's data folder (`04` addendum); it is test-only and copied locally.
R14 checked and not found. R15 partly materialised: calibration did not transfer.
