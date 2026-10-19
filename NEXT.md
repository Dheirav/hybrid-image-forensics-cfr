# NEXT: live handover for nitin-work

Standalone project. It is not a continuation of `Deepfake_Detection`; `reference/` holds old audit
reports about that repo and none of their numbers are results here.

Brief: `PROMPT.md`. Reports `reports/00` to `reports/15` are all written. Code: `src/hybrid_forensics/`. Tests: `tests/`.
Interpreter: `.venv/bin/python` (never system python). Not a git repo.

## State as of 2026-10-19, ~03:45 IST

Everything the brief lists has been run at least once; the exceptions are named below. 42 training runs (9 headline variants x 3 seeds in `runs/`, 15 stage 3 runs in `runs_extra/`),
calibration, selective prediction and robustness on `runs/cfr_adaptive_s0`, a zero-shot test on 1200 AI-edited, AI-generated and real images, the end-to-end matrix, reports 00 to 15.

Headline (details in `reports/12` and `reports/14`):
- The pipeline works end to end (image file in, label, probability, mask, overlay and JSON out). 26 software tests pass.
- Model A (visual) alone: 0.994 test AUROC on seen tools. A learned concatenation head is level with it (0.993) and the learned gate scores 0.991; both learn to ignore Model B. Every other pair variant is 0.957 to 0.985. Two Model A runs averaged: 0.995.
- Adaptive CFR against the pre-declared rule: inconclusive (+0.0034 vs mutual learning, all seeds positive, interval includes 0; -0.0019 vs mutual learning + hard-example mining, all seeds negative, interval includes 0).
  Extra findings: alignment weight 4 is clearly harmful and 0.25 is level with none; CFR without alignment is nominally better than with it.
- Known weaknesses: unseen removal (0.71 to 0.73 for pairs, 0.795 for A), blur (59 percent of authentic images flagged), calibration does not transfer, near-chance zero-shot detection of AI-edited and AI-generated images (AUROC 0.55 and 0.59), max-over-windows inflates false alarms on large images.
- Not built: a jointly trained parameter-matched control. Not done: a literature check for novelty; whether Model A uses semantic cues.

## Next phase, 2026-10-19 (work stopped by request partway through; read `NEXT_PHASE_RESULTS.md` first)

Done: baseline verification, a tool-disjoint test set (`forge_hard`), five diagnoses, per-tool metrics for all 42 runs. Main findings: Model A is at chance (0.51) on Poisson-blended edits of the same scenes and 0.895 on a content-preserving re-paste, so its 0.994 is one tool's hard edge;
Model B adds no information (stacked weight on B about 0.05, no gain on any set); CFR acts only on B and its weights there are nearly hard-example weights (Spearman 0.96 to 0.98); the external AI failure is not a real-photo shift, edit size or window rule.
Evidence: `reports/evidence/next_phase/`. Plan and pre-declared criteria: `NEXT_PHASE_PLAN.md`.

Erratum found while preparing the repo (`reports/13`, entry 19): `extra_analysis.md` had mixed a learning-rate probe run into `indep` seed 0. Fixed and the quoted figures corrected; no conclusion changed, and `comparison.md` was never affected.

Not done: Stage 2, the headline comparison retrained on `forge2` (36 runs queued in `tools/queue_forge2.sh`; only `single_a` seed 0 finished, unanalysed; `ml_lam0.25` seed 0 was interrupted and `runs_f2/ml_lam0.25_s0` is not usable, delete it before resuming).
To resume: build the data (`python -m hybrid_forensics.make_forge2`), run `tools/queue_forge2.sh`, then `tools/score_f2.sh` and `python -m hybrid_forensics.f2_analysis`. Progress: `tools/queue_f2_progress.py --watch`.
Memory: the forge2 test split (2000 images) needed more than 6 GB in `train.py` until `predict` was changed to store float16 masks and uint8 ground truth. A run needs about 3 GB of free RAM after that change; check before running other jobs.

## The one next action

Run Stage 2 (above), check after two runs that Model A is not at ceiling on the four seen tools of `forge2`, then judge `cfr_adaptive` against `ml`, `ml_lam0.25` and `ml_hem`, and the fusions against `single_a`, with the rule in `NEXT_PHASE_PLAN.md`.
After that, the longer-term step stays the same: a benchmark with AI-edited images for training and testing (`reports/15`). Do not change the decision rule after looking at results.

## Reproducing

    .venv/bin/python -m pytest -q
    PYTHONPATH=src .venv/bin/python -m hybrid_forensics.compare > reports/evidence/comparison.md
    PYTHONPATH=src .venv/bin/python -m hybrid_forensics.mutual_analysis > reports/evidence/mutual_analysis.md
    .venv/bin/python tools/e2e_matrix.py runs/cfr_adaptive_s0/best.pt          # without the calibration file; see reports/11
    tools/queue_stage2.sh      # 24 headline runs, about 3 hours, tools/queue_progress.py --watch
    tools/queue_stage3.sh      # 15 extra runs into runs_extra/, about 2.3 hours, tools/queue3_progress.py --watch
    tools/build_external.py    # copies the external AI test set into data/external (needs the old project's data folder; see reports/04 addendum)
    tools/external_all.sh runs/* runs_extra/*     # zero-shot AI-image evaluation, about 1 min per run
    PYTHONPATH=src .venv/bin/python -m hybrid_forensics.extra_analysis > reports/evidence/extra_analysis.md
    tools/pause_runs.sh stop|cont   # freeze or thaw the stage 3 queue (SIGSTOP)

Single image: `PYTHONPATH=src .venv/bin/python -m hybrid_forensics.infer <image> --ckpt runs/cfr_adaptive_s0/best.pt`

## Constraints

- `data/external` is a copy of an OpenSDI slice taken from the old Deepfake_Detection project's data folder. Test only, never trained on, not for redistribution.
- `data/forge` is built from COCO images with mixed licences (many non-commercial or no-derivatives). Private research use only; do not redistribute.
- The network is about 100 KB/s per connection, so large dataset downloads are impractical.
- Shared machine (8 cores, 7.8 GiB RAM). Only the training data loader keeps persistent workers; more than that nearly exhausted memory once (`reports/13`).

## Known design choices worth challenging

- `BayarConv` output is divided by 8 for scale; that constant is a guess, not tuned.
- DCT is luma only, log-compressed, so quantisation-table cues in chroma are dropped.
- Model B sees the image at the resolution it is given; resizing before the transforms destroys most forensic traces, so preprocessing must crop, not resize, for the forensic path.
- For two classes the controller's "complementary errors" and "disagreement" checks are the same quantity.
- The validation-fitted temperature in `calibration.json` worsens test calibration; do not use it for the shipped checkpoint.
