# Hybrid image manipulation detection and competitive forensic reasoning (CFR)

A standalone research project: a visual model (Model A, EfficientNet-B0) and a forensic-residual model (Model B, SRM + Bayar + block DCT) for detecting and localising
image manipulation, trained alone, side by side, with mutual learning, and with a "competitive forensic reasoning" objective. The brief is `PROMPT.md`.

## Status in one paragraph

The pipeline works end to end and is tested. The research claim is **not** supported: on the synthetic COCO-Forge benchmark Model A alone (0.994 test AUROC) is not beaten by any fusion,
mutual-learning or CFR variant; CFR is not distinguishable from hard-example mining; Model B adds no information; and every trained model is at chance on the same edits made with Poisson blending and near chance on
zero-shot AI-edited and AI-generated images. Start with `NEXT.md` (live handover), then `NEXT_PHASE_RESULTS.md` (latest findings), then `reports/14_final_feasibility_verdict.md`.
A follow-up retraining on a harder benchmark (`forge2`) was stopped after 1 of 36 runs; `NEXT.md` says how to resume it.

## Layout

| path | what |
|---|---|
| `PROMPT.md` | the original brief |
| `NEXT.md`, `NEXT_PHASE_PLAN.md`, `NEXT_PHASE_RESULTS.md` | handover, next-phase plan with pre-declared criteria, interim results |
| `reports/00` to `reports/15` | the required reports; `reports/evidence/` holds the tables and JSON behind them, `reports/evidence/next_phase/` the latest |
| `src/hybrid_forensics/` | models, losses, controller, training, evaluation, analysis, dataset builders, inference |
| `tools/` | queue runners, progress readers, data fetch, audits |
| `tests/` | 26 software tests (these are not research evidence) |
| `runs/`, `runs_extra/`, `runs_next/`, `runs_f2/` | per-run `results.json`, `probs.npz` and external/hard-set scores. Checkpoints (`best.pt`) are not in git, except `runs/cfr_adaptive_s0/best.pt` for a demo |
| `data/*/manifest.csv`, `data/coco_raw/selection.json` | metadata only. **No images are included** |

## Requirements

- Python 3.12, and an **NVIDIA GPU with CUDA** for training (`train.py` uses CUDA directly). Inference accepts `device="cpu"` in the Python API.
- About 8 GB of RAM; a training run needs roughly 3 GB free on top of whatever else is running.
- The pinned versions are in `requirements.txt`. It points at the CUDA 12.8 PyTorch wheels; change the index line if your driver needs another build.

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 26 tests, no data needed
```

`tools/seed_venv_from_local.py` and `tools/fetch_wheels.py` exist only because the original machine had a very slow network; ignore them.

## Data (not included)

COCO-Forge is built from COCO val2017 photographs, which carry mixed per-image Flickr licences (many non-commercial or no-derivatives). Private research use only; do not redistribute the images or the built datasets.

```
.venv/bin/python tools/fetch_coco_subset.py --n 2000          # COCO val2017 subset into data/coco_raw
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_dataset      # data/forge       (training and main test)
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_hard         # data/forge_hard  (tool-disjoint test, five tools)
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_forge2       # data/forge2      (harder training benchmark, only needed for Stage 2)
```

The builders are seeded, but JPEG encoding can differ slightly between OpenCV builds, so rebuilt images may not be byte-identical to the originals and numbers may move in the third decimal.
The external AI-edited and AI-generated set (`data/external`) was a slice of OpenSDI copied from another local project and is **not** reproducible from this repo; anything that needs it (`external_eval`, `diag_external`, the AI-image sections of the reports) cannot be rerun without a comparable set.

## Common commands

```
# rebuild the headline tables from the included per-run results (no GPU, no images)
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.compare > /tmp/comparison.md
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.extra_analysis > /tmp/extra_analysis.md

# single image with the included checkpoint
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.infer <image.jpg> --ckpt runs/cfr_adaptive_s0/best.pt

# train one run (about 5 to 10 minutes on a laptop-class GPU), then the full queues
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.train --variant single_a --seed 0
tools/queue_stage2.sh    # 24 headline runs, about 3 hours;      progress: tools/queue_progress.py --watch
tools/queue_forge2.sh    # 36 runs for the harder benchmark;     progress: tools/queue_f2_progress.py --watch
```

`compare` and `extra_analysis` read `data/forge/manifest.csv` and each run's `probs.npz`, so they work without the images. Anything that scores a checkpoint (`eval_manifest`, `diag_*`, `verify_baseline`) needs the built datasets.

## Rules this project follows

Pre-declared primary metric and decision rule (`reports/02`, `NEXT_PHASE_PLAN.md`), validation-only model selection, no tuning on test splits, and no numbers from the old `Deepfake_Detection` project.
Synthetic-data results are not evidence about AI-generated images.
