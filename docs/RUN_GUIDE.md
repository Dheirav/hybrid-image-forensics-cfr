# Step-by-step run guide

Follow the steps in order. Each step says what you should see, so you know when to stop and ask. Times were measured on an RTX 4060 laptop (8 GB GPU, 7.8 GB RAM) and will differ on yours.

## 0. What you need

- Linux, or **Windows with WSL2 (Ubuntu)**. macOS is not supported because there is no CUDA. The scripts are bash. On native Windows, use WSL2 and keep the project inside the Linux filesystem (`~/`), not under `/mnt/c`, which is very slow.
- An **NVIDIA GPU** with a recent driver. Check with `nvidia-smi`. Training does not run on CPU.
- Python 3.12 and git. At least 8 GB of RAM and about 5 GB of free disk for data and runs.

## 1. Clone and install

```
git clone <repo-url> nitin-work
cd nitin-work
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` points at the CUDA 12.8 PyTorch wheels. If `nvidia-smi` shows an older driver and the install or the next step fails, change the `--extra-index-url` line to the wheel index that matches your driver (see pytorch.org) and reinstall.

Check that CUDA is visible:

```
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name. If it prints `False`, stop here: nothing later will work.

## 2. Run the software tests (no data needed)

```
.venv/bin/python -m pytest -q
```

Expected: `26 passed`. These are software checks (shapes, loss terms, controller logic). They are **not** research evidence.

## 3. Rebuild the tables from the included results (no data, no GPU)

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.compare > /tmp/comparison.md
diff /tmp/comparison.md reports/evidence/comparison.md && echo identical
```

Expected: `identical`. This proves the repo is intact and shows the headline table.

## 4. Get the data (about 10 to 30 minutes, depending on your network)

The photographs are not in the repo because their licences are mixed (private research use only; do not redistribute). The scripts download COCO val2017 images and build the manipulated datasets.

```
.venv/bin/python tools/fetch_coco_subset.py --n 2000        # about 2000 COCO photos into data/coco_raw (a few hundred MB)
.venv/bin/python tools/fetch_coco_subset.py --status        # in a second terminal: images done / total
ls data/coco_raw/images | wc -l                              # expect 2000
```

If it stops, run the same command again: it resumes and skips files that exist. The list of selected images is deterministic, and `data/coco_raw/selection.json` in the repo is what the author used.

Then build the datasets:

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_dataset      # data/forge: 4000 images, train 1300 pairs, val 250, test 250, ood_removal 200
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.audit_dataset     # integrity, leakage and shortcut audit, writes reports/evidence/dataset_audit.json
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_hard         # data/forge_hard: five new editing tools, val and test (about 1 minute)
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.make_forge2       # data/forge2: harder training benchmark (about 1 minute)
```

Expected: `make_dataset` prints a table of counts per split and tool; `audit_dataset` ends without an error and reports zero source overlap between splits. JPEG encoders differ slightly between OpenCV builds, so your images may not be byte-identical to the author's, and metrics can move in the third decimal.

## 5. Smoke test: one epoch (about 40 seconds)

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.train --variant single_a --seed 0 --epochs 1 --out runs_smoke
```

Expected: a line like `epoch 1/1 ... val_auroc {'A': 0.93}` and `finished runs_smoke/single_a_s0`. If you get an out-of-memory error from the GPU, lower the batch size with `--bs 8`.

## 6. Reproduce one headline run (about 5 minutes)

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.train --variant single_a --seed 0 --out runs_mine
PYTHONPATH=src .venv/bin/python -c "from hybrid_forensics.analysis import primary; print(round(primary('runs_mine/single_a_s0','A'), 4))"
```

The number printed is the primary metric: test AUROC on authentic, splice and copy-move images. The author got **0.9966** for this run and 0.9961 when it was retrained, and across seeds 0, 1, 2 the range was 0.991 to 0.997. Anything from about 0.99 to 0.997 is a successful reproduction. Validation AUROC should climb from about 0.92 at epoch 1 to about 0.987 at epoch 12.

Use `--out runs_mine` so you do not overwrite the results that came with the repo.

## 7. Reproduce the pair variants (about 10 minutes each)

```
for v in indep ml ml_hem cfr_adaptive; do
  PYTHONPATH=src .venv/bin/python -m hybrid_forensics.train --variant $v --seed 0 --out runs_mine
done
```

The primary metric for these (use key `fused`) was 0.979 to 0.985 for the author:

```
PYTHONPATH=src .venv/bin/python -c "from hybrid_forensics.analysis import primary; [print(v, round(primary(f'runs_mine/{v}_s0'),4)) for v in ('indep','ml','ml_hem','cfr_adaptive')]"
```

The full headline study is 24 runs over three seeds and takes about 3 hours: `tools/queue_stage2.sh`, with progress from `tools/queue_progress.py --watch`. It writes into `runs/`, which already holds the author's runs, so copy or rename `runs/` first if you want to keep them.

## 8. Test on the tool-disjoint set `forge_hard`

This is the test that showed every model is at chance on Poisson-blended edits. It needs a trained checkpoint, so use the run from step 6.

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.eval_manifest score data/forge_hard/manifest.csv test hard runs_mine/single_a_s0
```

It prints AUROC per tool. Expected (author, seed 0): `splice_feather` about 0.994, `copy_move_feather` about 0.990, `self_resample` about 0.907, and **`splice_poisson` and `copy_move_poisson` about 0.52 (chance)**. The two feathered tools are the training tool rebuilt, so they are the control: if they are far from 0.99, something in your data build differs.

## 9. Stage 2: the unfinished experiment (about 3 hours on a free machine)

This is the work that was stopped after 1 of 36 runs. It retrains the main comparison on `forge2`, which has headroom, using the rule written in `NEXT_PHASE_PLAN.md` before any result existed.

```
tools/queue_forge2.sh                       # 36 runs into runs_f2/, seed-major
tools/queue_f2_progress.py --watch          # in another terminal: runs finished / 36, elapsed, ETA from the measured rate
```

Before you leave it running, make sure about 3 GB of RAM is free (close browsers and other heavy programs). After the first two runs finish, check that Model A is not already at ceiling on the four seen tools:

```
tools/score_f2.sh
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.f2_analysis > reports/evidence/next_phase/f2_analysis.md
```

Then read `reports/evidence/next_phase/f2_analysis.md`. If `single_a` is at 0.99 or above on the primary metric, the benchmark is saturated again and the comparison is not informative. Stop and report that rather than continuing. Delete the unusable `runs_f2/ml_lam0.25_s0` folder if it is present before starting.

When it finishes, `f2_analysis` prints the paired comparisons and a verdict per pair under the pre-declared rule. Do not change the rule after you look at the numbers.

## 10. Try the application on one image

```
PYTHONPATH=src .venv/bin/python -m hybrid_forensics.infer path/to/image.jpg --ckpt runs/cfr_adaptive_s0/best.pt
```

It prints a label, a probability and writes an overlay and JSON. Images are processed in 384 px windows and never resized. Allowed types are JPEG and PNG, 128 to 4096 px per side. It was trained only on synthetic edits, so treat the answer as a demonstration, not a verdict.

## Common problems

| symptom | cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | driver too old or wrong wheel; reinstall torch with the index that matches `nvidia-smi` |
| `CUDA out of memory` while training | add `--bs 8`; close other GPU programs |
| the machine freezes or Python is killed near the end of a run | system RAM is exhausted; close other programs and rerun that one run |
| `FileNotFoundError: data/forge/...` | step 4 not finished, run the builders from the repo root |
| download stops | rerun `fetch_coco_subset.py`, it resumes |
| a run is slower than the times above | normal on a slower GPU; use the progress readers for a measured ETA |

## What to send back

The `results.json` and `probs.npz` files inside each run folder (they are small), plus the output of step 3 and step 6, and anything that failed with the exact error text. Please do not send the images.
