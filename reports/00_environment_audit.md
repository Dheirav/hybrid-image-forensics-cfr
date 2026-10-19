# 00 Environment audit

Run on 2026-10-18 at 10:16 IST. Everything below was read or executed in this pass. Nothing was trained, installed or modified outside `reports/`.

## What `nitin-work` contains

| path | content |
|---|---|
| `PROMPT.md` | the validation brief (12 phases) |
| `reference/deepfake-project-audit/` | four reports about a different repository, `Deepfake_Detection/deepfake-project`. Its own README says not to quote them as results about this project |

There is no source code, no config, no notebook, no manifest, no checkpoint, no dataset, no tests and no experiment log. Phase 0 asks me to inspect the repository, and this one has nothing to inspect. That is the finding, and it decides everything downstream (see `01`).

## Search for the project elsewhere on this machine

- `grep` for "competitive forensic" and "mutual learning" across `~/Code` (md, py, txt) found nothing outside `nitin-work` itself and the unrelated PokerBot trees. The PokerBot `cfr` hits are counterfactual regret minimisation, a different CFR.
- No CASIA, Columbia, CoMoFoD, DEFACTO, NIST16, IMD2020 or COCOGlide directory under `/home/dheirav` or the Windows user folders. `~/opensdi_probe` exists and belongs to the Deepfake_Detection work.
- `Deepfake_Detection/deepfake-project` is the nearest code. It has a CLIP-based three-class detector with a mask decoder and cross-generator evaluation, but no second (forensic) branch for mutual learning, no adaptive transition and no CFR. Whether it is the intended base is a decision for you, not something I should assume.

## Hardware and software

| item | value |
|---|---|
| GPU | RTX 4060 Laptop, 8188 MiB total, 1745 MiB in use before any work of mine |
| System RAM | 7.8 GiB total, 5.3 GiB available (825 MiB free, the rest is page cache) |
| CPU | 8 logical cores |
| Disk | `/` 1007 GB, 673 GB free. `/mnt/c` not used for working files |
| Python | system 3.12.3. This directory has no venv, so no project interpreter exists yet |
| Dependencies | none for this project. The torch stack lives in the Deepfake_Detection venvs and has not been touched |

### Compute is shared

Several `PokerBot/scripts/chipzen_duel.py` processes are running at roughly 100 percent CPU each, which is most of the 8 cores, plus about 12 percent of RAM each as reported by `ps`. Any training here competes with them. The earlier audit also recorded that GPU jobs started from a Claude Code shell get killed by the harness's low-memory heuristic on this box and have to be started detached. I did not touch the PokerBot processes.

## Pretrained weights and licences

Nothing is cached for this project. From the earlier audit of the neighbouring repo (not re-verified now): CLIP ViT-B/16, B/32 and L/14 are in `~/.cache/clip/` under the MIT licence. EfficientNet-B0 and ConvNeXt-T would be downloaded through torchvision, with the usual ImageNet research-use caveat. `timm` was not installed there.

## Not executed, and why

- No dataset audit, forward pass or training: there is no data or code to run them on.
- No installs, because the dependency set depends on which base is chosen.
