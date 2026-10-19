## Leakage and integrity

- unreadable images or masks: 0 of 3000 images
- source overlap forge_hard val with test: 0; with COCO-Forge train: 0
- forge_hard val sources equal COCO-Forge val sources: True; test equal COCO-Forge test: True
- splice donors outside the split of the image they were pasted into: 0
- manipulated rows without a mask: 0; authentic rows with a mask: 0
- near-duplicate pairs (64-bit dHash Hamming at most 6) between forge_hard authentic crops and COCO-Forge TRAIN authentic crops: 0

## Per-tool integrity: mask size and where pixels changed (test split, against the same source's authentic crop)

| tool | n | mask area median | changed >30 inside mask | mean abs change inside mask | changed >30 far outside |
|---|---|---|---|---|---|
| copy_move_feather | 250 | 0.098 | 0.83 | 93.0 | 0.0012 |
| copy_move_poisson | 250 | 0.081 | 0.094 | 12.7 | 0.0013 |
| self_resample | 250 | 0.126 | 0.349 | 34.5 | 0.0012 |
| splice_feather | 250 | 0.105 | 0.904 | 105.0 | 0.0017 |
| splice_poisson | 250 | 0.074 | 0.098 | 13.2 | 0.0011 |

## Compression and metadata shortcuts

JPEG quality against label, chi-square p = 0.30 (quality is drawn independently of the label).

| tool | file size alone (test AUROC) | quality + size, fitted on forge_hard val |
|---|---|---|
| copy_move_feather | 0.508 | 0.504 |
| copy_move_poisson | 0.487 | 0.465 |
| self_resample | 0.483 | 0.541 |
| splice_feather | 0.491 | 0.501 |
| splice_poisson | 0.515 | 0.494 |

## Local-statistics baseline: 32 x 32 patch features, fitted on forge_hard VALIDATION (all tools pooled), scored on test per tool

A hand-built detector, not a deep model. `inside` takes patches from the edited region (a positive control: can a simple feature see this edit at all), `outside` takes patches at least 15 px from it (should sit at 0.5 if the edit is local).

| tool | inside patches, image AUROC | outside patches, image AUROC |
|---|---|---|
| copy_move_feather | 0.851 | 0.474 |
| copy_move_poisson | 0.549 | 0.549 |
| self_resample | 0.843 | 0.489 |
| splice_feather | 0.84 | 0.525 |
| splice_poisson | 0.573 | 0.516 |
