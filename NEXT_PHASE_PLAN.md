# Next phase plan (written 2026-10-19, IST morning)

Baseline: the 42 finished runs in `runs/` and `runs_extra/`, the reports `00` to `15`, and `NEXT.md`. Nothing in them is rerun
unless it is named below. New outputs go to `runs_next/`, `data/forge_hard/` and `reports/evidence/next_phase/`.

One thing I had seen before writing the criteria below: a two-run smoke test of the hard tool set (single_a seed 0 and cfr_adaptive seed 0)
showed chance-level AUROC on Poisson-blended edits. The criteria were written after that glimpse, so they are not blind, and the full
42-run table is what counts, not those two runs. Every other criterion below was written before its result existed.

## Baseline numbers to verify (Experiment A)

The brief lists A 0.9936, B 0.9108, learned fusion 0.9929, mutual learning 0.9791, mutual learning plus hard-example mining 0.9844,
adaptive CFR 0.9824, external 0.554 and 0.594. All of the COCO-Forge numbers are the three-seed mean of the primary metric (fused test AUROC on authentic, splice
and copy-move; test split). The two external numbers are on a different dataset (`data/external`, zero-shot, 400 real against 400 AI-edited or
AI-generated) and are not comparable with them. The checks: recompute each from the checkpoints through a second code path, compare with the stored
probabilities, and re-train `single_a` seed 0 once to see how far a rerun moves.

## Diagnoses and the evidence that decides each

| question | evidence | pre-declared reading |
|---|---|---|
| D1. Why is A so strong? Seam, scene content, or shortcut | `forge_hard` (5 editing tools on the same test sources, one authentic crop each): the training tool rebuilt, Poisson-blended splice and copy-move (same scene content, no hard seam), and `self_resample` (own object re-pasted in place, so no new scene content) | A is **seam-driven** if its AUROC falls by more than 0.10 from `splice_feather` to `splice_poisson`. It is **not content-driven** if `self_resample` AUROC stays above 0.80 with no object added. Pipeline control: the rebuilt training tools must score within 0.02 of COCO-Forge test, else the new pipeline is not comparable. |
| D2. Why is B weaker, and why does fusion not help | stacked logistic regression on A and B logits, fitted on validation and scored on test (single_a and single_b of the same seed); B's AUROC on the images A gets wrong or is unsure about; linear CKA between A and B embeddings; a linear probe on A embeddings against A+B embeddings; B per tool on `forge_hard` | B **adds information** if the stacked A+B test AUROC beats A alone with a paired 95 percent interval excluding 0 in at least two of three seeds. |
| D3. Do the models learn different evidence | logit correlation, error overlap, CKA, and whether B beats A on any tool or stratum | reported, no pass mark |
| D4. Why does CFR not beat hard-example mining | per-sample CFR and hard-example weights computed on the training images from the finished checkpoints: share of samples with a real challenge, effective sample size, rank correlation between the two weightings, alignment gate occupancy, epochs actually spent at full CFR | CFR is **effectively hard-example mining** if the Spearman correlation of the two weight vectors is above 0.9 for both models. |
| D5. Why does external detection fail | score distributions on external real against COCO-Forge authentic; leave-one-generator-out linear probes on frozen A and B embeddings; recall by edit area; mean against max window aggregation (diagnostic only, not adopted) | **Head or domain shift** if a probe on frozen embeddings reaches AUROC above 0.75 where the trained head is near 0.55. **Representation limit** if the probe is also below 0.65. |

## Follow-up experiments

| id | experiment | cost | status rule |
|---|---|---|---|
| A | baseline verification (above) plus one `single_a` retrain | about 6 min GPU | always |
| B | shortcut and leakage: source and near-duplicate overlap for `forge_hard`, donor split check, JPEG quality against label on the new set, file-size baseline, plus the D1 pipeline control | minutes, CPU | always |
| C | forensic-branch diagnosis (D2, D3) | minutes, GPU inference | always |
| D | fusion: stacked fusion and learned concat against A alone, same images, validation-only fitting | minutes | always |
| E | CFR against mutual learning and hard-example mining. Stage 1: on `forge_hard` for all existing seeds (tool-disjoint, no retraining). Stage 2: **only if** Stage 1 shows headroom for A on tool-disjoint data (A AUROC below 0.95 on at least two of the new tools): retrain on a training set that includes the new tools (`forge2`), tool-held-out test, at least 5 seeds for `single_a`, `ml_lam0.25`, `ml_hem`, `cfr_adaptive`, `joint_concat` | stage 2 about 25 runs at 5 to 10 min each, 2 to 4 h GPU | conditional |
| F | external generalisation, conventional against AI-edited against AI-generated, with the D5 probes | minutes | always |

## Success criteria (fixed before the Stage 2 result exists)

- A proposed change counts as an improvement only if it beats its comparator on the same test images, positive in every seed, with a 95 percent paired bootstrap interval over images excluding 0, and without worse authentic false-positive rate or ECE. Same rule as `reports/02`.
- CFR is "shown to help" only if that holds against **both** mutual learning at the better of weights 0.25 and 1, and hard-example mining. Otherwise the report says the current CFR formulation has not demonstrated its claimed benefit.
- A fusion is "shown to help" only if it beats Model A alone under the same rule.
- Nothing on the COCO-Forge benchmark counts as evidence about AI-edited or AI-generated images. Those are scored on `data/external` separately and are only ever tested there.
- Thresholds and model selection use validation only. Test splits are scored once per run. Threshold 0.5 stays the primary protocol, a validation-chosen threshold is a second column.
- Reported per scored comparison: AUROC, macro-F1, balanced accuracy, false-positive rate, and Dice and IoU where a mask exists, with per-seed values.

## Decision options at the end

Retain and refine (A-plus-B shows a reproducible gain somewhere), revise CFR (the diagnosis points to a fixable defect), or drop the current formulation
for the better-supported approach (a single strong visual model, trained for tool diversity, with the forensic branch only if D2 says it adds information).
