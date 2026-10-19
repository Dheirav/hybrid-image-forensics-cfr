# 07 Mutual learning and alignment

Question: when Model B is trained with a KL term toward Model A (mutual learning, `ml`), does it stay aligned with A, does it
get better, and does the alignment cost A anything? Three seeds per variant, test split, authentic + splice + copy_move.
Computed by `src/hybrid_forensics/mutual_analysis.py`; the table is in `reports/evidence/mutual_analysis.md`.
Descriptive only: the pre-declared decision rule is applied in `12`, not here.

## Setup

`indep` trains A and B on the same batches with no alignment term. `ml` adds a symmetric temperature-2 KL (partner detached,
T^2 scaling, lambda = 1) from the start, with the ground-truth cross-entropy and the mask loss unchanged. So `indep` against
`ml` isolates the effect of alignment alone.

## What the numbers show (mean over 3 seeds)

| variant | A AUROC | B AUROC | A and B agree | corr of logits | mean abs gap in p | share of B's errors that are also A's |
|---|---|---|---|---|---|---|
| indep | 0.993 | 0.899 | 0.828 | 0.729 | 0.261 | 0.130 |
| ml | 0.988 | 0.899 | 0.838 | 0.837 | 0.152 | 0.168 |
| single_b (B trained alone) | | 0.911 | | | | |

1. **B does become aligned with A.** The logit correlation rises from 0.73 to 0.84 and the mean probability gap falls from 0.26
   to 0.15. The two models are not collapsed: they still disagree on about 16 percent of images, and the gap does not go to zero.
2. **Alignment did not make B better.** B scores 0.899 in both `indep` and `ml`, and B trained entirely alone scores 0.911.
   The alignment pulled B toward A's outputs without raising B's own AUROC, which means the agreement gained is mostly
   agreement in calibration and confidence, not new detection ability.
3. **B copies a little of A's mistakes.** Of the images B gets wrong, 17 percent are also wrong for A under `ml`, against 13 percent
   under `indep`. This is a small shift, and since A is wrong on only about 4 percent of images, B's errors are mostly B's own
   (it errs on 18 percent). So B is not inheriting A's error pattern in any large way, but the direction of the change is toward sharing errors.
4. **Alignment cost A a little.** A falls from 0.993 (`indep`) to 0.988 (`ml`). With a per-seed sd of about 0.003 this is
   roughly the size of the noise, so I treat it as a possible small cost, not a proven one. The paired result in `12` shows the
   fused score of `ml` below `indep` by 0.0050, bootstrap interval [-0.0118, -0.0004]. That interval just excludes zero, on three seeds, so it is
   weak evidence against mutual learning on this benchmark, not for it.
5. **No calibration gain.** Fused test ECE is 0.061 for `ml` and 0.060 for `indep`.

## Reading

Mutual learning in this setup behaves as designed mechanically (alignment rises, no collapse), but it did not deliver the
benefit that motivates it: a stronger Model B, or a stronger fused decision. This is consistent with the main limit seen
in `06`: A is near ceiling, B is far below it, and a KL term toward a much stronger partner cannot give B information
B's inputs do not contain. A caveat on interpretation: all of this is one dataset with one set of synthetic edits and 12 epochs
of training, and I did not tune lambda or the temperature.

## Update: alignment strength, error correction and macro-F1

The brief asks for different alignment strengths and for a check on whether alignment makes wrong predictions harder to correct. Both were added.
Runs: `runs_extra/ml_lam0.25_s*` and `ml_lam4_s*` (alignment weight 0.25 and 4, everything else as `ml`, which uses 1; `indep` is the weight-0 case). Three seeds each.
Source: `reports/evidence/extra_analysis.md`.

### Alignment strength

| alignment weight | variant | primary AUROC (mean +/- sd) | FPR authentic | removal AUROC |
|---|---|---|---|---|
| 0 | `indep` | 0.9841 +/- 0.0029 | 0.033 | 0.704 |
| 0.25 | `ml_lam0.25` | 0.9858 +/- 0.0019 | 0.031 | 0.728 |
| 1 | `ml` | 0.9791 +/- 0.0023 | 0.053 | 0.716 |
| 4 | `ml_lam4` | 0.9568 +/- 0.0068 | 0.089 | 0.705 |

| comparison | per seed | mean | 95% interval |
|---|---|---|---|
| lam 0.25 - lam 1 (`ml`) | +0.0075, +0.0024, +0.0103 | +0.0067 | [+0.0010, +0.0131] |
| lam 4 - lam 1 (`ml`) | -0.0206, -0.0187, -0.0274 | -0.0224 | [-0.0317, -0.0143] |
| lam 0.25 - `indep` | +0.0002, +0.0029, +0.0022 | +0.0013 | [-0.0028, +0.0057] |
| lam 4 - `indep` | -0.0279, -0.0183, -0.0355 | -0.0278 | [-0.0408, -0.0169] |

The effect of alignment strength on the fused score is monotone once the weight is above about 0.25: more alignment is worse. Weight 4 costs 2.8 points of
AUROC against no alignment and raises the false-positive rate on authentic images from 3 to 9 percent, both well outside the seed spread. Weight 0.25 is level with no
alignment (+0.0013) and better than the weight I used for the headline runs (+0.0067, interval excludes zero). So **the headline mutual-learning weight of 1 was already on the harmful
side of the curve**, and that weakens the `ml` control in `09` and `12` (it makes `ml` a somewhat easier baseline for CFR to beat than a tuned mutual-learning would be). I did not retune the headline
runs, because that would have meant choosing a setting after seeing test-based comparisons; the headline weight stays at the value fixed in advance, and this table is reported next to it.

### Does alignment make wrong predictions harder to correct

Test, seen tools, mean over seeds. "B fixes A's miss" is the share of images A gets wrong that B gets right; "A fixes B's miss" the reverse.

| variant | A macro-F1 | B macro-F1 | A wrong | B fixes A's miss | B wrong | A fixes B's miss | fused macro-F1 |
|---|---|---|---|---|---|---|---|
| indep | 0.958 | 0.815 | 0.040 | 0.417 | 0.178 | 0.870 | 0.957 |
| ml | 0.958 | 0.805 | 0.040 | 0.231 | 0.183 | 0.832 | 0.925 |
| ml_hem | 0.962 | 0.774 | 0.037 | 0.327 | 0.199 | 0.879 | 0.933 |
| cfr_adaptive | 0.962 | 0.825 | 0.037 | 0.307 | 0.169 | 0.848 | 0.943 |
| cfr_sudden | 0.963 | 0.835 | 0.035 | 0.315 | 0.159 | 0.849 | 0.937 |
| cfr_fixed | 0.964 | 0.823 | 0.034 | 0.325 | 0.171 | 0.865 | 0.952 |
| cfr_scratch | 0.961 | 0.784 | 0.038 | 0.443 | 0.197 | 0.891 | 0.948 |

The direction is that alignment makes A's mistakes harder for B to correct, with a caveat on size: A is wrong on only about 17 of the 417 images per seed (about 50 images over three seeds), so each share rests on roughly 50 events and differences of 0.1 are not reliable. I did not bootstrap these. Without alignment (`indep`) B gets 42 percent of A's mistakes right; under mutual learning that falls to 23 percent, so B has been pulled
toward A's errors. The competitive variants recover part of it (31 to 33 percent for the aligned CFR runs) and `cfr_scratch` is back at or above the `indep` level (44 percent against 42). That is what the ground-truth gating in CFR
is meant to do, and it is the one place where the CFR design shows its intended mechanism at work. The other direction hardly changes (A corrects 83 to 89 percent of B's mistakes in every variant),
because A is simply right most of the time. The cost is visible in the fused macro-F1 at 0.5: 0.957 for `indep` against 0.925 for `ml`, with the CFR variants in between (0.937 to 0.952).
