# 08 Adaptive transition from mutual learning to CFR

Controller: `src/hybrid_forensics/controller.py`, unit-tested in `tests/test_controller.py`. Runs: `cfr_adaptive`, `cfr_sudden`,
`cfr_fixed` (CFR weight switched on at epoch 5 regardless of state) and `cfr_scratch` (CFR from the first step), seeds 0 to 2.
Per-epoch check logs are in `runs/*/results.json` under `transition`; the summary below is `reports/evidence/mutual_analysis.md`.

## How the controller decides

An epoch is eligible when all of these hold: past a minimum warm-up (3 epochs), both models above a minimum AUROC (0.60),
validation AUROC of both stable (change under 0.03 and loss not rising by more than 10 percent), training stable,
disagreement inside a window (0.03 to 0.35), and ECE at most 0.20. After 2 consecutive eligible epochs ("patience") the CFR
weight ramps up over 4 epochs (gamma 0.25, 0.5, 0.75, 1.0). It never forces a transition. `cfr_sudden` uses the same trigger
and then sets gamma to 1 at once.

A caveat about the checks: with two classes, "complementary errors" and "disagreement" are the same quantity, so those
two checks are one window, not two independent tests. I did not fix this.

## What happened

| run | trigger epoch (CFR weight first applied, 0-indexed), seeds 0 / 1 / 2 |
|---|---|
| cfr_adaptive | 7 / 6 / 8 |
| cfr_sudden | 7 / 10 / 8 |
| cfr_fixed | 5 / 5 / 5 (by design) |

The controller always triggered, and always later than the fixed epoch 5. The check that held it back longest was the
validation-stability check: it failed in nearly every epoch up to the trigger (for example seed 2 failed it at epochs 3 to 5).
The early epochs fail on warm-up, B not yet above 0.60 AUROC, and ECE above 0.20, which is the intended behaviour: B starts at chance
(`05`), so the controller correctly refused to align a model that had not learned anything.

Because the trigger fell at epoch 6 to 8 of 12, CFR was active for about 4 to 6 epochs including the ramp, and fully on for
roughly 1 to 3 of them. That is a short exposure for a method whose effect I am trying to measure.

## Does adaptive beat the alternatives

Primary metric (fused test AUROC, seen tools), paired over seeds, from `reports/evidence/comparison.md`:

| comparison | mean difference | per seed | 95% paired bootstrap |
|---|---|---|---|
| adaptive - sudden | -0.0007 | -0.0037, +0.0032, -0.0017 | [-0.0041, +0.0009] |
| adaptive - fixed epoch 5 | -0.0012 | -0.0044, +0.0027, -0.0018 | [-0.0049, +0.0001] |
| adaptive - CFR from scratch | -0.0026 | -0.0042, -0.0015, -0.0019 | [-0.0102, -0.0012] |

The adaptive schedule is not better than a sudden switch or a fixed switch epoch; both differences are inside the noise.
It is slightly worse than CFR applied from the first step, and that interval excludes zero (negative in all three seeds).
On unseen-tool removal, adaptive is behind sudden (-0.013) and fixed (-0.018) in all three seeds, and ahead of scratch by +0.004
with seeds disagreeing in sign.

So on this benchmark the transition schedule has not earned its complexity. The controller does what it is
designed to do (refuses early alignment, triggers once both models are stable), but triggering later and with a gentler ramp did not
turn into better results than switching early or switching abruptly.

## What this does not establish

- Only one trigger configuration was run. The thresholds are my defaults and were not tuned, deliberately, to avoid tuning on results.
- With 12 epochs there is little room between "late enough to be stable" and "early enough to matter". A longer schedule might behave differently.
- The `cfr_scratch` win is the opposite of the controller's premise (that warm-up helps). It is one dataset and three seeds, so I report
  it and do not build on it.

## Update: validation metrics before and after the transition, and the loss weight log

The brief asks that the log shows eligibility, the trigger decision, the transition epoch, the loss weights and the validation metrics before and after.
Eligibility and trigger per epoch are in `runs/*/results.json` under `transition` (summarised above); the weights and the before/after metrics are below.
`before` is the epoch just before the CFR weight first applied, `final` is the last epoch. Validation split only. Source: `reports/evidence/extra_analysis.md`, section 4.

| run | trigger epoch | gamma by epoch (0 to 11) | fused val AUROC before / final | A before / final | B before / final | disagreement before / final |
|---|---|---|---|---|---|---|
| adaptive s0 | 7 | 0 0 0 0 0 0 0 .25 .5 .75 1 1 | 0.955 / 0.974 | 0.974 / 0.988 | 0.847 / 0.891 | 0.228 / 0.168 |
| adaptive s1 | 6 | 0 0 0 0 0 0 .25 .5 .75 1 1 1 | 0.962 / 0.972 | 0.972 / 0.985 | 0.830 / 0.885 | 0.310 / 0.176 |
| adaptive s2 | 8 | 0 0 0 0 0 0 0 0 .25 .5 .75 1 | 0.964 / 0.975 | 0.984 / 0.987 | 0.863 / 0.896 | 0.210 / 0.180 |
| sudden s0 | 7 | 0 0 0 0 0 0 0 1 1 1 1 1 | 0.956 / 0.973 | 0.976 / 0.988 | 0.847 / 0.888 | 0.204 / 0.182 |
| sudden s1 | 10 | 0 x10, then 1 1 | 0.964 / 0.970 | 0.982 / 0.986 | 0.880 / 0.883 | 0.196 / 0.190 |
| sudden s2 | 8 | 0 x8, then 1 1 1 1 | 0.961 / 0.978 | 0.983 / 0.989 | 0.872 / 0.897 | 0.194 / 0.184 |
| fixed s0 | 5 | 0 x5, then .25 .5 .75 1 1 1 1 | 0.961 / 0.970 | 0.974 / 0.985 | 0.821 / 0.890 | 0.382 / 0.192 |
| fixed s1 | 5 | same | 0.958 / 0.976 | 0.975 / 0.989 | 0.816 / 0.879 | 0.256 / 0.192 |
| fixed s2 | 5 | same | 0.957 / 0.976 | 0.970 / 0.983 | 0.784 / 0.884 | 0.290 / 0.198 |

Every run improved on validation after the transition, and the disagreement between the branches fell, mostly into the 0.17 to 0.20 range. This is not evidence that the transition helped,
because validation AUROC also rises over epochs in the mutual-learning runs as the one-cycle learning rate decays; there is no matched no-transition curve at the same epoch in this table.
The cleaner comparison is the end-of-training one against `ml` in `09`. The fixed run shows the largest disagreement just before its switch (0.26 to 0.38), which the adaptive controller avoids
(0.21 to 0.31) by waiting for stability, which is a small observed effect of the controller doing what it was designed to do.
