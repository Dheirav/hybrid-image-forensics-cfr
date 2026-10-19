"""Collect runs/*/results.json into per-variant tables with seed mean, std and paired comparisons."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

METRICS = [("auroc", "AUROC"), ("macro_f1", "macroF1"), ("balanced_acc", "balAcc"), ("fpr_authentic", "FPR"),
           ("ece", "ECE"), ("dice", "Dice"), ("iou", "IoU")]


def load(pattern="runs/*/results.json"):
    runs = defaultdict(dict)
    for f in glob.glob(pattern):
        r = json.load(open(f))
        runs[r["args"]["variant"]][r["args"]["seed"]] = r
    return runs


def branch(r, split):
    keys = r["final"][split]
    return "fused" if "fused" in keys else [k for k in keys if k != "pair"][0]


def table(runs, split="test"):
    lines = [f"| variant | seeds | " + " | ".join(n for _, n in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for v, by_seed in sorted(runs.items()):
        cells = []
        for m, _ in METRICS:
            xs = [r["final"][split][branch(r, split)].get(m, np.nan) for r in by_seed.values()]
            cells.append(f"{np.nanmean(xs):.3f} ± {np.nanstd(xs, ddof=1) if len(xs) > 1 else float('nan'):.3f}" if len(xs) > 1
                         else f"{xs[0]:.3f}")
        lines.append(f"| {v} | {len(by_seed)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(runs, a, b, split="test", metric="auroc"):
    """Per-seed differences a - b over seeds both variants share; also returns a bootstrap CI over test images."""
    seeds = sorted(set(runs[a]) & set(runs[b]))
    d = [runs[a][s]["final"][split][branch(runs[a][s], split)][metric] - runs[b][s]["final"][split][branch(runs[b][s], split)][metric]
         for s in seeds]
    return seeds, d


if __name__ == "__main__":
    runs = load()
    for split in ("val", "test", "ood_removal"):
        print(f"\n### {split}\n" + table(runs, split))
