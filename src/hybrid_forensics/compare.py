"""Headline comparison with the decision rule declared in reports/02 (addendum).

Primary: fused AUROC on test restricted to authentic + splice + copy_move. Per-seed paired differences, and a
paired bootstrap over test images on seed-averaged probabilities. Also secondary metrics and pair diagnostics.
usage: python -m hybrid_forensics.compare > reports/evidence/comparison.md"""
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from .analysis import SEEN, by_tool, split_frame, subset
from .metrics import classification_metrics

VARIANTS = ["single_a", "single_b", "indep", "ml", "ml_hem", "cfr_adaptive", "cfr_sudden", "cfr_fixed", "cfr_scratch"]
PAIRS = [("cfr_adaptive", "ml"), ("cfr_adaptive", "ml_hem"), ("ml", "indep"), ("ml_hem", "ml"), ("cfr_adaptive", "cfr_sudden"),
         ("cfr_adaptive", "cfr_fixed"), ("cfr_adaptive", "cfr_scratch"), ("cfr_adaptive", "indep"), ("cfr_adaptive", "single_a")]


def key_of(v):
    return {"single_a": "A", "single_b": "B"}.get(v, "fused")


def runs():
    out = {}
    for f in sorted(glob.glob("runs/*/results.json")):
        r = json.load(open(f))
        out.setdefault(r["args"]["variant"], {})[r["args"]["seed"]] = os.path.dirname(f)
    return out


def prim(run, v, split="test", tools=SEEN):
    p, y, _ = subset(run, split, tools, key_of(v))
    return roc_auc_score(y, p)


def boot_diff(R, a, b, n=2000, seed=0, tools=SEEN):
    seeds = sorted(set(R[a]) & set(R[b]))
    pa = np.mean([subset(R[a][s], "test", tools, key_of(a))[0] for s in seeds], 0)
    pb = np.mean([subset(R[b][s], "test", tools, key_of(b))[0] for s in seeds], 0)
    y = subset(R[a][seeds[0]], "test", tools, key_of(a))[1]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), np.percentile(d, [2.5, 97.5])


def main():
    R = runs()
    print("## Headline: fused test AUROC, authentic + splice + copy_move (primary)\n")
    print("| variant | seeds | primary AUROC mean ± sd | per-seed | removal (test) | ood_removal | val AUROC | test ECE | FPR authentic | Dice | pair disagreement | corr(A,B) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        if v not in R:
            continue
        seeds = sorted(R[v])
        P = [prim(R[v][s], v) for s in seeds]
        rm = [by_tool(R[v][s], key_of(v))["removal"] for s in seeds]
        res = [json.load(open(f"{R[v][s]}/results.json"))["final"] for s in seeds]
        k = key_of(v)
        g = lambda sp, m: np.mean([r[sp][k][m] for r in res])
        ood = np.mean([r["ood_removal"][k]["auroc"] for r in res])
        dis = np.mean([r["test"]["pair"]["disagreement"] for r in res]) if "pair" in res[0]["test"] else float("nan")
        cor = np.mean([r["test"]["pair"]["prob_corr"] for r in res]) if "pair" in res[0]["test"] else float("nan")
        print(f"| {v} | {len(seeds)} | {np.mean(P):.4f} ± {np.std(P, ddof=1) if len(P) > 1 else float('nan'):.4f} | "
              f"{', '.join(f'{p:.4f}' for p in P)} | {np.mean(rm):.3f} | {ood:.3f} | {g('val', 'auroc'):.4f} | {g('test', 'ece'):.3f} | "
              f"{g('test', 'fpr_authentic'):.3f} | {g('test', 'dice'):.3f} | {dis:.3f} | {cor:.3f} |")
    print("\n## Paired comparisons (A minus B), primary metric\n")
    print("| comparison | shared seeds | per-seed difference | mean | 95% paired bootstrap over test images (seed-averaged probs) | positive in every seed |")
    print("|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if a not in R or b not in R:
            continue
        seeds = sorted(set(R[a]) & set(R[b]))
        d = [prim(R[a][s], a) - prim(R[b][s], b) for s in seeds]
        pt, ci = boot_diff(R, a, b)
        print(f"| {a} - {b} | {len(seeds)} | {', '.join(f'{x:+.4f}' for x in d)} | {np.mean(d):+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {'yes' if all(x > 0 for x in d) else 'no'} |")
    print("\n## Same comparisons on unseen-tool removal (test removal + authentic; secondary, headroom exists here)\n")
    print("| comparison | per-seed difference | mean |")
    print("|---|---|---|")
    for a, b in PAIRS:
        if a not in R or b not in R:
            continue
        seeds = sorted(set(R[a]) & set(R[b]))
        d = [by_tool(R[a][s], key_of(a))["removal"] - by_tool(R[b][s], key_of(b))["removal"] for s in seeds]
        print(f"| {a} - {b} | {', '.join(f'{x:+.4f}' for x in d)} | {np.mean(d):+.4f} |")
    print("\n## Transition controller\n")
    for v in ("cfr_adaptive", "cfr_sudden", "cfr_fixed"):
        for s in sorted(R.get(v, {})):
            r = json.load(open(f"{R[v][s]}/results.json"))
            print(f"- {v} seed {s}: trigger epoch {r['trigger_epoch']} (0-indexed epoch at which CFR weight first applies; None = never triggered)")


if __name__ == "__main__":
    main()
