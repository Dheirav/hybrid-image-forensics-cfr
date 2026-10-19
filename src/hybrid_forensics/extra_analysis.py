"""Analyses the brief asks for that the first pass left out (reports 07, 08, 09, 11, 12 addenda).

 1. mask-size strata   : detection by small / medium / large manipulated region (test, tercile cut on mask area)
 2. selective prediction: accuracy vs coverage when abstaining on low confidence vs on branch disagreement
 3. alignment and correction: does alignment make A's mistakes harder for B to correct (and the reverse)
 4. transition before/after: validation metrics either side of the CFR trigger, with the loss weight per epoch
 5. compute-matched control (variant J): parameters/FLOPs, and a two-Model-A ensemble from single_a seeds
 6. extra variants (joint fusion, no-alignment CFR, alignment-strength sweep) vs their natural comparators
Everything is computed from saved probabilities / results.json; nothing is retrained and nothing is tuned.
usage: python -m hybrid_forensics.extra_analysis > reports/evidence/extra_analysis.md"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .analysis import SEEN, split_frame, subset

PAIRS = ["indep", "ml", "ml_hem", "cfr_adaptive", "cfr_sudden", "cfr_fixed", "cfr_scratch"]


def all_runs():
    out = {}
    # COCO-Forge runs only: runs_f2 and runs_next score other manifests
    for f in sorted(glob.glob("runs/*/results.json") + glob.glob("runs_extra/*/results.json")):
        a = json.load(open(f))["args"]
        out.setdefault(a["variant"] + a.get("tag", ""), {})[a["seed"]] = os.path.dirname(f)
    return out


def key_of(v):
    return {"single_a": "A", "single_b": "B"}.get(v, "fused")


def mask_strata():
    df = split_frame("test")
    area = np.array([(cv2.imread(m, 0) > 127).mean() if isinstance(m, str) and m else np.nan for m in df.mask_path])
    man = df.manipulation.isin(["splice", "copy_move"]).values
    cuts = np.nanquantile(area[man], [1 / 3, 2 / 3])
    return df, area, cuts


def table_mask_strata(R):
    df, area, cuts = mask_strata()
    seen = df.manipulation.isin(SEEN).values
    strata = {"small": lambda a: a <= cuts[0], "medium": lambda a: (a > cuts[0]) & (a <= cuts[1]), "large": lambda a: a > cuts[1]}
    print(f"## 1. Detection by manipulated-region size (test, splice + copy-move vs all authentic)\n")
    print(f"Terciles of mask area cut at {cuts[0]:.3f} and {cuts[1]:.3f} of the frame. AUROC of each stratum against all 250 authentic images, and recall at 0.5.\n")
    print("| variant | small AUROC | medium AUROC | large AUROC | small recall | medium recall | large recall |\n|---|---|---|---|---|---|---|")
    for v in ["single_a", "single_b", "indep", "ml", "ml_hem", "cfr_adaptive"]:
        if v not in R:
            continue
        au, rc = {k: [] for k in strata}, {k: [] for k in strata}
        for s, run in R[v].items():
            z = np.load(os.path.join(run, "probs.npz")); p = z[f"test_p_{key_of(v)}"]; y = z["test_y"]
            for k, f in strata.items():
                sel = (df.label.values == 0) | (seen & ~np.isnan(area) & f(np.nan_to_num(area)))
                au[k].append(roc_auc_score(y[sel], p[sel]))
                m = (df.label.values == 1) & seen & f(np.nan_to_num(area))
                rc[k].append((p[m] >= .5).mean())
        print(f"| {v} | " + " | ".join(f"{np.mean(au[k]):.3f}" for k in strata) + " | " + " | ".join(f"{np.mean(rc[k]):.3f}" for k in strata) + " |")
    print()


def selective(R):
    print("## 2. Selective prediction (test, seen tools), accuracy at coverage, mean over seeds\n")
    print("Two abstention rules for the fused score: lowest confidence |p - 0.5| first, and branch disagreement first "
          "(ties broken by confidence). Coverage 0.8 means the 20 percent most uncertain images are withheld. "
          "The random-abstention accuracy is the full-coverage accuracy.\n")
    print("| variant | acc @1.0 | confidence @0.8 | confidence @0.6 | disagreement-first @0.8 | AURC (confidence) |\n|---|---|---|---|---|---|")
    for v in PAIRS:
        if v not in R:
            continue
        rows = []
        for s, run in R[v].items():
            pf, y, _ = subset(run, "test", SEEN, "fused"); pa, _, _ = subset(run, "test", SEEN, "A"); pb, _, _ = subset(run, "test", SEEN, "B")
            correct = (pf >= .5) == y.astype(bool)
            conf = np.abs(pf - .5)
            def acc_at(order, cov):
                keep = order[: int(round(cov * len(y)))]
                return correct[keep].mean()
            by_conf = np.argsort(-conf)
            dis = ((pa >= .5) != (pb >= .5)).astype(float)
            by_dis = np.lexsort((-conf, dis))   # agree first (dis=0), then more confident first
            risk = [1 - correct[by_conf[:k]].mean() for k in range(1, len(y) + 1)]
            rows.append([correct.mean(), acc_at(by_conf, .8), acc_at(by_conf, .6), acc_at(by_dis, .8), np.mean(risk)])
        print(f"| {v} | " + " | ".join(f"{np.mean([r[i] for r in rows]):.3f}" for i in range(5)) + " |")
    print()


def correction(R):
    print("## 3. Does alignment make errors harder to correct (test, seen tools, mean over seeds)\n")
    print("`B fixes A's miss`: of images A gets wrong, the share B gets right. `A fixes B's miss`: the reverse. "
          "A model that has copied its partner would show both near zero. The independent pair is the reference.\n")
    print("| variant | A macro-F1 | B macro-F1 | A wrong | B fixes A's miss | B wrong | A fixes B's miss | fused macro-F1 |\n|---|---|---|---|---|---|---|---|")
    for v in PAIRS:
        if v not in R:
            continue
        rows = []
        for s, run in R[v].items():
            pa, y, _ = subset(run, "test", SEEN, "A"); pb, _, _ = subset(run, "test", SEEN, "B"); pf, _, _ = subset(run, "test", SEEN, "fused")
            ya = y.astype(bool); wa, wb = (pa >= .5) != ya, (pb >= .5) != ya
            rows.append([f1_score(y, pa >= .5, average="macro"), f1_score(y, pb >= .5, average="macro"), wa.mean(),
                         (~wb[wa]).mean() if wa.any() else np.nan, wb.mean(), (~wa[wb]).mean() if wb.any() else np.nan,
                         f1_score(y, pf >= .5, average="macro")])
        print(f"| {v} | " + " | ".join(f"{np.nanmean([r[i] for r in rows]):.3f}" for i in range(7)) + " |")
    print()


def transition():
    print("## 4. Validation metrics either side of the CFR trigger (val split, from the training history)\n")
    print("`before` is the epoch just before the CFR weight first applied, `final` the last epoch. Weights are the blend gamma "
          "between the mutual-learning and CFR terms. Validation, never test.\n")
    print("| run | trigger epoch | gamma by epoch | fused val AUROC before / final | A before / final | B before / final | disagreement before / final |\n|---|---|---|---|---|---|---|")
    for v in ("cfr_adaptive", "cfr_sudden", "cfr_fixed"):
        for s in range(3):
            f = f"runs/{v}_s{s}/results.json"
            if not os.path.exists(f):
                continue
            r = json.load(open(f)); h = r["history"]; t = r["trigger_epoch"]
            b = h[max((t or 1) - 1, 0)]["val"]; l = h[-1]["val"]
            g = " ".join(f"{e['gamma']:.2f}" for e in h)
            print(f"| {v} s{s} | {t} | {g} | {b['fused']['auroc']:.3f} / {l['fused']['auroc']:.3f} | {b['A']['auroc']:.3f} / {l['A']['auroc']:.3f} | "
                  f"{b['B']['auroc']:.3f} / {l['B']['auroc']:.3f} | {b['pair']['disagreement']:.3f} / {l['pair']['disagreement']:.3f} |")
    print()


def compute_matched(R):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from .models import ForensicBranch, FusionHead, VisualBranch
    print("## 5. Compute-matched control (variant J)\n")
    x = torch.rand(1, 3, 384, 384)
    info = {}
    for name, m in (("Model A (EfficientNet-B0 + decoder)", VisualBranch(2)), ("Model B (forensic CNN + decoder)", ForensicBranch(2))):
        m.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        info[name] = (sum(p.numel() for p in m.parameters()) / 1e6, fc.get_total_flops() / 2e9)
    print("| model | parameters (M) | GMACs per 384 px image |\n|---|---|---|")
    for k, (p, f) in info.items():
        print(f"| {k} | {p:.2f} | {f:.2f} |")
    pa, fa = list(info.values())[0]; pb, fb = list(info.values())[1]
    print(f"| A + B pair | {pa + pb:.2f} | {fa + fb:.2f} |")
    print(f"| two Model A (the compute-matched alternative) | {2 * pa:.2f} | {2 * fa:.2f} |\n")
    if "single_a" in R and len(R["single_a"]) >= 3:
        seeds = sorted(R["single_a"]); aucs = []
        for i, s in enumerate(seeds):
            t = seeds[(i + 1) % len(seeds)]
            p1, y, _ = subset(R["single_a"][s], "test", SEEN, "A"); p2, _, _ = subset(R["single_a"][t], "test", SEEN, "A")
            aucs.append(roc_auc_score(y, (p1 + p2) / 2))
        base = [roc_auc_score(*(lambda p, y, _: (y, p))(*subset(R["single_a"][s], "test", SEEN, "A"))) for s in seeds]
        print("Post-hoc ensemble of two independently trained Model A runs (seed pairs 0+1, 1+2, 2+0), averaged probabilities, "
              "primary metric (test, seen tools):\n")
        print(f"- single Model A: {np.mean(base):.4f} (per seed {', '.join(f'{a:.4f}' for a in base)})")
        print(f"- two-Model-A ensemble: {np.mean(aucs):.4f} (per pair {', '.join(f'{a:.4f}' for a in aucs)})")
        print(f"- best A+B pair variant for reference (mean over seeds): see the headline table in `comparison.md`.\n")
        print("This control gets about 2x A's compute from a stronger visual model instead of a weak forensic one; it is not trained "
              "jointly, so it is a ceiling on 'more of A', not a matched training procedure.\n")


def extras(R):
    from .compare import boot_diff
    print("## 6. Extra variants (3 seeds each where finished)\n")
    names = ["joint_concat", "joint_gate", "cfr_scratch_noalign", "ml_lam0.25", "ml_lam4"]
    have = [n for n in names if n in R]
    print("| variant | seeds | primary AUROC mean ± sd | per seed | removal (test) | FPR authentic | test ECE | Dice |\n|---|---|---|---|---|---|---|---|")
    for v in ["single_a", "ml", "cfr_scratch", "cfr_adaptive"] + have:
        if v not in R:
            continue
        pr, rem, fpr, ec, di = [], [], [], [], []
        for s, run in sorted(R[v].items()):
            p, y, _ = subset(run, "test", SEEN, key_of(v)); pr.append(roc_auc_score(y, p))
            q, yy, _ = subset(run, "test", ("authentic", "removal"), key_of(v)); rem.append(roc_auc_score(yy, q))
            r = json.load(open(os.path.join(run, "results.json")))["final"]["test"][key_of(v)]
            fpr.append(r["fpr_authentic"]); ec.append(r["ece"]); di.append(r.get("dice", np.nan))
        print(f"| {v} | {len(pr)} | {np.mean(pr):.4f} ± {np.std(pr, ddof=1) if len(pr) > 1 else 0:.4f} | {', '.join(f'{x:.4f}' for x in pr)} | "
              f"{np.mean(rem):.3f} | {np.mean(fpr):.3f} | {np.mean(ec):.3f} | {np.nanmean(di):.3f} |")
    print("\nPaired comparisons on the primary metric (per-seed difference, mean, 95% paired bootstrap over test images on seed-averaged probabilities). "
          "These are descriptive; the pre-declared decision rule in `02` applies only to cfr_adaptive vs ml and ml_hem.\n")
    print("| A minus B | shared seeds | per-seed | mean | 95% bootstrap |\n|---|---|---|---|---|")
    pairs = [("joint_concat", "single_a"), ("joint_concat", "indep"), ("joint_gate", "single_a"), ("joint_gate", "indep"),
             ("cfr_scratch", "cfr_scratch_noalign"), ("ml_lam0.25", "ml"), ("ml_lam4", "ml"), ("ml_lam0.25", "indep"), ("ml_lam4", "indep")]
    for a, b in pairs:
        if a not in R or b not in R:
            continue
        seeds = sorted(set(R[a]) & set(R[b]))
        if not seeds:
            continue
        d = [roc_auc_score(*(lambda p, y, _: (y, p))(*subset(R[a][s], "test", SEEN, key_of(a)))) -
             roc_auc_score(*(lambda p, y, _: (y, p))(*subset(R[b][s], "test", SEEN, key_of(b)))) for s in seeds]
        m, ci = boot_diff(R, a, b)
        print(f"| {a} - {b} | {len(seeds)} | {', '.join(f'{x:+.4f}' for x in d)} | {m:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    print()
    for v in ("joint_gate",):
        gs = []
        for s, run in sorted(R.get(v, {}).items()):
            f = os.path.join(run, "gate_stats.json")
            if os.path.exists(f):
                gs.append(json.load(open(f)))
        if gs:
            print(f"Learned gate, mean weight on Model A over test images (per seed): {', '.join(f'{g['mean_gate_A']:.3f}' for g in gs)}\n")


if __name__ == "__main__":
    R = all_runs()
    table_mask_strata(R)
    selective(R)
    correction(R)
    transition()
    compute_matched(R)
    extras(R)
