"""Stage 2 analysis on forge2 (runs in runs_f2/, scored with `eval_manifest score data/forge2/manifest.csv test f2 <run>`).

Primary metric, declared in NEXT_PHASE_PLAN.md before any forge2 run existed: fused test AUROC on authentic + the four SEEN tools
(splice_feather, copy_move_feather, splice_poisson, copy_move_poisson), mean over seeds. Secondary: AUROC on authentic + the three UNSEEN tools
(self_resample, removal_telea, removal_ns), macro-F1, balanced accuracy and authentic false-positive rate at 0.5 and at the validation-chosen threshold,
ECE, Dice. Decision rule = reports/02: positive in every seed, paired 95 percent bootstrap over test images (seed-averaged scores) excluding 0, and no worse
ECE or authentic false-positive rate.
usage: PYTHONPATH=src .venv/bin/python -m hybrid_forensics.f2_analysis > reports/evidence/next_phase/f2_analysis.md
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .diag_fusion import md, logit
from .eval_manifest import val_threshold
from .metrics import classification_metrics
from .make_forge2 import SEEN, UNSEEN

ORDER = ["single_a", "single_b", "indep", "ml_lam0.25", "ml", "ml_hem", "cfr_adaptive", "joint_concat"]
COMPARE = [("cfr_adaptive", "ml"), ("cfr_adaptive", "ml_lam0.25"), ("cfr_adaptive", "ml_hem"), ("cfr_adaptive", "single_a"),
           ("ml_hem", "ml_lam0.25"), ("ml_hem", "single_a"), ("ml_lam0.25", "single_a"), ("ml", "ml_lam0.25"), ("joint_concat", "single_a")]


def load_runs():
    R = {}
    for f in sorted(glob.glob("runs_f2/*/f2_test_probs.npz")):
        run = os.path.dirname(f)
        v, s = os.path.basename(run).rsplit("_s", 1)
        z = np.load(f, allow_pickle=True)
        R.setdefault(v, {})[int(s)] = dict(run=run, p=z["p_fused"], y=z["label"], tool=z["tool"], dice=z["dice"], src=z["source_id"])
    return R


def sets(tool):
    return {"seen": np.isin(tool, ("authentic",) + SEEN), "unseen": np.isin(tool, ("authentic",) + UNSEEN)}


def auc(r, which):
    k = sets(r["tool"])[which]
    return roc_auc_score(r["y"][k], r["p"][k])


def boot(R, a, b, which, n=2000, seed=0):
    seeds = sorted(set(R[a]) & set(R[b]))
    r0 = R[a][seeds[0]]; k = sets(r0["tool"])[which]; y = r0["y"][k]
    pa = np.mean([R[a][s]["p"][k] for s in seeds], 0); pb = np.mean([R[b][s]["p"][k] for s in seeds], 0)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    per = [auc(R[a][s], which) - auc(R[b][s], which) for s in seeds]
    return seeds, per, float(np.mean(per)), np.percentile(d, [2.5, 97.5])


def pooled_metrics(r, which, thr=0.5):
    k = sets(r["tool"])[which]
    m = classification_metrics(r["p"][k], r["y"][k], thr)
    return m


def main():
    R = load_runs()
    variants = [v for v in ORDER if v in R]
    print("## Headline on forge2 (tool-disjoint). Primary = authentic + 4 seen tools; unseen = authentic + self_resample, removal_telea, removal_ns\n")
    cols = ["variant", "seeds", "primary AUROC (seen) mean ± sd", "per seed", "unseen-tool AUROC", "macro-F1 seen", "bal. acc seen", "FPR authentic", "macro-F1 seen (val thr)", "FPR (val thr)", "ECE seen", "Dice seen tools"]
    rows = []
    for v in variants:
        rs = [R[v][s] for s in sorted(R[v])]
        pa = [auc(r, "seen") for r in rs]; ua = [auc(r, "unseen") for r in rs]
        m = [pooled_metrics(r, "seen") for r in rs]
        mv = [pooled_metrics(r, "seen", val_threshold(r["run"])) for r in rs]
        dice = [np.nanmean(r["dice"][np.isin(r["tool"], SEEN)]) for r in rs]
        rows.append([v, len(rs), f"{np.mean(pa):.4f} ± {np.std(pa):.4f}", ", ".join(f"{x:.4f}" for x in pa), f"{np.mean(ua):.4f}",
                     f"{np.mean([x['macro_f1'] for x in m]):.3f}", f"{np.mean([x['balanced_acc'] for x in m]):.3f}", f"{np.mean([x['fpr_authentic'] for x in m]):.3f}",
                     f"{np.mean([x['macro_f1'] for x in mv]):.3f}", f"{np.mean([x['fpr_authentic'] for x in mv]):.3f}",
                     f"{np.mean([x['ece'] for x in m]):.3f}", f"{np.mean(dice):.3f}"])
    print(md(pd.DataFrame(rows, columns=cols)))

    print("\n## AUROC per tool against the same test authentic images (mean over seeds)\n")
    tools = list(SEEN + UNSEEN)
    rows = []
    for v in variants:
        row = [v]
        for t in tools:
            vals = [roc_auc_score(r["y"][np.isin(r["tool"], ("authentic", t))], r["p"][np.isin(r["tool"], ("authentic", t))]) for r in R[v].values()]
            row.append(f"{np.mean(vals):.3f}")
        rows.append(row)
    print(md(pd.DataFrame(rows, columns=["variant"] + [("[unseen] " if t in UNSEEN else "") + t for t in tools])))

    print("\n## Paired comparisons under the pre-declared rule (primary metric; unseen-tool metric shown as secondary)\n")
    rows = []
    for a, b in COMPARE:
        if a not in R or b not in R:
            continue
        seeds, per, mean, ci = boot(R, a, b, "seen")
        ma = [pooled_metrics(R[a][s], "seen") for s in seeds]; mb = [pooled_metrics(R[b][s], "seen") for s in seeds]
        d_ece = np.mean([x["ece"] for x in ma]) - np.mean([x["ece"] for x in mb])
        d_fpr = np.mean([x["fpr_authentic"] for x in ma]) - np.mean([x["fpr_authentic"] for x in mb])
        allpos, allneg = all(p > 0 for p in per), all(p < 0 for p in per)
        excl = ci[0] > 0 or ci[1] < 0
        verdict = "supported" if allpos and ci[0] > 0 and d_ece <= 0 and d_fpr <= 0 else ("unsupported (negative, same strength)" if allneg and ci[1] < 0 else "inconclusive")
        _, _, umean, uci = boot(R, a, b, "unseen")
        rows.append([f"{a} - {b}", len(seeds), ", ".join(f"{p:+.4f}" for p in per), f"{mean:+.4f}", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                     "yes" if allpos else "no", f"{d_ece:+.3f}", f"{d_fpr:+.3f}", verdict, f"{umean:+.4f} [{uci[0]:+.4f}, {uci[1]:+.4f}]"])
    print(md(pd.DataFrame(rows, columns=["A - B", "seeds", "per-seed difference", "mean", "95% paired bootstrap", "positive in all seeds", "ECE diff", "FPR diff", "verdict", "unseen-tool diff [95%]"])))
    print("\n`verdict` applies the rule literally to the pair shown. CFR is shown to help only if cfr_adaptive is `supported` against ml (weight 1), ml_lam0.25 AND ml_hem; a fusion helps only if it is `supported` against single_a.")

    # Does B add information on forge2? Stacking fitted on forge2 validation, scored on forge2 test.
    if "single_a" in R and "single_b" in R:
        print("\n## Does Model B add information on forge2? Logistic stack of A and B logits (same seed), fitted on forge2 validation\n")
        from sklearn.linear_model import LogisticRegression
        rows, ca = [], []
        for s in sorted(set(R["single_a"]) & set(R["single_b"])):
            za, zb = np.load(f"runs_f2/single_a_s{s}/probs.npz"), np.load(f"runs_f2/single_b_s{s}/probs.npz")
            st = LogisticRegression(C=100, max_iter=2000).fit(np.c_[logit(za["val_p_A"]), logit(zb["val_p_B"])], za["val_y"])
            ra, rb = R["single_a"][s], R["single_b"][s]
            ps = st.predict_proba(np.c_[logit(ra["p"]), logit(rb["p"])])[:, 1]
            row = {"seed": s, "stack weight A": round(st.coef_[0][0], 2), "stack weight B": round(st.coef_[0][1], 2)}
            for which in ("seen", "unseen"):
                k = sets(ra["tool"])[which]
                row[f"{which}: A"] = round(roc_auc_score(ra["y"][k], ra["p"][k]), 4); row[f"{which}: B"] = round(roc_auc_score(rb["y"][k], rb["p"][k]), 4)
                row[f"{which}: stack"] = round(roc_auc_score(ra["y"][k], ps[k]), 4)
            rows.append(row)
        print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
