"""Experiments C and D, and diagnoses D2 and D3: does Model B hold information Model A lacks, and is it used?

Uses the single_a and single_b checkpoints of the same seed (trained alone, so nothing was aligned or fused during training) and the
pair checkpoints for representation similarity. Everything that is FITTED (stacking weights, probes) is fitted on COCO-Forge VALIDATION
(seen tools only) and then scored on test and on the forge_hard tools, which no model or fit has seen.
usage: PYTHONPATH=src .venv/bin/python -m hybrid_forensics.diag_fusion > reports/evidence/next_phase/diag_fusion.md
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .eval_manifest import forward_all
from .external_eval import load

CF, HARD = pd.read_csv("data/forge/manifest.csv"), pd.read_csv("data/forge_hard/manifest.csv")
CACHE = "outputs/diag_cache"
SEEN = ("authentic", "splice", "copy_move")


def md(df):
    """Markdown table without the tabulate dependency (the venv is not to be modified)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def get(run, man, mname, split):
    """Per-branch probabilities, fused probabilities and embeddings, cached so the GPU is only needed once per checkpoint."""
    f = f"{CACHE}/{os.path.basename(run)}_{mname}_{split}.npz"
    if not os.path.exists(f):
        os.makedirs(CACHE, exist_ok=True)
        models, head = load(run)
        df, P, fused, _, _, E = forward_all(models, head, man, split)
        np.savez(f, fused=fused, tool=df.manipulation.values, y=df.label.values,
                 **{f"p_{k}": v for k, v in P.items()}, **{f"e_{k}": v for k, v in E.items()})
        del models, head
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float(np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def boot_diff(y, p1, p2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), np.percentile(d, [2.5, 97.5])


def eval_sets(cf_test, hard_test):
    """name -> (y, mask over rows, source dict) for every evaluation set used below."""
    sets = {}
    t = cf_test["tool"]
    sets["COCO-Forge seen tools"] = (cf_test, np.isin(t, SEEN))
    sets["COCO-Forge removal (unseen)"] = (cf_test, np.isin(t, ("authentic", "removal")))
    for tool in ("splice_feather", "copy_move_feather", "splice_poisson", "copy_move_poisson", "self_resample"):
        sets[f"hard: {tool}"] = (hard_test, np.isin(hard_test["tool"], ("authentic", tool)))
    return sets


def main():
    seeds = (0, 1, 2)
    D = {}
    for s in seeds:
        a, b = f"runs/single_a_s{s}", f"runs/single_b_s{s}"
        D[s] = {"A": {sp: get(a, m, n, sp) for sp, m, n in (("val", CF, "cf"), ("test", CF, "cf"))},
                "B": {sp: get(b, m, n, sp) for sp, m, n in (("val", CF, "cf"), ("test", CF, "cf"))}}
        D[s]["A"]["hard"], D[s]["B"]["hard"] = get(a, HARD, "hard", "test"), get(b, HARD, "hard", "test")

    # ---- 1. stacked fusion and linear probes, fitted on validation only
    print("## 1. Does B add information? Fusion weights and probes fitted on COCO-Forge validation, scored on unseen data\n")
    print("A alone is the Model A softmax. `fixed avg` is the 0.5/0.5 probability average used by the headline pairs. `stack` is a logistic regression on "
          "the two logits (A_s with B_s of the same seed). `probe A` and `probe A+B` are logistic regressions on frozen penultimate embeddings (C chosen by 5-fold CV on validation). "
          "AUROC, mean over 3 seeds (per seed in brackets).\n")
    names = list(eval_sets(D[0]["A"]["test"], D[0]["A"]["hard"]))
    methods = ["A alone", "B alone", "fixed avg", "stack A+B", "probe A", "probe A+B"]
    res = {m: {n: [] for n in names} for m in methods}
    scores = {m: {n: [] for n in names} for m in methods}   # for the paired bootstrap on seed-averaged scores
    coefs = []
    for s in seeds:
        A, B = D[s]["A"], D[s]["B"]
        yv = A["val"]["y"]
        la_v, lb_v = logit(A["val"]["p_A"]), logit(B["val"]["p_B"])
        stack = LogisticRegression(C=100, max_iter=2000).fit(np.c_[la_v, lb_v], yv)
        coefs.append(stack.coef_[0])
        sc_a, sc_ab = StandardScaler().fit(A["val"]["e_A"]), StandardScaler().fit(np.c_[A["val"]["e_A"], B["val"]["e_B"]])
        pr_a = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=5, max_iter=3000, scoring="roc_auc").fit(sc_a.transform(A["val"]["e_A"]), yv)
        pr_ab = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=5, max_iter=3000, scoring="roc_auc").fit(sc_ab.transform(np.c_[A["val"]["e_A"], B["val"]["e_B"]]), yv)
        for n, (src, keep) in eval_sets(A["test"], A["hard"]).items():
            ra, rb = (A["test"], B["test"]) if not n.startswith("hard") else (A["hard"], B["hard"])
            y = ra["y"][keep]
            pa, pb = ra["p_A"][keep], rb["p_B"][keep]
            cand = {"A alone": pa, "B alone": pb, "fixed avg": 0.5 * pa + 0.5 * pb,
                    "stack A+B": stack.predict_proba(np.c_[logit(pa), logit(pb)])[:, 1],
                    "probe A": pr_a.predict_proba(sc_a.transform(ra["e_A"][keep]))[:, 1],
                    "probe A+B": pr_ab.predict_proba(sc_ab.transform(np.c_[ra["e_A"][keep], rb["e_B"][keep]]))[:, 1]}
            for m, p in cand.items():
                res[m][n].append(roc_auc_score(y, p)); scores[m][n].append(p)
    print("| evaluation set | " + " | ".join(methods) + " |\n|" + "---|" * (len(methods) + 1))
    for n in names:
        print(f"| {n} | " + " | ".join(f"{np.mean(res[m][n]):.3f} ({', '.join(f'{x:.3f}' for x in res[m][n])})" for m in methods) + " |")
    print("\nStack weights on (logit A, logit B), per seed: " + "; ".join(f"A {c[0]:.2f}, B {c[1]:.2f}" for c in coefs))

    print("\nPaired bootstrap over test images on seed-averaged scores (A+B minus A alone), 95 percent interval:\n")
    print("| evaluation set | stack A+B minus A | probe A+B minus probe A | probe A+B minus A alone |\n|---|---|---|---|")
    for n, (src, keep) in eval_sets(D[0]["A"]["test"], D[0]["A"]["hard"]).items():
        y = (D[0]["A"]["test"] if not n.startswith("hard") else D[0]["A"]["hard"])["y"][keep]
        avg = lambda m: np.mean(scores[m][n], 0)
        def cell(m1, m2):
            d, (lo, hi) = boot_diff(y, avg(m1), avg(m2))
            return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"| {n} | {cell('stack A+B', 'A alone')} | {cell('probe A+B', 'probe A')} | {cell('probe A+B', 'A alone')} |")

    # ---- 2. where A is unsure or wrong
    print("\n## 2. Is B right where A is unsure? (COCO-Forge test, seen tools, pooled over 3 seeds)\n")
    rows = []
    for s in seeds:
        A, B = D[s]["A"]["test"], D[s]["B"]["test"]
        k = np.isin(A["tool"], SEEN)
        pa, pb, y = A["p_A"][k], B["p_B"][k], A["y"][k]
        wrong_a, wrong_b = (pa >= .5) != y, (pb >= .5) != y
        unsure = (pa > .05) & (pa < .95)
        rows.append(dict(seed=s, n=len(y), a_wrong=int(wrong_a.sum()), b_wrong=int(wrong_b.sum()), both=int((wrong_a & wrong_b).sum()),
                         b_fixes=int((wrong_a & ~wrong_b).sum()), a_fixes=int((~wrong_a & wrong_b).sum()), n_unsure=int(unsure.sum()),
                         auc_a_unsure=roc_auc_score(y[unsure], pa[unsure]), auc_b_unsure=roc_auc_score(y[unsure], pb[unsure]),
                         err_phi=float(np.corrcoef(wrong_a, wrong_b)[0, 1]), logit_corr=float(np.corrcoef(logit(pa), logit(pb))[0, 1])))
    print(md(pd.DataFrame(rows).round(3)))
    print("\n`b_fixes` counts images A gets wrong and B gets right at 0.5; `a_fixes` the reverse. `unsure` = A probability between 0.05 and 0.95. "
          "`err_phi` is the correlation of the two error indicators (0 = independent errors, 1 = identical errors).")

    # ---- 3. representation similarity
    print("\n## 3. Do A and B learn different representations? Linear CKA on test (seen tools) embeddings\n")
    print("1.0 means the two embedding spaces are linearly equivalent, near 0 means unrelated. References are the same architecture trained with another seed.\n")
    t = lambda run: get(run, CF, "cf", "test")
    k = np.isin(t("runs/single_a_s0")["tool"], SEEN)
    rows = []
    def add(label, e1, e2): rows.append({"pair": label, "CKA": round(linear_cka(e1[k], e2[k]), 3)})
    add("A seed0 vs A seed1 (reference)", t("runs/single_a_s0")["e_A"], t("runs/single_a_s1")["e_A"])
    add("B seed0 vs B seed1 (reference)", t("runs/single_b_s0")["e_B"], t("runs/single_b_s1")["e_B"])
    add("A vs B, trained alone (seed 0)", t("runs/single_a_s0")["e_A"], t("runs/single_b_s0")["e_B"])
    for v in ("indep", "ml", "ml_hem", "cfr_adaptive"):
        z = t(f"runs/{v}_s0")
        add(f"A vs B inside {v} (seed 0)", z["e_A"], z["e_B"])
    z = t("runs_extra/ml_lam4_s0"); add("A vs B inside ml_lam4 (seed 0)", z["e_A"], z["e_B"])
    print(md(pd.DataFrame(rows)))
    print("\nB inside the aligned pairs is compared with the A it was aligned to; if alignment made B a copy of A the pair CKA would rise toward the A-vs-A reference.")


if __name__ == "__main__":
    main()
