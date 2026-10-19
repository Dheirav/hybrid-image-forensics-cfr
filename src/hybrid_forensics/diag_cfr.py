"""Diagnosis D4: what do the CFR weights actually do once the models are trained, and how different are they from hard-example weights?

For each finished checkpoint the CFR and the hard-example-mining (HEM) weights are computed on the TRAINING images (no augmentation) with
losses.cfr_terms / losses.hem_terms, exactly the functions train.py uses. This is the end-of-training state of the best-validation checkpoint,
not the state at every step, so it shows where the weights end up, not the path. The per-run history gives the epochs spent under each regime.
usage: PYTHONPATH=src .venv/bin/python -m hybrid_forensics.diag_cfr > reports/evidence/next_phase/diag_cfr.md
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .diag_fusion import CACHE, CF, get, md
from .losses import cfr_terms, hem_terms

VARIANTS = ("ml", "ml_hem", "cfr_adaptive", "cfr_scratch")


def logits_of(p):
    p = torch.tensor(np.clip(p, 1e-6, 1 - 1e-6), dtype=torch.float64)
    return torch.stack([torch.log(1 - p), torch.log(p)], 1)   # softmax of these recovers p exactly


def ess_fraction(w):
    """Effective sample size / n: 1 means all samples count equally, small means a few samples carry the loss."""
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def main():
    rows = []
    for v in VARIANTS:
        for s in (0, 1, 2):
            run = f"runs/{v}_s{s}"
            z = get(run, CF, "cf", "train")
            y = torch.tensor(z["y"])
            res = json.load(open(f"{run}/results.json"))
            gam = [h["gamma"] for h in res["history"]]
            for m, o in (("A", "B"), ("B", "A")):
                ls, lp = logits_of(z[f"p_{m}"]), logits_of(z[f"p_{o}"])
                wc, ac = cfr_terms(ls, lp, y)
                wh, _ = hem_terms(ls, lp, y)
                pt = np.where(z["y"] == 1, z[f"p_{m}"], 1 - z[f"p_{m}"])
                pp = np.where(z["y"] == 1, z[f"p_{o}"], 1 - z[f"p_{o}"])
                c = np.clip(pp - pt, 0, None)
                wc, wh, ac = wc.numpy(), wh.numpy(), ac.numpy()
                rows.append(dict(variant=v, seed=s, model=m, train_auroc=roc_auc_score(z["y"], z[f"p_{m}"]), train_acc=float(((z[f"p_{m}"] >= .5) == z["y"]).mean()),
                                 challenged_5pct=float((c > .05).mean()), challenged_50pct=float((c > .5).mean()),
                                 cfr_w_max=float(wc.max()), cfr_ess=ess_fraction(wc), hem_ess=ess_fraction(wh),
                                 spearman_cfr_hem=float(spearmanr(wc, wh)[0]), gate_share=float((ac > 0.25 + 1e-6).mean()), align_mean=float(ac.mean()),
                                 epochs_gamma_pos=sum(g > 0 for g in gam), epochs_gamma_1=sum(g >= 1 for g in gam)))
    df = pd.DataFrame(rows)
    df.to_csv("reports/evidence/next_phase/diag_cfr_rows.csv", index=False)
    g = df.groupby(["variant", "model"]).mean(numeric_only=True).drop(columns="seed").round(4).reset_index()
    print("## CFR and hard-example weights on the training images, end-of-training checkpoint, mean over 3 seeds\n")
    print("`challenged_5pct` is the share of training images where the partner is more right than the model by more than 0.05 (the only images CFR up-weights). "
          "`cfr_ess` and `hem_ess` are the effective sample size as a fraction of n (1 = uniform weights). `spearman_cfr_hem` is the rank correlation of the two "
          "weight vectors on the same checkpoint. `gate_share` is the share of images where the alignment weight is above its floor (the partner is more right). "
          "`epochs_gamma_pos` and `epochs_gamma_1` count epochs with any CFR weight and with the full CFR weight (of 12).\n")
    print(md(g))


if __name__ == "__main__":
    main()
