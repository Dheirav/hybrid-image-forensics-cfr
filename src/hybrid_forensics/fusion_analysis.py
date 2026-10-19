"""Fusion comparison and branch-reliance tests for one trained pair run.

Fusions are fitted on VALIDATION probabilities only and scored on test:
  A alone, B alone, fixed 0.5 average, best fixed weight (grid on val), logistic stacker on the two logits (val-fit).
Reliance: re-run the checkpoint with one branch's input destroyed (diagnostic intervention only).
usage: python -m hybrid_forensics.fusion_analysis runs/<run>
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .analysis import SEEN, subset
from .calibrate import _logit
from .data import ForgeDataset
from .metrics import classification_metrics


def fusion_table(run):
    va_a, y_va, _ = subset(run, "val", ("authentic", "splice", "copy_move"), "A")
    va_b, _, _ = subset(run, "val", ("authentic", "splice", "copy_move"), "B")
    te_a, y_te, _ = subset(run, "test", SEEN, "A")
    te_b, _, _ = subset(run, "test", SEEN, "B")
    grid = np.linspace(0, 1, 21)
    w_best = grid[int(np.argmax([roc_auc_score(y_va, w * va_a + (1 - w) * va_b) for w in grid]))]
    stack = LogisticRegression(C=1.0).fit(np.c_[_logit(va_a), _logit(va_b)], y_va)
    rows = {"A alone": te_a, "B alone": te_b, "fixed 0.5": .5 * te_a + .5 * te_b,
            f"val-best weight (w_A={w_best:.2f})": w_best * te_a + (1 - w_best) * te_b,
            "logistic stacker (val-fit)": stack.predict_proba(np.c_[_logit(te_a), _logit(te_b)])[:, 1]}
    return {k: {"auroc": float(roc_auc_score(y_te, p)), "macro_f1": classification_metrics(p, y_te)["macro_f1"]} for k, p in rows.items()}, \
        {"stacker_coef_A_B": stack.coef_[0].tolist()}


@torch.no_grad()
def reliance(run, n=None):
    from .models import ForensicBranch, VisualBranch
    sd = torch.load(f"{run}/best.pt", map_location="cpu")
    models = {"A": VisualBranch(2), "B": ForensicBranch(2)}
    for k, m in models.items():
        m.load_state_dict(sd[k]); m.cuda().eval()
    import pandas as pd
    df = pd.read_csv("data/forge/manifest.csv")
    ds = ForgeDataset(df, "test")
    keep = [i for i in range(len(ds)) if ds.df.manipulation[i] in SEEN]
    g = torch.Generator().manual_seed(0)
    res = {}
    for cond in ("both intact", "A input destroyed", "B input destroyed"):
        pa, pb, ys = [], [], []
        for s in range(0, len(keep), 32):
            b = [ds[i] for i in keep[s:s + 32]]
            x = torch.stack([t["x"] for t in b]).cuda()
            noise = torch.rand(x.shape, generator=g).cuda()  # uniform noise carries no image content
            xa = noise if cond == "A input destroyed" else x
            xb = noise if cond == "B input destroyed" else x
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pa.append(F.softmax(models["A"](xa)["logits"].float(), 1)[:, 1].cpu()); pb.append(F.softmax(models["B"](xb)["logits"].float(), 1)[:, 1].cpu())
            ys += [t["y"] for t in b]
        pa, pb, ys = torch.cat(pa).numpy(), torch.cat(pb).numpy(), np.array(ys)
        res[cond] = {"fused_auroc": float(roc_auc_score(ys, .5 * pa + .5 * pb)), "A_auroc": float(roc_auc_score(ys, pa)),
                     "B_auroc": float(roc_auc_score(ys, pb))}
    return res


if __name__ == "__main__":
    run = sys.argv[1]
    t, extra = fusion_table(run)
    out = {"fusion": t, **extra, "reliance": reliance(run)}
    json.dump(out, open(f"{run}/fusion_analysis.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
