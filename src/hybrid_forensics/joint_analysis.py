"""Branch-reliance and gate diagnostics for the learned-fusion runs (joint_concat, joint_gate).

Branch-reliance as in the brief: one branch's input is replaced by uniform noise at evaluation. These are diagnostic
interventions on a trained model, not proof that a branch is intrinsically useful. For joint_gate the mean gate weight on
Model A is also recorded (written to <run>/gate_stats.json). usage: python -m hybrid_forensics.joint_analysis runs_extra/joint_gate_s0 ..."""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .analysis import SEEN
from .data import ForgeDataset
from .external_eval import load


@torch.no_grad()
def analyse(run):
    models, head = load(run)
    ds = ForgeDataset(pd.read_csv("data/forge/manifest.csv"), "test")
    keep = [i for i in range(len(ds)) if ds.df.manipulation[i] in SEEN]
    g = torch.Generator().manual_seed(0)
    res, gates = {}, []
    for cond in ("both intact", "A input destroyed", "B input destroyed"):
        pf, ys = [], []
        for s in range(0, len(keep), 32):
            b = [ds[i] for i in keep[s:s + 32]]
            x = torch.stack([t["x"] for t in b]).cuda()
            noise = torch.rand(x.shape, generator=g).cuda()
            xa = noise if cond == "A input destroyed" else x
            xb = noise if cond == "B input destroyed" else x
            with torch.autocast("cuda", dtype=torch.bfloat16):
                oa, ob = models["A"](xa), models["B"](xb)
            pf.append(F.softmax(head(oa, ob), 1)[:, 1].cpu())
            if cond == "both intact" and head.kind == "gate":
                gates.append(head.last_gate[:, 0].cpu())
            ys += [t["y"] for t in b]
        res[cond] = float(roc_auc_score(np.array(ys), torch.cat(pf).numpy()))
    if gates:
        gt = torch.cat(gates).numpy()
        json.dump({"mean_gate_A": float(gt.mean()), "sd_gate_A": float(gt.std())}, open(f"{run}/gate_stats.json", "w"))
        res["mean_gate_A"], res["sd_gate_A"] = float(gt.mean()), float(gt.std())
    json.dump(res, open(f"{run}/reliance.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(r, analyse(r))
