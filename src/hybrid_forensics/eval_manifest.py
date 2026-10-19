"""Score existing runs on any 384 px manifest (used for forge_hard) and aggregate per-tool metrics.

Nothing here trains or tunes. Threshold 0.5 is the protocol used everywhere else in this project; a second set of columns uses
the threshold that maximises balanced accuracy on the run's own COCO-Forge VALIDATION split (probs.npz), never on the scored split.
Each tool is scored against the same authentic images, so tools are paired by source.
usage: python -m hybrid_forensics.eval_manifest score data/forge_hard/manifest.csv test hard runs/<run> ...   -> runs/<run>/hard_test.json + _probs.npz
       python -m hybrid_forensics.eval_manifest report hard test                                           -> markdown tables over runs/ and runs_extra/
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import ForgeDataset
from .metrics import classification_metrics, dice_iou
from .external_eval import load


@torch.no_grad()
def forward_all(models, head, man, split, bs=32):
    """Per-image fused P(manipulated), per-branch probabilities, fused heat-map Dice/IoU, embeddings. Order follows the manifest rows of `split`."""
    ds = ForgeDataset(man, split)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    keys = list(models)
    P = {k: [] for k in keys}; fused, dice, iou, emb = [], [], [], {k: [] for k in keys}
    for b in loader:
        x = b["x"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = {k: m(x) for k, m in models.items()}
        for k, o in outs.items():
            P[k].append(F.softmax(o["logits"].float(), 1)[:, 1].cpu()); emb[k].append(o["embedding"].float().cpu())
        if head is not None:
            with torch.autocast("cuda", enabled=False):
                f = F.softmax(head(outs["A"], outs["B"]), 1)[:, 1].cpu()
        elif len(keys) == 2:
            f = 0.5 * P["A"][-1] + 0.5 * P["B"][-1]
        else:
            f = P[keys[0]][-1]
        fused.append(f)
        heat = sum(torch.sigmoid(o["mask_logits"].float()) for o in outs.values()).cpu() / len(outs)
        manip = b["y"] == 1
        if manip.any():
            d, i = dice_iou((heat[manip, 0] >= .5).numpy(), (b["mask"][manip, 0] >= .5).numpy())
        dd, ii = np.full(len(b["y"]), np.nan), np.full(len(b["y"]), np.nan)
        if manip.any():
            dd[manip.numpy()], ii[manip.numpy()] = d, i
        dice.append(dd); iou.append(ii)
    cat = lambda v: torch.cat(v).numpy()
    return ds.df, {k: cat(v) for k, v in P.items()}, cat(fused), np.concatenate(dice), np.concatenate(iou), {k: cat(v) for k, v in emb.items()}


def val_threshold(run):
    """Threshold maximising balanced accuracy on the run's own COCO-Forge validation split. Falls back to 0.5 if probs are missing."""
    f = os.path.join(run, "probs.npz")
    if not os.path.exists(f):
        return 0.5
    z = np.load(f)
    key = "val_p_fused" if "val_p_fused" in z else [k for k in z.files if k.startswith("val_p_")][0]
    y, p = z["val_y"], z[key]
    cands = np.unique(np.round(p, 4))
    best = max(cands, key=lambda t: (classification_metrics(p, y, t)["balanced_acc"], -abs(t - .5)))
    return float(best)


def score_run(run, manifest, split, name):
    man = pd.read_csv(manifest)
    models, head = load(run)
    df, P, fused, dice, iou, _ = forward_all(models, head, man, split)
    np.savez(os.path.join(run, f"{name}_{split}_probs.npz"), p_fused=fused, dice=dice, iou=iou, label=df.label.values,
             source_id=df.source_id.values, tool=df.manipulation.values, **{f"p_{k}": v for k, v in P.items()})
    return summarise(run, df, fused, dice, iou, name, split)


def summarise(run, df, fused, dice, iou, name, split):
    thr_v = val_threshold(run)
    auth = (df.label == 0).values
    out = {"thr_val": thr_v, "n_authentic": int(auth.sum())}
    for tool in sorted(set(df.manipulation) - {"authentic"}):
        sel = (df.manipulation == tool).values
        m = auth | sel
        y, p = df.label.values[m], fused[m]
        r = classification_metrics(p, y)
        rv = classification_metrics(p, y, thr_v)
        out[tool] = {"auroc": r["auroc"], "macro_f1": r["macro_f1"], "balanced_acc": r["balanced_acc"], "fpr": r["fpr_authentic"],
                     "recall": float((fused[sel] >= .5).mean()), "dice": float(np.nanmean(dice[sel])), "iou": float(np.nanmean(iou[sel])),
                     "macro_f1_valthr": rv["macro_f1"], "balanced_acc_valthr": rv["balanced_acc"], "fpr_valthr": rv["fpr_authentic"],
                     "recall_valthr": float((fused[sel] >= thr_v).mean()), "mean_score": float(fused[sel].mean())}
    out["mean_score_authentic"] = float(fused[auth].mean())
    json.dump(out, open(os.path.join(run, f"{name}_{split}.json"), "w"), indent=1)
    return out


def runs_by_variant(name, split):
    by = {}
    for f in sorted(glob.glob(f"runs*/*/{name}_{split}.json")):
        run = os.path.dirname(f)
        by.setdefault(os.path.basename(run).rsplit("_s", 1)[0], []).append(run)
    return by


def report(name, split, metric="auroc"):
    by = runs_by_variant(name, split)
    first = json.load(open(glob.glob(f"runs*/*/{name}_{split}.json")[0]))
    tools = [t for t, v in first.items() if isinstance(v, dict)]
    print(f"### {metric} per tool, mean +/- sd over seeds (each tool against the same {split} authentic images)\n")
    print("| variant | seeds | " + " | ".join(tools) + " |\n|" + "---|" * (len(tools) + 2))
    for v, runs in by.items():
        js = [json.load(open(f"{r}/{name}_{split}.json")) for r in runs]
        cells = [f"{np.mean([j[t][metric] for j in js]):.3f} ± {np.std([j[t][metric] for j in js]):.3f}" for t in tools]
        print(f"| {v} | {len(runs)} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    if sys.argv[1] == "score":
        manifest, split, name = sys.argv[2:5]
        for r in sys.argv[5:]:
            o = score_run(r, manifest, split, name)
            print(r, {k: round(v["auroc"], 3) for k, v in o.items() if isinstance(v, dict)}, flush=True)
    else:
        name, split = sys.argv[2:4]
        for m in sys.argv[4:] or ["auroc"]:
            report(name, split, m)
