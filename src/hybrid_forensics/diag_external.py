"""Diagnosis D5 and Experiment F: why do the COCO-Forge models fail on AI-edited and AI-generated images?

Four checks, each aimed at one explanation:
  1. real-image shift: do external REAL photos look different to the models than COCO-Forge authentic crops? (domain shift in what 'normal' is)
  2. frozen-feature probes: is the real/fake information in the embeddings at all? A leave-one-generator-out logistic regression on frozen embeddings
     separates 'the head does not transfer' (probe works) from 'the features do not carry it' (probe fails). The ImageNet-pretrained, never fine-tuned
     Model A backbone is the control: if it probes as well as the fine-tuned one, fine-tuning on COCO-Forge added nothing for this task.
  3. edit area: are AI edits missed because they are small?
  4. window aggregation: tiled max against one central window (the control that was already run).
Caveat for 2: the probe is trained on external labels, so it measures separability, not detection ability. Folder names are the only labels.
usage: PYTHONPATH=src .venv/bin/python -m hybrid_forensics.diag_external > reports/evidence/next_phase/diag_external.md
"""
import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .diag_fusion import md
from .external_eval import load
from .infer import TILE
from .models import VisualBranch

MAN = pd.read_csv("data/external/manifest.csv")


@torch.no_grad()
def embed(model_fn, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        xs = []
        for p in paths[i:i + bs]:
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            o = (rgb.shape[0] - TILE) // 2; q = (rgb.shape[1] - TILE) // 2
            xs.append(torch.from_numpy(np.ascontiguousarray(rgb[o:o + TILE, q:q + TILE])).permute(2, 0, 1).float() / 255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model_fn(torch.stack(xs).cuda())["embedding"].float().cpu())
    return torch.cat(out).numpy()


def logo_probe(E, kind, rng_seed=0):
    """Leave-one-generator-out: train on the other generators' fakes + half of the real images, test on the held-out generator + the other half."""
    real = np.where(MAN.kind.values == "real")[0]
    rng = np.random.default_rng(rng_seed); rng.shuffle(real)
    halves = (real[: len(real) // 2], real[len(real) // 2:])
    gens = sorted(MAN[MAN.kind == kind].generator_or_tool.unique())
    aucs = []
    for g in gens:
        fake_g = np.where((MAN.kind.values == kind) & (MAN.generator_or_tool.values == g))[0]
        fake_o = np.where((MAN.kind.values == kind) & (MAN.generator_or_tool.values != g))[0]
        for tr_real, te_real in (halves, halves[::-1]):
            tr, te = np.r_[fake_o, tr_real], np.r_[fake_g, te_real]
            ytr, yte = (MAN.label.values[tr]), (MAN.label.values[te])
            sc = StandardScaler().fit(E[tr])
            clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1], cv=3, max_iter=2000, scoring="roc_auc").fit(sc.transform(E[tr]), ytr)
            aucs.append(roc_auc_score(yte, clf.predict_proba(sc.transform(E[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    paths = MAN.path.tolist()
    # ---- 1. real-image shift
    print("## 1. Do external real photos look different to the models than COCO-Forge authentic crops?\n")
    print("One 384 px window per image (no tiling, so the numbers are comparable). COCO-Forge authentic = test split. Threshold 0.5.\n")
    cf = pd.read_csv("data/forge/manifest.csv").query("split == 'test' and manipulation == 'authentic'")
    rows = []
    for v in ("single_a", "ml_hem", "cfr_adaptive"):
        fp_e, fp_c, ms_e, ms_c, auc_e = [], [], [], [], {"ai_edited": [], "ai_generated": []}
        for s in (0, 1, 2):
            run = f"runs/{v}_s{s}"
            z = np.load(f"{run}/external_crop_probs.npz"); key = "p_fused" if "p_fused" in z else "p_A"
            p = z[key]; real = MAN.kind.values == "real"
            fp_e.append((p[real] >= .5).mean()); ms_e.append(p[real].mean())
            t = np.load(f"{run}/probs.npz"); pk = "test_p_fused" if "test_p_fused" in t.files else [k for k in t.files if k.startswith("test_p_")][0]
            df = pd.read_csv("data/forge/manifest.csv").query("split == 'test'").reset_index(drop=True)
            ca = (df.manipulation == "authentic").values
            fp_c.append((t[pk][ca] >= .5).mean()); ms_c.append(t[pk][ca].mean())
            for k in auc_e:
                sel = real | (MAN.kind.values == k); auc_e[k].append(roc_auc_score(MAN.label.values[sel], p[sel]))
        rows.append({"variant": v, "FPR external real": round(np.mean(fp_e), 3), "FPR COCO-Forge authentic": round(np.mean(fp_c), 3),
                     "mean score external real": round(np.mean(ms_e), 3), "mean score COCO-Forge authentic": round(float(np.mean(ms_c)), 3),
                     "central-window AUROC edited": round(np.mean(auc_e["ai_edited"]), 3), "central-window AUROC generated": round(np.mean(auc_e["ai_generated"]), 3)})
    print(md(pd.DataFrame(rows)))

    # ---- 2. probes on frozen features
    print("\n## 2. Is the real/fake information in the frozen embeddings? Leave-one-generator-out probes on the central 384 px window\n")
    print("Mean AUROC over held-out generators and the two real-image halves (sd in brackets). Chance is 0.5. Trained head = the model's own output on the same window, for reference.\n")
    feats = {}
    pre = VisualBranch(2, pretrained=True).cuda().eval()
    feats["ImageNet EfficientNet-B0, never fine-tuned"] = embed(pre, paths); del pre
    for s in (0, 1, 2):
        mA, _ = load(f"runs/single_a_s{s}"); feats[f"Model A trained alone, seed {s}"] = embed(mA["A"], paths); del mA
    mB, _ = load("runs/single_b_s0"); feats["Model B trained alone, seed 0"] = embed(mB["B"], paths); del mB
    feats["A seed 0 + B seed 0 (concatenated)"] = np.c_[feats["Model A trained alone, seed 0"], feats["Model B trained alone, seed 0"]]
    rows = []
    for name, E in feats.items():
        r = {"features": name}
        for kind in ("ai_edited", "ai_generated"):
            m, sd = logo_probe(E, kind); r[f"{kind} probe AUROC"] = f"{m:.3f} ({sd:.3f})"
        rows.append(r)
    for v in ("single_a", "cfr_adaptive"):
        r = {"features": f"trained head: {v} (3 seeds, central window)"}
        for kind in ("ai_edited", "ai_generated"):
            r[f"{kind} probe AUROC"] = f"{np.mean([json.load(open(f'runs/{v}_s{s}/external_crop.json'))[f'{kind}_auroc'] for s in (0, 1, 2)]):.3f}"
        rows.append(r)
    print(md(pd.DataFrame(rows)))

    # ---- 3. edit area
    print("\n## 3. AI-edited images by size of the edited region (tiled score, seed-averaged probabilities)\n")
    ed = MAN[MAN.kind == "ai_edited"].copy()
    ed["area"] = [(cv2.imread(m, 0) > 127).mean() for m in ed.mask_path]
    ed["bin"] = pd.qcut(ed.area, 3, labels=["small", "medium", "large"])
    cuts = ed.groupby("bin", observed=True).area.agg(["min", "max"])
    print("Terciles of mask area: " + ", ".join(f"{b} {r['min']:.3f} to {r['max']:.3f}" for b, r in cuts.iterrows()) + " of the frame.\n")
    rows = []
    real = MAN.kind.values == "real"
    for v in ("single_a", "cfr_adaptive"):
        p = np.mean([np.load(f"runs/{v}_s{s}/external_probs.npz")["p_fused"] for s in (0, 1, 2)], 0)
        for b in ("small", "medium", "large"):
            idx = ed.index[ed.bin == b].values
            sel = np.r_[np.where(real)[0], idx]
            rows.append({"variant": v, "edit size": b, "n": len(idx), "AUROC vs real": round(roc_auc_score(MAN.label.values[sel], p[sel]), 3), "recall at 0.5": round(float((p[idx] >= .5).mean()), 3)})
    print(md(pd.DataFrame(rows)))

    # ---- 4. aggregation
    print("\n## 4. Tiled max over windows against one central window (3 seed mean AUROC)\n")
    rows = []
    for v in ("single_a", "cfr_adaptive"):
        r = {"variant": v}
        for tag, f in (("tiled max", "external.json"), ("central window", "external_crop.json")):
            for kind in ("ai_edited", "ai_generated"):
                r[f"{tag}, {kind}"] = round(np.mean([json.load(open(f"runs/{v}_s{s}/{f}"))[f"{kind}_auroc"] for s in (0, 1, 2)]), 3)
            r[f"{tag}, FPR real"] = round(np.mean([json.load(open(f"runs/{v}_s{s}/{f}"))["fpr_real"] for s in (0, 1, 2)]), 3)
        rows.append(r)
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
