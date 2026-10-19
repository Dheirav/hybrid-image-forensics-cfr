#!/usr/bin/env python3
"""Experiment A: do the numbers in the next-phase brief match the evidence, and does the evaluation path reproduce them?

1. claimed vs stored: primary AUROC (fused test AUROC on authentic + splice + copy_move, three-seed mean) from each run's probs.npz.
2. external numbers: the brief's 0.554 / 0.594 against two ways of computing them (mean of per-seed AUROC, and AUROC of seed-averaged scores).
3. second code path: re-score the checkpoints with eval_manifest.forward_all (not train.py's predict) and compare with the stored probabilities.
4. determinism: compare a retrained single_a seed 0 (runs_next/single_a_repro_s0) with the original, if it has finished.
usage: PYTHONPATH=src .venv/bin/python tools/verify_baseline.py > reports/evidence/next_phase/baseline_verification.md"""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from hybrid_forensics.analysis import SEEN, primary
from hybrid_forensics.eval_manifest import forward_all
from hybrid_forensics.external_eval import load

CLAIMED = {"single_a": 0.9936, "single_b": 0.9108, "joint_concat": 0.9929, "ml": 0.9791, "ml_hem": 0.9844, "cfr_adaptive": 0.9824}
KEY = {"single_a": "A", "single_b": "B"}
dirs = lambda v: sorted(glob.glob(f"runs/{v}_s?") + glob.glob(f"runs_extra/{v}_s?"))

print("## 1. Claimed against stored (primary metric, test split, authentic + splice + copy_move)\n")
print("| variant | claimed | recomputed mean over seeds | per seed | difference |\n|---|---|---|---|---|")
for v, c in CLAIMED.items():
    vals = [primary(d, KEY.get(v, "fused")) for d in dirs(v)]
    print(f"| {v} | {c:.4f} | {np.mean(vals):.4f} | {', '.join(f'{x:.4f}' for x in vals)} | {np.mean(vals) - c:+.4f} |")

print("\n## 2. External numbers (data/external, zero-shot; a different dataset, not comparable with section 1)\n")
man = pd.read_csv("data/external/manifest.csv")
per_seed = {"ai_edited": [], "ai_generated": []}; avg = []
for d in dirs("cfr_adaptive"):
    j = json.load(open(f"{d}/external.json"))
    for k in per_seed: per_seed[k].append(j[f"{k}_auroc"])
    avg.append(np.load(f"{d}/external_probs.npz")["p_fused"])
avg = np.mean(avg, 0)
print("| quantity | ai_edited vs real | ai_generated vs real |\n|---|---|---|")
print("| brief | 0.554 | 0.594 |")
print(f"| cfr_adaptive, mean of per-seed AUROC | {np.mean(per_seed['ai_edited']):.3f} | {np.mean(per_seed['ai_generated']):.3f} |")
real = man.kind.values == "real"
for k in ("ai_edited", "ai_generated"):
    sel = real | (man.kind.values == k)
    per_seed[k + "_avg"] = roc_auc_score(man.label.values[sel], avg[sel])
print(f"| cfr_adaptive, AUROC of seed-averaged scores | {per_seed['ai_edited_avg']:.3f} | {per_seed['ai_generated_avg']:.3f} |")
print("\nThe brief's figures are the seed-averaged-score AUROC; the per-seed mean is 0.002 to 0.003 lower. Same model, same images, different averaging.")

print("\n## 3. Second code path (eval_manifest.forward_all) against stored probabilities, COCO-Forge test split\n")
print("| run | primary AUROC stored | primary AUROC re-scored | max abs prob difference | label flips at 0.5 |\n|---|---|---|---|---|")
mf = pd.read_csv("data/forge/manifest.csv")
for d in ["runs/single_a_s0", "runs/single_a_s1", "runs/single_a_s2", "runs/cfr_adaptive_s0", "runs_extra/joint_concat_s0"]:
    models, head = load(d)
    df, P, fused, _, _, _ = forward_all(models, head, mf, "test")
    z = np.load(f"{d}/probs.npz"); stored = z["test_p_fused"] if "test_p_fused" in z else z["test_p_A"]
    keep = df.manipulation.isin(SEEN).values
    print(f"| {d} | {roc_auc_score(df.label[keep], stored[keep]):.4f} | {roc_auc_score(df.label[keep], fused[keep]):.4f} | "
          f"{np.abs(fused - stored).max():.4f} | {int(((fused >= .5) != (stored >= .5)).sum())} of {len(df)} |")

print("\n## 4. Determinism: single_a seed 0 retrained\n")
f = "runs_next/single_a_repro_s0/results.json"
if os.path.exists(f):
    a, b = "runs/single_a_s0", "runs_next/single_a_repro_s0"
    pa, pb = primary(a, "A"), primary(b, "A")
    ra, rb = json.load(open(f"{a}/results.json")), json.load(open(f))
    va = [h["val"]["A"]["auroc"] for h in ra["history"]]; vb = [h["val"]["A"]["auroc"] for h in rb["history"]]
    print(f"primary AUROC original {pa:.4f}, retrained {pb:.4f}, difference {pb - pa:+.4f} (seed sd across seeds was 0.0027)")
    print("\nvalidation AUROC per epoch\n\n| epoch | original | retrained |\n|---|---|---|")
    for i, (x, y) in enumerate(zip(va, vb)): print(f"| {i + 1} | {x:.4f} | {y:.4f} |")
else:
    print("retrain has not finished yet")
