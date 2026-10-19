#!/usr/bin/env python3
"""Experiment B: leakage, integrity and shortcut checks for data/forge_hard, per tool.
The set has one authentic crop and five edits per source, so the pair-completeness check of audit_dataset does not apply and is replaced by per-tool checks.
usage: PYTHONPATH=src .venv/bin/python tools/audit_hard.py > reports/evidence/next_phase/audit_hard.md"""
import os, sys
import cv2, numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from hybrid_forensics.audit_dataset import dhash, patch_feats
from hybrid_forensics.diag_fusion import md

H = pd.read_csv("data/forge_hard/manifest.csv"); C = pd.read_csv("data/forge/manifest.csv")
H["file_size"] = [os.path.getsize(p) for p in H.path]
rng = np.random.default_rng(0)

print("## Leakage and integrity\n")
unread = [p for p in H.path if cv2.imread(p) is None] + [m for m in H.mask_path.dropna() if m and cv2.imread(m, 0) is None]
src = {s: set(g.source_id) for s, g in H.groupby("split")}
tr_src = set(C[C.split == "train"].source_id)
print(f"- unreadable images or masks: {len(unread)} of {len(H)} images")
print(f"- source overlap forge_hard val with test: {len(src['val'] & src['test'])}; with COCO-Forge train: {len((src['val'] | src['test']) & tr_src)}")
cf_split = {s: set(g.source_id) for s, g in C.groupby("split")}
print(f"- forge_hard val sources equal COCO-Forge val sources: {src['val'] == cf_split['val']}; test equal COCO-Forge test: {src['test'] == cf_split['test']}")
don = H.dropna(subset=["donor_source_id"])
print(f"- splice donors outside the split of the image they were pasted into: {sum(int(d) not in src[s] for s, d in zip(don.split, don.donor_source_id))}")
print(f"- manipulated rows without a mask: {int(((H.label == 1) & (H.mask_path.isna() | (H.mask_path == ''))).sum())}; authentic rows with a mask: {int(((H.label == 0) & H.mask_path.notna() & (H.mask_path != '')).sum())}")
tr_au = C[(C.split == "train") & (C.label == 0)]
Ht = np.array([dhash(cv2.imread(p)) for p in tr_au.path])
Hh = np.array([dhash(cv2.imread(p)) for p in H[H.label == 0].path])
d = (Hh[:, None, :] != Ht[None, :, :]).sum(2)
print(f"- near-duplicate pairs (64-bit dHash Hamming at most 6) between forge_hard authentic crops and COCO-Forge TRAIN authentic crops: {int((d <= 6).sum())}")

print("\n## Per-tool integrity: mask size and where pixels changed (test split, against the same source's authentic crop)\n")
t = H[H.split == "test"]; rows = []
for tool in sorted(set(t.manipulation) - {"authentic"}):
    a_in, a_out, ar, md_in = [], [], [], []
    for r in t[t.manipulation == tool].itertuples():
        au = cv2.imread(t[(t.source_id == r.source_id) & (t.label == 0)].path.iloc[0]).astype(int); x = cv2.imread(r.path).astype(int)
        m = cv2.imread(r.mask_path, 0) > 127; diff = np.abs(x - au).max(2)
        grown = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        a_in.append((diff[m] > 30).mean()); a_out.append((diff[~grown] > 30).mean()); ar.append(m.mean()); md_in.append(diff[m].mean())
    rows.append({"tool": tool, "n": len(ar), "mask area median": round(float(np.median(ar)), 3), "changed >30 inside mask": round(float(np.mean(a_in)), 3),
                 "mean abs change inside mask": round(float(np.mean(md_in)), 1), "changed >30 far outside": round(float(np.mean(a_out)), 4)})
print(md(pd.DataFrame(rows)))

print("\n## Compression and metadata shortcuts\n")
ct = pd.crosstab(H.label, H.jpeg_quality)
print(f"JPEG quality against label, chi-square p = {chi2_contingency(ct)[1]:.2f} (quality is drawn independently of the label).\n")
va, te = H[H.split == "val"], H[H.split == "test"]
rows = []
for tool in sorted(set(te.manipulation) - {"authentic"}):
    sel = lambda d: d[d.manipulation.isin(["authentic", tool])]
    g = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(sel(va)[["jpeg_quality", "file_size"]], sel(va).label)
    rows.append({"tool": tool, "file size alone (test AUROC)": round(roc_auc_score(sel(te).label, sel(te).file_size), 3),
                 "quality + size, fitted on forge_hard val": round(roc_auc_score(sel(te).label, g.predict_proba(sel(te)[["jpeg_quality", "file_size"]])[:, 1]), 3)})
print(md(pd.DataFrame(rows)))

print("\n## Local-statistics baseline: 32 x 32 patch features, fitted on forge_hard VALIDATION (all tools pooled), scored on test per tool\n")
print("A hand-built detector, not a deep model. `inside` takes patches from the edited region (a positive control: can a simple feature see this edit at all), `outside` takes patches at least 15 px from it (should sit at 0.5 if the edit is local).\n")
def patches(sub, where, k=6, s=32):
    X, Y, I = [], [], []
    for r in sub.itertuples():
        im = cv2.imread(r.path); mk = cv2.imread(r.mask_path, 0) > 127 if isinstance(r.mask_path, str) and r.mask_path else None
        far = None if mk is None else ~cv2.dilate(mk.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        for _ in range(k):
            for _try in range(40):
                y, x = rng.integers(0, 384 - s, 2)
                if where == "outside" and (far is None or far[y:y + s, x:x + s].all()): break
                if where == "inside" and r.label == 1 and mk[y:y + s, x:x + s].mean() > 0.8: break
                if where == "inside" and r.label == 0: break
            else:
                continue
            X.append(patch_feats(im[y:y + s, x:x + s])); Y.append(r.label); I.append(r.path)
    return np.array(X), np.array(Y), np.array(I)
rows = {tool: {"tool": tool} for tool in sorted(set(te.manipulation) - {"authentic"})}
for where in ("inside", "outside"):
    Xv, Yv, _ = patches(va, where)
    g = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xv, Yv)
    for tool in rows:
        sub = te[te.manipulation.isin(["authentic", tool])]
        Xt, Yt, It = patches(sub, where)
        p = pd.Series(g.predict_proba(Xt)[:, 1]).groupby(It).mean(); y = pd.Series(Yt).groupby(It).first()
        rows[tool][f"{where} patches, image AUROC"] = round(roc_auc_score(y, p), 3)
print(md(pd.DataFrame(rows.values())))
