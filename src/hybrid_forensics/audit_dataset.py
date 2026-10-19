"""Dataset integrity, leakage, alignment, balance and shortcut audit. Writes reports/evidence/dataset_audit.json."""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def patch_feats(p):
    f = p.astype(np.float32)
    g = f.mean(2)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx, gy = np.diff(g, axis=1), np.diff(g, axis=0)
    return np.r_[f.mean((0, 1)), f.std((0, 1)), np.abs(lap).mean(), lap.var(), np.abs(gx).mean(), np.abs(gy).mean(),
                 np.abs(np.diff(g[:, ::2], axis=1)).mean()]


def main(manifest="data/forge/manifest.csv", out="reports/evidence/dataset_audit.json", seed=0):
    df = pd.read_csv(manifest)
    R = {"n_images": len(df)}
    rng = np.random.default_rng(seed)

    # 1. files, dimensions, masks
    bad, dims, cover = [], set(), {}
    imgs, masks = {}, {}
    for r in df.itertuples():
        im = cv2.imread(r.path)
        if im is None:
            bad.append(r.path); continue
        dims.add(im.shape[:2])
        imgs[r.path] = im
        if isinstance(r.mask_path, str) and r.mask_path:
            m = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != im.shape[:2]:
                bad.append(r.mask_path); continue
            masks[r.path] = m > 127
            cover.setdefault(r.manipulation, []).append(float(masks[r.path].mean()))
        elif r.label == 1:
            bad.append("missing mask: " + r.path)
    R["unreadable_or_missing"] = bad
    R["image_shapes"] = sorted(map(str, dims))
    R["mask_area_fraction"] = {k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}
                               for k, v in cover.items()}
    R["empty_masks"] = int(sum(v == 0 for vs in cover.values() for v in vs))
    R["authentic_with_mask"] = int(((df.label == 0) & df.mask_path.notna() & (df.mask_path != "")).sum())

    # 2. split separation, pair completeness, donors
    sets = {s: set(g.source_id) for s, g in df.groupby("split")}
    R["source_overlap_between_splits"] = {f"{a}&{b}": len(sets[a] & sets[b]) for a in sets for b in sets if a < b}
    donors = df.dropna(subset=["donor_source_id"])
    R["splice_donor_in_other_split"] = int(sum(int(d) not in sets[s] for s, d in zip(donors.split, donors.donor_source_id)))
    per_src = df.groupby("source_id").label.agg(["sum", "count"])
    R["sources_not_exactly_one_authentic_one_manipulated"] = int(((per_src["sum"] != 1) | (per_src["count"] != 2)).sum())
    R["counts"] = {f"{s}/{m}": int(n) for (s, m), n in df.groupby(["split", "manipulation"]).size().items()}
    R["class_balance_by_split"] = {s: float(g.label.mean()) for s, g in df.groupby("split")}

    # 3. near duplicates across splits (authentic images only, 64-bit dHash)
    au = df[df.label == 0].reset_index(drop=True)
    H = np.array([dhash(imgs[p]) for p in au.path])
    d = (H[:, None, :] != H[None, :, :]).sum(2)
    sp = au.split.values
    cross = (sp[:, None] != sp[None, :]) & np.triu(np.ones_like(d, bool), 1)
    R["near_duplicate_pairs_across_splits_hamming_le_6"] = int(((d <= 6) & cross).sum())
    R["near_duplicate_pairs_within_splits_hamming_le_6"] = int(((d <= 6) & ~cross & np.triu(np.ones_like(d, bool), 1)).sum())

    # 4. mask alignment: pixels differ from the authentic twin inside the mask, not outside it
    al = {}
    for r in df[df.label == 1].itertuples():
        a = imgs[r.path.replace(f"_{r.manipulation}.jpg", "_authentic.jpg")]
        diff = np.abs(a.astype(np.int16) - imgs[r.path].astype(np.int16)).max(2)
        m = masks[r.path]
        grown = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        al.setdefault(r.manipulation, []).append(
            [float((diff[m] > 30).mean()), float((diff[~grown] > 30).mean()), float(diff[m].mean()), float(diff[~grown].mean())])
    R["mask_alignment"] = {k: {"frac_changed_inside_mask": float(np.mean([x[0] for x in v])),
                               "frac_changed_far_outside_mask": float(np.mean([x[1] for x in v])),
                               "mean_diff_inside": float(np.mean([x[2] for x in v])),
                               "mean_diff_far_outside": float(np.mean([x[3] for x in v])),
                               "images_with_>1pct_change_far_outside": int(sum(x[1] > 0.01 for x in v))} for k, v in al.items()}

    # 5. compression and size by class
    df["file_size"] = [os.path.getsize(p) for p in df.path]
    ct = pd.crosstab(df.label, df.jpeg_quality)
    R["jpeg_quality_by_class"] = ct.to_dict()
    R["jpeg_quality_vs_label_chi2_p"] = float(chi2_contingency(ct)[1])

    # 6. shortcut baselines, trained on train and scored on val
    tr, va = df[df.split == "train"], df[df.split == "val"]
    meta = ["jpeg_quality", "file_size"]
    m = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(tr[meta], tr.label)
    R["shortcut_metadata_only_val_auroc"] = float(roc_auc_score(va.label, m.predict_proba(va[meta])[:, 1]))
    m = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(tr[["jpeg_quality"]], tr.label)
    R["shortcut_jpeg_quality_only_val_auroc"] = float(roc_auc_score(va.label, m.predict_proba(va[["jpeg_quality"]])[:, 1]))

    def patches(sub, where, k=6, s=32):
        X, Y, I = [], [], []
        for r in sub.itertuples():
            im, mk = imgs[r.path], masks.get(r.path)
            far = None if mk is None else ~cv2.dilate(mk.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            for _ in range(k):
                for _try in range(40):
                    y, x = rng.integers(0, 384 - s, 2)
                    inside = None if mk is None else mk[y:y + s, x:x + s].mean()
                    if where == "outside" and (far is None or far[y:y + s, x:x + s].all()):
                        break
                    if where == "inside":
                        if r.label == 0 or inside > 0.8: break
                else:
                    continue
                X.append(patch_feats(im[y:y + s, x:x + s])); Y.append(r.label); I.append(r.path)
        return np.array(X), np.array(Y), np.array(I)

    for where in ("outside", "inside"):
        Xt, Yt, _ = patches(tr, where); Xv, Yv, Iv = patches(va, where)
        g = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xt, Yt)
        p = g.predict_proba(Xv)[:, 1]
        img_p = pd.Series(p).groupby(Iv).mean()
        img_y = pd.Series(Yv).groupby(Iv).first()
        R[f"patch32_{where}_mask_val_auroc_patch"] = float(roc_auc_score(Yv, p))
        R[f"patch32_{where}_mask_val_auroc_image"] = float(roc_auc_score(img_y, img_p))
    R["note"] = ("'outside' = manipulated-image patches taken >=15 px from the mask vs authentic patches; "
                 "should sit near 0.5 if the only difference is the edit. 'inside' is a positive control.")
    json.dump(R, open(out, "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
