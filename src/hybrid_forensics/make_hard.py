"""Build 'forge_hard': new editing tools applied to the SAME val/test source images as COCO-Forge.

Why it exists: Model A scores 0.994 on COCO-Forge, so that benchmark cannot separate any two methods and cannot say what
Model A is reading. Each source image here yields one authentic crop and one edit per tool, so every tool is compared against
the same authentic images and differs from the others only by how the edit was made.

Tools (none of them is the feathered alpha paste used for COCO-Forge training):
  splice_feather     control: the training tool, rebuilt through this pipeline. It checks that the new pipeline did not change difficulty.
  copy_move_feather  control, same reason.
  splice_poisson     donor object blended by gradient-domain (Poisson) cloning. Same semantics as splice_feather, no hard seam.
  copy_move_poisson  own object cloned elsewhere with Poisson blending.
  self_resample      the crop's own object re-pasted at the SAME place, 1.08x larger. Scene content is unchanged (no object appears
                     where it does not belong), so a detector that only reads "this does not belong" has nothing to find. What stays is
                     resampling and edge traces. This is the content-matched test.
Splits are the existing COCO-Forge val and test source ids, so nothing here touches a training source. Crops are drawn afresh.
"""
import json
import os
import random

import cv2
import numpy as np
import pandas as pd

from .make_dataset import CROP, JPEG_Q, _load

TOOLS = ("splice_feather", "copy_move_feather", "splice_poisson", "copy_move_poisson", "self_resample")


def _bbox_patch(src, src_mask):
    ys, xs = np.nonzero(src_mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return src[y1:y2, x1:x2], src_mask[y1:y2, x1:x2], (y1 + y2) // 2, (x1 + x2) // 2


def _resize_patch(patch, pm, s):
    nw, nh = max(8, int(patch.shape[1] * s)), max(8, int(patch.shape[0] * s))
    if nw >= CROP - 8 or nh >= CROP - 8:
        nw, nh = patch.shape[1], patch.shape[0]
    return cv2.resize(patch, (nw, nh)), cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)


def _random_place(nw, nh, rng, avoid):
    for _ in range(50):
        px, py = rng.randint(3, CROP - nw - 3), rng.randint(3, CROP - nh - 3)
        place = np.zeros((CROP, CROP), np.uint8)
        place[py:py + nh, px:px + nw] = 1
        if avoid is None or (place & avoid).sum() == 0:
            break
    return px, py


def poisson_paste(dst, src, src_mask, rng, avoid=None):
    """Poisson-blended paste. The returned mask is the pasted object mask; pixels outside it are untouched."""
    patch, pm, _, _ = _bbox_patch(src, src_mask)
    patch, pm = _resize_patch(patch, pm, rng.uniform(0.8, 1.2))
    nh, nw = pm.shape
    px, py = _random_place(nw, nh, rng, avoid)
    # cv2.seamlessClone wants a mask that does not touch the patch border, so the mask is padded by a 1 px frame inside the patch box
    m = (pm * 255).astype(np.uint8)
    m[0, :] = m[-1, :] = 0; m[:, 0] = m[:, -1] = 0
    cloned = cv2.seamlessClone(patch, dst, m, (px + nw // 2, py + nh // 2), cv2.NORMAL_CLONE)
    place = np.zeros((CROP, CROP), np.uint8)
    place[py:py + nh, px:px + nw] = (m > 0)
    # seamlessClone also alters pixels well outside the mask (about as many as inside it), which would make the
    # ground-truth mask wrong; only the masked pixels are taken from the clone
    return np.where(place[..., None] > 0, cloned, dst), place


def feather_paste(dst, src, src_mask, rng, avoid=None):
    """The COCO-Forge training tool, kept here as a control (same alpha feathering as make_dataset._paste)."""
    patch, pm, _, _ = _bbox_patch(src, src_mask)
    patch, pm = _resize_patch(patch, pm, rng.uniform(0.8, 1.2))
    nh, nw = pm.shape
    px, py = _random_place(nw, nh, rng, avoid)
    place = np.zeros((CROP, CROP), np.uint8); place[py:py + nh, px:px + nw] = pm
    full = np.zeros_like(dst); full[py:py + nh, px:px + nw] = patch
    alpha = cv2.GaussianBlur(place.astype(np.float32), (5, 5), 1.0)[..., None]
    return (dst * (1 - alpha) + full * alpha).round().astype(np.uint8), place


def self_resample(img, obj, scale=1.08):
    """Re-paste the object over itself, slightly larger and centred where it was. No new content, only resampling and edge traces."""
    patch, pm, cy, cx = _bbox_patch(img, obj)
    nw, nh = int(round(patch.shape[1] * scale)), int(round(patch.shape[0] * scale))
    patch, pm = cv2.resize(patch, (nw, nh)), cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)
    py = int(np.clip(cy - nh // 2, 0, CROP - nh)); px = int(np.clip(cx - nw // 2, 0, CROP - nw))
    place = np.zeros((CROP, CROP), np.uint8); place[py:py + nh, px:px + nw] = pm
    full = np.zeros_like(img); full[py:py + nh, px:px + nw] = patch
    alpha = cv2.GaussianBlur(place.astype(np.float32), (5, 5), 1.0)[..., None]
    return (img * (1 - alpha) + full * alpha).round().astype(np.uint8), place


def build(raw_dir="data/coco_raw", out_dir="data/forge_hard", base_manifest="data/forge/manifest.csv", seed=7, splits=("val", "test")):
    recs = {r["image_id"]: r for r in json.load(open(os.path.join(raw_dir, "selection.json")))}
    base = pd.read_csv(base_manifest)
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    rows = []
    for split in splits:
        ids = sorted(base[(base.split == split) & (base.label == 0)].source_id.unique())
        pool = [recs[i] for i in ids]
        for i, r in enumerate(pool):
            img, obj = _load(r, raw_dir, rng)
            donor = pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)]
            dimg, dobj = _load(donor, raw_dir, rng)
            variants = [("authentic", img, None, None)]
            for tool in TOOLS:
                if tool == "splice_feather": man, m = feather_paste(img, dimg, dobj, rng); d = donor["image_id"]
                elif tool == "splice_poisson": man, m = poisson_paste(img, dimg, dobj, rng); d = donor["image_id"]
                elif tool == "copy_move_feather": man, m = feather_paste(img, img, obj, rng, avoid=obj); d = None
                elif tool == "copy_move_poisson": man, m = poisson_paste(img, img, obj, rng, avoid=obj); d = None
                else: man, m = self_resample(img, obj); d = None
                variants.append((tool, man, m, d))
            for kind, arr, m, d in variants:
                q = rng.choice(JPEG_Q)
                name = f"{r['image_id']:012d}_{kind}"
                path = os.path.join(out_dir, "images", name + ".jpg")
                cv2.imwrite(path, arr, [cv2.IMWRITE_JPEG_QUALITY, q])
                mpath = ""
                if m is not None:
                    mpath = os.path.join(out_dir, "masks", name + ".png")
                    cv2.imwrite(mpath, m * 255)
                rows.append({"path": path, "source_id": r["image_id"], "label": int(kind != "authentic"), "mask_path": mpath,
                             "generator_or_tool": kind, "manipulation": kind, "format": "jpeg", "jpeg_quality": q,
                             "dimensions": f"{CROP}x{CROP}", "split": split, "donor_source_id": d, "coco_license": r["license"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby(["split", "manipulation"]).size())
