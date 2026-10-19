"""Build the 'COCO-Forge' manipulation dataset from the fetched COCO subset.

Every source image yields a pair: the untouched crop (authentic) and one manipulated copy made from
the *same crop window*, so the pair differs only by the manipulation. Both are written as JPEG with a
quality drawn independently of the label, so compression is not a shortcut. Splits are by source image;
a splice donor always comes from the same split, so no pixels cross a split boundary.

Manipulations (all synthetic and conventional; there are no AI-edited or AI-generated images here):
  splice     object from another source pasted into this crop
  copy_move  this crop's own object copied to another place
  removal    this crop's object inpainted away (cv2 Telea). Held out of training as an unseen tool.
"""
import json
import os
import random

import cv2
import numpy as np
import pandas as pd

CROP = 384
JPEG_Q = (75, 85, 95)
SPLIT_SIZES = {"train": 1300, "val": 250, "test": 250, "ood_removal": 200}
TYPES_IN_DIST = ("splice", "copy_move")


def _crop_window(rec, rng):
    """Random CROP x CROP window that keeps the chosen object's box inside, with an 8 px margin."""
    def axis(start, size, total):
        lo, hi = max(0, int(start + size) + 8 - CROP), min(int(start) - 8, total - CROP)
        return rng.randint(lo, hi) if lo <= hi else min(max(0, int(start) - 20), total - CROP)
    x, y, w, h = rec["bbox"]
    return axis(x, w, rec["width"]), axis(y, h, rec["height"])


def _load(rec, raw_dir, rng):
    img = cv2.imread(os.path.join(raw_dir, "images", rec["file"]))
    x0, y0 = _crop_window(rec, rng)
    crop = img[y0:y0 + CROP, x0:x0 + CROP].copy()
    seg = [[v - (x0 if i % 2 == 0 else y0) for i, v in enumerate(p)] for p in rec["segmentation"]]
    full = np.zeros((CROP, CROP), np.uint8)
    for poly in seg:
        cv2.fillPoly(full, [np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return crop, full


def _paste(dst, src, src_mask, rng, avoid=None):
    """Paste src's masked object into dst at a random place; returns new image and binary mask."""
    ys, xs = np.nonzero(src_mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch, pm = src[y1:y2, x1:x2], src_mask[y1:y2, x1:x2]
    s = rng.uniform(0.8, 1.2)
    nw, nh = max(8, int(patch.shape[1] * s)), max(8, int(patch.shape[0] * s))
    if nw >= CROP - 4 or nh >= CROP - 4:
        nw, nh = patch.shape[1], patch.shape[0]
    patch, pm = cv2.resize(patch, (nw, nh)), cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)
    for _ in range(50):
        px, py = rng.randint(0, CROP - nw), rng.randint(0, CROP - nh)
        place = np.zeros((CROP, CROP), np.uint8)
        place[py:py + nh, px:px + nw] = pm
        if avoid is None or (place & avoid).sum() == 0:
            break
    full = np.zeros_like(dst)
    full[py:py + nh, px:px + nw] = patch
    alpha = cv2.GaussianBlur(place.astype(np.float32), (5, 5), 1.0)[..., None]  # feathered edge, like a careful editor
    out = (dst * (1 - alpha) + full * alpha).round().astype(np.uint8)
    return out, place


def build(raw_dir="data/coco_raw", out_dir="data/forge", seed=0):
    recs = json.load(open(os.path.join(raw_dir, "selection.json")))
    recs = [r for r in recs if os.path.exists(os.path.join(raw_dir, "images", r["file"]))]
    rng = random.Random(seed)
    rng.shuffle(recs)
    need = sum(SPLIT_SIZES.values())
    if len(recs) < need:
        scale = len(recs) / need
        sizes = {k: int(v * scale) for k, v in SPLIT_SIZES.items()}
    else:
        sizes = dict(SPLIT_SIZES)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    rows, start = [], 0
    for split, n in sizes.items():
        pool = recs[start:start + n]
        start += n
        for i, r in enumerate(pool):
            img, obj = _load(r, raw_dir, rng)
            mtype = "removal" if split == "ood_removal" else TYPES_IN_DIST[i % 2] if split != "test" else ("splice", "copy_move", "removal")[i % 3]
            if mtype == "splice":
                donor = pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)]
                dimg, dobj = _load(donor, raw_dir, rng)
                man, mask = _paste(img, dimg, dobj, rng)
                donor_id = donor["image_id"]
            elif mtype == "copy_move":
                man, mask = _paste(img, img, obj, rng, avoid=obj)
                donor_id = None
            else:
                grown = cv2.dilate(obj, np.ones((7, 7), np.uint8))
                man, mask, donor_id = cv2.inpaint(img, grown, 3, cv2.INPAINT_TELEA), grown, None
            for label, arr, kind, m in ((0, img, "authentic", None), (1, man, mtype, mask)):
                q = rng.choice(JPEG_Q)
                name = f"{r['image_id']:012d}_{kind}"
                path = os.path.join(out_dir, "images", name + ".jpg")
                cv2.imwrite(path, arr, [cv2.IMWRITE_JPEG_QUALITY, q])
                mpath = ""
                if m is not None:
                    mpath = os.path.join(out_dir, "masks", name + ".png")
                    cv2.imwrite(mpath, m * 255)
                rows.append({"path": path, "source_id": r["image_id"], "label": label, "mask_path": mpath,
                             "generator_or_tool": {"authentic": "none", "splice": "paste_feathered",
                                                   "copy_move": "paste_feathered", "removal": "cv2_telea"}[kind],
                             "manipulation": kind, "format": "jpeg", "jpeg_quality": q, "dimensions": f"{CROP}x{CROP}",
                             "split": split, "donor_source_id": donor_id, "coco_license": r["license"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby(["split", "manipulation"]).size())
